# Copyright (c) 2023, Tri Dao, Albert Gu.

# Copyright (c) 2024, Vida Vakilotojar.
# Adds support for multi-head SSM.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads: int=1,
        scalar_dt: bool=False,
        dense_matrices: bool=True,
        convolved_v: bool=True,
        complementary_b: bool=False, # input = I - forget
        multi_head_proj: bool=False,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.n_heads = n_heads
        self.scalar_dt = scalar_dt
        self.dense_matrices = dense_matrices
        self.convolved_v = convolved_v
        self.complementary_b = complementary_b
        self.multi_head_proj = multi_head_proj and n_heads > 1

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        if self.scalar_dt:
            assert self.dt_rank == 1
        assert self.d_inner % self.n_heads == 0
        assert self.d_state % self.n_heads == 0

        self.head_d_inner = self.d_inner // self.n_heads
        self.head_dt_rank = max(1, self.dt_rank // self.n_heads)
        if self.scalar_dt:
            self.head_dt_rank = 1
        self.head_d_state = self.d_state // self.n_heads
        self.x_proj_output_size = self.head_d_state * 2 + self.head_dt_rank

        # x_proj is columnar block-diagonal when multi-head
        self.x_proj = nn.Linear(
            self.head_d_inner, self.x_proj_output_size * self.n_heads, bias=False, **factory_kwargs
        )

        # dt_proj is columnar block-diagonal when multi-head, even when we have dt_scalar.
        if not self.scalar_dt:
            self.dt_proj = nn.Linear(self.head_dt_rank, self.d_inner, bias=True, **factory_kwargs)
        else:
            self.dt_proj = nn.Linear(self.head_dt_rank, self.n_heads, bias=True, **factory_kwargs) # 1 x n_heads

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.head_dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # SoftPlus(x) = log(1 + exp(x)) is smooth approximation of ReLU
        # expm1 = SoftPlus Inverse: SoftPlus_Inverse(x) = log(exp(x) - 1).
        # Inverse of SoftPlus only defined for (0, infinity)

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.dt_proj.bias.shape[-1], **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt)) # log(1 - exp(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        if not self.scalar_dt:
            A = repeat(
                torch.arange(1, self.head_d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous() # d_inner x head_d_state
        else:
            A = repeat(torch.arange(1, 2, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.n_heads,
            ).contiguous()  # n_heads x 1

        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        if not self.multi_head_proj:
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        else:
            self.out_proj = nn.Linear(self.head_d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, in_state: Optional[Tensor] = None, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, head_d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
            out, out_state = mamba_inner_fn(
                xz,
                in_state,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                head_d_state=self.head_d_state,
                n_heads=self.n_heads,
                scalar_dt=self.scalar_dt,
                dense_matrices=self.dense_matrices, 
                convolved_v=self.convolved_v,
                multi_head_proj=self.multi_head_proj
            )
        else:
            x, z = xz.chunk(2, dim=1)

            if not self.convolved_v:
                pre_conv_x = x # bdl
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.

            # x_proj is columnar block-diagonal
            x_dbl = torch.einsum("hod,shd->sho",
                        rearrange(self.x_proj.weight, "(h d_o_h) d -> h d_o_h d", h=self.n_heads),
                        rearrange(x, "b (h d) l -> (b l) h d", h=self.n_heads)) # (bl h d)
            dt, B, C = torch.split(x_dbl, [self.head_dt_rank, self.head_d_state, self.head_d_state], dim=-1) # (bl h d)

            # dt_proj is columnar block-diagonal
            dt = rearrange(
                    torch.einsum("hod,shd->sho",
                        rearrange(self.dt_proj.weight, "(h d_o_h) d -> h d_o_h d", h=self.n_heads),
                        dt), # h d (bl)
                        "(b l) h d -> b (h d) l", l=seqlen) # b (hd) l
            B = rearrange(B, "(b l) h dstate -> b (h dstate) l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) h dstate -> b (h dstate) l", l=seqlen).contiguous()

            if self.scalar_dt and self.dense_matrices:
                # resize dt/dt_bias/A to the standard size expected
                dt = repeat(dt, "b h l -> b (h d) l", d=self.head_d_inner)
                A = repeat(A, "h1 h2 -> (h1 d) (h2 n)", d=self.head_d_inner, n=self.head_d_state)
                dt_bias = repeat(dt_bias, "h -> (h d)", d=self.head_d_inner)

            assert self.activation in ["silu", "swish"]
            # At this point, dt/B/C have nice dimensions for any head_size.
            # But x/A/delta_bias have dimensions that have to be carefully rearranged in
            # subsequent functions
            y = selective_scan_fn(
                x if (self.convolved_v) else pre_conv_x,  # bdl
                in_state,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
                head_d_state=self.head_d_state,
                n_heads=self.n_heads,
                scalar_dt=self.scalar_dt and not self.dense_matrices
            )

            # if ssm_state is not None:   # from now on, we always return the last_state!
            y, out_state = y
            if ssm_state is not None:
                ssm_state.copy_(out_state)  # write over last_state into the cache
            if not self.multi_head_proj:
                y = rearrange(y, "b d l -> b l d")
                out = self.out_proj(y)
            else:
                out = rearrange(torch.einsum("shd,hod->sho",
                                        rearrange(y, "b (h d) l -> (b l) h d", h=self.n_heads),
                                        rearrange(self.out_proj.weight, "(h d_o_h) d -> h d_o_h d", h=self.n_heads)), # (bl h d)
                    "(b l) h d -> b l (h d)", l=seqlen)
                if self.out_proj.bias is not None:
                    out = out + self.out_proj.bias
        return out, out_state

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        if not self.convolved_v:
            pre_conv_x = x # bdl

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = torch.einsum("hod,bhd->bho",
                    rearrange(self.x_proj.weight, "(h d_o_h) d -> h d_o_h d", h=self.n_heads),
                    rearrange(x, "b (h d) -> b h d", h=self.n_heads)) # (b h d)

        dt, B, C = torch.split(x_db, [self.head_dt_rank, self.head_d_state, self.head_d_state], dim=-1) # (b h d)
        # Don't add dt_bias here

        dt = rearrange(torch.einsum("hod,bhd->bho",
                                rearrange(self.dt_proj.weight, "(h d_o_h) d -> h d_o_h d", h=self.n_heads), dt),
                        "b h d -> b (h d") # b (h d)
        B = rearrange(B, "b h d -> b (h d)") # b (h d)
        C = rearrange(C, "b h d -> b (h d)") # b (h d)
        A = -torch.exp(self.A_log.float())  # (d_inner, head_d_state)

        if self.scalar_dt:
            # resize dt/dt_bias/A to the standard size expected
            dt = repeat(dt, "b h -> b (h d)", d=self.head_d_inner)
            A = repeat(A, "h1 h2 -> (h1 d) (h2 n)", d=self.head_d_inner, n=self.head_d_state)
            dt_bias = repeat(dt_bias, "h -> (h d)", d=self.head_d_inner)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype)) # bd/b(hd)
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = rearrange(torch.einsum('bhd,bhn->bhdn', 
                                rearrange(dt, "b (h d) -> b h d", h=self.n_heads), 
                                rearrange(B, "b (h n) -> b h n", h=self.n_heads)), 'b h d n->b (h d) n')
            ssm_state.copy_(ssm_state * dA + 
                            rearrange(x if (self.convolved_v) else pre_conv_x, "b d -> b d 1") * dB)
            # For multi-head, ssm_state is columnar block-diagonal and multiplying
            # it with C is special, because ssm_state is b(h d/h)(n/h), and C is b(h n/h)
            y = rearrange(
                torch.einsum("bhdn,bhn->bhd",
                    rearrange(ssm_state.to(dtype), "b (h d) n -> b h d n", h=self.n_heads),
                    rearrange(C, "b (h n) -> b h n", h=self.n_heads)),
                    "b h d -> b (h d)"
            )
            y = y + self.D.to(dtype) * x if (self.convolved_v) else pre_conv_x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, 
                x if (self.convolved_v) else pre_conv_x, 
                dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True,
                n_heads=self.n_heads
            )

        if not self.multi_head_proj:
            out = self.out_proj(y)
        else:
            out = rearrange(torch.einsum("bhd,hod->bho",
                                rearrange(y, "b (h d) -> b h d", h=self.n_heads),
                                rearrange(self.out_proj.weight, "(h d_o_h) d -> h d_o_h d", h=self.n_heads)), # (bl h d)
                         "b h d -> b (h d)")
            if self.out_proj.bias is not None:
                out = out + self.out_proj.bias
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.head_d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.head_d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, in_state: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states, out_state = self.mixer(hidden_states, in_state, inference_params=inference_params)
        return hidden_states, residual, out_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
