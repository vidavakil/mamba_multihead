# Copyright (c) 2023, Tri Dao, Albert Gu.

import torch
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn
    import causal_conv1d_cuda
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_cuda = None

import selective_scan_cuda

class SelectiveScanFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                return_last_state=False, 
                head_d_state=0, n_heads=1, scalar_dt=False):

        is_variable_B = B.dim() >= 3

        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            ctx.squeeze_C = True

        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus,
                                                n_heads, head_d_state, scalar_dt)
        ctx.delta_softplus = delta_softplus

        ctx.n_heads = n_heads
        ctx.head_d_state = head_d_state
        ctx.scalar_dt = scalar_dt

        ctx.has_z = z is not None
        last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)

        # The cuda kernel does a peculiar optimization of not multiplying the state 
        # by B if B is not variable! This does not impact MambaInnerFn, because it
        # never returns the state. But SelectiveScanFn may needd to return the 
        # last state! Hence the following is needed.
        if not is_variable_B:
            last_state = torch.einsum('bdn,dn->bdn', last_state, B)

        if not ctx.has_z:
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out if not return_last_state else (out, last_state)
        else:
            ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
            out_z = rest[0]
            return out_z if not return_last_state else (out_z, last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        if not ctx.has_z:
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            z = None
            out = None
        else:
            u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        # Here we just pass in None and dz will be allocated in the C++ code.
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, z, delta_bias, dout, x, out, None, ctx.delta_softplus,
            ctx.n_heads, ctx.head_d_state, ctx.scalar_dt,
            False  # option to recompute out_z, not used here
        )
        dz = rest[0] if ctx.has_z else None
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC,
                dD if D is not None else None,
                dz,
                ddelta_bias if delta_bias is not None else None,
                None,
                None, 
                None, None, None)


def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                     return_last_state=False,
                     head_d_state=0, n_heads=1, scalar_dt=False):
    """if return_last_state is True, returns (out, last_state)
    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
    not considered in the backward pass.
    """
    return SelectiveScanFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state,
                                 head_d_state, n_heads, scalar_dt)


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False, 
                      head_d_state=0, n_heads=1, scalar_dt=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L) 
    # BGNL means that inner_dim is divided into G groups,
    # and each of them interacts/outer-multiplies with its correspondig group in B
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """

    # for simplicity, expand delta/delta_bias/A to have the full shape.
    if scalar_dt:
        head_d_inner = u.shape[1] // n_heads # Note
        delta = repeat(delta, "b h l -> b (h d) l", d=head_d_inner)
        A = repeat(A, "h1 h2 -> (h1 d) (h2 m)", d=head_d_inner, m=head_d_state)
        if delta_bias is not None:
            delta_bias = repeat(delta_bias, "h -> (h d)", d=head_d_inner)

    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, d_state = u.shape[0], A.shape[0], A.shape[1] # Note that because of making A dense, d_state = n_heads * head_d_state!
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()
    x = A.new_zeros((batch, dim, d_state))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))

    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            B = repeat(rearrange(B, "B (h N) L -> B h N L", h=n_heads), "B h N L -> B (h H) N L", H=dim // n_heads)
        else:
            B = repeat(rearrange(B, "B G (h N) L -> B (G h) N L", h=n_heads), "B h N L -> B (h H) N L", H=dim // B.shape[1] // n_heads)
        deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = repeat(rearrange(C, "B G (h N) L -> B (G h) N L", h=n_heads), "B h N L -> B (h H) N L", H=dim // C.shape[1] // n_heads)

    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = rearrange(
                    torch.einsum("bhdn,bhn->bhd",
                        rearrange(x, "b (h d) n -> b h d n", h=n_heads),
                        rearrange(C[:, :, i], "b (h n) -> b h n", h=n_heads)),
                        "b h d -> b (h d)"
                )
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2) # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


class MambaInnerFn(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                out_proj_weight, out_proj_bias,
                A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
                C_proj_bias=None, delta_softplus=True,
                head_d_state=None, n_heads=1, scalar_dt=False, dense_matrices=True, convolved_v=True, multi_head_proj=False, # head_d_state cannot be extracted from A's shape anymore
                checkpoint_lvl=1):
        """
             xz: (batch, dim, seqlen)
        """
        assert causal_conv1d_cuda is not None, "causal_conv1d_cuda is not available. Please install causal-conv1d."
        assert checkpoint_lvl in [0, 1]
        L = xz.shape[-1]

        delta_rank = delta_proj_weight.shape[1]
        assert head_d_state is not None

        # Since A could be block-diagonal, we cannot extract head_d_state from A.
        # That is why head_d_state has to be passed in.
        original_d_state = head_d_state
        head_d_state = head_d_state * (1 if not A.is_complex() else 2)

        if torch.is_autocast_enabled():
            x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            delta_proj_weight = delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_weight = out_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_bias = (out_proj_bias.to(dtype=torch.get_autocast_gpu_dtype())
                             if out_proj_bias is not None else None)
        if xz.stride(-1) != 1:
            xz = xz.contiguous()
        conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
        x, z = xz.chunk(2, dim=1)
        conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
        conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
            x, conv1d_weight, conv1d_bias, None, None, None, True
        )
        # We're being very careful here about the layout, to avoid extra transposes.
        # We want delta to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.

        # x_proj_weight is columnar block-diagonal.
        x_dbl = torch.einsum("shd,hod->sho",
                            rearrange(conv1d_out, "b (h d) l -> (b l) h d", h=n_heads),
                            rearrange(x_proj_weight, "(h d_o_h) d -> h d_o_h d", h=n_heads)) # (bl h d)

        delta = rearrange(torch.einsum("shd,hod->sho",
                                    x_dbl[:, :, :delta_rank],
                                    rearrange(delta_proj_weight, "(h d_o_h) d -> h d_o_h d", h=n_heads)),
                        "(b l) h d -> b (h d) l", l=L).contiguous() # b (hd) l
        if scalar_dt and dense_matrices:
            # resize dt/dt_bias/A to the standard size expected
            head_d_inner = x.shape[1] // n_heads # Note
            delta = repeat(delta, "b h l -> b (h d) l", d=head_d_inner).contiguous()
            A = repeat(A, "h1 h2 -> (h1 d) (h2 m)", d=head_d_inner, m=original_d_state).contiguous()
            if delta_bias is not None:
                delta_bias = repeat(delta_bias, "h -> (h d)", d=head_d_inner).contiguous()

        ctx.is_variable_B = B is None
        ctx.is_variable_C = C is None
        ctx.B_proj_bias_is_None = B_proj_bias is None
        ctx.C_proj_bias_is_None = C_proj_bias is None
        if B is None:  # variable B
            B = rearrange(x_dbl[:, :, delta_rank:delta_rank + head_d_state], 
                            "s h dstate -> s (h dstate)").contiguous()
            if B_proj_bias is not None:
                B = B + B_proj_bias.to(dtype=B.dtype)
            if not A.is_complex():
                B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if B.stride(-1) != 1:
                B = B.contiguous()
        if C is None:  # variable C
            C = rearrange(x_dbl[:, :, -head_d_state:], "s h dstate -> s (h dstate)").contiguous()
            if C_proj_bias is not None:
                C = C + C_proj_bias.to(dtype=C.dtype)
            if not A.is_complex():
                C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if C.stride(-1) != 1:
                C = C.contiguous()
        if D is not None:
            D = D.contiguous()

        out, scan_intermediates, out_z = selective_scan_cuda.fwd(
            conv1d_out if convolved_v else x,
            delta, A, B, C, D, z, delta_bias, delta_softplus,
            n_heads, original_d_state, scalar_dt and not dense_matrices,
        )
        ctx.delta_softplus = delta_softplus

        ctx.n_heads = n_heads
        ctx.head_d_state = head_d_state
        ctx.original_d_state = original_d_state
        ctx.dense_matrices = dense_matrices
        ctx.scalar_dt = scalar_dt
        ctx.convolved_v = convolved_v
        ctx.multi_head_proj= multi_head_proj

        ctx.out_proj_bias_is_None = out_proj_bias is None
        ctx.checkpoint_lvl = checkpoint_lvl
        if checkpoint_lvl >= 1:  # Will recompute conv1d_out and delta in the backward pass
            conv1d_out, delta = None, None
        ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
                              delta_proj_weight, out_proj_weight, conv1d_out, delta,
                              A, B, C, D, delta_bias, scan_intermediates, out)
        if not ctx.multi_head_proj:
            out = F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)
        else:
            out = rearrange(torch.einsum("shd,hod->sho",
                                rearrange(out_z, "b (h d) l -> (b l) h d", h=n_heads),
                                rearrange(out_proj_weight, "(h d_o_h) d -> h d_o_h d", h=n_heads)), # (bl h d)
                                "(b l) h d -> b l (h d)", l=L) 
            if out_proj_bias is not None:
                out = out + out_proj_bias
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, dout):
        # dout: (batch, seqlen, dim)
        assert causal_conv1d_cuda is not None, "causal_conv1d_cuda is not available. Please install causal-conv1d."
        (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, out_proj_weight,
         conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, out) = ctx.saved_tensors
        L = xz.shape[-1]

        delta_rank = delta_proj_weight.shape[1]
        head_d_state = ctx.head_d_state
        original_d_state = ctx.original_d_state

        x, z = xz.chunk(2, dim=1)
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if ctx.checkpoint_lvl == 1:
            conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
                x, conv1d_weight, conv1d_bias, None, None, None, True
            )
            # recompute delta
            delta = rearrange(torch.einsum("shd,hod->sho",
                                        x_dbl[:, :, :delta_rank],
                                        rearrange(delta_proj_weight, "(h d_o_h) d -> h d_o_h d", h=ctx.n_heads)),
                            "(b l) h d -> b (h d) l", l=L).contiguous() # b (hd) l
            if ctx.scalar_dt and ctx.dense_matrices:
                # resize dt/A to the standard size expected. dt_bias was cached.
                head_d_inner = x.shape[1] // ctx.n_heads # Note
                delta = repeat(delta, "b h l -> b (h d) l", d=head_d_inner).contiguous()

        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        dxz = torch.empty_like(xz)  # (batch, dim, seqlen)
        dx, dz = dxz.chunk(2, dim=1)
        if not ctx.multi_head_proj:
            dout = rearrange(dout, "b l e -> e (b l)")
            dout_y = rearrange(out_proj_weight.t() @ dout, "d (b l) -> b d l", l=L)
        else:
            dout = rearrange(dout, "b l (h d) -> (b l) h d", h=ctx.n_heads)
            dout_y = rearrange(torch.einsum("Bhd,hdr->Bhr", dout,
                                            rearrange(out_proj_weight, "(h d) r -> h d r", h=ctx.n_heads)),
                                "(b l) h d -> b (h d) l", l=L)
        du, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = selective_scan_cuda.bwd(
            conv1d_out if ctx.convolved_v else x, delta, A, B, C, D, z, delta_bias, dout_y, scan_intermediates, out, dz,
            ctx.delta_softplus,
            ctx.n_heads, original_d_state, ctx.scalar_dt and not ctx.dense_matrices,
            True  # option to recompute out_z
        )
        if ctx.convolved_v:
            dconv1d_out = du
        else:
            dx_1 = du
        if not ctx.multi_head_proj:
            dout_proj_weight = torch.einsum("eB,dB->ed", dout, rearrange(out_z, "b d l -> d (b l)"))
        else:
            dout_proj_weight = rearrange(torch.einsum("Bhd,Bhr->hdr", 
                                                    dout, rearrange(out_z, "b (h d) l -> (b l) h d", h=ctx.n_heads)),
                                           "h d r -> (h d) r")
            dout = rearrange(dout, "(b l) h d -> (b l) (h d)", l=L) # only for dout_proj_bias
        dout_proj_bias = dout.sum(dim=(0, 1)) if not ctx.out_proj_bias_is_None else None
        dD = dD if D is not None else None
        dx_dbl = torch.empty_like(x_dbl)
        dB_proj_bias = None
        if ctx.is_variable_B:
            if not A.is_complex():
                dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
            dB = rearrange(dB, "s (h dstate) -> s h dstate", h=ctx.n_heads)
            dx_dbl[:, :, delta_rank:delta_rank + head_d_state] = dB # (bl h d)
            dB = None
        dC_proj_bias = None
        if ctx.is_variable_C:
            if not A.is_complex():
                dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
            dC = rearrange(dC, "s (h dstate) -> s h dstate", h=ctx.n_heads)
            dx_dbl[:, :, -head_d_state:] = dC # (bl h d)
            dC = None

        if ctx.scalar_dt and ctx.dense_matrices:
            # resize dt/dt_bias/A to the standard size expected
            head_d_inner = xz.shape[1] // ctx.n_heads # Note
            ddelta = rearrange(ddelta, "b (h d) l -> b h d l", h=ctx.n_heads)
            ddelta = torch.sum(ddelta, dim=2, keepdim=False).contiguous()
            if ddelta_bias is not None:
                ddelta_bias = rearrange(ddelta_bias, "(h d) -> h d", h=ctx.n_heads)
                ddelta_bias = torch.sum(ddelta_bias, dim=-1, keepdim=False).contiguous()
            dA = torch.sum(dA, dim=1, keepdim=True)
            dA = rearrange(dA, "(h d) n -> h d n", h=ctx.n_heads)
            dA = torch.sum(dA, dim=1, keepdim=False).contiguous()

        # delta_proj_weight is columnar block-diagonal
        # computes ddelta_proj_weight and dx_dbl
        # ddelta_proj_weight : d_inner x rank
        # dx_dbl[:, :, :delta_rank] : d h r
        ddelta = rearrange(ddelta, "b (h d) l -> h d (b l)", h=ctx.n_heads)
        ddelta_proj_weight = rearrange(torch.einsum("hdB,Bhr->hdr", 
                                                ddelta, x_dbl[:, :, :delta_rank]),
                                        "h d r -> (h d) r")
        dx_dbl[:, :, :delta_rank] = torch.einsum("hdB,hdr->Bhr", ddelta,
                                                rearrange(delta_proj_weight, "(h d) r -> h d r", h=ctx.n_heads))
        if ctx.convolved_v:
            dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")

        dx_proj_weight = rearrange(torch.einsum("Bhr,Bhd->hrd", dx_dbl,
                                                rearrange(conv1d_out, "b (h d) l -> (b l) h d", h=ctx.n_heads)),
                                    "h r d -> (h r) d")
        dconv =  rearrange(torch.einsum("Bhr,hrd->Bhd", dx_dbl,
                                        rearrange(x_proj_weight, "(h r) d -> h r d", h=ctx.n_heads)),
                            "B h d -> (h d) B")
        if ctx.convolved_v:
            dconv1d_out = dconv1d_out + dconv
        else:
            dconv1d_out = dconv

        dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
        # backward of conv1d with the backward of chunk).
        dx, dconv1d_weight, dconv1d_bias, *_ = causal_conv1d_cuda.causal_conv1d_bwd(
            x, conv1d_weight, conv1d_bias, dconv1d_out, None, None, None, dx, False, True
        )

        if not ctx.convolved_v:
            dx = dx + dx_1

        dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
        dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
                dout_proj_weight, dout_proj_bias,
                dA, dB, dC, dD,
                ddelta_bias if delta_bias is not None else None,
                dB_proj_bias, dC_proj_bias, None,
                None, None, None, None, None, None,
                None)


def mamba_inner_fn(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    out_proj_weight, out_proj_bias,
    A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True,
    head_d_state=None, n_heads=1, scalar_dt=False, dense_matrices=True, convolved_v=True, multi_head_proj=False,

):
    return MambaInnerFn.apply(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                              out_proj_weight, out_proj_bias,
                              A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus,
                              head_d_state, n_heads, scalar_dt, dense_matrices, convolved_v, multi_head_proj)

def mamba_inner_ref(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    out_proj_weight, out_proj_bias,
    A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True,
    head_d_state=None, n_heads=1, scalar_dt=False, dense_matrices=True, convolved_v=True, multi_head_proj=False,
):
    assert causal_conv1d_fn is not None, "causal_conv1d_fn is not available. Please install causal-conv1d."
    L = xz.shape[-1]

    delta_rank = delta_proj_weight.shape[1]

    # Since A could be block-diagonal, we cannot extract head_d_state from A.
    # That is why head_d_state has to be passed in.
    assert head_d_state is not None
    original_d_state = head_d_state
    head_d_state = head_d_state * (1 if not A.is_complex() else 2)

    x, z = xz.chunk(2, dim=1)

    if not convolved_v:
        pre_conv_x = x

    x = causal_conv1d_fn(x, rearrange(conv1d_weight, "d 1 w -> d w"), conv1d_bias, activation="silu")
    # We're being very careful here about the layout, to avoid extra transposes.
    # We want delta to have d as the slowest moving dimension
    # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.

    # x_proj_weight is columnar block-diagonal.
    # Note that in the above code, F.Linear was called and the dimensions of
    # conv1d were set in a special way to make it fast! However, I have the extra
    # h dimension!

    # TODO : Here, to make sure you do the right thing, implement the block-diagonal
    # multiplicaton using a for-loop over the heads, instead of using fancy einsum!
    x_dbl = torch.einsum("shd,hod->sho",
                        rearrange(x, "b (h d) l -> (b l) h d", h=n_heads),
                        rearrange(x_proj_weight, "(h d_o_h) d -> h d_o_h d", h=n_heads))  # (bl h d)
    delta = x_dbl[:, :, :delta_rank]

    # TODO: is this efficient?
    delta = rearrange(torch.einsum("hod,shd->sho",
                                rearrange(delta_proj_weight, "(h d_o_h) d -> h d_o_h d", h=n_heads),
                                delta),
                        "(b l) h d -> b (h d) l", l=L).contiguous() # b (hd) l

    # For simplicity, let's make below matrices dense.
    if scalar_dt and dense_matrices:
        # resize dt/dt_bias/A to the standard size expected
        head_d_inner = xz.shape[1] // n_heads // 2 # Note
        delta = repeat(delta, "b h l -> b (h d) l", d=head_d_inner)
        A = repeat(A, "h1 h2 -> (h1 d) (h2 m)", d=head_d_inner, m=original_d_state)
        if delta_bias is not None:
            delta_bias = repeat(delta_bias, "h -> (h d)", d=head_d_inner)

    if B is None:  # variable B
        B = rearrange(x_dbl[:, :, delta_rank:delta_rank + head_d_state], "s h d -> s (h d)").contiguous()
        if B_proj_bias is not None:
            B = B + B_proj_bias.to(dtype=B.dtype)
        if not A.is_complex():
            B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            B = rearrange(B, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2).contiguous()
    if C is None:  # variable B
        C = rearrange(x_dbl[:, :, -head_d_state:], "s h d -> s (h d)").contiguous()
        if C_proj_bias is not None:
            C = C + C_proj_bias.to(dtype=C.dtype)
        if not A.is_complex():
            C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            C = rearrange(C, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2).contiguous()

    # VIP: below I changed the call from selective_scan_fn to selective_scan_ref!
    y = selective_scan_fn(x if convolved_v else pre_conv_x,
                          delta, A, B, C, D, z=z, delta_bias=delta_bias, delta_softplus=True,
                          head_d_state=original_d_state, n_heads=n_heads, scalar_dt=scalar_dt and not dense_matrices)
    if not multi_head_proj:
        out = F.linear(rearrange(y, "b d l -> b l d"), out_proj_weight, out_proj_bias)
    else:
        out = rearrange(torch.einsum("shd,hod->sho",
                            rearrange(y, "b (h d) l -> (b l) h d", h=n_heads),
                            rearrange(out_proj_weight, "(h d_o_h) d -> h d_o_h d", h=n_heads)), # (bl h d)
                            "(b l) h d -> b l (h d)", l=L) # TODO: this is not as fast of F.Liner, because of the extra "+ bias"
        if out_proj_bias is not None:
            out = out + out_proj_bias
    return out
