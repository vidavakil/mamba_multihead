# Copyright (c) 2023, Tri Dao.

"""We want triton==2.1.0 for this
"""

import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from einops import rearrange, repeat


@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics({"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])})
@triton.jit
def _selective_scan_update_kernel(
    # Pointers to matrices
    state_ptr, x_ptr, dt_ptr, dt_bias_ptr, A_ptr, B_ptr, C_ptr, D_ptr, z_ptr, out_ptr,
    # Matrix dimensions
    batch, dim, dstate, # notice that this dstate is in fact head_d_state
    # Strides
    stride_state_batch, stride_state_dim, stride_state_dstate,
    stride_x_batch, stride_x_dim,
    stride_dt_batch, stride_dt_dim,
    stride_dt_bias_dim,
    stride_A_dim, stride_A_dstate,
    stride_B_batch, stride_B_dstate,
    stride_C_batch, stride_C_dstate,
    stride_D_dim,
    stride_z_batch, stride_z_dim,
    stride_out_batch, stride_out_dim,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    state_ptr += pid_b * stride_state_batch
    x_ptr += pid_b * stride_x_batch
    dt_ptr += pid_b * stride_dt_batch
    B_ptr += pid_b * stride_B_batch
    C_ptr += pid_b * stride_C_batch
    if HAS_Z:
        z_ptr += pid_b * stride_z_batch
    out_ptr += pid_b * stride_out_batch

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    state_ptrs = state_ptr + (offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate)
    x_ptrs = x_ptr + offs_m * stride_x_dim
    dt_ptrs = dt_ptr + offs_m * stride_dt_dim
    if HAS_DT_BIAS:
        dt_bias_ptrs = dt_bias_ptr + offs_m * stride_dt_bias_dim
    A_ptrs = A_ptr + (offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate)
    B_ptrs = B_ptr + offs_n * stride_B_dstate
    C_ptrs = C_ptr + offs_n * stride_C_dstate
    if HAS_D:
        D_ptrs = D_ptr + offs_m * stride_D_dim
    if HAS_Z:
        z_ptrs = z_ptr + offs_m * stride_z_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim

    state = tl.load(state_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0)
    x = tl.load(x_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    dt = tl.load(dt_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    if HAS_DT_BIAS:
        dt += tl.load(dt_bias_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    if DT_SOFTPLUS:
        dt = tl.where(dt <= 20.0, tl.math.log1p(tl.exp(dt)), dt)
    A = tl.load(A_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
    dA = tl.exp(A * dt[:, None])
    B = tl.load(B_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
    C = tl.load(C_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
    if HAS_D:
        D = tl.load(D_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    if HAS_Z:
        z = tl.load(z_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)

    dB = B[None, :] * dt[:, None]
    state = state * dA + dB * x[:, None] # TODO Question: does triton jit do broadcasting if dA is dx1 instead of dxn?
    tl.store(state_ptrs, state, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate))
    out = tl.sum(state * C[None, :], axis=1)
    if HAS_D:
        out += x * D
    if HAS_Z:
        out *= z * tl.sigmoid(z)
    tl.store(out_ptrs, out, mask=offs_m < dim)


def selective_state_update(state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False,
                           n_heads=1):
    """
    Argument:
        state: (batch, dim, dstate)
        x: (batch, dim)
        dt: (batch, dim)
        A: (dim, dstate)
        B: (batch, dstate)
        C: (batch, dstate)
        D: (dim,)
        z: (batch, dim)
        dt_bias: (dim,)
    Return:
        out: (batch, dim)
    """
    batch, dim, dstate = state.shape # notice that this dstate is in fact head_d_state
    assert x.shape == (batch, dim)
    assert A.shape == (dim, dstate)
    assert B.shape == (batch, dstate * n_heads)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (dim,)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (dim,)
    out = torch.empty_like(x)
    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE_M']), batch)
    z_strides = ((z.stride(0), z.stride(1)) if z is not None else (0, 0))

    # We don't want autotune since it will overwrite the state
    # We instead tune by hand.
    BLOCK_SIZE_M, num_warps = ((32, 4) if dstate <= 16
                            else ((16, 4) if dstate <= 32 else
                                    ((8, 4) if dstate <= 64 else
                                    ((4, 4) if dstate <= 128 else
                                    ((4, 8))))))
    # Have a loop over the heads here, so that each head's data is packed
    # together and sent to the triton kernel.
    head_rows = dim // n_heads
    head_columns = dstate
    for i in range(n_heads):
        row_start = i * head_rows
        row_end = (i + 1) * head_rows
        column_start = i * head_columns
        column_end = (i + 1) * head_columns

        # In multi-head mode, state and A are columnar block-diagonal
        with torch.cuda.device(x.device.index):
            _selective_scan_update_kernel[grid](
                state[:, row_start:row_end, :], 
                x[:, row_start:row_end],
                dt[:, row_start:row_end],
                dt_bias[row_start:row_end] if dt_bias is not None else None,
                A[row_start:row_end, :],
                B[:, column_start:column_end],
                C[:, column_start:column_end],
                D[row_start:row_end] if D is not None else None,
                z[:, row_start:row_end] if z is not None else None,
                out[:, row_start:row_end],
                batch, dim // n_heads, dstate,
                state.stride(0), state.stride(1), state.stride(2),
                x.stride(0), x.stride(1),
                dt.stride(0), dt.stride(1),
                dt_bias.stride(0) if dt_bias is not None else 0,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                C.stride(0), C.stride(1),
                D.stride(0) if D is not None else 0,
                z_strides[0], z_strides[1],
                out.stride(0), out.stride(1),
                dt_softplus,
                BLOCK_SIZE_M,
                num_warps=num_warps,
            )
    return out


def selective_state_update_ref(state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False,
                               n_heads=1):
    """
    Argument:
        state: (batch, dim, dstate)
        x: (batch, dim)
        dt: (batch, dim)
        A: (dim, dstate)
        B: (batch, dstate)
        C: (batch, dstate)
        D: (dim,)
        z: (batch, dim)
        dt_bias: (dim,)
    Return:
        out: (batch, dim)
    """
    batch, dim, dstate = state.shape # notice that this dstate is in fact head_d_state
    assert x.shape == (batch, dim)
    assert A.shape == (dim, dstate)
    assert B.shape == (batch, dstate * n_heads)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (dim,)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (dim,)
        dt = dt + dt_bias
    dt = F.softplus(dt) if dt_softplus else dt
    dA = torch.exp(rearrange(dt, "b d -> b d 1") * A)  # (batch, dim, dstate)

    # For multi-head, make ssm_state block diagonal before multiplying
    # it with C, because in this case ssm_state is b(h d/h)(n/h), and C is b(h n/h)
    dB = rearrange(rearrange(dt, "b (h d) -> b h d 1", h=n_heads) * rearrange(B, "b (h n) -> b h 1 n", h=n_heads), 
                    "b h d n -> b (h d) n", h=n_heads)  # (batch, dim, dstate)
    state.copy_(state * dA + dB * rearrange(x, "b d -> b d 1"))  # (batch, dim, dstate
    out = rearrange(
        torch.einsum("bhdn,bhn->bhd",
            rearrange(state, "b (h d) n -> b h d n", h=n_heads),
            rearrange(C, "b (h n) -> b h n", h=n_heads)),
            "b h d -> b (h d)"
    )

    # out = torch.einsum("bdn,bn->bd", state.to(C.dtype), C)
    if D is not None:
        out += (x * D).to(out.dtype)
    return (out if z is None else out * F.silu(z)).to(x.dtype)
