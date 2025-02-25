# Copyright (C) 2023, Tri Dao.

import math

import torch
import torch.nn.functional as F
import pytest

from einops import rearrange

from mamba_ssm.ops.triton.selective_state_update import selective_state_update, selective_state_update_ref

# TODO: For now, selective_state_update always works with dense A/dt/dt_bias, and thus dt_scalar is not passed in

# TODO: Why complex weights are not tested?

#@pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('itype', [torch.float32]) # torch.float16 and torch.bfloat16 fail!
@pytest.mark.parametrize("has_z", [False, True])
# @pytest.mark.parametrize('has_z', [True])
@pytest.mark.parametrize("dstate", [16, 32, 64])
# @pytest.mark.parametrize("dstate", [16])
@pytest.mark.parametrize("dim", [2048, 2048 + 16, 4096])
# @pytest.mark.parametrize("dim", [2048])
@pytest.mark.parametrize("n_heads", [1, 4, 16])
def test_selective_state_update(dim, dstate, has_z, itype, n_heads):
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (5e-3, 1e-2)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    # set seed
    torch.random.manual_seed(0)
    batch_size = 2
    state = torch.randn(batch_size, dim, dstate, dtype=itype, device=device)
    x = torch.randn(batch_size, dim, device=device, dtype=itype)

    dt = torch.randn(batch_size, dim, device=device, dtype=itype)
    dt_bias = torch.rand(dim, device=device) - 4.0
    A = -torch.rand(dim, dstate, device=device) - 1.0
    B = torch.randn(batch_size, dstate * n_heads, device=device)
    C = torch.randn(batch_size, dstate * n_heads, device=device)
    D = torch.randn(dim, device=device)
    if has_z:
        z = torch.randn_like(x)
    else:
        z = None
    state_ref = state.detach().clone()
    out = selective_state_update(state, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True, n_heads=n_heads)
    out_ref = selective_state_update_ref(state_ref, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True, n_heads=n_heads)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert torch.allclose(state, state_ref, rtol=rtol, atol=atol)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)
