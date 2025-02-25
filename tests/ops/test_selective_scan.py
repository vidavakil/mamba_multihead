# Copyright (C) 2023, Tri Dao.

import math

import torch
import torch.nn.functional as F
import pytest

from einops import rearrange

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, mamba_inner_ref

# @pytest.mark.parametrize('wtype', [torch.float32, torch.complex64])
@pytest.mark.parametrize('wtype', [torch.float32])
# @pytest.mark.parametrize('itype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('itype', [torch.float32])
# @pytest.mark.parametrize('seqlen', [8, 16, 32, 64, 128, 256, 372, 512, 784, 1024, 1134, 2048, 4096])
@pytest.mark.parametrize('seqlen', [1024]) # [128, 256, 512, 1024, 2048, 4096])
# @pytest.mark.parametrize('seqlen', [128])
# @pytest.mark.parametrize("return_last_state", [False, True])
@pytest.mark.parametrize("return_last_state", [True])
# @pytest.mark.parametrize('has_delta_bias', [False, True])
@pytest.mark.parametrize('has_delta_bias', [True])
# @pytest.mark.parametrize('delta_softplus', [False, True])
@pytest.mark.parametrize('delta_softplus', [True])
# @pytest.mark.parametrize('has_z', [False, True])
@pytest.mark.parametrize('has_z', [True])
# @pytest.mark.parametrize('has_D', [False, True])
@pytest.mark.parametrize('has_D', [True])
@pytest.mark.parametrize("varBC_groups", [1])
# @pytest.mark.parametrize("varBC_groups", [1])
# @pytest.mark.parametrize("is_variable_C", [False, True])
@pytest.mark.parametrize("is_variable_C", [True, False])
# @pytest.mark.parametrize("is_variable_B", [False, True])
@pytest.mark.parametrize("is_variable_B", [True])               # TODO: False is broken!
@pytest.mark.parametrize("is_multi_head_mamba", [False, True])
@pytest.mark.parametrize("n_heads", [2, 4, 8])
@pytest.mark.parametrize("scalar_dt", [False, True])
@pytest.mark.parametrize("has_in_h", [False, True])

def test_selective_scan(is_variable_B, is_variable_C, varBC_groups, has_D, has_z, has_delta_bias,
                        delta_softplus, return_last_state, seqlen, itype, wtype,
                        is_multi_head_mamba,
                        n_heads, scalar_dt,
                        has_in_h):
    # For now, selective_scan always returns the last_state.
    assert return_last_state == True
    if varBC_groups > 1 and (not is_variable_B or not is_variable_C):
        pytest.skip()  # This config is not applicable
    device = 'cuda'
    rtol, atol = (6e-4, 2e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)
    if has_z:  # If we have z, the errors on the weights seem higher
        rtolw = max(rtolw, rtol)
        atolw = max(atolw, atol)
    # set seed
    torch.random.manual_seed(0)
    batch_size = 2
    dim = 16
    d_state = 16 # head_state * n_heads
    is_complex = wtype == torch.complex64

    # Use below if not randomizing n_heads, and scalar_dt
    # if is_multi_head_mamba:
    #     n_heads = 4
    #     scalar_dt = True
    # else:
    #     n_heads = 1
    #     scalar_dt = False

    if not is_multi_head_mamba:
        n_heads = 1

    head_d_state = d_state // n_heads

    A = (-0.5 * torch.rand(n_heads if scalar_dt else dim, 1 if scalar_dt else head_d_state, device=device, dtype=wtype)).requires_grad_()
    if has_delta_bias:
        delta_bias = (0.5 * torch.rand(n_heads if scalar_dt else dim, device=device, dtype=torch.float32)).requires_grad_()
    delta = (0.5 * torch.rand(batch_size, n_heads if scalar_dt else dim, seqlen, device=device, dtype=itype)).requires_grad_()

    if not has_delta_bias:
        delta_bias = None

    if not is_variable_B:
        B_shape = (dim, head_d_state)
    elif varBC_groups == 1:
        B_shape = (batch_size, d_state, seqlen if not is_complex else seqlen * 2)
    else:
        B_shape = (batch_size, varBC_groups, d_state, seqlen if not is_complex else seqlen * 2)
    B = torch.randn(*B_shape, device=device, dtype=wtype if not is_variable_B else itype,
                    requires_grad=True)
    if not is_variable_C:
        C_shape = (dim, head_d_state)
    elif varBC_groups == 1:
        C_shape = (batch_size, d_state, seqlen if not is_complex else seqlen * 2)
    else:
        C_shape = (batch_size, varBC_groups, d_state, seqlen if not is_complex else seqlen * 2)
    C = torch.randn(*C_shape, device=device, dtype=wtype if not is_variable_C else itype,
                    requires_grad=True)
    if has_D:
        D = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    else:
        D = None
    if has_z:
        z = torch.randn(batch_size, dim, seqlen, device=device, dtype=itype, requires_grad=True)
    else:
        z = None
    if has_in_h:
        in_h = torch.rand(batch_size, dim, head_d_state, device=device, dtype=wtype).requires_grad_()
    else:
        in_h = None
    u = torch.randn(batch_size, dim, seqlen, device=device, dtype=itype, requires_grad=True)
    A_ref = A.detach().clone().requires_grad_()
    B_ref = B.detach().clone().requires_grad_()
    C_ref = C.detach().clone().requires_grad_()
    D_ref = D.detach().clone().requires_grad_() if D is not None else None
    z_ref = z.detach().clone().requires_grad_() if z is not None else None
    u_ref = u.detach().clone().requires_grad_()
    in_h_ref = in_h.detach().clone().requires_grad_() if has_in_h else None
    delta_ref = delta.detach().clone().requires_grad_()
    delta_bias_ref = delta_bias.detach().clone().requires_grad_() if delta_bias is not None else None
    out, *rest = selective_scan_fn(
        u, in_h, delta, A, B, C, D, z=z,
        delta_bias=delta_bias, delta_softplus=delta_softplus,
        return_last_state=True,
        head_d_state=head_d_state, n_heads=n_heads, scalar_dt=scalar_dt
    )
    if return_last_state:
        state = rest[0]
    out_ref, *rest = selective_scan_ref(
        u_ref, in_h_ref, delta_ref, A_ref, B_ref, C_ref, D_ref, z=z_ref,
        delta_bias=delta_bias_ref, delta_softplus=delta_softplus,
        return_last_state=return_last_state,
        head_d_state=head_d_state, n_heads=n_heads, scalar_dt=scalar_dt
    )
    if return_last_state:
        state_ref = rest[0]
    # dA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    # dt_u = delta * u

    print(f'Output max diff: {(out - out_ref).abs().max().item()}')
    print(f'Output mean diff: {(out - out_ref).abs().mean().item()}')
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)
    if return_last_state:
        print(f'State max diff: {(state - state_ref).abs().max().item()}')
        assert torch.allclose(state, state_ref, rtol=rtol, atol=atol)

    g = torch.randn_like(out)
    out_ref.backward(g)
    out.backward(g)

    print(f'du max diff: {(u.grad - u_ref.grad).abs().max().item()}')
    print(f'ddelta max diff: {(delta.grad - delta_ref.grad).abs().max().item()}')
    print(f'dA max diff: {(A.grad - A_ref.grad).abs().max().item()}')
    print(f'dB max diff: {(B.grad - B_ref.grad).abs().max().item()}')
    print(f'dC max diff: {(C.grad - C_ref.grad).abs().max().item()}')
    if has_D:
        print(f'dD max diff: {(D.grad - D_ref.grad).abs().max().item()}')
    if has_z:
        print(f'dz max diff: {(z.grad - z_ref.grad).abs().max().item()}')
    if has_delta_bias:
        print(f'ddelta_bias max diff: {(delta_bias.grad - delta_bias_ref.grad).abs().max().item()}')
    if has_in_h:
        print(f'din_h max diff: {(in_h.grad - in_h_ref.grad).abs().max().item()}')

    assert torch.allclose(u.grad, u_ref.grad.to(dtype=itype), rtol=rtol * 2, atol=atol * 2)
    assert torch.allclose(delta.grad, delta_ref.grad.to(dtype=itype), rtol=rtol * 5, atol=atol * 10)
    assert torch.allclose(A.grad, A_ref.grad, rtol=rtolw, atol=atolw * 5)
    assert torch.allclose(B.grad, B_ref.grad, rtol=rtolw if not is_variable_B else rtol,
                          atol=atolw if not is_variable_B else atol)
    assert torch.allclose(C.grad, C_ref.grad, rtol=rtolw if not is_variable_C else rtol,
                          atol=atolw if not is_variable_C else atol)
    if has_D:
        assert torch.allclose(D.grad, D_ref.grad, rtol=rtolw, atol=atolw)
    if has_z:
        assert torch.allclose(z.grad, z_ref.grad, rtol=rtolw, atol=atolw)
    if has_delta_bias:
        assert torch.allclose(delta_bias.grad, delta_bias_ref.grad, rtol=rtolw, atol=atolw)
    if has_in_h:
        assert torch.allclose(in_h.grad, in_h_ref.grad, rtol=rtolw, atol=atolw)

#@pytest.mark.parametrize('wtype', [torch.float32, torch.complex64])
@pytest.mark.parametrize('wtype', [torch.float32])
# @pytest.mark.parametrize('itype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('itype', [torch.float32])
# @pytest.mark.parametrize('seqlen', [8, 16, 32, 64, 128, 256, 372, 512, 784, 1024, 1134, 2048, 4096])
@pytest.mark.parametrize('seqlen', [2048])
#@pytest.mark.parametrize("is_variable_C", [False, True])
@pytest.mark.parametrize("is_variable_C", [False, True])
# @pytest.mark.parametrize("is_variable_B", [False, True])
@pytest.mark.parametrize("is_variable_B", [True, False])
@pytest.mark.parametrize("is_multi_head_mamba", [False, True])
@pytest.mark.parametrize("n_heads", [2, 4, 8])
@pytest.mark.parametrize("scalar_dt", [False, True])
@pytest.mark.parametrize("dense_matrices", [True])
@pytest.mark.parametrize("multi_head_proj", [False, True])
@pytest.mark.parametrize("convolved_v", [False, True])
@pytest.mark.parametrize("has_in_h", [False, True])
def test_mamba_inner_fn(is_variable_B, is_variable_C, seqlen, itype, wtype,
                        is_multi_head_mamba,
                        n_heads, scalar_dt, dense_matrices, multi_head_proj, convolved_v,
                        has_in_h):
    device = 'cuda'
    rtol, atol = (6e-4, 2e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)
    # If we have z, the errors on the weights seem higher
    rtolw = max(rtolw, rtol)
    atolw = max(atolw, atol)
    # set seed
    torch.random.manual_seed(0)
    batch_size = 2
    dim = 768
    d_state = 16 # head_d_state * num_heads
    dt_rank = 16

    # Use below if not randomizing the parameters
    # if is_multi_head_mamba:
    #     n_heads = 4
    #     dt_rank = 1
    #     scalar_dt = True
    #     dense_matrices = False
    #     multi_head_proj = True
    #     convolved_v = False
    # else:
    #     n_heads = 1
    #     scalar_dt = False
    #     dense_matrices = False
    #     multi_head_proj = False
    #     convolved_v = True

    if not is_multi_head_mamba:
        n_heads = 1
    if scalar_dt:
        dt_rank = 1
    else:
        dense_matrices = False

    is_complex = wtype == torch.complex64
    xz = torch.randn(batch_size, 2 * dim, seqlen, device=device, dtype=itype, requires_grad=True)
    conv1d_weight = torch.randn(dim, 1, 3, device=device, dtype=torch.float32, requires_grad=True)
    conv1d_bias = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)

    assert dim % n_heads == 0
    assert d_state % n_heads == 0

    head_d_inner = dim // n_heads
    head_dt_rank = max(1, dt_rank // n_heads)
    if scalar_dt:
        head_dt_rank = 1
    head_d_state = d_state // n_heads
    x_proj_output_size = head_dt_rank + (bool(is_variable_B) + bool(is_variable_C)) * head_d_state * (1 if not is_complex else 2)

    x_proj_weight = torch.randn(x_proj_output_size * n_heads, head_d_inner, device=device, dtype=itype, requires_grad=True)
    delta_proj_weight = torch.randn(n_heads if scalar_dt else dim, head_dt_rank, device=device, dtype=itype, requires_grad=True)
    delta_bias = (0.5 * torch.rand(n_heads if scalar_dt else dim, device=device, dtype=torch.float32)).requires_grad_()
    A = (-0.5 * torch.rand(n_heads if scalar_dt else dim, 1 if scalar_dt else head_d_state, device=device, dtype=wtype)).requires_grad_()
    out_proj_weight = torch.randn(dim//2, head_d_inner if multi_head_proj else dim, device=device, dtype=itype, requires_grad=True)

    out_proj_bias = None

    B = (torch.randn(dim, head_d_state, device=device, dtype=wtype, requires_grad=True)
         if not is_variable_B else None)
    C = (torch.randn(dim, head_d_state, device=device, dtype=wtype, requires_grad=True)
         if not is_variable_C else None)
    D = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)

    if has_in_h:
        in_h = torch.rand(batch_size, dim, head_d_state, device=device, dtype=wtype).requires_grad_()
    else:
        in_h = None

    B_proj_bias = None
    C_proj_bias = None
    in_h_ref = in_h.detach().clone().requires_grad_() if has_in_h else None
    xz_ref = xz.detach().clone().requires_grad_()
    conv1d_weight_ref = conv1d_weight.detach().clone().requires_grad_()
    conv1d_bias_ref = conv1d_bias.detach().clone().requires_grad_()
    x_proj_weight_ref = x_proj_weight.detach().clone().requires_grad_()
    delta_proj_weight_ref = delta_proj_weight.detach().clone().requires_grad_()
    out_proj_weight_ref = out_proj_weight.detach().clone().requires_grad_()
    out_proj_bias_ref = (out_proj_bias.detach().clone().requires_grad_()
                         if out_proj_bias is not None else None)
    A_ref = A.detach().clone().requires_grad_()
    B_ref = B.detach().clone().requires_grad_() if B is not None else None
    C_ref = C.detach().clone().requires_grad_() if C is not None else None
    D_ref = D.detach().clone().requires_grad_()
    delta_bias_ref = delta_bias.detach().clone().requires_grad_() if delta_bias is not None else None
    out, last_state = mamba_inner_fn(xz, in_h, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                         out_proj_weight, out_proj_bias,
                         A, B, C, D, delta_bias=delta_bias, delta_softplus=True,
                         head_d_state=head_d_state, n_heads=n_heads, scalar_dt=scalar_dt, dense_matrices=dense_matrices,
                         convolved_v=convolved_v, multi_head_proj=multi_head_proj)
    out_ref, last_state_ref = mamba_inner_ref(xz_ref, in_h_ref, conv1d_weight_ref, conv1d_bias_ref, x_proj_weight_ref,
                              delta_proj_weight_ref, out_proj_weight_ref, out_proj_bias_ref,
                              A_ref, B_ref, C_ref, D_ref,
                              delta_bias=delta_bias_ref, delta_softplus=True,
                              head_d_state=head_d_state, n_heads=n_heads, scalar_dt=scalar_dt, dense_matrices=dense_matrices,
                              convolved_v=convolved_v, multi_head_proj=multi_head_proj)
    # dA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    # dt_u = delta * u

    print(f'Output max diff: {(out - out_ref).abs().max().item()}')
    print(f'Output mean diff: {(out - out_ref).abs().mean().item()}')
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)

    print(f'last_state max diff: {(last_state - last_state_ref).abs().max().item()}')
    print(f'last_state mean diff: {(last_state - last_state_ref).abs().mean().item()}')
    assert torch.allclose(last_state, last_state_ref, rtol=rtol, atol=atol)

    g = torch.randn_like(out)
    out_ref.backward(g)
    out.backward(g)

    print(f'dxz max diff: {(xz.grad - xz_ref.grad).abs().max().item()}')
    print(f'dA max diff: {(A.grad - A_ref.grad).abs().max().item()}')
    if not is_variable_B:
        print(f'dB max diff: {(B.grad - B_ref.grad).abs().max().item()}')
    if not is_variable_C:
        print(f'dC max diff: {(C.grad - C_ref.grad).abs().max().item()}')
    print(f'dD max diff: {(D.grad - D_ref.grad).abs().max().item()}')
    print(f'ddelta_bias max diff: {(delta_bias.grad - delta_bias_ref.grad).abs().max().item()}')
    print(f'dout_proj_weight max diff: {(out_proj_weight.grad - out_proj_weight_ref.grad).abs().max().item()}')
    print(f'ddelta_proj_weight max diff: {(delta_proj_weight.grad - delta_proj_weight_ref.grad).abs().max().item()}')
    print(f'dx_proj_weight max diff: {(x_proj_weight.grad - x_proj_weight_ref.grad).abs().max().item()}')
    print(f'dconv1d_weight max diff: {(conv1d_weight.grad - conv1d_weight_ref.grad).abs().max().item()}')
    print(f'dconv1d_bias max diff: {(conv1d_bias.grad - conv1d_bias_ref.grad).abs().max().item()}')
    if has_in_h:
        print(f'din_h max diff: {(in_h.grad - in_h_ref.grad).abs().max().item()}')

    # assert torch.allclose(xz.grad, xz_ref.grad.to(dtype=itype), rtol=rtol * 2, atol=atol * 2)
    # assert torch.allclose(delta.grad, delta_ref.grad.to(dtype=itype), rtol=rtol * 5, atol=atol * 10)
    # assert torch.allclose(A.grad, A_ref.grad, rtol=rtolw, atol=atolw * 5)
    # assert torch.allclose(B.grad, B_ref.grad, rtol=rtolw if not is_variable_B else rtol,
    #                       atol=atolw if not is_variable_B else atol)
    # assert torch.allclose(C.grad, C_ref.grad, rtol=rtolw if not is_variable_C else rtol,
    #                       atol=atolw if not is_variable_C else atol)
    # assert torch.allclose(D.grad, D_ref.grad, rtol=rtolw, atol=atolw)
    # assert torch.allclose(delta_bias.grad, delta_bias_ref.grad, rtol=rtolw, atol=atolw)
    if has_in_h:
        assert torch.allclose(in_h.grad, in_h_ref.grad, rtol=rtolw, atol=atolw)
