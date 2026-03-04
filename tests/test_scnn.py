from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from pytspl.scnn.chebyshev import assemble_powers, normalize_laplacian, normalize_like
from pytspl.scnn.scnn import (
    Coboundary,
    SimplicialConvolution,
    SimplicialConvolution2,
    coo2tensor,
)
from pytspl.scnn.utils import (
    ensure_float_sparse,
    laplacian_sanity_check,
    scipy_to_torch_sparse,
    torch_sparse_mm,
)


# Helpers
def _seed_all(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_spd_sparse(M: int, density: float = 0.15, seed: int = 0) -> sp.csr_matrix:
    """
    Make a symmetric positive definite sparse matrix (good as Laplacian-like operator).
    """
    rng = np.random.default_rng(seed)
    A = sp.random(M, M, density=density, data_rvs=rng.standard_normal, format="csr")
    A = (A + A.T) * 0.5
    # Make diagonally dominant => SPD-ish
    diag = np.asarray(np.abs(A).sum(axis=1)).ravel()
    A = A + sp.diags(diag + 1.0, format="csr")
    return A.tocsr()


def _naive_assemble_powers(K: int, L: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Naive reference implementation of power basis:
        [x, Lx, L^2x, ..., L^(K-1)x]
    x: (B, C, M)
    L: (M, M) sparse
    returns: (B, C, M, K)
    """
    B, C, M = x.shape
    outs = []
    cur = x
    for k in range(K):
        outs.append(cur.unsqueeze(-1))  # (B,C,M,1)
        if k < K - 1:
            # Apply L to each (b,c) separately
            next_cur = torch.empty_like(cur)
            for b in range(B):
                for c in range(C):
                    v = cur[b, c, :].reshape(M, 1)
                    next_cur[b, c, :] = torch.sparse.mm(L, v).reshape(-1)
            cur = next_cur
    return torch.cat(outs, dim=-1)


# Tests: utils
def test_scipy_to_torch_sparse_matches_dense():
    _seed_all(1)
    M = 12
    A = sp.random(M, M, density=0.2, format="csr", random_state=1)
    A = A + sp.eye(M, format="csr")  # ensure diagonal exists
    T = scipy_to_torch_sparse(A)

    assert T.is_sparse
    assert tuple(T.shape) == (M, M)

    dense_t = T.to_dense().cpu().numpy()
    dense_a = A.toarray()
    np.testing.assert_allclose(dense_t, dense_a, rtol=0, atol=1e-7)


def test_coo2tensor_accepts_csr_and_coo():
    _seed_all(2)
    M = 10
    A_csr = sp.random(M, M, density=0.2, format="csr", random_state=2)
    A_coo = A_csr.tocoo()

    t1 = coo2tensor(A_csr)
    t2 = coo2tensor(A_coo)

    np.testing.assert_allclose(t1.to_dense().cpu().numpy(), A_csr.toarray(), atol=1e-7)
    np.testing.assert_allclose(t2.to_dense().cpu().numpy(), A_csr.toarray(), atol=1e-7)


def test_ensure_float_sparse_casts_int_and_respects_dtype():
    A_int = sp.csr_matrix(np.array([[1, 0], [0, 2]], dtype=np.int64))
    A_out = ensure_float_sparse(A_int)
    assert A_out.dtype == np.float32

    A_f32 = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32))
    A_same = ensure_float_sparse(A_f32)
    assert A_same.dtype == np.float32

    A_f64 = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64))
    A_to_f64 = ensure_float_sparse(A_f64, dtype=np.float64)
    assert A_to_f64.dtype == np.float64


def test_laplacian_sanity_check_raises_on_invalid_inputs():
    with pytest.raises(TypeError):
        laplacian_sanity_check(np.eye(3))

    with pytest.raises(ValueError):
        laplacian_sanity_check(sp.csr_matrix((0, 0)))

    with pytest.raises(ValueError):
        laplacian_sanity_check(sp.csr_matrix(np.ones((2, 3))))


def test_torch_sparse_mm_rejects_invalid_tensor_types():
    A = scipy_to_torch_sparse(sp.eye(4, format="csr"))
    X = torch.randn(4, 2)
    Y = torch_sparse_mm(A, X)
    assert tuple(Y.shape) == (4, 2)

    with pytest.raises(TypeError):
        torch_sparse_mm(torch.eye(4), X)

    with pytest.raises(TypeError):
        torch_sparse_mm(A, A)


# Tests: chebyshev / assemble
def test_assemble_powers_shape_and_values_match_naive():
    _seed_all(3)
    B, C, M, K = 2, 3, 25, 4

    L_scipy = _make_spd_sparse(M, density=0.12, seed=3)
    L = scipy_to_torch_sparse(L_scipy).coalesce()

    x = torch.randn(B, C, M)

    X_fast = assemble_powers(K, L, x)
    X_ref = _naive_assemble_powers(K, L, x)

    assert tuple(X_fast.shape) == (B, C, M, K)
    assert tuple(X_ref.shape) == (B, C, M, K)

    # Power basis should match numerically (small float error)
    torch.testing.assert_close(X_fast, X_ref, rtol=1e-5, atol=1e-6)


def test_assemble_powers_k1_returns_input_unsqueezed():
    _seed_all(33)
    B, C, M = 2, 2, 7
    x = torch.randn(B, C, M)
    L = scipy_to_torch_sparse(_make_spd_sparse(M, density=0.2, seed=33)).coalesce()

    X = assemble_powers(1, L, x)
    assert tuple(X.shape) == (B, C, M, 1)
    torch.testing.assert_close(X[..., 0], x, rtol=0, atol=0)


def test_assemble_powers_raises_on_invalid_inputs():
    x = torch.randn(2, 2, 5)
    L = scipy_to_torch_sparse(_make_spd_sparse(5, density=0.3, seed=34)).coalesce()

    with pytest.raises(ValueError):
        assemble_powers(0, L, x)

    with pytest.raises(TypeError):
        assemble_powers(2, torch.eye(5), x)

    with pytest.raises(ValueError):
        assemble_powers(2, L, torch.randn(2, 5))

    with pytest.raises(ValueError):
        assemble_powers(2, L, torch.randn(2, 2, 6))


def test_normalize_laplacian_shapes_and_diagonal_shift():
    _seed_all(4)
    M = 20
    L = _make_spd_sparse(M, density=0.1, seed=4)

    Lh = normalize_laplacian(L, half_interval=False)
    assert Lh.shape == (M, M)

    # Check that half_interval=False corresponds to scaled(L) - I on diagonal:
    # diag(Lh) = (2/lam_max)*diag(L) - 1
    # We don't need perfect equality due to float operations, but should be close.
    from scipy.sparse.linalg import eigsh

    lam_max = float(eigsh(L, k=1, which="LM", return_eigenvectors=False)[0])
    diag_expected = (2.0 / lam_max) * L.diagonal() - 1.0
    np.testing.assert_allclose(Lh.diagonal(), diag_expected, rtol=1e-6, atol=1e-6)

    Lh_half = normalize_laplacian(L, half_interval=True)
    diag_expected_half = (1.0 / lam_max) * L.diagonal()
    np.testing.assert_allclose(Lh_half.diagonal(), diag_expected_half, rtol=1e-6, atol=1e-6)


def test_normalize_like_uses_ref_eigenvalue_and_part_shape():
    _seed_all(5)
    M = 18
    L_ref = _make_spd_sparse(M, density=0.15, seed=5)

    # Make a "part" matrix with same shape but different scale
    L_part = (0.25 * L_ref).tocsr()

    Lp_hat = normalize_like(L_ref, L_part, half_interval=False)
    assert Lp_hat.shape == (M, M)

    from scipy.sparse.linalg import eigsh

    lam_ref = float(eigsh(L_ref, k=1, which="LM", return_eigenvectors=False)[0])
    diag_expected = (2.0 / lam_ref) * L_part.diagonal() - 1.0
    np.testing.assert_allclose(Lp_hat.diagonal(), diag_expected, rtol=1e-6, atol=1e-6)


def test_normalize_like_half_interval_diagonal():
    _seed_all(55)
    M = 14
    L_ref = _make_spd_sparse(M, density=0.15, seed=55)
    L_part = (0.3 * L_ref).tocsr()

    from scipy.sparse.linalg import eigsh

    lam_ref = float(eigsh(L_ref, k=1, which="LM", return_eigenvectors=False)[0])
    out = normalize_like(L_ref, L_part, half_interval=True)
    np.testing.assert_allclose(
        out.diagonal(),
        (1.0 / lam_ref) * L_part.diagonal(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_normalize_laplacian_eps_guard_avoids_division_by_zero():
    L_zero = sp.csr_matrix((5, 5), dtype=np.float64)
    out = normalize_laplacian(L_zero, half_interval=True, eps=1e-3)
    np.testing.assert_allclose(out.toarray(), np.zeros((5, 5)), rtol=0, atol=0)


# Tests: scnn
def test_simplicial_convolution_forward_shape_and_grad():
    _seed_all(6)
    B, C_in, C_out, M, K = 2, 2, 4, 30, 3

    L_scipy = _make_spd_sparse(M, density=0.12, seed=6)
    L = scipy_to_torch_sparse(L_scipy).coalesce()

    x = torch.randn(B, C_in, M, requires_grad=True)

    layer = SimplicialConvolution(K=K, C_in=C_in, C_out=C_out, variance=0.1)
    y = layer(L, x)

    assert tuple(y.shape) == (B, C_out, M)

    loss = y.pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(layer.theta.grad).all()


def test_simplicial_convolution_no_bias_works():
    _seed_all(66)
    B, C_in, C_out, M, K = 2, 2, 3, 12, 3
    L = scipy_to_torch_sparse(_make_spd_sparse(M, density=0.2, seed=66)).coalesce()
    x = torch.randn(B, C_in, M)

    layer = SimplicialConvolution(
        K=K, C_in=C_in, C_out=C_out, enable_bias=False, variance=0.1
    )
    y = layer(L, x)
    assert tuple(y.shape) == (B, C_out, M)
    assert layer.bias.ndim == 0


def test_simplicial_convolution_input_validation():
    L = scipy_to_torch_sparse(_make_spd_sparse(8, density=0.2, seed=67)).coalesce()
    x = torch.randn(2, 1, 8)

    with pytest.raises(ValueError):
        SimplicialConvolution(K=0, C_in=1, C_out=2)

    with pytest.raises(ValueError):
        SimplicialConvolution(K=2, C_in=1, C_out=2, groups=2)

    layer = SimplicialConvolution(K=2, C_in=1, C_out=2)

    with pytest.raises(TypeError):
        layer(torch.eye(8), x)

    with pytest.raises(ValueError):
        layer(L, torch.randn(2, 8))

    with pytest.raises(ValueError):
        layer(L, torch.randn(2, 2, 8))

    with pytest.raises(ValueError):
        layer(L, torch.randn(2, 1, 7))


def test_simplicial_convolution2_forward_shape_and_grad():
    _seed_all(7)
    B, C_in, C_out, M, K1, K2 = 2, 1, 3, 28, 2, 4

    L_full = _make_spd_sparse(M, density=0.12, seed=7)
    # Make two "parts" (not necessarily a true Hodge split, but shape-correct)
    Ll = (0.6 * L_full).tocsr()
    Lu = (0.4 * L_full).tocsr()

    Ll_t = scipy_to_torch_sparse(Ll).coalesce()
    Lu_t = scipy_to_torch_sparse(Lu).coalesce()

    x = torch.randn(B, C_in, M, requires_grad=True)

    layer = SimplicialConvolution2(K1=K1, K2=K2, C_in=C_in, C_out=C_out, variance=0.1)
    y = layer(Ll_t, Lu_t, x)

    assert tuple(y.shape) == (B, C_out, M)

    loss = y.abs().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(layer.theta.grad).all()


def test_simplicial_convolution2_no_bias_and_validation():
    _seed_all(77)
    M = 10
    L = _make_spd_sparse(M, density=0.2, seed=77)
    Ll_t = scipy_to_torch_sparse((0.5 * L).tocsr()).coalesce()
    Lu_t = scipy_to_torch_sparse((0.5 * L).tocsr()).coalesce()
    x = torch.randn(2, 1, M)

    layer = SimplicialConvolution2(
        K1=2, K2=2, C_in=1, C_out=2, enable_bias=False, variance=0.1
    )
    y = layer(Ll_t, Lu_t, x)
    assert tuple(y.shape) == (2, 2, M)
    assert layer.bias.ndim == 0

    with pytest.raises(ValueError):
        SimplicialConvolution2(K1=0, K2=1, C_in=1, C_out=2)

    with pytest.raises(ValueError):
        SimplicialConvolution2(K1=1, K2=1, C_in=1, C_out=2, groups=2)

    with pytest.raises(TypeError):
        layer(torch.eye(M), Lu_t, x)

    with pytest.raises(TypeError):
        layer(Ll_t, torch.eye(M), x)

    with pytest.raises(ValueError):
        layer(Ll_t, Lu_t, torch.randn(2, M))

    with pytest.raises(ValueError):
        layer(Ll_t, Lu_t, torch.randn(2, 2, M))

    with pytest.raises(ValueError):
        layer(Ll_t, Lu_t, torch.randn(2, 1, M + 1))


def test_coboundary_matches_naive_loop():
    _seed_all(8)
    B, C_in, C_out, M, N = 2, 3, 4, 15, 11

    # Random sparse D (N x M)
    rng = np.random.default_rng(8)
    D = sp.random(N, M, density=0.25, data_rvs=rng.standard_normal, format="csr")
    D_t = scipy_to_torch_sparse(D).coalesce()

    x = torch.randn(B, C_in, M, requires_grad=True)

    layer = Coboundary(C_in=C_in, C_out=C_out, variance=0.1)
    y_fast = layer(D_t, x)
    assert tuple(y_fast.shape) == (B, C_out, N)

    # Naive reference following the repo structure
    X_list = []
    for b in range(B):
        X12 = []
        for c in range(C_in):
            v = x[b, c, :].unsqueeze(1)              # (M,1)
            out = torch.sparse.mm(D_t, v).T          # (1,N)
            X12.append(out)
        X12 = torch.cat(X12, dim=0)                  # (C_in,N)
        X_list.append(X12.unsqueeze(0))              # (1,C_in,N)

    X_ref = torch.cat(X_list, dim=0)                 # (B,C_in,N)
    y_ref = torch.einsum("oi,bin->bon", (layer.theta, X_ref)) + layer.bias

    torch.testing.assert_close(y_fast, y_ref, rtol=1e-5, atol=1e-6)

    # Gradient sanity
    loss = y_fast.pow(2).mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_coboundary_no_bias_and_validation():
    _seed_all(88)
    B, C_in, C_out, M, N = 2, 2, 3, 9, 7
    D = sp.random(N, M, density=0.3, random_state=88, format="csr")
    D_t = scipy_to_torch_sparse(D).coalesce()
    x = torch.randn(B, C_in, M)

    layer = Coboundary(C_in=C_in, C_out=C_out, enable_bias=False, variance=0.1)
    y = layer(D_t, x)
    assert tuple(y.shape) == (B, C_out, N)
    assert layer.bias.ndim == 0

    with pytest.raises(ValueError):
        Coboundary(C_in=0, C_out=2)

    with pytest.raises(TypeError):
        layer(torch.eye(N, M), x)

    with pytest.raises(ValueError):
        layer(D_t, torch.randn(B, M))

    with pytest.raises(ValueError):
        layer(D_t, torch.randn(B, C_in + 1, M))

    with pytest.raises(ValueError):
        layer(D_t, torch.randn(B, C_in, M + 1))
