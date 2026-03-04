from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import eigsh

from .utils import ensure_float_sparse, laplacian_sanity_check, torch_sparse_mm


def normalize_laplacian(
    L: sparse.spmatrix,
    half_interval: bool = False,
    eps: float = 1e-12,
) -> sparse.spmatrix:
    """
    Normalize a (Hodge) Laplacian so its spectrum is bounded for polynomial filters.
    If half_interval=False:
       Roughly maps eigenvalues to [-1, 1]
    Else:
        Roughly maps eigenvalues to [0, 1]
        
    Args:
        L: SciPy sparse square matrix.
        half_interval: whether to skip the "-I" shift and 2x scaling.
        eps: safeguard if eigenvalue is tiny.

    Returns:
        Normalized SciPy sparse matrix with same shape as L.
    """
    laplacian_sanity_check(L)
    L = ensure_float_sparse(L).tocsr()

    # Largest magnitude eigenvalue
    topeig = float(eigsh(L, k=1, which="LM", return_eigenvectors=False)[0])
    topeig = max(topeig, eps)

    if half_interval:
        return (1.0 / topeig) * L

    # full interval scaling: 2/topeig * L - I
    M = L.shape[0]
    ret = (2.0 / topeig) * L
    # subtract identity
    ret = ret - sparse.identity(M, format="csr", dtype=ret.dtype)
    return ret


def normalize_like(
    L: sparse.spmatrix,
    Lx: sparse.spmatrix,
    *,
    half_interval: bool = False,
    eps: float = 1e-12,
) -> sparse.spmatrix:
    """
    Normalize Lx using top eigenvalue computed from L.

    Args:
        L: SciPy sparse square matrix to compute lambda_max from.
        Lx: SciPy sparse square matrix to scale.
        half_interval: same meaning as normalize_laplacian
        eps: eigenvalue floor

    Returns:
        Normalized Lx.
    """
    laplacian_sanity_check(L)
    laplacian_sanity_check(Lx)
    L = ensure_float_sparse(L).tocsr()
    Lx = ensure_float_sparse(Lx).tocsr()

    topeig = float(eigsh(L, k=1, which="LM", return_eigenvectors=False)[0])
    topeig = max(topeig, eps)

    if half_interval:
        return (1.0 / topeig) * Lx

    M = Lx.shape[0]
    ret = (2.0 / topeig) * Lx
    ret = ret - sparse.identity(M, format="csr", dtype=ret.dtype)
    return ret


def assemble_powers(
    K: int,
    L: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Build the stack [x, Lx, L^2 x, ..., L^(K-1) x] used by the SCNN polynomial filter.

    Args:
        K: number of powers (K >= 1)
        L: torch sparse (M x M) Laplacian
        x: dense tensor (B x C_in x M)

    Returns:
        X: dense tensor (B x C_in x M x K)
    """
    if K < 1:
        raise ValueError("K must be >= 1")
    if not L.is_sparse:
        raise TypeError("L must be a torch sparse tensor (M x M)")
    if x.is_sparse:
        raise TypeError("x must be dense (B x C x M)")
    if x.ndim != 3:
        raise ValueError(f"x must have shape (B, C, M), got {tuple(x.shape)}")

    B, C, M = x.shape
    if L.shape != (M, M):
        raise ValueError(f"L shape {tuple(L.shape)} must match (M,M)=({M},{M})")

    Xk = x.permute(2, 0, 1).reshape(M, B * C)

    outs = [Xk]  # list of (M x (B*C))

    for _ in range(1, K):
        Xk = torch_sparse_mm(L, Xk)  # (M x (B*C))
        outs.append(Xk)

    # Stack along last axis
    stacked = torch.stack(outs, dim=-1)

    # Reshape back to (B x C x M x K)
    stacked = stacked.reshape(M, B, C, K).permute(1, 2, 0, 3).contiguous()
    return stacked

def assemble_chebyshev(
    K: int,
    L_hat: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Build [T_0(L_hat)x, T_1(L_hat)x, ..., T_{K-1}(L_hat)x] using Chebyshev recurrence.

    Args:
        K: number of terms (K>=1)
        L_hat: torch sparse (M x M), should be normalized to have spectrum in [-1,1]
        x: dense (B x C x M)

    Returns:
        X: dense (B x C x M x K)
    """
    if K < 1:
        raise ValueError("K must be >= 1")
    if not L_hat.is_sparse:
        raise TypeError("L_hat must be a torch sparse tensor")
    if x.ndim != 3:
        raise ValueError(f"x must be (B,C,M), got {tuple(x.shape)}")

    B, C, M = x.shape
    if L_hat.shape != (M, M):
        raise ValueError(f"L_hat shape {tuple(L_hat.shape)} must match (M,M)=({M},{M})")

    # Columns: (M x (B*C))
    X0 = x.permute(2, 0, 1).reshape(M, B * C)  # T0
    outs = [X0]

    if K == 1:
        stacked = torch.stack(outs, dim=-1)
        return stacked.reshape(M, B, C, K).permute(1, 2, 0, 3).contiguous()

    X1 = torch_sparse_mm(L_hat, X0)            # T1 = L_hat * T0
    outs.append(X1)

    for _ in range(2, K):
        # Tk = 2 L_hat T_{k-1} - T_{k-2}
        X2 = 2.0 * torch_sparse_mm(L_hat, X1) - X0
        outs.append(X2)
        X0, X1 = X1, X2

    stacked = torch.stack(outs, dim=-1)        # (M x (B*C) x K)
    return stacked.reshape(M, B, C, K).permute(1, 2, 0, 3).contiguous()