from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from scipy import sparse


def scipy_to_torch_sparse(
    A: sparse.spmatrix,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Convert a SciPy sparse matrix to a PyTorch sparse COO tensor (coalesced).

    Args:
        A: SciPy sparse matrix (CSR/CSC/COO/etc.)
        dtype: torch dtype for values
        device: torch device

    Returns:
        torch.sparse_coo_tensor with shape A.shape
    """
    if not sparse.isspmatrix(A):
        raise TypeError("A must be a SciPy sparse matrix")
    A = A.tocoo()

    # indices: (2, nnz)
    idx = torch.tensor(
        np.vstack((A.row, A.col)),
        dtype=torch.long,
        device=device,
    )
    val = torch.tensor(A.data, dtype=dtype, device=device)

    t = torch.sparse_coo_tensor(idx, val, size=A.shape, device=device)
    return t.coalesce()


def torch_sparse_mm(A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """
    Multiply sparse matrix A (M x M) with dense X (M x K) -> (M x K).
    A must be torch sparse COO/CSR tensor; X dense.
    """
    if not A.is_sparse:
        raise TypeError("A must be a torch sparse tensor")
    if X.is_sparse:
        raise TypeError("X must be a dense tensor")
    return torch.sparse.mm(A, X)


def laplacian_sanity_check(L: sparse.spmatrix) -> None:
    """Basic sanity checks for Laplacian-like operators."""
    if not sparse.isspmatrix(L):
        raise TypeError("L must be a SciPy sparse matrix")
    m, n = L.shape
    if m != n:
        raise ValueError(f"L must be square, got {L.shape}")
    if m == 0:
        raise ValueError("L is empty (0x0)")


def ensure_float_sparse(A: sparse.spmatrix, dtype=np.float32) -> sparse.spmatrix:
    """Ensure SciPy sparse matrix is float64/float32 (not int/bool)."""
    if not sparse.isspmatrix(A):
        raise TypeError("A must be a SciPy sparse matrix")
    if A.dtype.kind in ("i", "u", "b") or A.dtype != np.dtype(dtype):
        return A.astype(dtype, copy=False)
    return A
