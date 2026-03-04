from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from .chebyshev import assemble_powers, assemble_chebyshev
from .utils import scipy_to_torch_sparse, torch_sparse_mm


def coo2tensor(
    A: sp.spmatrix,
    *,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if not sp.isspmatrix(A):
        raise TypeError("A must be a SciPy sparse matrix")
    return scipy_to_torch_sparse(A, dtype=dtype, device=device)


class SimplicialConvolution(nn.Module):
    """
    Polynomial filter on a single Laplacian.

    Input:
        L: torch sparse (M x M)
        x: torch dense  (B x C_in x M)

    Output:
        y: torch dense (B x C_out x M)
    """
    def __init__(
        self,
        K: int,
        C_in: int,
        C_out: int,
        enable_bias: bool = True,
        variance: float = 1.0,
        groups: int = 1,
    ):
        if groups != 1:
            raise ValueError("Only groups=1 is currently supported.")
        super().__init__()

        if C_in <= 0 or C_out <= 0 or K <= 0:
            raise ValueError("C_in, C_out, and K must be > 0")

        self.C_in = int(C_in)
        self.C_out = int(C_out)
        self.K = int(K)
        self.enable_bias = bool(enable_bias)

        self.theta = nn.Parameter(variance * torch.randn((self.C_out, self.C_in, self.K)))

        if self.enable_bias:
            self.bias = nn.Parameter(torch.zeros((1, self.C_out, 1)))
        else:
            self.register_buffer("bias", torch.tensor(0.0), persistent=False)

    def forward(self, L: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must have shape (B, C_in, M), got {tuple(x.shape)}")
        if not L.is_sparse:
            raise TypeError("L must be a torch sparse tensor")

        B, C_in, M = x.shape
        if C_in != self.C_in:
            raise ValueError(f"C_in mismatch: got {C_in}, expected {self.C_in}")
        if L.shape != (M, M):
            raise ValueError(f"L shape {tuple(L.shape)} must match (M,M)=({M},{M})")

        X = assemble_powers(self.K, L, x)  # (B, C_in, M, K)
        y = torch.einsum("bimk,oik->bom", (X, self.theta))  # (B, C_out, M)
        return y + self.bias


class SimplicialConvolution2(nn.Module):
    """
    Polynomial filter on two Laplacians (e.g., lower & upper Laplacian).
    It concatenates the power stacks and learns one weight tensor over both.

    Input:
        Ll: torch sparse (M x M)
        Lu: torch sparse (M x M)
        x : torch dense  (B x C_in x M)

    Output:
        y : torch dense  (B x C_out x M)
    """
    def __init__(
        self,
        K1: int,
        K2: int,
        C_in: int,
        C_out: int,
        *,
        enable_bias: bool = True,
        variance: float = 1.0,
        groups: int = 1,
    ):
        if groups != 1:
            raise ValueError("Only groups=1 is currently supported.")
        super().__init__()

        if C_in <= 0 or C_out <= 0 or K1 <= 0 or K2 <= 0:
            raise ValueError("C_in, C_out, K1, K2 must be > 0")

        self.C_in = int(C_in)
        self.C_out = int(C_out)
        self.K1 = int(K1)
        self.K2 = int(K2)
        self.enable_bias = bool(enable_bias)

        self.theta = nn.Parameter(
            variance * torch.randn((self.C_out, self.C_in, self.K1 + self.K2))
        )

        if self.enable_bias:
            self.bias = nn.Parameter(torch.zeros((1, self.C_out, 1)))
        else:
            self.register_buffer("bias", torch.tensor(0.0), persistent=False)

    def forward(self, Ll: torch.Tensor, Lu: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must have shape (B, C_in, M), got {tuple(x.shape)}")
        if (not Ll.is_sparse) or (not Lu.is_sparse):
            raise TypeError("Ll and Lu must be torch sparse tensors")

        B, C_in, M = x.shape
        if C_in != self.C_in:
            raise ValueError(f"C_in mismatch: got {C_in}, expected {self.C_in}")
        if Ll.shape != (M, M) or Lu.shape != (M, M):
            raise ValueError("Ll and Lu must both have shape (M, M) matching x")

        X1 = assemble_powers(self.K1, Ll, x)  # (B, C_in, M, K1)
        X2 = assemble_powers(self.K2, Lu, x)  # (B, C_in, M, K2)
        X = torch.cat((X1, X2), dim=3)        # (B, C_in, M, K1+K2)

        y = torch.einsum("bimk,oik->bom", (X, self.theta))  # (B, C_out, M)
        return y + self.bias

# This class does not yet implement the
# Laplacian-power-pre/post-composed with the coboundary. It can be
# simulated by just adding more layers anyway, so keeping it simple
# for now.
#
# Note: You can use this for a adjoints of coboundaries too. Just feed
# a transposed D.
class Coboundary(nn.Module):
    """
    Applies a coboundary/boundary-like operator D, then mixes channels.

    Input:
        D: torch sparse (N x M)
        x: torch dense  (B x C_in x M)

    Output:
        y: torch dense  (B x C_out x N)
    """
    def __init__(
        self,
        C_in: int,
        C_out: int,
        *,
        enable_bias: bool = True,
        variance: float = 1.0,
    ):
        super().__init__()

        if C_in <= 0 or C_out <= 0:
            raise ValueError("C_in and C_out must be > 0")

        self.C_in = int(C_in)
        self.C_out = int(C_out)
        self.enable_bias = bool(enable_bias)

        self.theta = nn.Parameter(variance * torch.randn((self.C_out, self.C_in)))

        if self.enable_bias:
            self.bias = nn.Parameter(torch.zeros((1, self.C_out, 1)))
        else:
            self.register_buffer("bias", torch.tensor(0.0), persistent=False)

    def forward(self, D: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if not D.is_sparse:
            raise TypeError("D must be a torch sparse tensor")
        if x.ndim != 3:
            raise ValueError(f"x must have shape (B, C_in, M), got {tuple(x.shape)}")

        B, C_in, M = x.shape
        if C_in != self.C_in:
            raise ValueError(f"C_in mismatch: got {C_in}, expected {self.C_in}")
        if D.shape[1] != M:
            raise ValueError(f"D second dimension must be M={M}, got {D.shape[1]}")

        N = D.shape[0]

        X0 = x.permute(2, 0, 1).reshape(M, B * C_in)      # (M x (B*C_in))
        Y0 = torch_sparse_mm(D, X0)                       # (N x (B*C_in))
        Y0 = Y0.reshape(N, B, C_in).permute(1, 2, 0)      # (B x C_in x N)

        y = torch.einsum("oi,bin->bon", (self.theta, Y0))  # (B x C_out x N)
        return y + self.bias