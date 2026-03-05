from __future__ import annotations

import warnings
from typing import Optional, Sequence

import scipy.sparse as sp
import torch
import torch.nn as nn

from .chebyshev import assemble_chebyshev, assemble_powers
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


class SimplicialConv(nn.Module):
    """
    Unified simplicial polynomial convolution over one or more operators.

    Input:
        x: torch dense (B x C_in x M)
        operators: list of sparse operators [(M x M), ...]

    Output:
        y: torch dense (B x C_out x M)
    """

    def __init__(
        self,
        orders: int | Sequence[int],
        C_in: int,
        C_out: int,
        *,
        basis: str = "power",
        operators: Optional[Sequence[torch.Tensor]] = None,
        enable_bias: bool = True,
        variance: float = 1.0,
        groups: int = 1,
    ):
        super().__init__()
        if groups != 1:
            raise ValueError("Only groups=1 is currently supported.")
        if C_in <= 0 or C_out <= 0:
            raise ValueError("C_in and C_out must be > 0")
        if isinstance(orders, int):
            orders = [orders]
        if len(orders) == 0 or any(k <= 0 for k in orders):
            raise ValueError("orders must be a non-empty list of positive ints")
        if basis not in {"power", "chebyshev"}:
            raise ValueError("basis must be 'power' or 'chebyshev'")

        self.C_in = int(C_in)
        self.C_out = int(C_out)
        self.orders = tuple(int(k) for k in orders)
        self.basis = basis
        self.enable_bias = bool(enable_bias)
        self.variance = float(variance)

        self.theta = nn.Parameter(torch.empty((self.C_out, self.C_in, sum(self.orders))))
        if self.enable_bias:
            self.bias = nn.Parameter(torch.zeros((1, self.C_out, 1)))
        else:
            self.register_buffer("bias", torch.tensor(0.0), persistent=False)

        self._registered_ops_count = 0
        if operators is not None:
            self._register_operators(operators)

        self.reset_parameters()

    def _register_operators(self, operators: Sequence[torch.Tensor]) -> None:
        if len(operators) != len(self.orders):
            raise ValueError("operators length must match number of orders")
        for i, op in enumerate(operators):
            if not op.is_sparse:
                raise TypeError("All operators must be torch sparse tensors")
            self.register_buffer(f"_op_{i}", op.coalesce(), persistent=False)
        self._registered_ops_count = len(operators)

    def _get_registered_operators(self) -> list[torch.Tensor]:
        return [getattr(self, f"_op_{i}") for i in range(self._registered_ops_count)]

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.theta)
        if self.variance != 1.0:
            with torch.no_grad():
                self.theta.mul_(self.variance)
        if self.enable_bias:
            nn.init.zeros_(self.bias)

    def extra_repr(self) -> str:
        return (
            f"C_in={self.C_in}, C_out={self.C_out}, orders={list(self.orders)}, "
            f"basis='{self.basis}', bias={self.enable_bias}"
        )

    def _resolve_operators(
        self,
        operators: Optional[Sequence[torch.Tensor]],
        L: Optional[torch.Tensor],
        Ll: Optional[torch.Tensor],
        Lu: Optional[torch.Tensor],
    ) -> list[torch.Tensor]:
        if operators is not None:
            ops = list(operators)
        elif L is not None:
            ops = [L]
        elif (Ll is not None) or (Lu is not None):
            if (Ll is None) or (Lu is None):
                raise ValueError("Both Ll and Lu must be provided together")
            ops = [Ll, Lu]
        else:
            ops = self._get_registered_operators()

        if len(ops) != len(self.orders):
            raise ValueError(f"Expected {len(self.orders)} operators, got {len(ops)}")
        for op in ops:
            if not op.is_sparse:
                raise TypeError("All operators must be torch sparse tensors")
        return ops

    def forward(
        self,
        x: torch.Tensor,
        operators: Optional[Sequence[torch.Tensor]] = None,
        *,
        L: Optional[torch.Tensor] = None,
        Ll: Optional[torch.Tensor] = None,
        Lu: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must have shape (B, C_in, M), got {tuple(x.shape)}")

        _, C_in, M = x.shape
        if C_in != self.C_in:
            raise ValueError(f"C_in mismatch: got {C_in}, expected {self.C_in}")

        ops = self._resolve_operators(operators, L, Ll, Lu)
        for op in ops:
            if op.shape != (M, M):
                raise ValueError(f"Operator shape {tuple(op.shape)} must match ({M}, {M})")

        assemble = assemble_powers if self.basis == "power" else assemble_chebyshev
        stacks = [assemble(k, op, x) for k, op in zip(self.orders, ops)]
        X = stacks[0] if len(stacks) == 1 else torch.cat(stacks, dim=3)
        y = torch.einsum("bimk,oik->bom", (X, self.theta))
        return y + self.bias


class SimplicialConvolution(nn.Module):
    """Deprecated alias for SimplicialConv."""

    def __init__(
        self,
        K: int,
        C_in: int,
        C_out: int,
        enable_bias: bool = True,
        variance: float = 1.0,
        groups: int = 1,
        basis: str = "power",
        L: Optional[torch.Tensor] = None,
    ):
        warnings.warn(
            "SimplicialConvolution is deprecated; use SimplicialConv instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()
        self._impl = SimplicialConv(
            orders=K,
            C_in=C_in,
            C_out=C_out,
            basis=basis,
            operators=[L] if L is not None else None,
            enable_bias=enable_bias,
            variance=variance,
            groups=groups,
        )

    @property
    def theta(self) -> nn.Parameter:
        return self._impl.theta

    @property
    def bias(self) -> torch.Tensor:
        return self._impl.bias

    def forward(self, *args, **kwargs) -> torch.Tensor:
        if len(args) == 2:
            L, x = args
            return self._impl(x, L=L, **kwargs)
        if len(args) == 1:
            (x,) = args
            return self._impl(x, **kwargs)
        raise TypeError("Expected forward(L, x) or forward(x, L=...)")


class SimplicialConvolution2(nn.Module):
    """Deprecated alias for SimplicialConv."""

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
        basis: str = "power",
        Ll: Optional[torch.Tensor] = None,
        Lu: Optional[torch.Tensor] = None,
    ):
        warnings.warn(
            "SimplicialConvolution2 is deprecated; use SimplicialConv instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()
        operators = None
        if (Ll is not None) and (Lu is not None):
            operators = [Ll, Lu]
        self._impl = SimplicialConv(
            orders=[K1, K2],
            C_in=C_in,
            C_out=C_out,
            basis=basis,
            operators=operators,
            enable_bias=enable_bias,
            variance=variance,
            groups=groups,
        )

    @property
    def theta(self) -> nn.Parameter:
        return self._impl.theta

    @property
    def bias(self) -> torch.Tensor:
        return self._impl.bias

    def forward(self, *args, **kwargs) -> torch.Tensor:
        if len(args) == 3:
            Ll, Lu, x = args
            return self._impl(x, Ll=Ll, Lu=Lu, **kwargs)
        if len(args) == 1:
            (x,) = args
            return self._impl(x, **kwargs)
        raise TypeError("Expected forward(Ll, Lu, x) or forward(x, Ll=..., Lu=...)")


# This class does not yet implement the
# Laplacian-power-pre/post-composed with the coboundary. It can be
# simulated by just adding more layers anyway, so keeping it simple
# for now.
#
# Note: You can use this for adjoints of coboundaries too. Just feed
# a transposed D.
class CoboundaryConv(nn.Module):
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

        X0 = x.permute(2, 0, 1).reshape(M, B * C_in)
        Y0 = torch_sparse_mm(D, X0)
        Y0 = Y0.reshape(N, B, C_in).permute(1, 2, 0)

        y = torch.einsum("oi,bin->bon", (self.theta, Y0))
        return y + self.bias


class Coboundary(CoboundaryConv):
    """Deprecated alias for CoboundaryConv."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "Coboundary is deprecated; use CoboundaryConv instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
