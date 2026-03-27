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
    """
    Convert a SciPy sparse matrix to a torch sparse COO tensor.

    Args:
        A: SciPy sparse matrix to convert.
        dtype: Target torch dtype for values.
        device: Optional target device.

    Returns:
        Torch sparse tensor with the same shape/non-zeros as `A`.
    """
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
    ):
        """
        Initialize a simplicial polynomial convolution layer.

        Args:
            orders: Polynomial order(s), one per operator.
            C_in: Number of input channels.
            C_out: Number of output channels.
            basis: Polynomial basis, either "power" or "chebyshev".
            operators: Optional default sparse operators to register.
            enable_bias: Whether to learn an additive bias.
            variance: Multiplicative scale applied after Xavier init.

        Returns:
            None.
        """
        super().__init__()
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
        """
        Register default sparse operators as non-persistent buffers.

        Args:
            operators: Sparse square operators, aligned with `self.orders`.

        Returns:
            None.
        """
        if len(operators) != len(self.orders):
            raise ValueError("operators length must match number of orders")
        for i, op in enumerate(operators):
            if not op.is_sparse:
                raise TypeError("All operators must be torch sparse tensors")
            self.register_buffer(f"_op_{i}", op.coalesce(), persistent=False)
        self._registered_ops_count = len(operators)

    def _get_registered_operators(self) -> list[torch.Tensor]:
        """
        Return operators previously registered on this module.

        Returns:
            List of registered sparse operators.
        """
        return [getattr(self, f"_op_{i}") for i in range(self._registered_ops_count)]

    def reset_parameters(self) -> None:
        """Initialize learnable parameters."""
        nn.init.xavier_uniform_(self.theta)
        if self.variance != 1.0:
            with torch.no_grad():
                self.theta.mul_(self.variance)
        if self.enable_bias:
            nn.init.zeros_(self.bias)

    def extra_repr(self) -> str:
        """
        Provide a compact representation string for module printing.
        
        Returns:
            Human-readable summary of key hyperparameters.
        """
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
        """
        Resolve runtime operators from explicit args or registered defaults.

        Args:
            operators: Explicit operator list, if provided.
            L: Single operator shortcut.
            Ll: Lower operator for two-operator mode.
            Lu: Upper operator for two-operator mode.

        Returns:
            List of sparse operators aligned with `self.orders`.
        """
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
        """
        Apply the convolution using one or multiple sparse operators.

        Args:
            x: Input tensor of shape (B, C_in, M).
            operators: Optional operator list to use in this call.
            L: Single-operator shortcut.
            Ll: Lower operator in two-operator mode.
            Lu: Upper operator in two-operator mode.

        Returns:
            Output tensor of shape (B, C_out, M).
        """
        if x.ndim != 3:
            raise ValueError(f"x must have shape (B, C_in, M), got {tuple(x.shape)}")

        _, C_in, M = x.shape
        if C_in != self.C_in:
            raise ValueError(f"C_in mismatch: got {C_in}, expected {self.C_in}")

        ops = self._resolve_operators(operators, L, Ll, Lu)
        for op in ops:
            if op.shape != (M, M):
                raise ValueError(f"Operator shape {tuple(op.shape)} must match ({M}, {M})")

        # Build per-operator polynomial features and concatenate along order axis.
        assemble = assemble_powers if self.basis == "power" else assemble_chebyshev
        stacks = [assemble(k, op, x) for k, op in zip(self.orders, ops)]
        X = stacks[0] if len(stacks) == 1 else torch.cat(stacks, dim=3)
        y = torch.einsum("bimk,oik->bom", (X, self.theta))
        return y + self.bias


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
        """
        Initialize a coboundary convolution layer.

        Args:
            C_in: Number of input channels.
            C_out: Number of output channels.
            enable_bias: Whether to learn an additive bias.
            variance: Scale for random initialization of channel mixing.

        Returns:
            None.
        """
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
        """
        Apply sparse coboundary operator `D` then channel mixing.

        Args:
            D: Sparse operator of shape (N, M).
            x: Input tensor of shape (B, C_in, M).

        Returns:
            Output tensor of shape (B, C_out, N).
        """
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

        # Flatten (B, C) into one axis so sparse matmul is done in one call.
        X0 = x.permute(2, 0, 1).reshape(M, B * C_in)
        Y0 = torch_sparse_mm(D, X0)
        Y0 = Y0.reshape(N, B, C_in).permute(1, 2, 0)

        y = torch.einsum("oi,bin->bon", (self.theta, Y0))
        return y + self.bias
