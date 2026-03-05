from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from .chebyshev import normalize_like
from .scnn import SimplicialConv
from .utils import scipy_to_torch_sparse


@dataclass
class SimplicialBatch:
    """
    Container for multi-rank SCNN inputs.

    Typical usage stores one tensor/operator per simplicial rank.
    """

    xs: list[torch.Tensor]
    Ls: Optional[list[torch.Tensor]] = None
    Lls: Optional[list[torch.Tensor]] = None
    Lus: Optional[list[torch.Tensor]] = None
    masks: Optional[list[list[int]]] = None
    xs_target: Optional[list[torch.Tensor]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> "SimplicialBatch":
        def _move(items: Optional[list[torch.Tensor]]) -> Optional[list[torch.Tensor]]:
            if items is None:
                return None
            return [t.to(device) for t in items]

        return SimplicialBatch(
            xs=_move(self.xs) or [],
            Ls=_move(self.Ls),
            Lls=_move(self.Lls),
            Lus=_move(self.Lus),
            masks=self.masks,
            xs_target=_move(self.xs_target),
            metadata=dict(self.metadata),
        )

    def mask_cochains(
        self,
        missing_pct: float,
        *,
        seed: Optional[int] = None,
        fill_strategy: str = "median",
    ) -> "SimplicialBatch":
        """
        Mask current cochains and store both masked tensors and known-index masks.
        """
        masked_xs, masks = mask_cochains(
            self.xs,
            missing_pct,
            seed=seed,
            fill_strategy=fill_strategy,
        )
        self.xs = masked_xs
        self.masks = masks
        self.metadata["missing_pct"] = float(missing_pct)
        self.metadata["fill_strategy"] = fill_strategy
        return self

    def build_normalized_operators(
        self,
        sc: Any,
        topdim: int,
        *,
        half_interval: bool = True,
        eps: float = 1e-12,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> "SimplicialBatch":
        """
        Build and store normalized lower/upper operators for this batch.
        """
        Lls, Lus = build_normalized_operators(
            sc,
            topdim,
            half_interval=half_interval,
            eps=eps,
            dtype=dtype,
            device=device,
        )
        self.Lls = Lls
        self.Lus = Lus
        self.metadata["topdim"] = int(topdim)
        self.metadata["half_interval"] = bool(half_interval)
        return self


def build_cochains(
    sc: Any,
    topdim: int,
    *,
    batch_size: int = 1,
    channels: int = 1,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    fill_value: float = 0.0,
) -> list[torch.Tensor]:
    """
    Build dense cochain tensors (B x C x M_d) for d=0..topdim from simplex features.
    """
    if topdim < 0:
        raise ValueError("topdim must be >= 0")
    if batch_size <= 0 or channels <= 0:
        raise ValueError("batch_size and channels must be > 0")

    if not hasattr(sc, "_simplices_by_dim"):
        raise AttributeError("sc must expose '_simplices_by_dim' to align simplex ordering")

    xs: list[torch.Tensor] = []
    for d in range(topdim + 1):
        fmap = sc.get_simplex_features(d)
        order = sc._simplices_by_dim[d]
        vals = np.array([fmap.get(s, fill_value) for s in order], dtype=np.float32)
        base = torch.tensor(vals, dtype=dtype, device=device).view(1, 1, -1)
        xs.append(base.repeat(batch_size, channels, 1))
    return xs


def build_normalized_operators(
    sc: Any,
    topdim: int,
    *,
    half_interval: bool = True,
    eps: float = 1e-12,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Build normalized lower/upper operators per rank and convert to torch sparse tensors.
    """
    if topdim < 0:
        raise ValueError("topdim must be >= 0")

    Lls: list[torch.Tensor] = []
    Lus: list[torch.Tensor] = []
    for d in range(topdim + 1):
        L_full = sc.hodge_laplacian_matrix(rank=d)

        if d == 0:
            m0 = sc.shape[0]
            L_down = sp.csr_matrix((m0, m0))
        else:
            L_down = sc.lower_laplacian_matrix(rank=d)

        L_up = sc.upper_laplacian_matrix(rank=d)

        Ll = normalize_like(L_full, L_down, half_interval=half_interval, eps=eps)
        Lu = normalize_like(L_full, L_up, half_interval=half_interval, eps=eps)

        Lls.append(scipy_to_torch_sparse(Ll, dtype=dtype, device=device).coalesce())
        Lus.append(scipy_to_torch_sparse(Lu, dtype=dtype, device=device).coalesce())
    return Lls, Lus


def mask_cochains(
    xs: Sequence[torch.Tensor],
    missing_pct: float,
    *,
    seed: Optional[int] = 1337,
    fill_strategy: str = "median",
) -> tuple[list[torch.Tensor], list[list[int]]]:
    """
    Randomly mask entries per rank and fill missing values.

    Returns:
        masked_xs: same shapes as input tensors
        known_indices: list of kept indices per rank
    """
    if not 0.0 <= missing_pct < 1.0:
        raise ValueError("missing_pct must be in [0, 1)")
    if fill_strategy not in {"median", "mean", "zero"}:
        raise ValueError("fill_strategy must be one of: 'median', 'mean', 'zero'")

    rng = np.random.default_rng(seed)
    masked_xs: list[torch.Tensor] = []
    known_indices: list[list[int]] = []

    for x in xs:
        if x.ndim != 3:
            raise ValueError(f"Each cochain tensor must be (B,C,M), got {tuple(x.shape)}")

        M = int(x.shape[2])
        idx = np.arange(M)
        known_size = max(1, int((1.0 - missing_pct) * M))
        known_idx = np.sort(rng.choice(M, size=known_size, replace=False))
        missing_idx = np.setdiff1d(idx, known_idx)

        x_out = x.clone()
        known_tensor = torch.tensor(known_idx, dtype=torch.long, device=x.device)
        missing_tensor = torch.tensor(missing_idx, dtype=torch.long, device=x.device)

        if fill_strategy == "zero":
            fill = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        elif fill_strategy == "mean":
            fill = x_out.index_select(2, known_tensor).mean()
        else:
            fill = x_out.index_select(2, known_tensor).median()

        if missing_tensor.numel() > 0:
            x_out.index_fill_(2, missing_tensor, fill)

        masked_xs.append(x_out)
        known_indices.append(known_idx.tolist())

    return masked_xs, known_indices


class SimplicialConvBlock(nn.Module):
    """
    Stack of SimplicialConv layers with activation between layers.
    """

    def __init__(
        self,
        orders: int | Sequence[int],
        C_in: int,
        C_hidden: int,
        C_out: int,
        *,
        depth: int = 3,
        basis: str = "power",
        variance: float = 1.0,
        activation: Optional[nn.Module] = None,
        enable_bias: bool = True,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        if depth == 1:
            channels = [C_in, C_out]
        else:
            channels = [C_in] + [C_hidden] * (depth - 1) + [C_out]

        self.layers = nn.ModuleList(
            [
                SimplicialConv(
                    orders=orders,
                    C_in=channels[i],
                    C_out=channels[i + 1],
                    basis=basis,
                    variance=variance,
                    enable_bias=enable_bias,
                )
                for i in range(len(channels) - 1)
            ]
        )
        self.activation = activation if activation is not None else nn.LeakyReLU()

    def forward(
        self,
        x: torch.Tensor,
        *,
        operators: Optional[Sequence[torch.Tensor]] = None,
        L: Optional[torch.Tensor] = None,
        Ll: Optional[torch.Tensor] = None,
        Lu: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out, operators=operators, L=L, Ll=Ll, Lu=Lu)
            if i < len(self.layers) - 1:
                out = self.activation(out)
        return out


class SimplicialConvStack(nn.Module):
    """
    Per-rank stack of SimplicialConvBlock modules.
    """

    def __init__(
        self,
        topdim: int,
        *,
        orders: int | Sequence[int],
        colors: int = 1,
        num_filters: int = 30,
        depth: int = 3,
        basis: str = "power",
        variance: float = 0.01,
        activation: Optional[nn.Module] = None,
        enable_bias: bool = True,
    ):
        super().__init__()
        if topdim < 0:
            raise ValueError("topdim must be >= 0")
        if colors <= 0 or num_filters <= 0:
            raise ValueError("colors and num_filters must be > 0")

        hidden = num_filters * colors
        self.topdim = int(topdim)
        self.blocks = nn.ModuleList(
            [
                SimplicialConvBlock(
                    orders=orders,
                    C_in=colors,
                    C_hidden=hidden,
                    C_out=colors,
                    depth=depth,
                    basis=basis,
                    variance=variance,
                    activation=activation,
                    enable_bias=enable_bias,
                )
                for _ in range(topdim + 1)
            ]
        )

    def forward(
        self,
        xs: Sequence[torch.Tensor],
        *,
        Ls: Optional[Sequence[torch.Tensor]] = None,
        Lls: Optional[Sequence[torch.Tensor]] = None,
        Lus: Optional[Sequence[torch.Tensor]] = None,
        operators_by_dim: Optional[Sequence[Sequence[torch.Tensor]]] = None,
    ) -> list[torch.Tensor]:
        if len(xs) != self.topdim + 1:
            raise ValueError(f"Expected {self.topdim + 1} cochains, got {len(xs)}")

        ys: list[torch.Tensor] = []
        for d in range(self.topdim + 1):
            if operators_by_dim is not None:
                y = self.blocks[d](xs[d], operators=operators_by_dim[d])
            elif (Lls is not None) or (Lus is not None):
                if (Lls is None) or (Lus is None):
                    raise ValueError("Both Lls and Lus must be provided together")
                y = self.blocks[d](xs[d], Ll=Lls[d], Lu=Lus[d])
            elif Ls is not None:
                y = self.blocks[d](xs[d], L=Ls[d])
            else:
                y = self.blocks[d](xs[d])
            ys.append(y)
        return ys


class MaskedReconstructionTrainer:
    """
    Utility trainer for masked cochain reconstruction with SCNN-style models.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        criterion: nn.Module,
        *,
        optimizer_type: str = "adam",
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        zero_grad_set_to_none: bool = False,
        topdim: Optional[int] = None,
        realizations: int = 1,
    ):
        if realizations <= 0:
            raise ValueError("realizations must be > 0")
        if optimizer_kwargs is None:
            optimizer_kwargs = {"lr": 1e-3}
        self.model = model
        self.criterion = criterion
        self.zero_grad_set_to_none = bool(zero_grad_set_to_none)
        self.topdim = topdim
        self.realizations = int(realizations)
        self.optimizer_type = optimizer_type
        self.optimizer_kwargs = dict(optimizer_kwargs)
        self.optimizer = (
            optimizer
            if optimizer is not None
            else self._make_optimizer(self.optimizer_type, self.optimizer_kwargs)
        )

    def _make_optimizer(
        self,
        optimizer_type: str,
        optimizer_kwargs: dict[str, Any],
    ) -> torch.optim.Optimizer:
        opts: dict[str, type[torch.optim.Optimizer]] = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
            "adagrad": torch.optim.Adagrad,
        }
        key = optimizer_type.lower()
        if key not in opts:
            raise ValueError(
                f"Unsupported optimizer_type '{optimizer_type}'. "
                f"Choose one of {sorted(opts.keys())}."
            )
        return opts[key](self.model.parameters(), **optimizer_kwargs)

    def set_optimizer(
        self,
        optimizer_type: str,
        *,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        if optimizer_kwargs is None:
            optimizer_kwargs = {"lr": 1e-3}
        self.optimizer_type = optimizer_type
        self.optimizer_kwargs = dict(optimizer_kwargs)
        self.optimizer = self._make_optimizer(optimizer_type, self.optimizer_kwargs)

    def set_zero_grad_mode(self, *, set_to_none: bool = False) -> None:
        self.zero_grad_set_to_none = bool(set_to_none)

    def set_realizations(self, realizations: int) -> None:
        if realizations <= 0:
            raise ValueError("realizations must be > 0")
        self.realizations = int(realizations)

    @staticmethod
    def _clone_batch(batch: SimplicialBatch, topdim: int) -> SimplicialBatch:
        return SimplicialBatch(
            xs=[x.clone() for x in batch.xs[: topdim + 1]],
            Lls=batch.Lls[: topdim + 1] if batch.Lls is not None else None,
            Lus=batch.Lus[: topdim + 1] if batch.Lus is not None else None,
            masks=batch.masks[: topdim + 1] if batch.masks is not None else None,
            xs_target=(
                [x.clone() for x in batch.xs_target[: topdim + 1]]
                if batch.xs_target is not None
                else None
            ),
            metadata=dict(batch.metadata),
        )

    def fit(
        self,
        batch: SimplicialBatch,
        *,
        num_epochs: int,
        print_every: Optional[int] = None,
        realizations: Optional[int] = None,
        optimizer_type: Optional[str] = None,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        zero_grad_set_to_none: Optional[bool] = None,
        realization_now: Optional[int] = None
    ) -> dict[str, Any]:
        if num_epochs <= 0:
            raise ValueError("num_epochs must be > 0")
        if (print_every is not None) and (print_every <= 0):
            raise ValueError("print_every must be > 0 when provided")
        if batch.xs_target is None:
            raise ValueError("batch.xs_target is required for reconstruction training")
        if batch.masks is None:
            raise ValueError("batch.masks is required for masked training")
        if (batch.Lls is None) or (batch.Lus is None):
            raise ValueError("batch.Lls and batch.Lus are required")

        tdim = self.topdim if self.topdim is not None else (len(batch.xs) - 1)
        n_real = self.realizations if realizations is None else int(realizations)
        if n_real <= 0:
            raise ValueError("realizations must be > 0")
        if optimizer_type is not None:
            self.set_optimizer(
                optimizer_type=optimizer_type,
                optimizer_kwargs=optimizer_kwargs,
            )
        if zero_grad_set_to_none is not None:
            self.zero_grad_set_to_none = bool(zero_grad_set_to_none)

        losses_by_realization: list[list[float]] = []

        for r in range(n_real):
            realization_losses: list[float] = []
            work_batch = self._clone_batch(batch, tdim)
            for epoch in range(num_epochs):
                self.optimizer.zero_grad(set_to_none=self.zero_grad_set_to_none)
                ys = self.model(work_batch)

                loss = torch.tensor(0.0, device=work_batch.xs[0].device)
                for d in range(tdim + 1):
                    idx = work_batch.masks[d]
                    loss = loss + self.criterion(
                        ys[d][:, :, idx],
                        work_batch.xs_target[d][:, :, idx],
                    )

                loss.backward()
                self.optimizer.step()

                loss_val = float(loss.detach().cpu())
                realization_losses.append(loss_val)
                if (print_every is not None) and (epoch % print_every == 0):
                    print(f"realization={realization_now} epoch={epoch} loss={loss_val}")

            losses_by_realization.append(realization_losses)

        return {
            "losses": losses_by_realization,
            "final_loss": [losses[-1] for losses in losses_by_realization],
            "num_epochs": num_epochs,
            "realizations": n_real,
        }
