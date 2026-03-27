from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import torch
import torch.nn as nn

from pytspl.scnn.chebyshev import assemble_chebyshev
from pytspl.scnn.highlevel import (
    MaskedReconstructionTrainer,
    SimplicialBatch,
    SimplicialConvBlock,
    SimplicialConvStack,
    build_cochains,
    build_normalized_operators,
    mask_cochains,
)
from pytspl.scnn.scnn import SimplicialConv, coo2tensor
from pytspl.scnn.utils import scipy_to_torch_sparse, torch_sparse_mm


def _make_spd_sparse(M: int, density: float = 0.2, seed: int = 0) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    A = sp.random(M, M, density=density, data_rvs=rng.standard_normal, format="csr")
    A = (A + A.T) * 0.5
    diag = np.asarray(np.abs(A).sum(axis=1)).ravel()
    return (A + sp.diags(diag + 1.0, format="csr")).tocsr()


def _dense_chebyshev_reference(K: int, L_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    B, C, M = x.shape
    Ld = L_hat.to_dense()

    X0 = x.permute(2, 0, 1).reshape(M, B * C)
    outs = [X0]
    if K == 1:
        stacked = torch.stack(outs, dim=-1)
        return stacked.reshape(M, B, C, K).permute(1, 2, 0, 3).contiguous()

    X1 = Ld @ X0
    outs.append(X1)
    for _ in range(2, K):
        X2 = 2.0 * (Ld @ X1) - X0
        outs.append(X2)
        X0, X1 = X1, X2

    stacked = torch.stack(outs, dim=-1)
    return stacked.reshape(M, B, C, K).permute(1, 2, 0, 3).contiguous()


class _DummySC:
    def __init__(self):
        self.shape = (3, 2)
        self._simplices_by_dim = [
            [(0,), (1,), (2,)],
            [(0, 1), (1, 2)],
        ]
        self._features = {
            0: {(0,): 1.0, (1,): 2.0, (2,): 3.0},
            1: {(0, 1): 10.0},
        }

    def get_simplex_features(self, rank: int):
        return self._features[rank]

    def hodge_laplacian_matrix(self, rank: int):
        if rank == 0:
            return sp.csr_matrix(
                np.array(
                    [
                        [2.0, -1.0, 0.0],
                        [-1.0, 2.0, -1.0],
                        [0.0, -1.0, 2.0],
                    ],
                    dtype=np.float64,
                )
            )
        return sp.csr_matrix(np.array([[2.0, -1.0], [-1.0, 2.0]], dtype=np.float64))

    def lower_laplacian_matrix(self, rank: int):
        if rank != 1:
            raise ValueError("Only rank-1 lower Laplacian is defined in this test helper")
        return sp.eye(2, format="csr", dtype=np.float64)

    def upper_laplacian_matrix(self, rank: int):
        if rank == 0:
            return sp.eye(3, format="csr", dtype=np.float64)
        return sp.csr_matrix((2, 2), dtype=np.float64)


class _NoSimplexOrderSC:
    def get_simplex_features(self, rank: int):
        return {}


class _ScaleBatchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch: SimplicialBatch):
        return [self.scale * x for x in batch.xs]


def test_scipy_to_torch_sparse_rejects_dense_and_honors_dtype():
    A = sp.eye(4, format="csr", dtype=np.float64)
    out = scipy_to_torch_sparse(A, dtype=torch.float64)
    assert out.dtype == torch.float64

    with pytest.raises(TypeError):
        scipy_to_torch_sparse(np.eye(4))


def test_torch_sparse_mm_supports_rectangular_sparse_operator():
    A = sp.csr_matrix(np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], dtype=np.float32))
    A_t = scipy_to_torch_sparse(A)
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)

    y = torch_sparse_mm(A_t, X)
    expected = torch.tensor([[11.0, 14.0], [9.0, 12.0]], dtype=torch.float32)
    torch.testing.assert_close(y, expected, rtol=0, atol=0)


def test_assemble_chebyshev_matches_dense_reference():
    B, C, M, K = 2, 2, 9, 5
    L = scipy_to_torch_sparse(_make_spd_sparse(M, density=0.15, seed=10)).coalesce()
    x = torch.randn(B, C, M)

    got = assemble_chebyshev(K, L, x)
    ref = _dense_chebyshev_reference(K, L, x)

    assert tuple(got.shape) == (B, C, M, K)
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)


def test_assemble_chebyshev_k1_and_validation():
    M = 6
    L = scipy_to_torch_sparse(_make_spd_sparse(M, density=0.2, seed=11)).coalesce()
    x = torch.randn(1, 1, M)

    out = assemble_chebyshev(1, L, x)
    assert tuple(out.shape) == (1, 1, M, 1)
    torch.testing.assert_close(out[..., 0], x, rtol=0, atol=0)

    with pytest.raises(ValueError):
        assemble_chebyshev(0, L, x)
    with pytest.raises(TypeError):
        assemble_chebyshev(2, torch.eye(M), x)
    with pytest.raises(ValueError):
        assemble_chebyshev(2, L, torch.randn(1, M))
    with pytest.raises(ValueError):
        assemble_chebyshev(2, L, torch.randn(1, 1, M + 1))


def test_coo2tensor_rejects_dense_and_can_set_dtype():
    A = sp.eye(3, format="csr", dtype=np.float64)
    t = coo2tensor(A, dtype=torch.float64)
    assert t.dtype == torch.float64

    with pytest.raises(TypeError):
        coo2tensor(np.eye(3))


def test_simplicial_conv_registered_operator_and_extra_repr():
    M = 7
    L = scipy_to_torch_sparse(_make_spd_sparse(M, density=0.2, seed=12)).coalesce()
    x = torch.randn(2, 1, M)

    layer = SimplicialConv(orders=3, C_in=1, C_out=2, operators=[L], basis="power")
    y_default = layer(x)
    y_explicit = layer(x, operators=[L])

    assert tuple(y_default.shape) == (2, 2, M)
    torch.testing.assert_close(y_default, y_explicit, rtol=1e-6, atol=1e-6)

    rep = layer.extra_repr()
    assert "orders=[3]" in rep
    assert "basis='power'" in rep


def test_simplicial_conv_chebyshev_basis_matches_manual_einsum():
    M = 8
    L = scipy_to_torch_sparse(_make_spd_sparse(M, density=0.2, seed=13)).coalesce()
    x = torch.randn(2, 1, M)

    layer = SimplicialConv(
        orders=4,
        C_in=1,
        C_out=1,
        basis="chebyshev",
        enable_bias=False,
        variance=0.2,
    )
    y = layer(x, L=L)
    X = assemble_chebyshev(4, L, x)
    y_ref = torch.einsum("bimk,oik->bom", (X, layer.theta)) + layer.bias

    torch.testing.assert_close(y, y_ref, rtol=1e-6, atol=1e-6)


def test_simplicial_conv_operator_resolution_validation():
    M = 5
    L = scipy_to_torch_sparse(_make_spd_sparse(M, density=0.25, seed=14)).coalesce()
    x = torch.randn(1, 1, M)

    with pytest.raises(ValueError):
        SimplicialConv(orders=[2, 2], C_in=1, C_out=1, operators=[L])
    with pytest.raises(ValueError):
        SimplicialConv(orders=2, C_in=1, C_out=1, basis="invalid")

    layer = SimplicialConv(orders=2, C_in=1, C_out=1)
    with pytest.raises(ValueError):
        layer(x)
    with pytest.raises(ValueError):
        layer(x, Ll=L)
    with pytest.raises(ValueError):
        layer(x, Lu=L)


def test_simplicial_batch_to_returns_new_batch_with_copied_metadata():
    x = torch.randn(1, 1, 3)
    L = scipy_to_torch_sparse(sp.eye(3, format="csr"))
    batch = SimplicialBatch(
        xs=[x],
        Ls=[L],
        masks=[[0, 1]],
        xs_target=[x.clone()],
        metadata={"tag": "original"},
    )

    moved = batch.to("cpu")
    assert moved is not batch
    assert moved.xs[0].device.type == "cpu"
    assert moved.Ls[0].device.type == "cpu"
    assert moved.metadata == {"tag": "original"}

    moved.metadata["tag"] = "changed"
    assert batch.metadata["tag"] == "original"


def test_build_cochains_validation_errors():
    with pytest.raises(ValueError):
        build_cochains(_DummySC(), topdim=-1)
    with pytest.raises(ValueError):
        build_cochains(_DummySC(), topdim=0, batch_size=0)
    with pytest.raises(ValueError):
        build_cochains(_DummySC(), topdim=0, channels=0)
    with pytest.raises(AttributeError):
        build_cochains(_NoSimplexOrderSC(), topdim=0)


def test_build_normalized_operators_raises_for_negative_topdim():
    with pytest.raises(ValueError):
        build_normalized_operators(_DummySC(), topdim=-1)


def test_mask_cochains_validation_and_minimum_known_index_count():
    x = torch.randn(1, 1, 4)
    with pytest.raises(ValueError):
        mask_cochains([x], missing_pct=1.0)
    with pytest.raises(ValueError):
        mask_cochains([x], missing_pct=-0.01)
    with pytest.raises(ValueError):
        mask_cochains([x], missing_pct=0.5, fill_strategy="unknown")
    with pytest.raises(ValueError):
        mask_cochains([torch.randn(1, 4)], missing_pct=0.5)

    masked, known = mask_cochains([x], missing_pct=0.99, seed=2)
    assert len(known[0]) == 1
    assert tuple(masked[0].shape) == tuple(x.shape)


def test_simplicial_conv_block_depth_one_and_depth_validation():
    M = 6
    L = scipy_to_torch_sparse(_make_spd_sparse(M, density=0.2, seed=15)).coalesce()
    x = torch.randn(2, 1, M)

    block = SimplicialConvBlock(orders=2, C_in=1, C_hidden=3, C_out=1, depth=1)
    assert len(block.layers) == 1
    y = block(x, L=L)
    assert tuple(y.shape) == tuple(x.shape)

    with pytest.raises(ValueError):
        SimplicialConvBlock(orders=2, C_in=1, C_hidden=3, C_out=1, depth=0)


def test_simplicial_conv_stack_operators_by_dim_and_validation():
    with pytest.raises(ValueError):
        SimplicialConvStack(topdim=-1, orders=1)
    with pytest.raises(ValueError):
        SimplicialConvStack(topdim=1, orders=1, colors=0)
    with pytest.raises(ValueError):
        SimplicialConvStack(topdim=1, orders=1, num_filters=0)

    stack = SimplicialConvStack(topdim=1, orders=1, colors=1, num_filters=2, depth=1)
    xs = [torch.randn(1, 1, 3), torch.randn(1, 1, 2)]
    L0 = scipy_to_torch_sparse(_make_spd_sparse(3, density=0.5, seed=16)).coalesce()
    L1 = scipy_to_torch_sparse(_make_spd_sparse(2, density=1.0, seed=17)).coalesce()

    ys = stack(xs, operators_by_dim=[[L0], [L1]])
    assert len(ys) == 2
    assert tuple(ys[0].shape) == tuple(xs[0].shape)
    assert tuple(ys[1].shape) == tuple(xs[1].shape)

    with pytest.raises(ValueError):
        stack([xs[0]], operators_by_dim=[[L0]])


def test_masked_reconstruction_trainer_config_and_fit_validations():
    with pytest.raises(ValueError):
        MaskedReconstructionTrainer(
            model=_ScaleBatchModel(),
            optimizer=None,
            criterion=nn.MSELoss(),
            realizations=0,
        )

    model = _ScaleBatchModel()
    trainer = MaskedReconstructionTrainer(
        model=model,
        optimizer=None,
        criterion=nn.MSELoss(reduction="sum"),
        topdim=0,
        realizations=1,
        optimizer_type="adam",
        optimizer_kwargs={"lr": 1e-3},
    )

    trainer.set_zero_grad_mode(set_to_none=True)
    assert trainer.zero_grad_set_to_none is True

    trainer.set_realizations(2)
    assert trainer.realizations == 2
    with pytest.raises(ValueError):
        trainer.set_realizations(0)
    with pytest.raises(ValueError):
        trainer.set_optimizer("does_not_exist")

    I = scipy_to_torch_sparse(sp.eye(3, format="csr")).coalesce()
    xs = [torch.tensor([[[1.0, 2.0, 3.0]]])]
    batch = SimplicialBatch(
        xs=[xs[0].clone()],
        xs_target=[xs[0].clone()],
        masks=[[0, 2]],
        Lls=[I],
        Lus=[I],
    )

    with pytest.raises(ValueError):
        trainer.fit(batch, num_epochs=1, print_every=0)
    with pytest.raises(ValueError):
        trainer.fit(batch, num_epochs=1, realizations=0)

    out = trainer.fit(
        batch,
        num_epochs=2,
        realizations=1,
        optimizer_type="sgd",
        optimizer_kwargs={"lr": 1e-2},
        zero_grad_set_to_none=False,
    )

    assert out["num_epochs"] == 2
    assert out["realizations"] == 1
    assert len(out["losses"]) == 1
    assert len(out["losses"][0]) == 2
    assert len(out["final_loss"]) == 1
