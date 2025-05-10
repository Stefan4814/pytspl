import numpy as np
import pytest

from pytspl.cell_complex import CellComplex


class TestCellComplex:
    @pytest.fixture
    def cc_mock(self):
        nodes = [0, 1, 2, 3]
        edges = [(0, 1), (1, 2), (2, 0), (0, 3), (3, 1)]
        polygons = [[0, 1, 2], [0, 3, 1]]
        return CellComplex(nodes=nodes, edges=edges, polygons=polygons)

    def test_nodes_edges_polygons(self, cc_mock: CellComplex):
        assert cc_mock.nodes == [0, 1, 2, 3]
        assert len(cc_mock.edges) == 5
        assert len(cc_mock.polygons) == 2

    def test_incidence_matrix(self, cc_mock: CellComplex):
        B1 = cc_mock.incidence_matrix(rank=1).toarray()
        assert B1.shape == (4, 5)

        B2 = cc_mock.incidence_matrix(rank=2).toarray()
        assert B2.shape == (5, 2)

        # B1 @ B2 should be zero
        assert np.allclose(B1 @ B2, np.zeros((4, 2)))

    def test_laplacian_matrix(self, cc_mock: CellComplex):
        lap = cc_mock.laplacian_matrix().toarray()
        assert lap.shape == (4, 4)

        B1 = cc_mock.incidence_matrix(rank=1)
        expected = B1 @ B1.T
        assert np.allclose(lap, expected.toarray())

    def test_lower_laplacian_matrix(self, cc_mock: CellComplex):
        L1 = cc_mock.lower_laplacian_matrix(rank=1).toarray()
        B1 = cc_mock.incidence_matrix(rank=1)
        expected = B1.T @ B1
        assert np.allclose(L1, expected.toarray())

        L2 = cc_mock.lower_laplacian_matrix(rank=2).toarray()
        B2 = cc_mock.incidence_matrix(rank=2)
        expected = B2.T @ B2
        assert np.allclose(L2, expected.toarray())

    def test_upper_laplacian_matrix(self, cc_mock: CellComplex):
        L0 = cc_mock.upper_laplacian_matrix(rank=0).toarray()
        B1 = cc_mock.incidence_matrix(rank=1)
        expected = B1 @ B1.T
        assert np.allclose(L0, expected.toarray())

        L1 = cc_mock.upper_laplacian_matrix(rank=1).toarray()
        B2 = cc_mock.incidence_matrix(rank=2)
        expected = B2 @ B2.T
        assert np.allclose(L1, expected.toarray())

    def test_hodge_laplacian_matrix(self, cc_mock: CellComplex):
        L0 = cc_mock.hodge_laplacian_matrix(rank=0).toarray()
        L0_expected = cc_mock.laplacian_matrix().toarray()
        assert np.allclose(L0, L0_expected)

        L1 = cc_mock.hodge_laplacian_matrix(rank=1).toarray()
        L1_expected = (
            cc_mock.lower_laplacian_matrix(rank=1)
            + cc_mock.upper_laplacian_matrix(rank=1)
        ).toarray()
        assert np.allclose(L1, L1_expected)

    def test_generate_coordinates(self, cc_mock: CellComplex):
        coords = cc_mock.generate_coordinates()
        assert isinstance(coords, dict)
        assert all(node in coords for node in cc_mock.nodes)
