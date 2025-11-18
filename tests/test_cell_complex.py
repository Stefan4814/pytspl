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

    def test_average_node_degree(self, cc_mock: CellComplex):
        avg_degree = cc_mock.average_node_degree()
        assert avg_degree == 2.5 

    def test_degree_distribution(self, cc_mock: CellComplex):
        degree_dist = cc_mock.degree_distribution()
        assert degree_dist == {0: 3, 1: 3, 2: 2, 3: 2}

    def test_average_polygon_size(self, cc_mock: CellComplex):
        avg_size = cc_mock.average_polygon_size()
        assert avg_size == 3.0

    def test_edge_participation_count(self, cc_mock: CellComplex):
        edge_counts = cc_mock.edge_participation_count()
        assert edge_counts == {
            (0, 1): 2,
            (1, 2): 1,
            (0, 2): 1,
            (0, 3): 1,
            (1, 3): 1,
        }

    def test_tocsr(self, cc_mock: CellComplex):
        B1_csr = cc_mock.tocsr(cc_mock.B1)
        assert isinstance(B1_csr, np.ndarray) is False
        assert B1_csr.shape == (4, 5)

    def test_print_summary(self, cc_mock: CellComplex, capsys):
        cc_mock.print_summary()
        captured = capsys.readouterr()
        assert "Num. of nodes: 4" in captured.out
        assert "Num. of edges: 5" in captured.out
        assert "Num. of triangles: 2" in captured.out
