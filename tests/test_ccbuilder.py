import pytest
import numpy as np

from pytspl.cell_complex.ccbuilder import CCBuilder


@pytest.fixture
def graph():
    # Sample graph with cycles (polygons)
    nodes = [0, 1, 2, 3, 4, 5, 6]
    edges = [
        (0, 1), (1, 2), (2, 0),  # triangle 0-1-2
        (0, 2), (2, 3), (0, 3),  # triangle 0-2-3
        (4, 5), (5, 6), (6, 4),  # triangle 4-5-6
    ]
    return CCBuilder(nodes=nodes, edges=edges)


class TestCCBuilder:
    def test_polygons(self, graph: CCBuilder):
        polygons = graph.polygons()
        expected_polygons = [(0, 1, 2), (0, 2, 3), (4, 5, 6)]

        # Normalize by converting to sorted lists and sets for order-insensitive comparison
        sorted_polygons = sorted([sorted(p) for p in polygons])
        expected_sorted = sorted([sorted(p) for p in expected_polygons])
        assert sorted_polygons == expected_sorted

    def test_to_cell_complex_all(self, graph: CCBuilder):
        cc = graph.to_cell_complex(condition="all")
        expected_polygons = [(0, 1, 2), (0, 2, 3), (4, 5, 6)]

        assert cc.nodes == graph.nodes
        assert cc.edges == graph.edges
        sorted_polygons = sorted([sorted(p) for p in cc.polygons])
        expected_sorted = sorted([sorted(p) for p in expected_polygons])
        assert sorted_polygons == expected_sorted

    def test_incidence_matrices(self, graph: CCBuilder):
        cc = graph.to_cell_complex()
        B1 = cc.incidence_matrix(rank=1).toarray()
        B2 = cc.incidence_matrix(rank=2).toarray()

        assert B1.shape == (len(cc.nodes), len(cc.edges))
        assert B2.shape == (len(cc.edges), len(cc.polygons))

        # Check boundary condition: B1 @ B2 ≈ 0
        assert np.allclose(B1 @ B2, np.zeros((len(cc.nodes), len(cc.polygons))))

    def test_laplacians(self, graph: CCBuilder):
        cc = graph.to_cell_complex()

        L0 = cc.hodge_laplacian_matrix(rank=0).toarray()
        L1 = cc.hodge_laplacian_matrix(rank=1).toarray()

        assert L0.shape == (len(cc.nodes), len(cc.nodes))
        assert L1.shape == (len(cc.edges), len(cc.edges))

        # L1 = L1_lower + L1_upper
        L1_calc = (
            cc.lower_laplacian_matrix(rank=1) + cc.upper_laplacian_matrix(rank=1)
        ).toarray()
        assert np.allclose(L1, L1_calc)
