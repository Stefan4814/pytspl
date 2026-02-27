"""SC builder module to build simplicial complex networks using
0-simplices (nodes), 1-simplices (edges) and 2-simplices (triangles).

The 2-simplices can be added in three ways:
    - Triangles passed as an argument.
    - All triangles in the simplicial complex.
    - Triangles based on a condition e.g. distance.

This builder also supports:
- Directly supplying higher-dimensional simplices through the ``simplices`` mapping
  (dim -> list of simplices). If provided, triangle inference is skipped.
- Automatic construction of all cliques (all simplices) when ``only_2d`` is False.
"""

from typing import Optional

import networkx as nx

from pytspl.simplicial_complex import SimplicialComplex


class SCBuilder:
    """SCBuilder is used to build a simplicial complex by defining the
    2-simplices using different ways."""

    def __init__(
        self,
        nodes: list,
        edges: list,
        node_features: dict = {},
        edge_features: dict = {},
        simplices: Optional[dict[int, list]] = None,
    ):
        """Initialize the SCBuilder object."""
        # 0-simplices - nodes
        self.nodes = nodes
        # 1-simplices - edges
        self.edges = edges

        # node and edge features
        self.node_features = node_features
        self.edge_features = edge_features
        # optional higher-dimensional simplices
        self.simplices = simplices

    def triangles(self) -> list:
        """
        Get a list of triangles in the graph.

        Returns:
            list: List of triangles.
        """
        g = nx.Graph()
        g.add_edges_from(self.edges)
        cliques = nx.enumerate_all_cliques(g)
        triangle_nodes = [x for x in cliques if len(x) == 3]
        # sort the triangles
        triangle_nodes = [sorted(tri) for tri in triangle_nodes]
        return triangle_nodes

    def triangles_dist_based(self, dist_col_name: str, epsilon: float) -> list:
        """
        Get a list of triangles in the graph that satisfy the condition:
            d(a, b) < epsilon, d(a, c) < epsilon, d(b, c) < epsilon

        Args:
            dist_col_name (str): Name of the column that contains the distance.
            epsilon (float, optional): Distance threshold to consider for
            triangles.

        Returns:
            list: List of triangles that satisfy the condition.
        """
        triangle_nodes = self.triangles()

        conditional_tri = []
        for a, b, c in triangle_nodes:
            if (
                self.edge_features[(a, b)][dist_col_name]
                and self.edge_features[(b, c)][dist_col_name]
                and self.edge_features[(a, c)][dist_col_name]
            ):
                dist_ab = self.edge_features[(a, b)][dist_col_name]
                dist_ac = self.edge_features[(b, c)][dist_col_name]
                dist_bc = self.edge_features[(a, c)][dist_col_name]

                if (
                    dist_ab < epsilon
                    and dist_ac < epsilon
                    and dist_bc < epsilon
                ):
                    conditional_tri.append([a, b, c])

        return conditional_tri

    def _all_simplices(self) -> dict[int, list]:
        """
        Enumerate all cliques in the underlying graph and return a simplices
        mapping keyed by dimension.

        Returns:
            dict[int, list]: Mapping of dimension -> list of simplices.
        """
        g = nx.Graph()
        g.add_nodes_from(self.nodes)
        g.add_edges_from(self.edges)

        simplices: dict[int, list] = {
            0: [(n,) for n in self.nodes],
            1: [tuple(e) for e in self.edges],
        }

        for clique in nx.enumerate_all_cliques(g):
            if len(clique) <= 2:
                continue
            dim = len(clique) - 1
            simplex = tuple(sorted(clique))
            simplices.setdefault(dim, []).append(simplex)
        return simplices

    def to_simplicial_complex(
        self,
        condition: str = "all",
        dist_col_name: str = "distance",
        dist_threshold: float = 1.5,
        triangles=None,
        simplices: Optional[dict[int, list]] = None,
        only_2d: bool = True,
    ) -> SimplicialComplex:
        """
        Convert the graph to a simplicial complex using the given condition
        of simplices. The simplicial complex will also have node and edge
        features.

        Args:
            condition (str, optional): Condition to build the 2-simplices
            (triangles). Defaults to "all".
            Options:
            - "all": All simplices.
            - "distance": Based on distance.

            dist_col_name (str, optional): Name of the column that contains
            the distance.
            dist_threshold (float, optional): Distance threshold to consider
            for simplices. Defaults to 1.5.
            simplices (dict[int, list], optional): Explicit mapping of
            dimension -> simplices to build higher-dimensional complexes.
            If provided, triangle inference is skipped.
            only_2d (bool, optional): If True (default), restrict to up to
            triangles; if False, construct all simplices via cliques.

        Returns:
            SimplicialComplex: Simplicial complex network.
        """
        if simplices is None and self.simplices is not None:
            simplices = self.simplices

        if simplices is not None:
            return SimplicialComplex(
                simplices=simplices,
                node_features=self.node_features,
                edge_features=self.edge_features,
            )

        if not only_2d:
            simplices = self._all_simplices()
            return SimplicialComplex(
                simplices=simplices,
                node_features=self.node_features,
                edge_features=self.edge_features,
            )

        if triangles is None:
            if condition == "all":
                # add all 2-simplices
                triangles = self.triangles()
            else:
                # add 2-simplices based on condition
                triangles = self.triangles_dist_based(
                    dist_col_name=dist_col_name, epsilon=dist_threshold
                )

        # create the simplicial complex
        sc = SimplicialComplex(
            nodes=self.nodes,
            edges=self.edges,
            triangles=triangles,
            node_features=self.node_features,
            edge_features=self.edge_features,
        )

        return sc
