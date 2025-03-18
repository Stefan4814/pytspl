from itertools import combinations
from typing import Hashable, Iterable

import numpy as np
from scipy.sparse import csr_matrix

class CellComplex:
    """Data structure class for cell complexes."""
    def __init__(
        self,
        nodes: list = [],
        edges: list = [],
        polygons: list = [],
        node_features: dict = {},
        edge_features: dict = {},
    ):
        """
        Create a cell complex from nodes, edges, and polygons.

        Args:
            nodes (list, optional): List of nodes. Defaults to [].
            edges (list, optional): List of edges. Defaults to [].
            polygons (list, optional): List of polygons. Defaults to [].
            node_features (dict, optional): Dict of node features.
            Defaults to {}.
            edge_features (dict, optional): Dict of edge features.
            Defaults to {}.
        """
        self.nodes = nodes
        self.edges = edges
        self.polygons = polygons

        self.node_features = node_features
        self.edge_features = edge_features

        self.B1 = self.compute_B1()
        self.B2 = self.compute_B2()

    def to_simplicial_complex(self):
        """
        Convert the cell complex into a simplicial complex.
        Only keeps triangles (3-node polygons) and computes incidence matrices.

        Returns:
            SimplicialComplex: The resulting simplicial complex.
        """
        from pytspl.simplicial_complex import SimplicialComplex  

        # Filter only the polygons that are triangles
        triangles = [p for p in self.polygons if len(p) == 3]

        return SimplicialComplex(
            nodes=self.nodes,
            edges=self.edges,
            triangles=triangles,  # Keep only triangles
            node_features=self.node_features,
            edge_features=self.edge_features,
        )
    
    def compute_B1(self):
        """Compute node-to-edge incidence matrix B1."""
        num_nodes = len(self.nodes)
        num_edges = len(self.edges)
        B1 = np.zeros((num_nodes, num_edges))

        for j, (u, v) in enumerate(self.edges):
            B1[u, j] = -1 
            B1[v, j] = 1   
        return B1
    
    def compute_B2(self):
        """Compute edge-to-polygon incidence matrix B2."""
        num_polygons = len(self.polygons)
        num_edges = len(self.edges)
        B2 = np.zeros((num_edges, num_polygons))

        # Create a dictionary to map each edge to its index
        edge_index = {edge: i for i, edge in enumerate(self.edges)}

        for j, polygon in enumerate(self.polygons):
            polygon_size = len(polygon)

            for index in range(polygon_size):
                # Wrap around for cyclic order
                u, v = polygon[index], polygon[(index + 1) % polygon_size]

                if (u, v) in edge_index:
                    edge_idx = edge_index[(u, v)]
                    B2[edge_idx, j] = 1  # Assign positive orientation
                elif (v, u) in edge_index:  # Reverse edge exists
                    edge_idx = edge_index[(v, u)]
                    B2[edge_idx, j] = -1  # Assign negative orientation

        return B2

    
    