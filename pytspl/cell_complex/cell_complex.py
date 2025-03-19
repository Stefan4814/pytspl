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

    def generate_coordinates(self) -> dict:
        """
        Generate the coordinates of the nodes using spring layout
        if the coordinates of the cc don't exist.

        Returns:
            dict: Coordinates of the nodes.
        """
        import networkx as nx

        print("WARNING: No coordinates found.")
        print("Generating coordinates using spring layout.")

        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(self.edges)

        coordinates = nx.spring_layout(G)
        return coordinates

    def tocsr(self, matrix: np.ndarray) -> csr_matrix:
        """
        Convert a numpy array to a csr_matrix.

        Args:
            matrix (np.ndarray): Numpy array to convert.

        Returns:
            csr_matrix: Compressed Sparse Row matrix.
        """
        return csr_matrix(matrix, dtype=float)
    
    def incidence_matrix(self, rank: int) -> csr_matrix:
        """
        Compute the incidence matrix of the simplicial complex.

        Args:
            rank (int): Rank of the incidence matrix.

        Returns:
            csr_matrix: Incidence matrix of the simplicial complex.
        """
        if rank == 0:
            return np.ones(len(self.nodes), dtype=float)
        elif rank == 1:
            return self.tocsr(self.B1)
        elif rank == 2:
            return self.tocsr(self.B2)
        else:
            raise ValueError(
                "Rank cannot be larger than the dimension of the complex."
            )

    def laplacian_matrix(self) -> csr_matrix:
        """
        Compute the Laplacian matrix of the cell complex.

        Returns:
            csr_matrix: Laplacian matrix of the cell complex.
        """
        B1 = self.tocsr(self.B1)
        return B1 @ B1.T
    
    def lower_laplacian_matrix(self, rank: int = 1) -> csr_matrix:
        """
        Compute the lower Laplacian matrix of the cell complex.

        Args:
            rank (int): Rank of the lower Laplacian matrix.

        ValueError:
            If the rank is not 1 or 2.

        Returns:
            csr_matrix: Lower Laplacian matrix of the cell complex.
        """
        if rank == 1:
            B1 = self.incidence_matrix(rank=1)
            return B1.T @ B1
        elif rank == 2:
            B2 = self.incidence_matrix(rank=2)
            return B2.T @ B2
        else:
            raise ValueError("Rank must be either 1 or 2.")
        
    def upper_laplacian_matrix(self, rank: int = 1) -> csr_matrix:
        """
        Compute the upper Laplacian matrix of the cell complex.

        Args:
            rank (int): Rank of the upper Laplacian matrix.

        ValueError:
            If the rank is not 0 or 1.

        Returns:
            csr_matrix: Upper Laplacian matrix of the cell complex.
        """
        if rank == 0:
            return self.laplacian_matrix()
        elif rank == 1:
            B2 = self.incidence_matrix(rank=2)
            return B2 @ B2.T
        else:
            raise ValueError("Rank must be either 0 or 1.")
        
    def hodge_laplacian_matrix(self, rank: int = 1) -> csr_matrix:
        """
        Compute the Hodge Laplacian matrix of the cell complex.

        Args:
            rank (int): Rank of the Hodge Laplacian matrix.

        ValueError:
            If the rank is not 0, 1, or 2.

        Returns:
            csr_matrix: Hodge Laplacian matrix of the cell complex.
        """
        if rank == 0:
            return self.laplacian_matrix()
        elif rank == 1:
            return self.lower_laplacian_matrix(
                rank=rank
            ) + self.upper_laplacian_matrix(rank=rank)
        else:
            raise ValueError("Rank must be between 0 and 2.")
    
    
    