"""Data structure for a simplicial complex that supports arbitrary dimension."""

from typing import Hashable, Iterable

import numpy as np
from scipy.sparse import csr_matrix

from pytspl.cell_complex.ccbuilder import CCBuilder

from pytspl.decomposition.eigendecomposition import (
    get_curl,
    get_curl_eigenpair,
    get_divergence,
    get_gradient_eigenpair,
    get_harmonic_eigenpair,
    get_total_variance,
)
from pytspl.decomposition.frequency_component import FrequencyComponent
from pytspl.decomposition.hodge_decomposition import (
    get_curl_flow,
    get_gradient_flow,
    get_harmonic_flow,
)


class SimplicialComplex:
    """Data structure class for a simplicial complex."""

    def __init__(
        self,
        simplices: dict[int, list] | None = None,
        nodes: list = None,
        edges: list = None,
        triangles: list = None,
        node_features: dict = None,
        edge_features: dict = None,
        simplex_features: dict[int, dict] | None = None,
    ):
        """
        Create a simplicial complex. Supports higher dimensions via the
        ``simplices`` argument while keeping compatibility with legacy
        ``nodes``/``edges``/``triangles`` inputs.

        Args:
            simplices (dict[int, list], optional): Mapping of dimension to
                simplices. Each simplex must be an ordered iterable of
                vertices, where the order encodes orientation. Missing faces
                are added automatically.
            nodes (list, optional): 0-simplices. Ignored if ``simplices`` is
                provided.
            edges (list, optional): 1-simplices. Ignored if ``simplices`` is
                provided.
            triangles (list, optional): 2-simplices. Ignored if ``simplices``
                is provided.
            node_features (dict, optional): Dict of node features.
            edge_features (dict, optional): Dict of edge features.
            simplex_features (dict[int, dict], optional): Mapping of dimension
                to a dict of features for that dimension's simplices. Keys must
                be simplices present in the complex. If provided for dim 0/1 it
                overrides auto-filled ``node_features``/``edge_features`` in
                that dimension.
        """
        node_features = node_features or {}
        edge_features = edge_features or {}
        nodes = nodes or []
        edges = edges or []
        triangles = triangles or []

        if simplices is not None:
            simplices_by_dim = self._build_from_mapping(simplices)
            original_simplices = [[tuple(s) if not isinstance(
                s, tuple) else s for s in simplices_by_dim[0]]]
            for dim_list in simplices_by_dim[1:]:
                original_simplices.append(
                    [list(s) if isinstance(s, tuple) else s for s in dim_list])
        else:
            simplices_by_dim = self._build_from_parts(nodes, edges, triangles)
            original_simplices = [
                [(n,) for n in simplices_by_dim[0]],
                [tuple(e) for e in simplices_by_dim[1]],
                [list(t) for t in simplices_by_dim[2]]
                if len(simplices_by_dim) > 2 else [],]

        self._simplices_by_dim = simplices_by_dim
        self._original_simplices_by_dim = original_simplices

        self.node_features = node_features
        self.edge_features = edge_features

        self.nodes = [s[0] for s in self._simplices_by_dim[0]]
        self.edges = list(
            self._simplices_by_dim[1]) if self.max_dim >= 1 else []
        self.triangles = (
            list(self._original_simplices_by_dim[2])
            if self.max_dim >= 2 and len(self._original_simplices_by_dim) > 2
            else []
        )
        if simplices is not None:
            self._validate_legacy_args(nodes, edges, triangles)

        self._features_by_dim = self._init_features_by_dim(simplex_features)
        self._incidence_matrices = self._compute_incidence_matrices()

    # Construction helpers
    def _build_from_parts(
        self, nodes: list, edges: list, triangles: list
    ) -> list[list[tuple]]:
        '''Build simplices from separate lists of nodes, edges, and triangles'''
        simplices_by_dim: list[list[tuple]] = [
            [(n,) for n in nodes],
            [tuple(e) for e in edges],
            [tuple(t) for t in triangles] if triangles else [],
        ]
        # Remove trailing empty dimensions
        while simplices_by_dim and simplices_by_dim[-1] == []:
            simplices_by_dim.pop()
        return self._ensure_closure(simplices_by_dim)

    def _build_from_mapping(
            self, simplices: dict[int, list]) -> list[list[tuple]]:
        '''Build simplices from a mapping of dimension to simplices'''
        max_dim = max(simplices.keys())
        simplices_by_dim: list[list[tuple]] = []
        for k in range(max_dim + 1):
            dim_simplices = simplices.get(k, [])
            if k == 0:
                simplices_by_dim.append([(v,) for v in dim_simplices])
            else:
                simplices_by_dim.append([tuple(s) for s in dim_simplices])
        return self._ensure_closure(simplices_by_dim)

    def _validate_legacy_args(self, nodes, edges, triangles):
        '''Validate that legacy nodes/edges/triangles are consistent with simplices if both are provided'''
        # nodes
        if nodes:
            if set(nodes) != set(self.nodes):
                raise ValueError(
                    "Inconsistent inputs: `simplices` implies a different node set than `nodes`."
                )

        # edges (ignore orientation)
        if edges:
            def undirected(e):
                u, v = e
                return (u, v) if u <= v else (v, u)

            if {undirected(e) for e in edges} != {undirected(e)
                                                  for e in self.edges}:
                raise ValueError(
                    "Inconsistent inputs: `simplices` implies a different edge set than `edges` "
                    "(orientation is ignored in this check).")

        # triangles (ignore orientation) only if given
        if triangles:
            def tri_key(t):
                a, b, c = t
                return tuple(sorted((a, b, c)))

            # self.triangles might be list-of-lists; normalize
            implied_tris = {tri_key(t) for t in self.triangles}
            given_tris = {tri_key(t) for t in triangles}

            if given_tris != implied_tris:
                raise ValueError(
                    "Inconsistent inputs: `simplices` implies a different triangle set than `triangles` "
                    "(orientation is ignored in this check).")

    def _ensure_closure(
            self, simplices_by_dim: list[list[tuple]]) -> list[list[tuple]]:
        """
        Ensure all faces exist for each simplex. Missing faces are appended
        with the induced orientation.
        """
        def _dedupe(seq):
            seen = set()
            result = []
            for item in seq:
                if item not in seen:
                    seen.add(item)
                    result.append(item)
            return result

        if not simplices_by_dim:
            return [[]]

        max_dim = len(simplices_by_dim) - 1
        for k in range(max_dim, 0, -1):
            faces = set(simplices_by_dim[k - 1])
            # undirected keys only for edges (1-simplices)
            undirected_faces = {
                tuple(
                    sorted(f)) for f in faces} if (
                k - 1) == 1 else set()
            for simplex in simplices_by_dim[k]:
                for i in range(len(simplex)):
                    face = simplex[:i] + simplex[i + 1:]

                    if (k - 1) == 1:
                        key = tuple(sorted(face))
                        if key in undirected_faces:
                            continue
                        simplices_by_dim[k - 1].append(face)
                        faces.add(face)
                        undirected_faces.add(key)
                    else:
                        if face not in faces:
                            simplices_by_dim[k - 1].append(face)
                            faces.add(face)
        # de-duplicate while preserving order
        simplices_by_dim = [_dedupe(dim_list) for dim_list in simplices_by_dim]
        # keep nodes ordered if they are sortable
        try:
            simplices_by_dim[0] = sorted(simplices_by_dim[0])
        except TypeError:
            pass
        return simplices_by_dim

    def _compute_incidence_matrices(self) -> dict[int, np.ndarray]:
        '''Compute the incidence matrices for each dimension.'''
        incidence = {}
        for k in range(1, len(self._simplices_by_dim)):
            lower = self._simplices_by_dim[k - 1]
            upper = self._simplices_by_dim[k]
            lower_index = {s: idx for idx, s in enumerate(lower)}
            Bk = np.zeros((len(lower), len(upper)))
            for col, simplex in enumerate(upper):
                for i in range(len(simplex)):
                    face = simplex[:i] + simplex[i + 1:]
                    sign = (-1) ** i
                    row = lower_index.get(face)

                    if row is None and len(face) == 2:
                        rev = (face[1], face[0])
                        row = lower_index.get(rev)
                        if row is not None:
                            sign *= -1

                    if row is None:
                        raise ValueError(
                            f"Missing face {face} for simplex {simplex}")

                    Bk[row, col] += sign
            incidence[k] = Bk
        return incidence

    # Basic properties
    def print_summary(self):
        """Print the summary of the simplicial complex."""
        for k, simplices in enumerate(self._simplices_by_dim):
            print(f"Num. of {k}-simplices: {len(simplices)}")
        print(f"Shape: {self.shape}")
        print(f"Max Dimension: {self.max_dim}")

    @property
    def shape(self) -> tuple:
        """Return the shape of the simplicial complex."""
        sizes = tuple(len(s) for s in self._simplices_by_dim)
        return sizes

    @property
    def max_dim(self) -> int:
        """Return the maximum dimension of the simplicial complex."""
        return len(self._simplices_by_dim) - 1

    @property
    def simplices(self) -> list:
        """
        Get all the simplices of the simplicial complex.

        Returns:
            list: List of simplices across dimensions.
        """
        simplices = []
        simplices.extend([(n,) for n in self.nodes])
        simplices.extend(list(self.edges))
        if self.triangles:
            simplices.extend(self.triangles)
        if self.max_dim > 2:
            for dim_list in self._simplices_by_dim[3:]:
                simplices.extend([list(s) for s in dim_list])
        return simplices

    # Features
    def _init_features_by_dim(
        self, simplex_features: dict[int, dict] | None
    ) -> list[dict]:
        """Initialize per-dimension simplex features with legacy compatibility."""
        features_by_dim: list[dict] = [dict() for _ in range(self.max_dim + 1)]

        if simplex_features:
            for dim, fmap in simplex_features.items():
                if dim < 0 or dim > self.max_dim:
                    raise ValueError(
                        f"Feature dimension {dim} exceeds complex dimension {self.max_dim}."
                    )
                normalized = {}
                simplices_set = set(self._simplices_by_dim[dim])
                for simplex, value in fmap.items():
                    simplex_t = tuple(simplex)
                    if simplex_t not in simplices_set:
                        raise ValueError(
                            f"Feature provided for simplex {simplex_t} which is not in the complex."
                        )
                    normalized[simplex_t] = value
                features_by_dim[dim].update(normalized)

        # Fill legacy features if that dimension has none set explicitly.
        if self.node_features and not features_by_dim[0]:
            features_by_dim[0] = {
                (k if isinstance(k, tuple) else (k,)): v
                for k, v in self.node_features.items()
            }
        if self.max_dim >= 1 and self.edge_features and not features_by_dim[1]:
            features_by_dim[1] = {tuple(k): v for k, v in self.edge_features.items()}

        return features_by_dim

    def get_simplex_features(self, dim: int, name: str | None = None) -> dict:
        """
        Return the feature mapping for the given dimension.

        Args:
            dim (int): Dimension of the simplices.
            name (str, optional): If provided, extract that field from feature dicts.

        Returns:
            dict: Mapping of simplex -> feature value(s).
        """
        if dim < 0 or dim > self.max_dim:
            raise ValueError(
                f"Dimension {dim} exceeds complex dimension {self.max_dim}."
            )
        features = self._features_by_dim[dim]
        if name:
            try:
                return {key: value[name] for key, value in features.items()}
            except KeyError:
                raise KeyError(
                    f"Simplex feature {name} does not exist in dimension {dim}."
                )
        return features

    def edge_feature_names(self) -> list[str]:
        """Return the list of edge feature names."""
        if len(self.get_edge_features()) == 0:
            return []
        return list(list(self.get_edge_features().values())[0].keys())

    def get_node_features(self) -> list[dict]:
        """Return the list of node features."""
        return self.node_features

    def get_edge_features(self, name: str = None) -> list[dict]:
        """Return the list of edge features."""
        edge_features = self.edge_features
        if name:
            try:
                return {key: value[name]
                        for key, value in edge_features.items()}
            except KeyError:
                raise KeyError(
                    f"Edge feature {name} does not exist in the simplicial complex."
                )
        return edge_features

    # Combinatorial helpers
    def get_faces(self, simplex: Iterable[Hashable]) -> list[tuple]:
        """
        Return the faces of the simplex of dimension len(simplex)-2.
        Faces are returned in sorted order (lexicographic).
        Args:
            simplex (Iterable[Hashable]): Simplex for which to find the faces.
        Returns:
            list[tuple]: List of faces of the simplex.
        """
        simplex = tuple(simplex)
        if len(simplex) == 1:
            return []
        faces = [simplex[:i] + simplex[i + 1:] for i in range(len(simplex))]
        return sorted(faces)

    # Matrices
    def identity_matrix(self) -> np.ndarray:
        """Identity matrix of the simplicial complex."""
        return np.eye(len(self.nodes))

    def tocsr(self, matrix: np.ndarray) -> csr_matrix:
        """Convert a numpy array to a csr_matrix."""
        return csr_matrix(matrix, dtype=float)

    def compute_B1(self) -> np.ndarray:
        """Return the node-edge incidence matrix."""
        return self._incidence_matrices.get(1, np.zeros((len(self.nodes), 0)))

    def compute_B2(self) -> np.ndarray:
        """Return the edge-triangle incidence matrix if present."""
        return self._incidence_matrices.get(2, np.zeros((len(self.edges), 0)))

    @property
    def B1(self) -> np.ndarray:
        return self.compute_B1()

    @property
    def B2(self) -> np.ndarray:
        return self.compute_B2()

    def incidence_matrix(self, rank: int) -> csr_matrix:
        """
        Compute the incidence matrix of the simplicial complex.
        """
        if rank == 0:
            return self.tocsr(np.ones(len(self.nodes), dtype=float))
        if rank < 0 or rank > self.max_dim:
            raise ValueError(
                "Rank cannot be larger than the dimension of the complex.")
        matrix = self._incidence_matrices.get(rank)
        if matrix is None:
            raise ValueError(f"No incidence matrix of rank {rank} available.")
        return self.tocsr(matrix)
    
    def incidence_matrix_sparse(self, rank: int) -> csr_matrix:
        """Return the sparse incidence matrix for a given rank."""
        return csr_matrix(self._incidence_matrices[rank], dtype=float)

    def adjacency_matrix(self) -> csr_matrix:
        """Compute the adjacency matrix of the simplicial complex."""
        adjacency_mat = np.zeros((self.B1.shape[0], self.B1.shape[0]))
        for col in range(self.B1.shape[1]):
            col_nonzero = np.where(self.B1[:, col] != 0)[0]
            from_node, to_node = col_nonzero[0], col_nonzero[1]
            adjacency_mat[from_node, to_node] = 1
            adjacency_mat[to_node, from_node] = 1
        return csr_matrix(adjacency_mat)

    def laplacian_matrix(self) -> csr_matrix:
        """Compute the 0-th Laplacian matrix of the simplicial complex."""
        B1 = self.incidence_matrix(rank=1)
        return B1 @ B1.T

    def lower_laplacian_matrix(self, rank: int = 1) -> csr_matrix:
        """Compute the lower Laplacian matrix."""
        if rank < 1 or rank > self.max_dim:
            raise ValueError(
                "Rank must be at least 1 and not exceed the max dimension.")
        Bk = self.incidence_matrix(rank=rank)
        return Bk.T @ Bk

    def upper_laplacian_matrix(self, rank: int = 1) -> csr_matrix:
        """Compute the upper Laplacian matrix."""
        if rank < 0 or rank > self.max_dim:
            raise ValueError("Rank must be between 0 and the max dimension.")
        next_rank = rank + 1
        if next_rank > self.max_dim:
            size = len(self._simplices_by_dim[rank])
            return self.tocsr(np.zeros((size, size)))
        Bk1 = self.incidence_matrix(rank=next_rank)
        return Bk1 @ Bk1.T

    def hodge_laplacian_matrix(self, rank: int = 1) -> csr_matrix:
        """Compute the Hodge Laplacian matrix of the simplicial complex."""
        if rank < 0 or rank > self.max_dim:
            raise ValueError("Rank must be between 0 and the max dimension.")
        if rank == 0:
            return self.laplacian_matrix()
        L_lower = self.lower_laplacian_matrix(rank=rank)
        L_upper = self.upper_laplacian_matrix(rank=rank)
        return L_lower + L_upper

    def hodge_laplacians(self) -> dict[int, csr_matrix]:
        """Compute the Hodge Laplacian matrices for all dimensions."""
        return {k: self.hodge_laplacian_matrix(rank=k) for k in range(self.max_dim + 1)}

    # Shifting and embeddings
    def apply_lower_shifting(
            self,
            flow: np.ndarray,
            steps: int = 1) -> np.ndarray:
        """
        Apply the lower shifting operator to the simplicial complex.
        Args:
            flow (np.ndarray): Flow on the simplicial complex.
            steps (int): Number of times to apply the lower shifting operator.
            Defaults to 1.
        Returns:
            np.ndarray: Lower shifted simplicial complex.
        """
        L_lower = self.lower_laplacian_matrix(rank=1)
        if steps == 1:
            return L_lower @ flow
        return L_lower @ (L_lower @ flow)

    def apply_upper_shifting(
            self,
            flow: np.ndarray,
            steps: int = 1) -> np.ndarray:
        """
        Apply the upper shifting operator to the simplicial complex.
        Args:
            flow (np.ndarray): Flow on the simplicial complex.
            steps (int): Number of times to apply the upper shifting operator.
            Defaults to 1.
        Returns:
            np.ndarray: Upper shifted simplicial complex.
        """
        L_upper = self.upper_laplacian_matrix(rank=1)
        if steps == 1:
            return L_upper @ flow
        return L_upper @ (L_upper @ flow)

    def apply_k_step_shifting(
            self,
            flow: np.ndarray,
            steps: int = 2) -> np.ndarray:
        """
        Apply the k-step shifting operator to the simplicial complex.
        Args:
            flow (np.ndarray): Flow on the simplicial complex.
        Returns:
            np.ndarray: K-step shifted simplicial complex.
        """
        lower_shift = self.apply_lower_shifting(flow, steps=steps)
        upper_shift = self.apply_upper_shifting(flow, steps=steps)
        return lower_shift + upper_shift

    def get_simplicial_embeddings(self, flow: np.ndarray) -> tuple:
        """
        Return harmonic, curl, and gradient embeddings.
        Args:
            flow (np.ndarray): Flow on the simplicial complex.
        Returns:
            np.ndarray: Simplicial embeddings of the simplicial complex.
            Harmonic embedding, curl embedding, and gradient embedding.
        """
        k = 1
        L1 = self.hodge_laplacian_matrix(rank=k).toarray()
        L1U = self.upper_laplacian_matrix(rank=k).toarray()
        L1L = self.lower_laplacian_matrix(rank=k).toarray()

        u_h, _ = get_harmonic_eigenpair(L1, tolerance=1e-3)
        u_c, _ = get_curl_eigenpair(L1U, 1e-3)
        u_g, _ = get_gradient_eigenpair(L1L, 1e-3)

        f_tilda_h = u_h.T @ flow
        f_tilda_c = u_c.T @ flow
        f_tilda_g = u_g.T @ flow
        return f_tilda_h, f_tilda_c, f_tilda_g

    def get_component_eigenpair(
        self,
        component: str = FrequencyComponent.HARMONIC.value,
        tolerance: float = 1e-3,
    ) -> tuple:
        """
         Return the eigendecomposition of the simplicial complex.
        Args:
            component (str, optional): Component of the eigendecomposition
            to return. Defaults to "harmonic".
            tolerance (float, optional): Tolerance for eigenvalues to be
            considered zero. Defaults to 1e-3.
        ValueError:
            If the component is not one of 'harmonic', 'curl', or 'gradient'.
        Returns:
            np.ndarray: Eigenvectors of the component.
            np.ndarray: Eigenvalues of the component.
        """
        if component == FrequencyComponent.HARMONIC.value:
            L1 = self.hodge_laplacian_matrix(rank=1).toarray()
            u_h, eig_h = get_harmonic_eigenpair(L1, tolerance)
            return u_h, eig_h
        if component == FrequencyComponent.CURL.value:
            L1U = self.upper_laplacian_matrix(rank=1).toarray()
            u_c, eig_c = get_curl_eigenpair(L1U, tolerance)
            return u_c, eig_c
        if component == FrequencyComponent.GRADIENT.value:
            L1L = self.lower_laplacian_matrix(rank=1).toarray()
            u_g, eig_g = get_gradient_eigenpair(L1L, tolerance)
            return u_g, eig_g
        raise ValueError(
            "Invalid component. Choose from 'harmonic',"
            + "'curl', or 'gradient'."
        )

    def get_total_variance(self) -> np.ndarray:
        """
        Get the total variance of the SC.
        Returns:
            np.ndarray: The total variance of the SC.
        """
        laplacian_matrix = self.laplacian_matrix()
        return get_total_variance(laplacian_matrix)

    def get_divergence(self, flow: np.ndarray) -> np.ndarray:
        """
        Get the divergence of the edge flow.
        Args:
            flow (np.ndarray): The edge flow defined over a SC.
        Returns:
            np.ndarray: The divergence of the edge flow.
        """
        B1 = self.incidence_matrix(rank=1)
        return get_divergence(B1, flow)

    def get_curl(self, flow: np.ndarray) -> np.ndarray:
        """
        Get the curl of the edge flow.
        Args:
            flow (np.ndarray): The edge flow defined over a SC.
        Returns:
            np.ndarray: The curl of the edge flow.
        """
        B2 = self.incidence_matrix(rank=2)
        return get_curl(B2, flow)

    def get_component_flow(
        self,
        flow: np.ndarray,
        component: str = FrequencyComponent.GRADIENT.value,
        round_fig: bool = True,
        round_sig_fig: int = 2,
    ) -> np.ndarray:
        """
        Return the component flow of the simplicial complex
        using the Hodge decomposition.
        Args:
            flow (np.ndarray): Flow on the simplicial complex.
            component (str, optional): Component of the Hodge decomposition.
            Defaults to FrequencyComponent.GRADIENT.value.
            round_fig (bool, optional): Round the hodgedecomposition to the
            Default to True.
            round_sig_fig (int, optional): Round to significant figure.
            Defaults to 2.
        Returns:
            np.ndarray: Hodge decomposition of the edge flow.
        """
        B1 = self.incidence_matrix(rank=1)
        B2 = self.incidence_matrix(rank=2)

        if component == FrequencyComponent.HARMONIC.value:
            return get_harmonic_flow(
                B1=B1,
                B2=B2,
                flow=flow,
                round_fig=round_fig,
                round_sig_fig=round_sig_fig,
            )
        if component == FrequencyComponent.CURL.value:
            return get_curl_flow(
                B2=B2,
                flow=flow,
                round_fig=round_fig,
                round_sig_fig=round_sig_fig,
            )
        if component == FrequencyComponent.GRADIENT.value:
            return get_gradient_flow(
                B1=B1,
                flow=flow,
                round_fig=round_fig,
                round_sig_fig=round_sig_fig,
            )
        raise ValueError(
            "Invalid component. Choose from 'harmonic',"
            + "'curl', or 'gradient'."
        )

    # Conversions
    def to_cell_complex(self):
        """
        Convert the simplicial complex into a cell complex by detecting larger polygons.
        """
        cc_builder = CCBuilder(
            nodes=self.nodes,
            edges=self.edges,
            node_features=self.node_features,
            edge_features=self.edge_features,
        )
        return cc_builder.to_cell_complex()

    # Utility
    def generate_coordinates(self) -> dict:
        """
        Generate the coordinates of the nodes using spring layout
        if the coordinates of the sc don't exist.
        """
        import networkx as nx

        print("WARNING: No coordinates found.")
        print("Generating coordinates using spring layout.")

        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(self.edges)

        coordinates = nx.spring_layout(G)
        return coordinates
