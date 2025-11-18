from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.sparse import csr_matrix, issparse
from gensim.models import Word2Vec

from pytspl.cell_complex import CellComplex

def _k_cell_adjacency(
    cc: CellComplex,
    k: int,
    multicount: bool = False,
) -> csr_matrix:
    """
    Build the (symmetric) adjacency matrix between k-cells in a CellComplex.

    k = 0 : nodes
    k = 1 : edges
    k = 2 : polygons

    For each k we combine:
        - lower adjacency  (share a (k-1)-cell)
        - upper adjacency  (share a (k+1)-cell, if any)

    Args:
        cc: CellComplex
        k:  dimension of cells to walk on (0, 1, or 2)
        multicount: if True, keep multiplicity of shared faces/cofaces
                    (used by "uniform-multicount" scheme).
                    If False, binarize adjacency.

    Returns:
        csr_matrix of shape (N_k, N_k) where N_k is the number of k-cells.
    """
    if k < 0 or k > 2:
        raise ValueError("k must be in {0, 1, 2} for this CellComplex implementation.")

    # Number of k-cells
    if k == 0:
        n = len(cc.nodes)
        if n == 0:
            return csr_matrix((0, 0), dtype=float)

        # Node adjacency via edges: if an edge (u, v) exists, connect u and v.
        A = np.zeros((n, n), dtype=float)
        for (u, v) in cc.edges:
            A[u, v] += 1.0
            A[v, u] += 1.0

        if not multicount:
            A = (A > 0).astype(float)

        np.fill_diagonal(A, 0.0)
        return csr_matrix(A)

    elif k == 1:
        m = len(cc.edges)
        if m == 0:
            return csr_matrix((0, 0), dtype=float)

        B1 = cc.B1  # shape: (num_nodes, num_edges)
        B2 = cc.B2  # shape: (num_edges, num_polygons)

        # lower adjacency: edges sharing a node 
        # (i,j) entry is number of common incident nodes (0, 1, or 2)
        A_down = B1.T @ B1  # shape (m, m)

        # upper adjacency: edges sharing a polygon 
        # (i,j) entry is number of common polygons
        if B2.size > 0:
            A_up = B2 @ B2.T  
        else:
            A_up = np.zeros((m, m))

        A = A_down + A_up

        # Remove diagonal (self-connections) for adjacency
        if issparse(A):
            A = A.tolil()
            A.setdiag(0.0)
            A = A.tocsr()
        else:
            np.fill_diagonal(A, 0.0)

        if not multicount:
            # Binarize if we don't want multiplicity
            if issparse(A):
                A = (A > 0).astype(float).tocsr()
            else:
                A = (A > 0).astype(float)

        return csr_matrix(A)

    else:  
        p = len(cc.polygons)
        if p == 0:
            return csr_matrix((0, 0), dtype=float)

        B2 = cc.B2  # shape: (num_edges, num_polygons)

        # lower adjacency for polygons: polygons sharing an edge
        A_down = B2.T @ B2 

        # no upper adjacency (no 3-cells)
        A = A_down

        # Remove diagonal
        if issparse(A):
            A = A.tolil()
            A.setdiag(0.0)
            A = A.tocsr()
        else:
            np.fill_diagonal(A, 0.0)

        if not multicount:
            if issparse(A):
                A = (A > 0).astype(float).tocsr()
            else:
                A = (A > 0).astype(float)

        return csr_matrix(A)

def assemble(
    cc: CellComplex,
    k: int,
    scheme: str = "uniform",
    laziness: Optional[float] = None,
) -> csr_matrix:
    """
    Assemble the transition matrix P for a random walk on k-cells of a CellComplex.

    Args:
        cc: CellComplex (or SimplicialComplex, which is a subclass)
        k: dimension of the cells where the walk lives (0, 1, or 2)
        scheme: one of {"uniform", "uniform-lazy", "uniform-multicount"}
            - "uniform"
                P_ij = 1 / deg(i) for each neighbor j
            - "uniform-multicount"
                P_ij ∝ number of shared faces/cofaces between i and j
            - "uniform-lazy"
                P_ii = laziness
                P_ij (j != i) uniform over neighbors with total mass 1 - laziness
        laziness: required if scheme == "uniform-lazy", value in [0, 1]

    Returns:
        P (csr_matrix): N_k x N_k transition matrix.
    """
    assert scheme in ["uniform", "uniform-lazy", "uniform-multicount"]
    if scheme == "uniform-lazy":
        if laziness is None:
            raise ValueError("laziness must be provided for scheme 'uniform-lazy'.")
        if laziness < 0.0 or laziness > 1.0:
            raise ValueError("laziness must be in [0, 1].")

    # Build adjacency with or without multiplicity
    multicount = (scheme == "uniform-multicount")
    A = _k_cell_adjacency(cc, k, multicount=multicount)  

    N = A.shape[0]
    if N == 0:
        return csr_matrix((0, 0), dtype=float)

    A = A.tocsr()
    row_inds: List[int] = []
    col_inds: List[int] = []
    data: List[float] = []

    for i in range(N):
        row = A.getrow(i)             
        w = row.toarray().ravel()     

        if scheme == "uniform-lazy":
            # Non-self neighbors total weight (if multicount) or count
            w_no_self = w.copy()
            w_no_self[i] = 0.0
            total = w_no_self.sum()

            if total == 0.0:
                # Isolated cell: stay put with probability 1
                row_inds.append(i)
                col_inds.append(i)
                data.append(1.0)
                continue

            # Each neighbor gets (1 - laziness) * (w_j / total)
            probs = np.zeros_like(w)
            probs[w_no_self > 0] = (1.0 - laziness) * (w_no_self[w_no_self > 0] / total)
            probs[i] = laziness

        else:
            # "uniform" or "uniform-multicount"
            total = w.sum()
            if total == 0.0:
                # self-loop with prob 1 for isolated cells
                probs = np.zeros_like(w)
                probs[i] = 1.0
            else:
                if scheme == "uniform":
                    # ignore multiplicity: uniform over neighbors
                    mask = (w > 0)
                    deg = mask.sum()
                    probs = np.zeros_like(w)
                    if deg == 0:
                        probs[i] = 1.0
                    else:
                        probs[mask] = 1.0 / float(deg)
                else:
                    # "uniform-multicount": probability proportional to w_j
                    probs = w / total

        # Store non-zero probabilities
        nonzero = np.nonzero(probs)[0]
        for j in nonzero:
            row_inds.append(i)
            col_inds.append(j)
            data.append(float(probs[j]))

    P = csr_matrix((data, (row_inds, col_inds)), shape=(N, N), dtype=float)
    return P

def walk(start_cell: int, walk_length: int, P: csr_matrix) -> List[int]:
    """
    Perform a single random walk on k-cells.

    Args:
        start_cell: starting k-cell index (0 <= start_cell < P.shape[0])
        walk_length: number of steps
        P: transition matrix (csr_matrix), rows are probability distributions

    Returns:
        List of visited cell indices [c_0, c_1, ..., c_{walk_length}]
    """
    N = P.shape[0]
    if start_cell < 0 or start_cell >= N:
        raise ValueError(f"start_cell must be in [0, {N-1}].")

    c = np.arange(N)
    rw: List[int] = [start_cell]

    current = start_cell
    for _ in range(walk_length):
        row = P.getrow(current).toarray().ravel()
        # if row does not sum exactly to 1, normalize
        s = row.sum()
        if s <= 0:
            # degenerate case: stay put
            next_cell = current
        else:
            if abs(s - 1.0) > 1e-8:
                row = row / s
            next_cell = int(np.random.choice(c, size=1, p=row)[0])

        rw.append(next_cell)
        current = next_cell

    return rw


def RandomWalks(
    walk_length: int,
    number_walks: int,
    P: csr_matrix,
    seed: Optional[int] = None,
) -> List[List[int]]:
    """
    Perform a fixed number of random walks starting from each k-cell.

    Args:
        walk_length: length of each walk
        number_walks: number of walks per starting cell
        P: transition matrix on k-cells
        seed: random seed used for shuffling the resulting walks (optional)

    Returns:
        List of walks; each walk is a list of k-cell indices.
    """
    N = P.shape[0]
    walks: List[List[int]] = []

    for _ in range(number_walks):
        for cell in range(N):
            walks.append(walk(cell, walk_length, P))

    if seed is not None:
        np.random.seed(seed)

    np.random.shuffle(walks)
    return walks

def save_random_walks(walks: List[List[int]], filename: str) -> None:
    """
    Save random walks to a text file, one walk per line as comma-separated ints.

    Example line:
        0,3,5,2,2,7
    """
    with open(filename, "w") as f:
        for walk in walks:
            line = ",".join(str(x) for x in walk)
            f.write(line + "\n")


def load_walks(filename: str) -> List[List[int]]:
    """
    Load random walks from a text file saved by `save_random_walks`.

    Each line must be a comma-separated list of integers.
    """
    walks: List[List[int]] = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            steps = [int(s) for s in line.split(",")]
            walks.append(steps)
    return walks

def Embedding(
    walks: List[List[int]],
    emb_dim: int,
    epochs: int = 5,
    filename: str = "k-simplex2vec_embedding.model",
) -> Word2Vec:
    """
    Train a word2vec model on k-cell random walks.

    Args:
        walks: list of walks, each a list of integer cell indices
        emb_dim: embedding dimension
        epochs: number of training epochs (gensim's 'iter' argument)
        filename: path to save the trained model

    Returns:
        Trained gensim Word2Vec model.
    """
    # Convert walks to strings for gensim
    walks_str: List[List[str]] = [
        [str(cell) for cell in walk] for walk in walks
    ]

    model = Word2Vec(
        sentences=walks_str,
        vector_size=emb_dim,   
        window=3,
        min_count=0,
        sg=1,                  
        workers=1,
        epochs=epochs,
    )
    model.save(filename)
    return model

def _num_k_cells(cc: CellComplex, k: int) -> int:
    """
    Helper: return the number of k-cells in the complex.
    k = 0: nodes
    k = 1: edges
    k = 2: polygons
    """
    if k == 0:
        return len(cc.nodes)
    elif k == 1:
        return len(cc.edges)
    elif k == 2:
        return len(cc.polygons)
    else:
        raise ValueError("k must be in {0, 1, 2}.")


def cell2vec(
    cc: CellComplex,
    k: int,
    emb_dim: int,
    walk_length: int = 20,
    number_walks: int = 10,
    scheme: str = "uniform",
    laziness: float | None = None,
    epochs: int = 5,
    seed: int | None = None,
    model_filename: str | None = None,
) -> np.ndarray:
    """
    End-to-end pipeline to compute k-cell embeddings (cell2vec / k-simplex2vec).

    Args:
        cc: CellComplex (or SimplicialComplex) instance.
        k:  cell dimension to embed:
                0 -> nodes
                1 -> edges
                2 -> polygons
        emb_dim: embedding dimension.
        walk_length: length of each random walk.
        number_walks: number of walks per k-cell.
        scheme: transition scheme, one of:
                "uniform", "uniform-lazy", "uniform-multicount".
        laziness: laziness parameter in [0,1] if scheme == "uniform-lazy",
                  else ignored.
        epochs: number of training epochs for word2vec.
        seed: random seed (used for:
              - shuffling walks,
              - and also set before walk generation for reproducibility).
        model_filename: if not None, save the gensim model to this path.
                        If None, a default name is used.

    Returns:
        embeddings: np.ndarray of shape (N_k, emb_dim),
                    where N_k is the number of k-cells.
                    Row i is the embedding of the i-th k-cell.
    """
    # Optional: make everything reproducible
    if seed is not None:
        np.random.seed(seed)

    # Build transition matrix on k-cells
    P = assemble(cc, k=k, scheme=scheme, laziness=laziness)

    # Generate random walks
    walks = RandomWalks(
        walk_length=walk_length,
        number_walks=number_walks,
        P=P,
        seed=seed,  # used for shuffling
    )

    # Train word2vec
    if model_filename is None:
        model_filename = f"k-simplex2vec_k{k}.model"

    model = Embedding(
        walks=walks,
        emb_dim=emb_dim,
        epochs=epochs,
        filename=model_filename,
    )

    # Build embedding matrix aligned with k-cell indices
    N_k = _num_k_cells(cc, k)
    embeddings = np.zeros((N_k, emb_dim), dtype=float)

    # gensim stores vectors with string keys "0", "1", ...
    for idx in range(N_k):
        key = str(idx)
        if key in model.wv:
            embeddings[idx, :] = model.wv[key]
        else:
            # If for some reason a key is missing, leave zeros (or add small noise)
            # embeddings[idx, :] = np.random.normal(scale=1e-3, size=emb_dim)
            pass

    return embeddings