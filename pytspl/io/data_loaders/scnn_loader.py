import numpy as np
import pkg_resources

from pytspl.simplicial_complex import SimplicialComplex

SCNN_DATA_FOLDER = pkg_resources.resource_filename("pytspl", "data/scnn_paper")


def _load_simplices(path: str) -> dict[int, list[tuple]]:
    """Load simplices.npy into a dim -> list of simplices mapping."""
    data = np.load(path, allow_pickle=True)
    simplices_by_dim: dict[int, list[tuple]] = {}
    for dim, simplices_dict in enumerate(data):
        ordered = [None] * len(simplices_dict)
        for simplex, idx in simplices_dict.items():
            if dim == 0:
                # vertices are stored as singletons; pass raw ids to avoid nesting
                vertex = next(iter(simplex))
                ordered[idx] = vertex
            else:
                ordered[idx] = tuple(sorted(simplex))
        simplices_by_dim[dim] = ordered
    return simplices_by_dim


def _load_cochains(path: str) -> dict[int, dict]:
    """Load cochains.npy into a dim -> {simplex: feature} mapping."""
    data = np.load(path, allow_pickle=True)
    features_by_dim: dict[int, dict] = {}
    for dim, cochain_dict in enumerate(data):
        features_by_dim[dim] = {tuple(sorted(simplex)): value for simplex, value in cochain_dict.items()}
    return features_by_dim


def load_scnn_paper(only_sc: bool = True, only_2d: bool = False) -> tuple:
    """
    Load the SCNN paper dataset (simplices + cochains saved as npy files).

    Args:
        only_sc (bool): Must be True. Loader builds a SimplicialComplex.
        only_2d (bool): Must be False. Dataset includes simplices above 2D.

    Returns:
        tuple: (SimplicialComplex, coordinates dict, empty flow dict)
    """
    if not only_sc:
        raise ValueError("SCNN paper loader only supports only_sc=True.")
    if only_2d:
        raise ValueError("SCNN paper loader requires only_2d=False to keep higher-dimensional simplices.")

    simplices_path = f"{SCNN_DATA_FOLDER}/150250_simplices.npy"
    cochains_path = f"{SCNN_DATA_FOLDER}/150250_cochains.npy"

    simplices = _load_simplices(simplices_path)
    simplex_features = _load_cochains(cochains_path)

    sc = SimplicialComplex(
        simplices=simplices,
        simplex_features=simplex_features,
    )

    coordinates = sc.generate_coordinates()
    flow = {}

    return sc, coordinates, flow
