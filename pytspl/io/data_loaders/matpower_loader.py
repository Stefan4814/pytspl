import pandas as pd
import pkg_resources
import numpy as np
from scipy.io import loadmat

from pytspl.simplicial_complex import SimplicialComplex
from pytspl.cell_complex import CellComplex


MATPOWER_DATA_FOLDER = pkg_resources.resource_filename(
    "pytspl", "data/powerNetwork"
)

BUS_I = 0

F_BUS = 0
T_BUS = 1
TAP = 8
PF = 13


def load_matpower_data(case_name: str = "case57", only_sc: bool = True) -> tuple:
    """
    Load a MATPOWER case and return the simplicial/cell complex and coordinates.

    Args:
        case_name (str): File must be named "{case_name}TS.m" under data/powerNetwork
        only_sc (bool, optional): if True return a simplicial complex,
                                  else return a cell complex.

    Returns:
        tuple:
            SimplicialComplex or CellComplex: the simplicial/cell complex.
            dict: coordinates of nodes (spring layout, since none are provided).
            dict: edge flow data as {(u,v): PF}.
    """
    filename = f"{MATPOWER_DATA_FOLDER}/{case_name}TS.m"

    mat = loadmat(filename, appendmat=False)

    print("Loading MATPOWER file:", filename)

    branch = np.asarray(mat["branchResults"], dtype=float)
    bus = np.asarray(mat["busResults"], dtype=float)

    branch0 = branch[0]  # (E, 17)
    bus0 = bus[0]        # (V, 13)

    bus_ids = bus0[:, BUS_I].astype(int)
    id2idx = {bid: i for i, bid in enumerate(bus_ids)}

    f_ids = branch0[:, F_BUS].astype(int)
    t_ids = branch0[:, T_BUS].astype(int)
    edges = [(id2idx[fi], id2idx[ti]) for fi, ti in zip(f_ids, t_ids)]

    nodes = list(range(len(bus_ids)))

    # Keep bus columns as node features
    node_features = {
        i: {f"bus_col_{c}": float(bus0[i, c]) for c in range(bus0.shape[1])}
        for i in nodes
    }

    # Keep branch columns as edge features
    edge_features = {}
    for j, (u, v) in enumerate(edges):
        feats = {f"branch_col_{c}": float(branch0[j, c]) for c in range(branch0.shape[1])}
        tap = float(branch0[j, TAP])
        feats["TAP_fixed"] = 1.0 if tap == 0.0 else tap 
        edge_features[(u, v)] = feats

    if only_sc:
        complex = SimplicialComplex(
            nodes=nodes,
            edges=edges,
            triangles=[],
            node_features=node_features,
            edge_features=edge_features,
        )
    else:
        complex = CellComplex(
            nodes=nodes,
            edges=edges,
            polygons=[],
            node_features=node_features,
            edge_features=edge_features,
        )

    coordinates = complex.generate_coordinates()

    # Flow: PF per edge, aligned with complex.edges (same pattern as forex/transportation)
    pf = branch0[:, PF]
    flow = {edge: float(pf[i]) for i, edge in enumerate(complex.edges)}

    return complex, coordinates, flow
