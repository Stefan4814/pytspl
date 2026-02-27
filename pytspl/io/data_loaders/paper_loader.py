import pandas as pd
import pkg_resources

from pytspl.io.network_reader import read_coordinates, read_csv

PAPER_DATA_FOLDER = pkg_resources.resource_filename(
    "pytspl", "data/paper_data"
)


def load_paper_data(only_sc: bool = True, only_2d: bool = True) -> tuple:
    """
    Read the paper data and return the simplicial/cell complex, coordinates
    and the flow.

    Args:
        only_sc(bool, optional): whether to return the simplicial complex or the cell complex 
        only_2d (bool, optional): if true (default) build up to triangles; if false, build all simplices.
    
    Returns:
        tuple:
            SimplicialComplex or CellComplex: The simplicial/cell complex of the paper data.
            dict: The coordinates of the nodes.
            dict: The flow data of the paper data.
    """
    # read network data
    filename = PAPER_DATA_FOLDER + "/edges.csv"
    delimiter = " "
    src_col = "Source"
    dest_col = "Target"
    feature_cols = ["Distance"]

    builder = read_csv(
        filename=filename,
        delimiter=delimiter,
        src_col=src_col,
        dest_col=dest_col,
        feature_cols=feature_cols,
        only_sc=only_sc,
    )

    if only_sc:
        sc_or_cc = builder.to_simplicial_complex(only_2d=only_2d)
    else:
        sc_or_cc = builder.to_cell_complex()

    # read coordinates data
    filename = PAPER_DATA_FOLDER + "/coordinates.csv"
    coordinates = read_coordinates(
        filename=filename,
        node_id_col="Id",
        x_col="X",
        y_col="Y",
        delimiter=" ",
    )

    # read flow data
    filename = PAPER_DATA_FOLDER + "/flow.csv"
    flow = pd.read_csv(filename, header=None).values[:, 0]
    flow = {edge: flow[i] for i, edge in enumerate(sc_or_cc.edges)}

    return sc_or_cc, coordinates, flow
