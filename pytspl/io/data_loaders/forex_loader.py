import pandas as pd
import pkg_resources

from pytspl.io.network_reader import read_B1_B2

FOREX_DATA_FOLDER = pkg_resources.resource_filename(
    "pytspl", "data/foreign_exchange"
)


def load_forex_data(only_sc: bool = True, only_2d: bool = True) -> tuple:
    """
    Load the forex data and return the simplicial complex and coordinates.

    Args:
        only_sc(bool, optional): if true return a simplicial complex, else return a cell complex
        only_2d (bool, optional): if true (default) build up to triangles; if false, build all simplices.

    Returns:
        tuple:
            SimplicialComplex or CellComplex: The simplicial/cell complex of the forex data.
            dict: The coordinates of the nodes. If the coordinates do not
            exist, the coordinates are generated using spring layout.
            dict: The flow data of the forex data.
    """
    B1_filename = f"{FOREX_DATA_FOLDER}/B1.csv"
    B2_filename = f"{FOREX_DATA_FOLDER}/B2t.csv"
    y_filename = f"{FOREX_DATA_FOLDER}/flow_FX_1538755200.csv"

    builder, triangles = read_B1_B2(
        B1_filename=B1_filename, B2_filename=B2_filename, only_sc=only_sc
    )

    if only_sc:
        complex = builder.to_simplicial_complex(triangles=triangles, only_2d=only_2d)
    else:
        complex = builder.to_cell_complex()

    # no coordinates for forex data - generate using spring layout
    coordinates = complex.generate_coordinates()

    # read flow data
    flow = pd.read_csv(y_filename, header=None).values[:, 0]
    flow = {edge: flow[i] for i, edge in enumerate(complex.edges)}

    return complex, coordinates, flow
