import pandas as pd
import pkg_resources

from pytspl.io.network_reader import read_B1_B2

LASTFM_DATA_FOLDER = pkg_resources.resource_filename(
    "pytspl", "data/lastfm-dataset-1K"
)


def load_lastfm_1k_artist(only_sc: bool = True) -> tuple:
    """
    Read the lastfm 1k artist data.

    Args:
        only_sc(bool, optional): If true, return a simplical complex, else return a cell complex
    
    Returns:
        tuple:
            SimplicialComplex or CellComplex: The simplicial/cell complex of the paper data.
            dict: The coordinates of the nodes.
            dict: The flow data of the paper data.
    """
    builder, triangles = read_B1_B2(
        f"{LASTFM_DATA_FOLDER}/B1-artist.csv",
        f"{LASTFM_DATA_FOLDER}/B2t-artist.csv",
    )
    if only_sc:
        complex = builder.to_simplicial_complex(triangles=triangles)
    else:
        complex = builder.to_cell_complex()

    # no coordinates for forex data - generate using spring layout
    coordinates = complex.generate_coordinates()

    flow_path = f"{LASTFM_DATA_FOLDER}/flow-artist.csv"
    flow = (
        pd.read_csv(flow_path, delimiter=",", header=None).to_numpy().flatten()
    )
    # create a dictionary of the flow data
    flow = {edge: flow[i] for i, edge in enumerate(complex.edges)}

    return complex, coordinates, flow
