import pickle

import numpy as np
import pkg_resources

from pytspl.simplicial_complex.scbuilder import SCBuilder
from pytspl.cell_complex.ccbuilder import CCBuilder

WSN_DATA_FOLDER = pkg_resources.resource_filename("pytspl", "data/wsn")


def load_wsn_data(only_sc: bool = True, only_2d: bool = True) -> tuple:
    """
    Load the water supply network data and return the simplicial complex
    and coordinates.

    Args:
        only_sc (bool, optional): if true return a simplicial complex, else return a cell complex.

    Returns:
        tuple:
            SimplicialComplex/CellComplex: The simplicial/cell complex of the water supply
            network data.
            dict: The coordinates of the nodes. If the coordinates do not
            exist, the coordinates are generated using spring layout.
            np.ndarray: The flow data of the water supply
    """

    with open(f"{WSN_DATA_FOLDER}/water_network.pkl", "rb") as f:
        B1, flow_rate, _, head, hr = pickle.load(f)

    num_edges = B1.shape[1]
    nodes = set()
    edges = []

    for j in range(num_edges):
        col = B1[:, j]
        from_node = np.where(col == -1)[0][0]
        to_node = np.where(col == 1)[0][0]

        nodes.add(from_node)
        nodes.add(to_node)

        edges.append((from_node, to_node))

    nodes = list(range(max(nodes) + 1))

    builder_cls = SCBuilder if only_sc else CCBuilder
    builder = builder_cls(nodes=nodes, edges=edges)

    if only_sc:
        complex = builder.to_simplicial_complex(only_2d=only_2d)
    else:
        complex = builder.to_cell_complex()

    # no coordinates
    coordinates = complex.generate_coordinates()

    # read flow data
    hr = hr.squeeze()
    head = np.asarray(head)

    flow_rate = np.asarray(flow_rate)
    sign = np.sign(flow_rate)
    flow_rate = -hr * sign * np.abs(flow_rate) ** 1.852
    hr[:] = 1

    y = np.concatenate((head, flow_rate))

    return complex, coordinates, y
