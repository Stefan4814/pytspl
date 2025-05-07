import os
import pkg_resources
import networkx as nx

from pytspl.simplicial_complex.scbuilder import SCBuilder
from pytspl.cell_complex.ccbuilder import CCBuilder

WEBKB_DATA_FOLDER = pkg_resources.resource_filename(
    "pytspl", "data/webkb"
)

def load_webkb_data() -> dict[str, nx.Graph]:
    """
    Load the raw hyperlink graphs for all three WebKB subdatasets.

    Returns:
        dict[str, nx.Graph]:
            A mapping from subset name (“cornell”, “texas”, “wisconsin”)
            to its corresponding NetworkX Graph.

    Raises:
        FileNotFoundError:
            If any expected `out1_graph_edges.txt` file is missing under
            `data/webkb/{subset}/`.
    """
    graphs: dict[str, nx.Graph] = {}
    for subset in ("cornell", "texas", "wisconsin"):
        edge_path = os.path.join(
            WEBKB_DATA_FOLDER, subset, "out1_graph_edges.txt"
        )
        if not os.path.isfile(edge_path):
            raise FileNotFoundError(
                f"Missing {subset} edges file at {edge_path}"
            )

        # networkx will parse the two-column tab-separated text into ints
        G = nx.read_edgelist(
            edge_path,
            nodetype=int,
            data=False,
            # create_using=nx.DiGraph()            
        )
        graphs[subset] = G

    return graphs


def _load_webkb_subset(subset: str, only_sc: bool = True):
    """
    Build and return a (simplicial or cell) complex for one WebKB split.

    Args:
        subset (str):
            One of “cornell”, “texas”, or “wisconsin” indicating which
            subdataset to load.
        only_sc (bool, optional):
            If True (default), returns a SimplicialComplex. If False, returns CellComplex.
    Returns:
        tuple:
            - complex (SimplicialComplex or CellComplex):
                The built complex for the chosen WebKB split.
            - coords (dict[int, tuple[float, float]]):
                Spring‐layout (x,y) positions for each node.
            - flow (dict):
                Empty dictionary, since WebKB has no flow data.

    Raises:
        ValueError:
            If subset is not one of the supported names.
    """
    graphs = load_webkb_data()
    try:
        G = graphs[subset]
    except KeyError:
        raise ValueError(f"Unknown WebKB split '{subset}'")

    # build the simplicial complex
    builder_cls = SCBuilder if only_sc else CCBuilder
    builder = builder_cls(
        nodes=list(G.nodes()), 
        edges=list(G.edges())
    )
    if only_sc:
        complex = builder.to_simplicial_complex()
    else:
        complex = builder.to_cell_complex()

    # 2D coords via spring layout
    pos = nx.spring_layout(G)
    coords = {n: (float(x), float(y)) for n, (x, y) in pos.items()}

    # no flow for WebKB
    flow = {}

    return complex, coords, flow