import os
import pkg_resources
import networkx as nx

from pytspl.simplicial_complex.scbuilder import SCBuilder

WEBKB_DATA_FOLDER = pkg_resources.resource_filename(
    "pytspl", "data/webkb"
)

def load_webkb_data() -> dict[str, nx.Graph]:
    """
    Load all three WebKB hyperlink graphs (Cornell, Texas, Wisconsin)
    directly from the two-column TXT files.
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


def _load_webkb_subset(subset: str):
    """
    Helper to turn one WebKB split into (SC, coords, flow) exactly like your
    other loaders, using SCBuilder.
    """
    graphs = load_webkb_data()
    try:
        G = graphs[subset]
    except KeyError:
        raise ValueError(f"Unknown WebKB split '{subset}'")

    # build the simplicial complex
    builder = SCBuilder(
        nodes=list(G.nodes()), 
        edges=list(G.edges())
    )
    sc = builder.to_simplicial_complex(condition="all")

    # 2D coords via spring layout
    pos = nx.spring_layout(G)
    coords = {n: (float(x), float(y)) for n, (x, y) in pos.items()}

    # no flow for WebKB
    flow = {}

    return sc, coords, flow