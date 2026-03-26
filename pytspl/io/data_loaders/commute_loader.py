import pkg_resources
import pandas as pd
from pytspl.simplicial_complex import SimplicialComplex

COMMUTING_DATA_FOLDER = pkg_resources.resource_filename(
    "pytspl", "data/commute"
)

def load_commuting_msoa():
    flows_file = f"{COMMUTING_DATA_FOLDER}/flows.csv"
    coords_file = f"{COMMUTING_DATA_FOLDER}/nodes.csv"

    df = pd.read_csv(flows_file)
    coords_df = pd.read_csv(coords_file)

    vertices = sorted(set(df["origin"]).union(df["destination"]))
    edges = list(set(tuple(sorted((row.origin, row.destination))) for _, row in df.iterrows()))

    simplices = {
        0: vertices,
        1: edges,
    }

    sc = SimplicialComplex(simplices=simplices)

    sc.print_summary()

    coordinates = {
        row["id"]: (row["lon"], row["lat"])
        for _, row in coords_df.iterrows()
    }

    flow = {
        tuple(sorted((row.origin, row.destination))): row.flow
        for _, row in df.iterrows()
    }

    return sc, coordinates, flow