"""Module for generating a random simplicial complex."""

import networkx as nx
import numpy as np

from pytspl.simplicial_complex.scbuilder import SCBuilder
from pytspl.simplicial_complex.simplicial_complex import SimplicialComplex

def generate_random_simplicial_complex(
    num_of_nodes: int,
    p: float,
    dist_threshold: float,
    seed: int,
    max_dim: int = 2,
) -> tuple[SimplicialComplex, dict]:
    """
    Generate a random simplicial complex.

    Args:
        num_of_nodes (int): Number of nodes in the graph.
        p (float): Probability of edge creation.
        dist_threshold (float): Threshold for simplicial complex construction.
        seed (int): Seed for random number generator.
        max_dim (int, optional): Maximum simplicial dimension to build.
            Defaults to 2 (triangles only for backward compatibility).

    Returns:
        SimplicialComplex: The generated simplicial complex.
        dict: The coordinates of the nodes.
    """
    G = nx.erdos_renyi_graph(n=num_of_nodes, p=p, seed=seed, directed=False)

    # get random weights
    import random

    weights = [random.random() for i in range(G.number_of_edges())]
    # set the weights
    for i, (u, v) in enumerate(G.edges()):
        G[u][v]["distance"] = weights[i]

    nodes = list(G.nodes())
    edges = list(G.edges())

    # get edge features
    edges_features = {}
    for u, v in G.edges():
        features = {k: v for k, v in G[u][v].items()}
        edges_features[(u, v)] = features

    builder = SCBuilder(
        nodes=nodes, edges=edges, edge_features=edges_features
    )
    # only_2d stays default True; if max_dim > 2 we enumerate all cliques
    only_2d = max_dim <= 2
    if only_2d:
        sc = builder.to_simplicial_complex(
            condition="distance", dist_threshold=dist_threshold, only_2d=True
        )
    else:
        # build all simplices via cliques then trim to max_dim
        simplices = builder._all_simplices()
        if max_dim is not None:
            simplices = {k: v for k, v in simplices.items() if k <= max_dim}
        sc = builder.to_simplicial_complex(simplices=simplices, only_2d=False)
    coordinates = nx.spring_layout(G)

    return sc, coordinates
