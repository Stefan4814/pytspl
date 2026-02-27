# wntr_loader.py

import os
import pkg_resources
import wntr
import networkx as nx

from pytspl.simplicial_complex.scbuilder import SCBuilder
from pytspl.cell_complex.ccbuilder import CCBuilder


WNTR_DATA_FOLDER = pkg_resources.resource_filename(
    "pytspl", "data/wntr"
)


def _value_at_time(series, t_sec: int) -> float:
    """
    Robustly sample a WNTR results Series at time t_sec (seconds).
    If exact time not present, use nearest.
    """
    if t_sec in series.index:
        return float(series.loc[t_sec])
    idx = series.index.get_indexer([t_sec], method="nearest")[0]
    return float(series.iloc[idx])


def _load_pipes_and_junctions(wn: wntr.network.WaterNetworkModel) -> nx.DiGraph:
    """
    Read all pipes and junctions from a WNTR model into a graph.
    Note that pumps, valves, tanks, reservoirs, etc. are ignored.
    """
    G = nx.DiGraph()

    # exclude pump source junctions
    pump_sources = {pump_data.start_node.name for _, pump_data in wn.pumps()}

    # nodes = junctions
    for name, data in wn.junctions():
        if name not in pump_sources:
            G.add_node(
                name,
                xpos=float(data.coordinates[0]),
                ypos=float(data.coordinates[1]),
                zpos=float(data.elevation),
            )

    # edges = pipes between junctions, oriented start->end
    for pipe_name, pipe in wn.pipes():
        u, v = pipe.start_node_name, pipe.end_node_name
        if u in G.nodes and v in G.nodes:
            G.add_edge(
                u, v,
                name=pipe_name,
                length=float(pipe.length),
                diameter=float(pipe.diameter),
                roughness=float(pipe.roughness),
            )

    return G


def _remove_non_junctions(wn, results, junction_names, pipe_btwn_junction_names):
    """
    Take flows on links that touch non-junction nodes and add/subtract them into junction demands.
    """
    demand, flowrate = results.node["demand"], results.link["flowrate"]
    external_flow = flowrate.drop(pipe_btwn_junction_names, axis=1, errors="ignore")

    for pipe_name in external_flow.columns:
        source = wn.links[pipe_name].start_node_name
        sink = wn.links[pipe_name].end_node_name

        if source in junction_names and sink not in junction_names:
            demand[source] += external_flow[pipe_name]
        elif source not in junction_names and sink in junction_names:
            demand[sink] -= external_flow[pipe_name]

    return demand


def load_wntr_data(
    network_name: str,
    snapshot_hour: int = 12,
    only_sc: bool = True,
):
    """
    Load a single snapshot from an EPANET .inp file using WNTR.

    Returns
        complex : SimplicialComplex or CellComplex
        coordinates : dict[int, tuple[float, float]]
        flow_dict : dict[(u,v), float]   # oriented edge flow aligned with complex.edges
    """

    inp_path = os.path.join(WNTR_DATA_FOLDER, f"{network_name}.inp")
    if not os.path.isfile(inp_path):
        raise FileNotFoundError(f"Missing {network_name}.inp in {WNTR_DATA_FOLDER}")

    if not (0 <= snapshot_hour <= 23):
        raise ValueError("snapshot_hour must be in [0, 23].")

    # Run 24h simulation to get features and flow
    N_HOURS = 24
    wn = wntr.network.WaterNetworkModel(inp_path)
    wn.options.time.duration = 3600 * (N_HOURS - 1)
    wn.options.hydraulic.accuracy = 1e-8

    # eliminate minor losses for more stable results
    for _, pipe in wn.pipes():
        pipe.minor_loss = 0.0

    sim = wntr.sim.WNTRSimulator(wn)
    results = sim.run_sim()

    t_sec = int(snapshot_hour * 3600)

    # Build graph (junctions + pipes)
    G = _load_pipes_and_junctions(wn)

    # remove non-junctions by modifying demand
    pipe_btwn_junction_names = {data["name"] for _, _, data in G.edges(data=True)}
    demand = _remove_non_junctions(
        wn, results, set(G.nodes), pipe_btwn_junction_names
    )

    # Reindex nodes to 0..N-1 (pytspl convention)
    # Use a stable ordering so runs are reproducible
    node_names = sorted(G.nodes())
    node_map = {name: i for i, name in enumerate(node_names)}
    nodes = list(range(len(node_names)))

    # coordinates dict
    coordinates = {
        node_map[name]: (float(G.nodes[name]["xpos"]), float(G.nodes[name]["ypos"]))
        for name in node_names
    }

    # node features at snapshot
    node_features = {}
    for name in node_names:
        i = node_map[name]
        node_features[i] = {
            "elevation": float(G.nodes[name]["zpos"]),
            "demand": _value_at_time(demand[name], t_sec),
            "pressure": _value_at_time(results.node["pressure"][name], t_sec),
            "head": _value_at_time(results.node["head"][name], t_sec),
        }

    # edges/features with correct orientation (u->v from EPANET)
    edges = []
    edge_features = {}
    edge_to_pipe = {}

    for u_name, v_name, data in G.edges(data=True):
        u = node_map[u_name]
        v = node_map[v_name]
        edges.append((u, v)) 

        edge_to_pipe[(u, v)] = data["name"]
        edge_features[(u, v)] = {
            "length": float(data["length"]),
            "diameter": float(data["diameter"]),
            "roughness": float(data["roughness"]),
        }

    # Build complex
    builder_cls = SCBuilder if only_sc else CCBuilder
    builder = builder_cls(
        nodes=nodes,
        edges=edges,
        node_features=node_features,
        edge_features=edge_features,
    )

    complex = builder.to_simplicial_complex() if only_sc else builder.to_cell_complex()

    # Flow dict aligned with complex.edges (edge->value), sign correct
    flow_dict = {}
    for (u, v) in complex.edges:
        pipe_name = edge_to_pipe[(u, v)]
        flow_dict[(u, v)] = _value_at_time(results.link["flowrate"][pipe_name], t_sec)

    # If coordinates are missing for some reason, follow library convention
    if not coordinates:
        coordinates = complex.generate_coordinates()

    return complex, coordinates, flow_dict