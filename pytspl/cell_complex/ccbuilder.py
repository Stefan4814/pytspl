import networkx as nx

from pytspl.cell_complex import CellComplex


class CCBuilder:

    def __init__(
        self,
        nodes: list,
        edges: list,
        node_features: dict = {},
        edge_features: dict = {},
    ):
        # 0-simplices - nodes
        self.nodes = nodes
        # 1-simplices - edges
        self.edges = edges

        # node and edge features
        self.node_features = node_features
        self.edge_features = edge_features

    def polygons(self) -> list:
        """
        Get a list of polygons (cycles) in the undirected graph.

        Returns:
            list: List of polygons as sets of nodes.
        """
        # Create an undirected graph
        g = nx.Graph()
        g.add_edges_from(self.edges)

        # Find all fundamental cycles (basis cycles)
        # Returns a list of cycles (each cycle is a list of nodes)
        all_cycles = nx.cycle_basis(g)

        # Sort nodes within each cycle and store as tuples to ensure uniqueness
        unique_cycles = {tuple((cycle)) for cycle in all_cycles}

        # Filter out non-minimal cycles
        simple_cycles = []
        for cycle in unique_cycles:
            if not any(set(cycle) > set(smaller_cycle)
                       for smaller_cycle in simple_cycles):
                simple_cycles.append(cycle)

        return simple_cycles

    def to_cell_complex(
        self,
        condition: str = "all",
        dist_col_name: str = "distance",
        dist_threshold: float = 1.5,
        polygons=None,
    ) -> CellComplex:
        """
        Convert the graph to a cell complex using the given condition.
        The cell complex will also have node and edge
        features.

        Args:
            condition (str, optional): Condition to build the simplices. Defaults to "all".
            Options:
            - "all": All simplices.
            - "distance": Based on distance.

            dist_col_name (str, optional): Name of the column that contains
            the distance.
            dist_threshold (float, optional): Distance threshold to consider
            for simplices. Defaults to 1.5.

        Returns:
            CellComplex: Cell complex network.
        """
        if polygons is None:
            if condition == "all":
                polygons = self.polygons()

        cc = CellComplex(
            nodes=self.nodes,
            edges=self.edges,
            polygons=polygons,
            node_features=self.node_features,
            edge_features=self.edge_features,
        )

        return cc
