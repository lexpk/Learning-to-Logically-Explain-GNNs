from collections import defaultdict
import networkx as nx


class ColorRefinement:
    """Class representing the color refinement algorithm.
    """
    def __init__(
        self,
        graph: nx.Graph,
        labels: dict[int, int] = None,
        max_depth: int = None
    ):
        """Initialize a ColorRefinement object.

        Args:
            graph: The graph on which to apply the algorithm.

        Returns:
            A ColorRefinement object.
        """
        self.graph = graph
        if labels is None:
            self.labels = [() for _ in self.graph.nodes()]
        else:
            self.labels = labels
        self.node_colors = []
        self.color_nodes = []
        self.depth = 0
        self.max_depth = max_depth
        self._init_colors()
        self._refine()

    def _init_colors(self):
        """Initialize the colors of the nodes.
        """
        next_node_colors = [None for _ in self.graph.nodes()]
        next_color_nodes = defaultdict(set)
        for node in self.graph.nodes():
            color = RefinementColor(labels=self.labels[node])
            next_node_colors[node] = color
            next_color_nodes[color].add(node)
        self.node_colors.append(tuple(next_node_colors))
        self.color_nodes.append(next_color_nodes)

    def _refine(self):
        """Apply the color refinement algorithm.
        """
        while True:
            next_node_colors = [None for _ in self.graph.nodes()]
            next_color_nodes = defaultdict(set)
            for node in self.graph.nodes():
                color = RefinementColor(*(
                    self.node_colors[-1][m] for m in self.graph.neighbors(node)
                ), labels=self.labels[node])
                next_node_colors[node] = color
                next_color_nodes[color].add(node)
            if len(next_color_nodes) == len(self.color_nodes[-1]):
                return
            self.color_nodes.append(next_color_nodes)
            self.node_colors.append(tuple(next_node_colors))
            self.depth += 1
            if self.depth == self.max_depth:
                return


class RefinementColor:
    """Class representing unordered rooted trees. Unordered rooted trees
    correspond exactly to the possible colors of Color Refinement and
    are used to represent the result of applyting the algorithm to a graph.
    """
    def __init__(self, *wl_type, labels: tuple = tuple()):
        """Initialize a WLType object. The trivial tree is represented by
        WLType(). Otherwise, the WL type is given by WLType(*children_wl_types)

        Args:
            *wl_type: The WL types of the children
            labels: The labels of the root node

        Returns:
            A WLType object
        """
        self.data = tuple(sorted(
            wl_type
        ))
        self.labels = labels
        self.hash = hash((self.data, self.labels))

    def __eq__(self, other: "RefinementColor"):
        """t1 == t2 iff t1 and t2 are isomorphic as rooted unordered trees.

        Args:
            other: The other WLType object

        Returns:
            True if t1 and t2 are isomorphic as rooted unordered trees, False
            otherwise
        """
        if self.hash != other.hash:
            return False
        if self.labels != other.labels:
            return False
        if self.data != other.data:
            return False
        return True

    def __ne__(self, other: "RefinementColor"):
        """t1 != t2 iff t1 and t2 are not isomorphic as rooted unordered trees.

        Args:
            other: The other WLType object

        Returns:
            True if t1 and t2 are not isomorphic as rooted unordered trees,
            False otherwise
        """
        return not self.__eq__(other)

    def __lt__(self, other: "RefinementColor"):
        if self.hash != other.hash:
            return self.hash < other.hash
        if self.labels != other.labels:
            return self.labels < other.labels
        if self.data != other.data:
            return self.data < other.data
        return False

    def __hash__(self):
        return self.hash

    def __repr__(self):
        return f"{{{', '.join([str(t) for t in self.data])}}}"

    def to_networkx(self):
        '''Convert the WL type to a networkx graph.

        Returns:
            A networkx graph representing the WL type.
        '''
        if self.data == tuple():
            g = nx.Graph()
            g.add_node(0)
            return g
        else:
            args = [t.to_networkx() for t in self.data]

            def yield_relabeled(graphs):
                first_label = 0
                for G in graphs:
                    yield nx.convert_node_labels_to_integers(
                        G, first_label=first_label, ordering="sorted"
                    )
                    first_label += len(G)

            h = nx.union_all(yield_relabeled(args))
            nx.relabel_nodes(h, lambda x: x+1, copy=False)
            h.add_node(0)
            index = 1
            for g in args:
                h.add_edge(0, index)
                index += g.number_of_nodes()
            return h
