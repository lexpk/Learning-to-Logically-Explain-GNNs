from collections import defaultdict
import networkx as nx
from c2 import Atom, Var, And, Not, GuardedExistsEq, GuardedExists


class ColorRefinement:
    """Class representing the color refinement algorithm.
    """
    def __init__(
        self,
        graph: nx.Graph,
        vertex_labels: dict[int, tuple] = None,
        max_depth: int = None
    ):
        """Initialize a ColorRefinement object.

        Args:
            graph: The graph on which to apply the algorithm.

        Returns:
            A ColorRefinement object.
        """
        if hasattr(graph, "color_refinement"):
            self = graph.color_refinement
        else:
            self.graph = graph
            if vertex_labels is None:
                self.vertex_labels = [() for _ in self.graph.nodes()]
            else:
                self.vertex_labels = vertex_labels
            self.node_colors = []
            self.color_nodes = []
            self.depth = 0
            self.max_depth = max_depth
            self._init_colors()
            self._refine()
            graph.color_refinement = self

    def _init_colors(self):
        """Initialize the colors of the nodes.
        """
        next_node_colors = [None for _ in self.graph.nodes()]
        next_color_nodes = defaultdict(set)
        for node in self.graph.nodes():
            color = RefinementColor(vertex_labels=self.vertex_labels[node])
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
                ), vertex_labels=self.vertex_labels[node])
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
    def __init__(self, *rc, vertex_labels: tuple = tuple()):
        """Initialize a RefinementColor object. The trivial tree is represented
        by RefinementColor(). Otherwise, the RefinementColor is given by
        RefinementColor(*children_wl_types)

        Args:
            *rc: The RefinementColors of the children
            vertex_labels: The vertex_labels of the root node

        Returns:
            A RefinementColor object
        """
        self.data = tuple(sorted(rc))
        self.vertex_labels = vertex_labels
        self.hash = hash((self.data, self.vertex_labels))

    def __eq__(self, other: "RefinementColor"):
        """t1 == t2 iff t1 and t2 are isomorphic as rooted unordered trees.

        Args:
            other: The other RefinementColor

        Returns:
            True if t1 and t2 are isomorphic, False otherwise
        """
        if self.hash != other.hash:
            return False
        if self.vertex_labels != other.vertex_labels:
            return False
        if self.data != other.data:
            return False
        return True

    def __ne__(self, other: "RefinementColor"):
        """t1 != t2 iff t1 and t2 are not isomorphic.

        Args:
            other: The other RefinementColor

        Returns:
            True if t1 and t2 are not isomorphic, False otherwise
        """
        return not self.__eq__(other)

    def __lt__(self, other: "RefinementColor"):
        if self.hash != other.hash:
            return self.hash < other.hash
        if self.vertex_labels != other.vertex_labels:
            return self.vertex_labels < other.vertex_labels
        if self.data != other.data:
            return self.data < other.data
        return False

    def __hash__(self):
        return self.hash

    def __repr__(self):
        return f"{self.vertex_labels}\n\t" +\
            "\n\t".join([repr(t) for t in self.data])

    def to_networkx(self):
        '''Convert the RefinementColor to a networkx graph.

        Returns:
            A networkx graph representing the RefinementColor.
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

    def chi_0(self, var: Var = Var.x):
        '''Gives a FO-formula decribing the vertex labels of the root node.

        Args:
            var: The free variable to use.
        '''
        return And(*[
            Atom(f"P{i}", var) if self.vertex_labels[i]
            else Not(Atom(f"P{i+1}", var))
            for i in range(len(self.vertex_labels))]
        )

    def chi(self, n, var: Var = Var.x):
        '''Gives a FO-formula describing a Refinement Color up to depth n.

        Args:
            n: The depth.
            var: The free variable to use.
        '''
        if n == 0:
            return self.chi_0(var)
        else:
            child_count = (
                (child, sum(1 for c in self.data if c == child))
                for child in set(self.data)
            )
            other = Var.y if var == Var.x else Var.x
            return And(*[
                self.chi_0(var),
                And(*[
                    GuardedExistsEq(count, var, child.chi(n-1, other))
                    for (child, count) in child_count
                ]),
                Not(GuardedExists(var, And(*[
                    Not(child.chi(n-1, other))
                    for (child, _) in child_count
                ])))
            ])
