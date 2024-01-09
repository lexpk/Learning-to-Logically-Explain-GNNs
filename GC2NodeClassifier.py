import numpy as np
from colorrefinement import color_refinement
from networkx import Graph


class GC2NodeClassifier:
    """Class representing the GC2 node classifier.
    """
    def __init__(
        self,
        graphs: list[Graph],
        vertices: list[int],
        vertex_labels: list[list[set]] = None,
        labels: np.array = None,
        max_depth: int = None
    ):
        """Initialize a GC2NodeClassifier object.

        Args:
            graph: The graph on which to apply the algorithm.

        Returns:
            A GC2NodeClassifier object.
        """
        self.graphs = graphs
        self.vertices = vertices
        self.vertex_labels = vertex_labels
        self.max_depth = max_depth
        self.color_refinement = []
        self._init_gc2_node_classifier()

    def _init_gc2_node_classifier(self):
        """Initialize the GC2 node classifier.
        """
        self.formulas = {}
        for graph, vertex, vertex_labels in zip(
            self.graphs,
            self.vertices,
            self.vertex_labels
        ):
            cr = color_refinement(
                graph, vertex_labels, self.max_depth
            )
            self.formulas |= cr.node_colors[cr.depth][vertex] \
                .chi(cr.depth).subformulas()
