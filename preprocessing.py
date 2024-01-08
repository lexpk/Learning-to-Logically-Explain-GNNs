import numpy as np


class CRPreprocessor:
    def __init__(self, formulas: list, with_nodes: bool = True):
        self.formulas = formulas
        self.with_nodes = with_nodes

    def __call__(self, graphs: list, nodes: list = None):
        if self.with_nodes:
            return self._call_with_nodes(graphs, nodes)
        else:
            return self._call_without_nodes(graphs)

    def _call_with_nodes(self, graphs: list, nodes: list):
        assert len(graphs) == len(nodes)
        result = np.array([
            [formula.evaluate(graph, node, node) for formula in self.formulas]
            for graph, node in zip(graphs, nodes)
        ])
        return result

    def _call_without_nodes(self, graphs: list):
        result = np.array([
            [formula.evaluate(graph, None, None) for formula in self.formulas]
            for graph in graphs
        ])
        return result
