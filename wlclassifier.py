from matplotlib import pyplot as plt
import networkx as nx
from sklearn import tree
from sklearn import naive_bayes


class WLClassifier():
    """Class representing a decision tree classifier that uses the
    Weisfeiler-Lehman algorithm to transform graphs into feature vectors.
    """
    def __init__(
        self,
        depth=2,
        comparisons=[
            # lambda x, y: x == y,
            lambda x, y: x >= y,
            # lambda x, y: x < y,
        ],
        nb=False
    ) -> None:
        """Initialize a WLClassifier object.

        Args:
            depth: The number of iterations of the Weisfeiler-Lehman algorithm
                to apply to the graphs.

        Returns:
            A WLClassifier object.
        """
        self.training_data = None
        self.eval_data = None
        self.features: set[(int, WLType)] = set([(0, WLType())])
        self.WL_training = None
        self.WL_eval = None
        self.WL_depth = depth
        self.nb = nb
        if self.nb:
            self.classifier = naive_bayes.GaussianNB()
        else:
            self.classifier = tree.DecisionTreeClassifier()
        self.comparisons = comparisons

    def _set_training_data(self, data: list[nx.Graph]):
        self.training_data = data
        self.WL_training = [[
            {v: WLType() for v in graph.nodes()}
            for graph in self.training_data
        ]]
        for i in range(0, self.WL_depth):
            self.WL_training.append([
                self._wl_(graph, prev_wl, i+1, True)
                for graph, prev_wl in zip(
                    self.training_data, self.WL_training[i]
                )
            ])
        self.WL_training = list(zip(*self.WL_training))
        self.features = sorted(self.features)

    def _set_eval_data(self, data: list[nx.Graph]):
        self.eval_data = data
        self.WL_eval = [[
            {v: WLType() for v in graph.nodes()}
            for graph in self.eval_data
        ]]
        for i in range(0, self.WL_depth):
            self.WL_eval.append([
                self._wl_(graph, prev_wl, i+1)
                for graph, prev_wl in zip(self.eval_data, self.WL_eval[i])
            ])
        self.WL_eval = list(zip(*self.WL_eval))

    def _wl_(self, graph, prev_wl, i, add_to_labels=False):
        new_wl = {}
        for node in graph.nodes():
            new_wl[node] = WLType(*(
                prev_wl[m] for m in graph.neighbors(node)
            ))
            if add_to_labels:
                self.features.add((i, new_wl[node]))
        return new_wl

    def _transform(self, training=False):
        if training:
            graphs = self.training_data
            wltypes = self.WL_training
        else:
            graphs = self.eval_data
            wltypes = self.WL_eval
        return [
            [
                len([
                    node for node in graph.nodes()
                    if comparison(wl[i][node], type)
                ])
                for comparison in self.comparisons
                for (i, type) in self.features
            ]
            for wl, graph in zip(wltypes, graphs)
        ]

    def apply(self, graphs: list[nx.Graph]):
        """Return the index of the leaf each sample falls into.

        Args:
            graphs: The graphs to classify.

        Returns:
            The index of the leaf each sample falls into.
        """
        self._set_eval_data(graphs)
        X = self._transform()
        return self.classifier.apply(X)

    def decision_path(self, graphs: list[nx.Graph]):
        """Return the decision path in the tree.

        Args:
            graphs: The graphs to classify.

        Returns:
            Return a node indicator CSR matrix where non zero elements
            indicates that the samples goes through the nodes.
        """
        assert not self.nb
        self._set_eval_data(graphs)
        X = self._transform()
        return self.classifier.decision_path(X)

    def fit(
        self,
        data: list[nx.Graph],
        labels: list,
        sample_weight: list[float] = None
    ):
        """Build a decision tree classifier from the training set.

        Args:
            data: The training input samples. A list of graphs.
            labels: The target values (class labels) as booleans, integers
            or strings.
            sample_weight: Sample weights. If None, then samples are equally
            weighted.
        """
        self._set_training_data(data)
        X = self._transform(training=True)
        y = labels
        self.classifier.fit(X, y, sample_weight=sample_weight)
        if not self.nb:
            self.tree_ = self.classifier.tree_

    def get_depth(self):
        """Return the depth of the decision tree.

        Returns:
            The depth of the decision tree.
        """
        return self.classifier.get_depth()

    def get_n_leaves(self):
        """Return the number of leaves of the decision tree.

        Returns:
            The number of leaves of the decision tree.
        """
        assert not self.nb
        return self.classifier.get_n_leaves()

    def predict(self, graphs: list[nx.Graph]):
        """Predict class or regression value for X.

        Args:
            graphs: The graphs to classify.

        Returns:
            The predicted classes, or the predict values.
        """
        self._set_eval_data(graphs)
        X = self._transform()
        return self.classifier.predict(X)

    def score(self, graphs: list[nx.Graph], labels: list):
        """Return the mean accuracy on the given test data and labels.

        Args:
            graphs: The graphs to classify.
            labels: The target values (class labels) as booleans, integers
                or strings.

        Returns:
            Mean accuracy of self.predict(graphs) wrt. labels.
        """
        self._set_eval_data(graphs)
        X = self._transform()
        return self.classifier.score(X, labels)

    def draw(self):
        """Draw the decision tree.
        """
        assert self.classifier is not None and not self.nb
        depth = self.classifier.tree_.max_depth
        dim_x, dim_y = 2**(depth), depth + 1
        figure = plt.figure(figsize=(2*dim_x, 2*dim_y))
        self._draw_rec(0, 1, 1, figure, dim_x, dim_y)
        plt.show()

    def _draw_rec(self, node, x_index, y_index, fig, dim_x, dim_y):
        x = (x_index-1) * (2**(dim_y - y_index)) + 1
        index = (2**(dim_y - y_index)) + (y_index-1)*2*dim_x + 2*x - 1
        ax = fig.add_subplot(
            dim_y,
            2*dim_x,
            (index, index + 1)
        )
        if self.classifier.tree_.feature[node] == -2:
            ax.text(
                0.5,
                0.5,
                repr([int(i) for i in self.classifier.tree_.value[node][0]]),
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes,
                fontsize=18,
            )
            ax.axis('off')
        else:
            ax.set_title(f"Contains >{self.classifier.tree_.threshold[node]}")
            nx.draw(
                self.features[
                    self.classifier.tree_.feature[node]
                ][1].to_networkx(),
                node_size=300/4**self.features[
                    self.classifier.tree_.feature[node]
                ][0],
                ax=ax
            )
        if self.classifier.tree_.children_left[node] != -1:
            self._draw_rec(
                self.classifier.tree_.children_left[node],
                2*(x_index-1) + 1,
                y_index + 1,
                fig,
                dim_x,
                dim_y
            )
        if self.classifier.tree_.children_right[node] != -1:
            self._draw_rec(
                self.classifier.tree_.children_right[node],
                2*(x_index-1) + 2,
                y_index + 1,
                fig,
                dim_x,
                dim_y
            )


class WLType:
    """Class representing unordered rooted trees. Unordered rooted trees
    correspond exactly to the possible colors of the Weisfeiler-Lehman
    algorithm and are used to represent the result of applyting the
    Weisfeiler-Lehman algorithm to a graph.
    """
    def __init__(self, *wl_type):
        """Initialize a WLType object. The trivial tree is represented by
        WLType(). Otherwise, the WL type is given by WLType(*children_wl_types)

        Args:
            *wl_type: The WL types of the children

        Returns:
            A WLType object
        """
        self.data = tuple(sorted(wl_type, reverse=True))

    def __eq__(self, other: "WLType"):
        """t1 == t2 iff t1 and t2 are isomorphic as rooted unordered trees.

        Args:
            other: The other WLType object

        Returns:
            True if t1 and t2 are isomorphic as rooted unordered trees, False
            otherwise
        """
        return self.data == other.data

    def __ne__(self, other: "WLType"):
        """t1 != t2 iff t1 and t2 are not isomorphic as rooted unordered trees.

        Args:
            other: The other WLType object

        Returns:
            True if t1 and t2 are not isomorphic as rooted unordered trees,
            False otherwise
        """
        return self.data != other.data

    def __ge__(self, other: "WLType"):
        """t1 >= t2 iff t1 contains t2 as a subgraph.

        Args:
            other: The other WLType object.

        Returns:
            True if self contains other as a subgraph, False otherwise.
        """
        return len(self.data) >= len(other.data) and all([
            self.data[i] >= other.data[i]
            for i in range(0, min(len(self.data), len(other.data)))
        ])

    def __gt__(self, other):
        """t1 > t2 iff t1 is a proper subgraph of t2.

        Args:
            other: The other WLType object.

        Returns:
            True if self is a proper subgraph of other, False otherwise.
        """
        return (self >= other) and (self != other)

    def __le__(self, other):
        """t1 <= t2 iff t2 contains t1 as a subgraph.

        Args:
            other: The other WLType object.

        Returns:
            True if other contains self as a subgraph, False otherwise.
        """
        return other.__ge__(self)

    def __lt__(self, other):
        '''t1 < t2 iff t1 is a proper subgraph of t2.

        Args:
            other: The other WLType object.

        Returns:
            True if self is a proper subgraph of other, False otherwise.
        '''
        return other.__gt__(self)

    def __hash__(self):
        return hash(self.data)

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
            h = nx.disjoint_union_all(args)
            nx.relabel_nodes(h, lambda x: x+1, copy=False)
            h.add_node(0)
            index = 1
            for g in args:
                h.add_edge(0, index)
                index += g.number_of_nodes()
            return h
