from math import ceil
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from c2 import And, Atom, GuardedExistsGeq, Not, Top, Var
from networkx import adjacency_matrix
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class GC2NodeClassifier:
    """Class representing the GC2 node classifier.
    """
    def __init__(
        self,
        graph,
        features,
        label,
        evaluation_depth=2,
        max_depth=1,
        criterion="gini",
        splitter="random",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
    ):
        """Initialize a GC2NodeClassifier object. Encode node features
        such that graph.nodes[vertex] yields a dictionary with the featurename
        as key and the feature value as value. The label should also be part
        of this dictionary.

        Args:
            data: A list of tuples containing a graph and a vertex.
            target: The key of the label.

        Returns:
            A GC2NodeClassifier object.
        """
        self.graph = graph
        self.features = [Top()] + [Atom(f, Var.x) for f in features]
        self.label = label

        self.adj = adjacency_matrix(self.graph)
        self.degree = self.adj.sum(axis=1).reshape(-1, 1)
        self.X = np.concatenate(
            (
                np.ones((len(graph), 1)),
                np.array([
                    [
                        self.graph.nodes[node][feature]
                        for feature in self.features[1:]
                    ]
                    for node in self.graph.nodes
                ], dtype=bool)
            ), axis=1
        )
        self.eval_depth = evaluation_depth
        self.y = np.array([
            [graph.nodes[node][self.label]]
            for node in graph.nodes
        ])
        p = np.mean(self.y)
        self.weights = self.y * (1 - p) + (1 - self.y) * p
        self._y = self.y
        for i in range(self.eval_depth):
            self._y = np.concatenate(
                (
                    self._y,
                    self.adj.dot(self._y[:, -1:])/self.degree
                ),
                axis=1
            )
        self._y = StandardScaler() \
            .fit(self._y) \
            .transform(self._y)
        self.tree_args = {
            "criterion": criterion,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "min_weight_fraction_leaf": min_weight_fraction_leaf,
            "max_features": max_features,
            "random_state": random_state,
            "max_leaf_nodes": max_leaf_nodes,
            "min_impurity_decrease": min_impurity_decrease,
        }

    def _extend(self):
        for level in range(self.eval_depth - 1, -1, -1):
            X_int = self.adj.dot(self.X)
            self.dt = DecisionTreeRegressor(max_depth=2)
            self.dt.fit(
                np.concatenate(
                    (self.X, X_int, self.degree - X_int),
                    axis=1
                ),
                self._y[:, level],
                sample_weight=self.weights.reshape(-1)
            )
            features = _new_formulas_from_decision_tree(self.dt)
            X_features = np.zeros((
                len(self.graph),
                len(features) + 4*len(features)**2
            ))
            new_formulas = []
            i = 0

            def check_unique(feature):
                for j in range(self.X.shape[1]):
                    if np.array_equal(feature, self.X[:, j]) or \
                            np.array_equal(feature, 1 - self.X[:, j]):
                        return False
                for j in range(i):
                    if np.array_equal(feature, X_features[:, j]) or \
                            np.array_equal(feature, 1 - X_features[:, j]):
                        return False
                return True

            for sign, id, threshold in features:
                result = self._to_feature_vector(id, threshold, X_int)
                if check_unique(result):
                    X_features[:, i] = result
                    new_formulas.append(
                        self._to_formula(sign, id, threshold).simplify()
                    )
                    i += 1
            for ((sign1, id1, threshold1), (sign2, id2, threshold2)) in \
                    [(f1, f2) for f1 in features for f2 in features]:
                result = self._to_feature_vector(id1, threshold1, X_int) \
                    * self._to_feature_vector(id2, threshold2, X_int)
                if check_unique(result):
                    X_features[:, i] = result
                    new_formulas.append(
                        And(
                            self._to_formula(sign1, id1, threshold1),
                            self._to_formula(sign2, id2, threshold2)
                        ).simplify()
                    )
                    i += 1
                result = self._to_feature_vector(id1, threshold1, X_int) \
                    * (1 - self._to_feature_vector(id2, threshold2, X_int))
                if check_unique(result):
                    X_features[:, i] = result
                    new_formulas.append(
                        And(
                            self._to_formula(sign1, id1, threshold1),
                            Not(self._to_formula(sign2, id2, threshold2))
                        ).simplify()
                    )
                    i += 1
                result = (1 - self._to_feature_vector(id1, threshold1, X_int))\
                    * self._to_feature_vector(id2, threshold2, X_int)
                if check_unique(result):
                    X_features[:, i] = result
                    new_formulas.append(
                        And(
                            Not(self._to_formula(sign1, id1, threshold1)),
                            self._to_formula(sign2, id2, threshold2)
                        ).simplify()
                    )
                    i += 1
                result = (1 - self._to_feature_vector(id1, threshold1, X_int))\
                    * (1 - self._to_feature_vector(id2, threshold2, X_int))
                if check_unique(result):
                    X_features[:, i] = result
                    new_formulas.append(
                        And(
                            Not(self._to_formula(sign1, id1, threshold1)),
                            Not(self._to_formula(sign2, id2, threshold2))
                        ).simplify()
                    )
                    i += 1
            if new_formulas:
                self.features += new_formulas
                self.X = np.concatenate((self.X, X_features[:, :i]), axis=1)
        if new_formulas:
            return True
        else:
            return False

    def _to_feature_vector(self, feature_id, threshold, X_int):
        """Convert an encoding of a geq_formula from a decision tree to the
        corresponding feature vector.

        Args:
            polarity: False if the formula is negated.
            feature_id: The integer encoding the feature.
            threshold: The integer encoding the threshold.
            X_int: The accumulated neighbor features.

        Returns:
            A feature vector.
        """
        if feature_id < len(self.features):
            return self.X[:, feature_id]
        elif feature_id < 2 * len(self.features):
            return X_int[:, feature_id - len(self.features)] > threshold
        else:
            return (self.degree - X_int)[
                :, feature_id - 2 * len(self.features)
            ] > threshold

    def _to_formula(self, sign, feature_id, threshold):
        """Convert an encoding of a geq_formula from a decision tree to a
        formula.

        Args:
            polarity: False if the formula is negated.
            feature_id: The integer encoding the feature.
            threshold: The integer encoding the threshold.

        Returns:
            A formula.
        """
        if feature_id < len(self.features):
            formula = self.features[feature_id]
        elif feature_id < 2 * len(self.features):
            feature = self.features[feature_id - len(self.features)]
            variable = Var.x if Var.y in feature.free_variables() else Var.y
            formula = GuardedExistsGeq(ceil(threshold), variable, feature)
        else:
            feature = self.features[feature_id - 2 * len(self.features)]
            variable = Var.x if Var.y in feature.free_variables() else Var.y
            formula = GuardedExistsGeq(ceil(threshold), variable, Not(feature))
        if sign:
            return formula
        else:
            return Not(formula)

    def formula(self):
        """Return a formula representing the classifier."""
        loss = np.mean((np.abs(self.X - self.y)*self.weights), axis=0)
        pos = np.argmin(loss)
        neg = np.argmax(loss)
        if loss[pos] > 1 - loss[neg]:
            return Not(self.features[neg])
        else:
            return self.features[pos]

    def predict(self, graph, vertex):
        """Predict the label of a vertex in a graph.

        Args:
            graph: The graph.
            vertex: The vertex.

        Returns:
            The predicted label.
        """
        return self.formula().evaluate(
            graph,
            vertex,
            vertex,
        )

    def accuracy(self):
        """Compute the accuracy of the classifier on the training data.

        Returns:
            The accuracy.
        """
        loss = np.mean((np.abs(self.X - self.y)*self.weights), axis=0)
        pos = np.argmin(loss)
        neg = np.argmax(loss)
        return max(loss[neg], 1 - loss[pos])


def _c2_formulas_from_decision_tree(clf: DecisionTreeClassifier):
    """Convert a decision tree to a formula tree represented as a tuple.
    """
    tree = clf.tree_

    def recurse(node_id=0):
        if tree.children_left[node_id] != tree.children_right[node_id]:
            formula = (
                tree.feature[node_id],
                tree.threshold[node_id]
            )
            return [
                ((False,) + formula, ) + conjunction
                for conjunction in recurse(tree.children_left[node_id])
            ] + [
                ((True,) + formula, ) + conjunction
                for conjunction in recurse(tree.children_right[node_id])
            ]
        else:
            return [(tree.value[node_id][0][1] > tree.value[node_id][0][0],),]
    return recurse()


def _new_formulas_from_decision_tree(clf: DecisionTreeClassifier):
    """Convert a decision tree to a list of formulas.
    """
    tree = clf.tree_
    formulas = []

    def recurse(node_id=0):
        if tree.children_left[node_id] != tree.children_right[node_id]:
            formulas.append(
                (
                    True,
                    tree.feature[node_id],
                    tree.threshold[node_id]
                )
            )
            recurse(tree.children_left[node_id])
            recurse(tree.children_right[node_id])
    recurse()
    return formulas
