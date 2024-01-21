from collections import defaultdict
from itertools import chain, combinations
import numpy as np
from c2 import And, Atom, GuardedExistsGeq, Not, Top, Var
from networkx import adjacency_matrix
from sklearn.tree import DecisionTreeClassifier


class GC2NodeClassifier:
    """Class representing the GC2 node classifier.
    """
    def __init__(
        self,
        graph,
        features,
        label,
        evaluation_depth=2,
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
                        self.graph.nodes[node][feature.name]
                        for feature in self.features[1:]
                    ]
                    for node in self.graph.nodes
                ], dtype=bool)
            ), axis=1
        )
        self.eval_depth = evaluation_depth + 1
        self.y = np.array([
            graph.nodes[node][self.label]
            for node in self.graph.nodes
        ])
        self.features_n = [self.features for _ in range(self.eval_depth)]
        self.X_n = [self.X for _ in range(self.eval_depth)]
        self.y_n = [None for _ in range(self.eval_depth)]
        self.known_splits_n = [
            defaultdict(lambda: []) for _ in range(self.eval_depth)
        ]
        self.weights_n = [None for _ in range(self.eval_depth)]
        self.best_n = [None for _ in range(self.eval_depth)]
        self.update(0)

    def update(self, level, counter_example_weight=0):
        if level >= self.eval_depth:
            return
        if level == 0:
            self.y_n[level] = self.y
            p = np.mean(self.y)
            self.weights_n[level] = self.y * (1 - p) + (1 - self.y) * p
        else:
            index, sign = self.best_n[level - 1]
            prediction = self.X_n[level - 1][:, index] if sign \
                else 1 - self.X_n[level - 1][:, index]
            support = (prediction != self.y_n[level - 1]) * \
                self.weights_n[level - 1]
            if np.any(support):
                y = self.adj.dot(
                    self.y_n[level-1] * (
                        1 + support * counter_example_weight
                    )
                )
                mean = np.mean(y)
                std = np.std(y)
                if std == 0:
                    y = np.zeros(len(self.y))
                else:
                    y = (y - mean)/std
                self.y_n[level] = y > 0
                self.weights_n[level] = np.abs(y)
            else:
                self.y_n[level] = np.zeros(len(self.y))
                self.weights_n[level] = np.zeros(len(self.y))
        self.best_n[level] = self.best(
            self.X_n[level], self.y_n[level], self.weights_n[level]
        )
        self.update(level + 1)

    def combine(self, level):
        if len(self.features_n[level]) < 3:
            return
        features = self.features_n[level] + list(chain.from_iterable((
            And(f1, f2), And(f1, Not(f2)),
            And(Not(f1), f2), And(Not(f1), Not(f2))
        ) for f1, f2 in combinations(self.features_n[level][1:], 2)))
        A = self.X_n[level][:, 1:]
        self.X_n[level] = np.concatenate(
            (
                self.X_n[level],
                np.array(
                    list(chain.from_iterable((
                        A[:, i] * A[:, j],
                        A[:, i] * (1 - A[:, j]),
                        (1 - A[:, i]) * A[:, j],
                        (1 - A[:, i]) * (1 - A[:, j])
                    ) for i, j in combinations(range(A.shape[1]), 2))),
                    dtype=bool
                ).transpose()
            ), axis=1
        )
        self.X_n[level], indices = np.unique(
            self.X_n[level], axis=1, return_index=True
        )
        self.features_n[level] = [features[i] for i in indices]
        self.best_n[level] = self.best(
            self.X_n[level], self.y_n[level], self.weights_n[level]
        )
        self.update(level + 1)

    def extend(self, level, only_misses=False, counter_example_weight=0):
        if level >= self.eval_depth:
            return
        X_neighbor = self.adj.dot(self.X_n[level+1])
        new_formulas = []
        new_features = np.zeros(
            (self.X_n[level].shape[0], X_neighbor.shape[1])
        )
        index, sign = self.best_n[level]
        prediction = self.X_n[level][:, index] if sign \
            else 1 - self.X_n[level][:, index]
        support = prediction != self.y_n[level]
        for feature in range(X_neighbor.shape[1]):
            splits = np.unique(X_neighbor[:, feature])
            best_split = 0
            min_loss = self.y.shape[0]
            for split in splits:
                if split in self.known_splits_n[level][feature]:
                    continue
                loss = self.loss(
                    X_neighbor[:, feature].reshape(-1, 1) >= split,
                    self.y_n[level],
                    self.weights_n[level] * (
                        1 + support * counter_example_weight
                    )
                )
                if loss < min_loss:
                    best_split = split
                    min_loss = loss
            free_variable = Var.x if Var.x not in \
                self.features_n[level+1][feature].free_variables() else Var.y
            new_formulas.append(
                GuardedExistsGeq(
                    int(best_split),
                    free_variable,
                    self.features_n[level+1][feature]
                )
            )
            new_features[:, feature] =\
                X_neighbor[:, feature] >= best_split
            self.known_splits_n[level][feature].append(best_split)
        self.X_n[level], indices = np.unique(
            np.concatenate((self.X_n[level], new_features), axis=1),
            axis=1,
            return_index=True
        )
        features = self.features_n[level] + new_formulas
        self.features_n[level] = [features[i] for i in indices]
        self.best_n[level] = self.best(
            self.X_n[level], self.y_n[level], self.weights_n[level]
        )
        self.update(level + 1)

    def reduce(self, level, size):
        loss = self.loss(
            self.X_n[level], self.y_n[level], self.weights_n[level]
        )
        complement = self.loss(
            self.X_n[level], 1 - self.y_n[level], self.weights_n[level]
        )
        indices = np.argsort(np.minimum(loss, complement))[:size]
        self.features_n[level] = [self.features_n[level][i] for i in indices]
        self.X_n[level] = self.X_n[level][:, indices]
        self.update(level + 1)

    def best(self, X, y, weights=None):
        losses = self.loss(X, y, weights)
        complement = self.loss(X, 1-y, weights)
        best = np.argmin(losses)
        worst = np.argmin(complement)
        if losses[best] < complement[worst]:
            return best, True
        else:
            return worst, False

    def loss(self, X, y, weights=None):
        if weights is None:
            return np.sum(X != y.reshape(-1, 1), axis=0)
        else:
            return np.sum(
                (X != y.reshape(-1, 1)) * weights.reshape(-1, 1),
                axis=0
            )

    def accuracy(self):
        index, sign = self.best_n[0]
        prediction = self.X_n[0][:, index] if sign \
            else 1 - self.X_n[0][:, index]
        return np.mean(prediction == self.y)

    def formula(self):
        index, sign = self.best_n[0]
        formula = (self.features_n[0][index] if sign
                   else Not(self.features_n[0][index])).simplify()
        return formula


def _contains_feature(Xs, feature):
    for X in Xs:
        for j in range(X.shape[1]):
            if np.array_equal(feature, X[:, j]) or \
                    np.array_equal(feature, 1 - X[:, j]):
                return False
    return True


def _dt_to_feature_ids(clf: DecisionTreeClassifier):
    """Convert a decision tree to a list of feature ids.
    """
    tree = clf.tree_
    features = []

    def recurse(node_id=0):
        if tree.children_left[node_id] != tree.children_right[node_id]:
            features.append(tree.feature[node_id])
            recurse(tree.children_left[node_id])
            recurse(tree.children_right[node_id])
    recurse()
    return features


def _dt_to_feature_tuples(clf: DecisionTreeClassifier):
    """Convert a decision tree to a list of feature tuples.
    """
    tree = clf.tree_
    formulas = []

    def recurse(node_id=0):
        if tree.children_left[node_id] != tree.children_right[node_id]:
            formulas.append(
                (
                    tree.feature[node_id],
                    tree.threshold[node_id]
                )
            )
            recurse(tree.children_left[node_id])
            recurse(tree.children_right[node_id])
    recurse()
    return formulas


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
