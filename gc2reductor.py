from math import ceil
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from helper import feature_negations, loss, negations

from c2 import GuardedExistsGeq, GuardedExistsLeq, Var


class BestSplitReductor():
    def __init__(self):
        return

    def reduce(self, formulas, X, y, w=None, outgoing=True):
        self._fit(X, y, w)
        new_formulas = self._new_formulas(formulas, outgoing)
        new_X = self._new_X(X)
        return new_formulas + negations(new_formulas), \
            np.concatenate([new_X, feature_negations(new_X)], axis=1)

    def _fit(self, X, y, w=None):
        self.best_splits = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            splits = np.unique(X[:, i])
            max_loss = y.shape[0]
            for split in splits:
                split_loss = loss(X[:, i] >= split, y, w)
                if split_loss < max_loss:
                    self.best_splits[i] = split
                    max_loss = split_loss
        return self.best_splits

    def _new_X(self, X):
        return np.concatenate([
            X >= self.best_splits,
            X <= self.best_splits
        ], axis=1)

    def _new_formulas(self, formulas, outgoing):
        variables = [
            Var.x if Var.x not in formula.free_variables()
            else Var.y for formula in formulas
        ]
        return [
            GuardedExistsGeq(ceil(split), variable, formula, outgoing)
            for split, variable, formula
            in zip(self.best_splits, variables, formulas)
        ] + [
            GuardedExistsLeq(int(split), variable, formula, outgoing)
            for split, variable, formula
            in zip(self.best_splits, variables, formulas)
        ]


class DecisionTreeReductor():
    def __init__(self, max_depth=3):
        self.depth = max_depth

    def reduce(
        self,
        formulas,
        X_neighbor,
        y,
        w=None,
        outgoing=True,
    ):
        self._fit(X_neighbor, y, w)
        new_formulas = self._new_formulas(formulas, outgoing)
        new_X = self._new_X(X_neighbor)
        return list(new_formulas), new_X

    def _fit(self, X, y, w=None):
        self.dt = DecisionTreeClassifier(max_depth=self.depth)
        self.dt.fit(X, y, sample_weight=w)
        tree = self.dt.tree_
        self.selected_features = []
        self.selected_splits = []

        def recurse(node_id=0):
            if tree.children_left[node_id] != \
                    tree.children_right[node_id]:
                self.selected_features.append(self.dt.tree_.feature[node_id])
                self.selected_splits.append(tree.threshold[node_id])
                recurse(tree.children_left[node_id])
                recurse(tree.children_right[node_id])
        recurse()

    def _new_X(self, X):
        X_selected = X[:, self.selected_features]
        return np.concatenate([
            X_selected >= self.selected_splits,
            X_selected <= self.selected_splits
        ], axis=1)

    def _new_formulas(self, formulas, outgoing):
        variables = [
            Var.x if Var.x not in formulas[i].free_variables()
            else Var.y for i in self.selected_features
        ]
        return [
            GuardedExistsGeq(ceil(split), variable, formulas[i], outgoing)
            for split, variable, i
            in zip(self.selected_splits, variables, self.selected_features)
        ] + [
            GuardedExistsLeq(int(split), variable, formulas[i], outgoing)
            for split, variable, i
            in zip(self.selected_splits, variables, self.selected_features)
        ]
