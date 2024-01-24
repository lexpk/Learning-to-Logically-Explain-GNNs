from collections import defaultdict
import numpy as np
from c2 import Atom, Top, Var
from gc2combinator import DecisionTreeCombinator
from gc2reductor import DecisionTreeReductor
from gc2selector import KBestSelector
from helper import best, feature_negations, loss, negations, unique


class GC2Explainer():
    def __init__(
        self,
        combinator=DecisionTreeCombinator(),
        reductor=DecisionTreeReductor(),
        selector=KBestSelector(),
        regularization=lambda x, _: x,
    ):
        self.combinator = combinator
        self.reductor = reductor
        self.selector = selector
        self.regularization = regularization

    def explain(self, adj, X, y, feature_names=None, depth=2):
        self.adj = adj
        self.directed = np.any(adj != adj.T)
        self.average_degree = np.mean(adj.sum(axis=1))
        if feature_names is None:
            feature_names = [f'x{str(i)}' for i in range(X.shape[1])]
        formulas = [Top()] + [
            Atom(feature_name, Var.x) for feature_name in feature_names
        ]
        formulas += list(negations(formulas))
        X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        X = np.concatenate([X, feature_negations(X)], axis=1)
        self.formulas, self.X = unique(
            [],
            np.zeros((X.shape[0], 0)),
            formulas,
            X,
            self.average_degree
        )
        self.y = y
        self._w = {}
        self._y = {}
        self._formulas = defaultdict(lambda: [f for f in self.formulas])
        self._X = defaultdict(lambda: self.X)
        p = y.mean()
        self._w[0, 1] = y * (1 - p) + (1 - y) * p
        self._y[0, 1] = y
        self._formulas[0, 1], self._X[0, 1] = self.selector.select(
            self._formulas[0, 1],
            self._X[0, 1],
            self._y[0, 1],
            self._w[0, 1],
            self.regularization,
            self.average_degree,
            self.adj.shape[0]
        )
        self._formulas[0, 1], self._X[0, 1] = self.combinator.combine(
            self._formulas[0, 1],
            self._X[0, 1],
            self._y[0, 1],
            self._w[0, 1],
        )
        for i in range(1, depth+1):
            self.expand(i)
            self.reduce(i)
        return self.explanation()

    def expand(self, depth):
        self._expand(0, 1, depth)

    def _expand(self, depth, s, max_depth):
        if depth >= max_depth:
            return
        index = best(
            self._X[depth, s],
            self._y[depth, s],
            self._w[depth, s],
            self._formulas[depth, s],
            self.regularization,
            self.average_degree,
            self.adj.shape[0]
        )
        for bit in range(2):
            mask = np.abs(self._X[depth, s][:, index] - bit)
            total = self.adj.dot(mask)
            true = self.adj.dot(mask * self._y[depth, s])
            quot = np.divide(
                true, total, out=np.zeros_like(true), where=total != 0)
            normal = np.divide(
                quot - np.mean(quot),
                np.std(quot),
                out=np.zeros_like(quot),
                where=np.std(quot) != 0
            )
            support = normal != 0
            if np.all(support == 0):
                self._w[depth+1, 2*s+bit] = 0
                continue
            self._y[depth+1, 2*s+bit] = normal > 0
            p = np.mean(self._y[depth+1, 2*s+bit])/np.mean(support)
            self._w[depth+1, 2*s+bit] = self._y[depth+1, 2*s+bit] * (1 - p) + \
                (1 - self._y[depth+1, 2*s+bit]) * p * support
            self._formulas[depth+1, 2*s+bit], self._X[depth+1, 2*s+bit] = \
                self.selector.select(
                    self._formulas[depth, s],
                    self._X[depth, s],
                    self._y[depth+1, 2*s+bit],
                    self._w[depth+1, 2*s+bit],
                    self.regularization,
                    self.average_degree,
                    self.adj.shape[0]
                )
            self._formulas[depth+1, 2*s+bit], self._X[depth+1, 2*s+bit] = \
                self.combinator.combine(
                    self._formulas[depth+1, 2*s+bit],
                    self._X[depth+1, 2*s+bit],
                    self._y[depth+1, 2*s+bit],
                    self._w[depth+1, 2*s+bit],
                    self.average_degree
                )
        if np.any(self._w[depth+1, 2*s] != 0):
            self._expand(depth+1, 2*s, max_depth)
        if np.any(self._w[depth+1, 2*s+1] != 0):
            self._expand(depth+1, 2*s+1, max_depth)

    def reduce(self, depth):
        self._reduce(0, 1, depth)

    def _reduce(self, depth, s, max_depth):
        if depth >= max_depth:
            return
        if np.all(self._w[depth, s] == 0):
            return
        self._reduce(depth+1, 2*s, max_depth)
        self._reduce(depth+1, 2*s+1, max_depth)
        new_formulas, new_X = [None, None], [None, None]
        index = best(
            self._X[depth, s],
            self._y[depth, s],
            self._w[depth, s],
            self._formulas[depth, s],
            self.regularization,
            self.average_degree,
            self.adj.shape[0]
        )
        for bit in range(2):
            mask = np.abs(self._X[depth, s][:, index] - bit)
            new_formulas[bit], new_X[bit] = self.reductor.reduce(
                self._formulas[depth+1, 2*s+bit],
                self.adj.dot(self._X[depth+1, 2*s+bit]),
                self._y[depth, s],
                self._w[depth, s] * mask,
                outgoing=True
            )
            if self.directed:
                reverse = self.reductor.reduce(
                    self._formulas[depth+1, 2*s+bit],
                    self.adj.T.dot(self._X[depth+1, 2*s+bit]),
                    self._y[depth, s],
                    self._w[depth, s] * mask,
                    outgoing=False
                )
                new_formulas[bit] += reverse[0]
                new_X[bit] = np.concatenate([new_X[bit], reverse[1]], axis=1)
            self._formulas[depth, s], self._X[depth, s] = unique(
                new_formulas[bit],
                new_X[bit],
                self._formulas[depth, s],
                self._X[depth, s],
                self.average_degree
            )
        self._formulas[depth, s], self._X[depth, s] =\
            self.combinator.combine(
                self._formulas[depth, s],
                self._X[depth, s],
                self._y[depth, s],
                self._w[depth, s],
                self.average_degree
            )

    def explanation(self):
        index = best(
            self._X[0, 1],
            self._y[0, 1],
            self._w[0, 1],
            self._formulas[0, 1],
            self.regularization,
            self.average_degree,
            self.adj.shape[0]
        )
        return self._formulas[0, 1][index]

    def accuracy(self):
        index = best(
            self._X[0, 1],
            self._y[0, 1],
            self._w[0, 1],
            self._formulas[0, 1],
            self.regularization,
            self.average_degree,
            self.adj.shape[0]
        )
        return np.mean(self._y[0, 1] == self._X[0, 1][:, index])

    def loss(self):
        index = best(
            self._X[0, 1],
            self._y[0, 1],
            self._w[0, 1],
            self._formulas[0, 1],
            self.regularization,
            self.average_degree,
            self.adj.shape[0]
        )
        return loss(
            self._y[0, 1],
            self._X[0, 1][:, index],
            self._w[0, 1],
            self._formulas[0, 1][index],
            self.regularization,
            self.average_degree,
            self.adj.shape[0]
        )
