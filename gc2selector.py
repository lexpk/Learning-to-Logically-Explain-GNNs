

import numpy as np

from helper import loss


class KBestSelector():
    def __init__(self, k=5):
        self.k = k

    def select(
        self,
        formulas,
        X,
        y,
        w=None,
        regularization=lambda x, _: x,
        unguarded_weight=0,
        guarded_weight=0
    ):
        index = self.index(formulas, X, y, w, regularization,
                           unguarded_weight, guarded_weight)
        return [formulas[i] for i in index], X[:, index]

    def index(
        self,
        formulas,
        X,
        y,
        w=None,
        regularization=lambda x, _: x,
        unguarded_weight=0,
        guarded_weight=0
    ):
        scores = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            scores[i] = loss(
                X[:, i],
                y,
                w,
                formulas[i],
                regularization,
                unguarded_weight,
                guarded_weight
            )
        return np.argsort(scores)[:self.k]
