from itertools import combinations, product
from math import comb
import numpy as np

from c2 import And, Not, Or


def best(
    X,
    y,
    weights=None,
    formulas=None,
    regularization=lambda x, _: x,
    guarded_weight=0,
    unguarded_weight=0
):
    if formulas is not None:
        complexity = np.array([
            formula.complexity(
                guarded_weight=guarded_weight,
                unguarded_weight=unguarded_weight
            )
            for formula in formulas
        ])
    else:
        complexity = np.zeros(X.shape[1])
    if weights is None:
        losses = np.mean(X != y.reshape(-1, 1), axis=0)
    else:
        losses = np.mean(
            weights.reshape(-1, 1) * (X != y.reshape(-1, 1)), axis=0)
    return min(enumerate(
        regularization(losses, complexity)
    ), key=lambda x: x[1])[0]


def loss(
    y_true,
    y_pred,
    w=None,
    formula=None,
    regularization=lambda x, _: x,
    unguarded_weight=0,
    guarded_weight=0,
):
    if formula is not None:
        complexity = formula.complexity(
            unguarded_weight=unguarded_weight,
            guarded_weight=guarded_weight
        )
    if w is None:
        cost = np.mean(y_true != y_pred)
    else:
        cost = np.mean(w*(y_true != y_pred))
    return regularization(cost, complexity)


def unique(
    new_formulas,
    new_X,
    formulas,
    X,
    guarded_weight=1,
    unguarded_weight=1
):
    seen = {X[:, i].tostring(): i for i in range(X.shape[1])}
    all_formulas = formulas + new_formulas

    def indices():
        for i in range(new_X.shape[1]):
            str = new_X[:, i].tostring()
            if str not in seen:
                seen[str] = X.shape[1] + i
            else:
                j = seen[str]
                if all_formulas[j].complexity(
                    guarded_weight=guarded_weight,
                    unguarded_weight=unguarded_weight
                ) > all_formulas[i].complexity(
                    guarded_weight=guarded_weight,
                    unguarded_weight=unguarded_weight
                ):
                    seen[str] = i
    indices()
    return [
        all_formulas[i] for i in seen.values()
    ], np.concatenate([X, new_X], axis=1)[:, list(seen.values())]


def negations(formulas):
    """
    Given a list of boolean formulas, return a list of all possible
    combinations of these formulas of length `length`.
    """
    for formula in formulas:
        yield Not(formula)


def boolean_combinations(formulas, length):
    """
    Given a list of boolean formulas, return a list of all possible
    conjunctions and disjunctions of these formulas and their negations
    of length `length`.
    """
    for i in range(2, length+1):
        for combination in combinations(formulas, i):
            for negation in product([False, True], repeat=i):
                def negate(x): return Not(x[0]).simplify() if x[0] else x[0]
                yield And(*[negate(x) for x in zip(combination, negation)])
                yield Or(*[negate(x) for x in zip(combination, negation)])


def conjunctions(formulas, length):
    """
    Given a list of boolean formulas, return a list of all possible
    combinations of these formulas of length `length`.
    """
    for i in range(2, length+1):
        for combination in combinations(formulas, i):
            yield And(*combination)


def disjunctions(formulas, length):
    """
    Given a list of boolean formulas, return a list of all possible
    combinations of these formulas of length `length`.
    """
    for i in range(2, length+1):
        for combination in combinations(formulas, length):
            yield Or(*combination)


def feature_combinations(X, length):
    """
    Given a feature matrix, return matrix containing all possible boolean
    combinations of the features of length `length`.
    """
    X_new = np.zeros((
        X.shape[0],
        sum((2**(i+1))*comb(X.shape[1], i) for i in range(2, length+1))
    ))

    def it():
        for i in range(2, length+1):
            for combination in combinations(range(X.shape[1]), i):
                for negation in product([1, 0], repeat=i):
                    yield np.all(
                        X[:, combination] == negation,
                        axis=1
                    )
                    yield np.any(
                        X[:, combination] == negation,
                        axis=1
                    )
    for i, x in enumerate(it()):
        X_new[:, i] = x
    return X_new


def feature_conjunctions(X, length):
    """
    Given a feature matrix, return matrix containing all possible boolean
    combinations of the features of up to length `length`.
    """
    X_new = np.zeros((
            X.shape[0],
            sum(comb(X.shape[1], i) for i in range(2, length+1))
        ))

    def it():
        for i in range(2, length+1):
            for combination in combinations(range(X.shape[1]), i):
                yield np.all(
                    X[:, combination],
                    axis=1
                )
    for i, x in enumerate(it()):
        X_new[:, i] = x
    return X_new


def feature_disjunctions(X, length):
    """
    Given a feature matrix, return matrix containing all possible boolean
    combinations of the features of up to length `length`.
    """
    X_new = np.zeros((
            X.shape[0],
            sum(comb(X.shape[1], i) for i in range(2, length+1))
        ))

    def it():
        for i in range(2, length+1):
            for combination in combinations(range(X.shape[1]), i):
                yield np.any(
                    X[:, combination],
                    axis=1
                )
    for i, x in enumerate(it()):
        X_new[:, i] = x
    return X_new


def feature_negations(X):
    """
    Given a feature matrix, return matrix containing all possible boolean
    combinations of the features of length `length`.
    """
    return 1 - X
