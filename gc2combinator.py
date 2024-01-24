from sklearn.tree import DecisionTreeClassifier

from helper import feature_combinations, boolean_combinations, unique


class DecisionTreeCombinator():
    def __init__(self, combination_size=2, depth=3):
        assert combination_size <= depth
        self.combination_size = combination_size
        self.dt = DecisionTreeClassifier(max_depth=depth)

    def combine(
        self,
        formulas,
        X,
        y,
        w,
        guarded_weight=1,
        unguarded_weight=1
    ):
        self._fit(X, y, sample_weight=w)
        return unique(
            list(self._new_formulas(formulas)),
            self._new_X(X),
            formulas,
            X,
            guarded_weight=guarded_weight,
            unguarded_weight=unguarded_weight
        )

    def _fit(self, X, y, sample_weight=None):
        self.dt.fit(X, y, sample_weight=sample_weight)
        tree = self.dt.tree_
        self.selected_features = []

        def recurse(node_id=0):
            if tree.children_left[node_id] != \
                    tree.children_right[node_id]:
                self.selected_features.append(tree.feature[node_id])
                recurse(tree.children_left[node_id])
                recurse(tree.children_right[node_id])
        recurse()
        return self.selected_features

    def _new_X(self, X):
        X_selected = X[:, self.selected_features]
        return feature_combinations(X_selected, self.combination_size)

    def _new_formulas(self, formulas):
        selected_formulas = [formulas[i] for i in self.selected_features]
        return boolean_combinations(selected_formulas, self.combination_size)
