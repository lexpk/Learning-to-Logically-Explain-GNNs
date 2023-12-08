from abc import ABC, abstractmethod
from enum import Enum
from networkx import Graph


class Var(Enum):
    x = 0
    y = 1


class Formula(ABC):
    @abstractmethod
    def is_atomic(self) -> bool:
        pass

    @abstractmethod
    def free_variables(self) -> set[Var]:
        pass

    def evaluate(self, graph: Graph) -> bool:
        fv = self.free_variables()
        if len(fv) == 0:
            return self._evaluate(graph, (0, 0))
        elif len(fv) == 1:
            if Var.x in fv:
                return all([
                    self._evaluate(graph, (i, 0))
                    for i in graph.nodes()
                ])
            else:
                return all([
                    self._evaluate(graph, (0, i))
                    for i in graph.nodes()
                ])
        else:
            return all([
                self._evaluate(graph, (i, j))
                for i in graph.nodes()
                for j in graph.nodes()
            ])

    @abstractmethod
    def _evaluate(self, graph: Graph, substititution: (int, int)) -> bool:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class And(Formula):
    def __init__(
        self,
        left: Formula,
        right: Formula
    ) -> None:
        self.left = left
        self.right = right

    def is_atomic(self) -> bool:
        return False

    def free_variables(self) -> set[Var]:
        return self.left.free_variables() | self.right.free_variables()

    def _evaluate(self, graph: Graph, substitution: (int, int)) -> bool:
        return self.left._evaluate(
            graph, substitution
        ) and self.right._evaluate(
            graph, substitution
        )

    def __str__(self) -> str:
        return f"({self.left} ∧ {self.right})"


class Or(Formula):
    def __init__(self, left: Formula, right: Formula) -> None:
        self.left = left
        self.right = right

    def is_atomic(self) -> bool:
        return False

    def free_variables(self) -> set[Var]:
        return self.left.free_variables() | self.right.free_variables()

    def _evaluate(self, graph: Graph, substitution: (int, int)) -> bool:
        return self.left._evaluate(
            graph, substitution
        ) or self.right._evaluate(
            graph, substitution
        )

    def __str__(self) -> str:
        return f"({self.left} ∨ {self.right})"


class Not(Formula):
    def __init__(self, formula: Formula) -> None:
        self.formula = formula

    def is_atomic(self) -> bool:
        return False

    def free_variables(self) -> set[Var]:
        return self.formula.free_variables()

    def _evaluate(self, graph: Graph, substitution: (int, int)) -> bool:
        return not self.formula._evaluate(graph, substitution)

    def __str__(self) -> str:
        return f"¬{self.formula}"


class Implies(Formula):
    def __init__(self, left: Formula, right: Formula) -> None:
        self.left = left
        self.right = right

    def is_atomic(self) -> bool:
        return False

    def free_variables(self) -> set[Var]:
        return self.left.free_variables() | self.right.free_variables()

    def _evaluate(self, graph: Graph, substitution: (int, int)) -> bool:
        return not self.left._evaluate(
            graph, substitution
        ) or self.right._evaluate(
            graph, substitution
        )

    def __str__(self) -> str:
        return f"({self.left} → {self.right})"


class Exists(Formula):
    def __init__(self, var: Var, formula: Formula, count: int = 1) -> None:
        self.var = var
        self.count = count
        self.formula = formula

    def is_atomic(self) -> bool:
        return False

    def free_variables(self) -> set[Var]:
        return self.formula.free_variables() - {self.var}

    def _evaluate(self, graph: Graph, substitution: (int, int)) -> bool:
        if self.var == Var.x:
            return self.count <= len([
                i for i in graph.nodes()
                if self.formula._evaluate(graph, (i, substitution[1]))
            ])
        else:
            return self.count <= len([
                i for i in graph.nodes()
                if self.formula._evaluate(graph, (substitution[0], i))
            ])

    def __str__(self) -> str:
        return f"∃{self.var}^{self.count}.{self.formula}"


class Forall(Formula):
    def __init__(self, var: Var, formula: Formula) -> None:
        self.var = var
        self.formula = formula

    def is_atomic(self) -> bool:
        return False

    def free_variables(self) -> set[Var]:
        return self.formula.free_variables() - {self.var}

    def _evaluate(self, graph: Graph, substitution: (int, int)) -> bool:
        if self.var == Var.x:
            return all([
                self.formula._evaluate(graph, (i, substitution[1]))
                for i in graph.nodes()
            ])
        else:
            return all([
                self.formula._evaluate(graph, (substitution[0], i))
                for i in graph.nodes()
            ])

    def __str__(self) -> str:
        return f"∀{self.var}.{self.formula}"


class E(Formula):
    def __init__(self, v: Var, w: Var) -> None:
        self.v = v
        self.w = w

    def is_atomic(self) -> bool:
        return True

    def free_variables(self) -> set[Var]:
        return {self.v, self.w}

    def _evaluate(self, graph: Graph, substitution: (int, int)) -> bool:
        return graph.has_edge(substitution[0], substitution[1])

    def __str__(self) -> str:
        return f"E({self.v}, {self.w})"


class Var1(Formula):
    def __init__(self) -> None:
        return

    def __str__(self) -> str:
        return "x"


class Var2(Formula):
    def __init__(self) -> None:
        return

    def __str__(self) -> str:
        return "y"
