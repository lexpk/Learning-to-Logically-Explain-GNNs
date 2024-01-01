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

    @abstractmethod
    def is_gc2(self) -> bool:
        pass

    def evaluate(self, graph: Graph) -> bool:
        if not hasattr(graph, "evaluations"):
            graph.evaluations = {}
        fv = self.free_variables()
        if len(fv) == 0:
            return self._evaluate(graph, 0, 0)
        elif len(fv) == 1:
            if Var.x in fv:
                for x in graph.nodes():
                    if not self._evaluate(graph, x, 0):
                        return False
                else:
                    return True
            else:
                for y in graph.nodes():
                    if not self._evaluate(graph, 0, y):
                        return False
                else:
                    return True
        else:
            for x in graph.nodes():
                for y in graph.nodes():
                    if not self._evaluate(graph, x, y):
                        return False
            else:
                return True

    @abstractmethod
    def _evaluate(self, graph: Graph, x, y) -> bool:
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
        self.hash = hash(str(self))

    def is_atomic(self) -> bool:
        return False

    def free_variables(self) -> set[Var]:
        return self.left.free_variables() | self.right.free_variables()

    def is_gc2(self) -> bool:
        return self.left.is_gc2() and self.right.is_gc2()

    def _evaluate(self, graph: Graph, x, y) -> bool:
        return self.left._evaluate(graph, x, y) \
            and self.right._evaluate(graph, x, y)

    def __str__(self) -> str:
        return f"({self.left} ∧ {self.right})"

    def __hash__(self) -> int:
        return self.hash


class Or(Formula):
    def __init__(self, left: Formula, right: Formula) -> None:
        self.left = left
        self.right = right
        self.hash = hash(str(self))

    def is_atomic(self) -> bool:
        return False

    def free_variables(self) -> set[Var]:
        return self.left.free_variables() | self.right.free_variables()

    def is_gc2(self) -> bool:
        return self.left.is_gc2() and self.right.is_gc2()

    def _evaluate(self, graph: Graph, x, y):
        return self.left._evaluate(graph, x, y) \
            or self.right._evaluate(graph, x, y)

    def __str__(self) -> str:
        return f"({self.left} ∨ {self.right})"

    def __hash__(self) -> int:
        return self.hash


class Not(Formula):
    def __init__(self, formula: Formula) -> None:
        self.formula = formula
        self.hash = hash(str(self))

    def is_atomic(self) -> bool:
        return False

    def free_variables(self) -> set[Var]:
        return self.formula.free_variables()

    def is_gc2(self) -> bool:
        return self.formula.is_gc2()

    def _evaluate(self, graph: Graph, x, y):
        return not self.formula._evaluate(graph, x, y)

    def __str__(self) -> str:
        return f"¬{self.formula}"

    def __hash__(self) -> int:
        return self.hash


class Implies(Formula):
    def __init__(self, left: Formula, right: Formula) -> None:
        self.left = left
        self.right = right
        self.hash = hash(str(self))

    def is_atomic(self) -> bool:
        return False

    def free_variables(self) -> set[Var]:
        return self.left.free_variables() | self.right.free_variables()

    def is_gc2(self) -> bool:
        return self.left.is_gc2() and self.right.is_gc2()

    def _evaluate(self, graph: Graph, x, y):
        return not self.left._evaluate(graph, x, y) \
            or self.right._evaluate(graph, x, y)

    def __str__(self) -> str:
        return f"({self.left} → {self.right})"

    def __hash__(self) -> int:
        return self.hash


class ExistsGeq(Formula):
    def __init__(self, count: int, var: Var, formula: Formula) -> None:
        self.var = var
        self.count = count
        self.formula = formula
        self.hash = hash(str(self))

    def is_atomic(self) -> bool:
        return False

    def free_variables(self) -> set[Var]:
        return self.formula.free_variables() - {self.var}

    def is_gc2(self) -> bool:
        return False

    def _evaluate(self, graph: Graph, x, y):
        if self.var == Var.x:
            if (self, y) in graph.evaluations:
                return graph.evaluations[self, y]
            count = 0
            for x in graph.nodes():
                if self.formula._evaluate(graph, x, y):
                    count += 1
                if count >= self.count:
                    graph.evaluations[self, y] = True
                    return True
            else:
                graph.evaluations[self, y] = False
                return False
        else:
            if (self, x) in graph.evaluations:
                return graph.evaluations[self, x]
            count = 0
            for y in graph.nodes():
                if self.formula._evaluate(graph, x, y):
                    count += 1
                if count >= self.count:
                    graph.evaluations[self, x] = True
                    return True
            else:
                graph.evaluations[self, x] = False
                return False

    def __str__(self) -> str:
        return f"∃≥{self.count}{self.var.name}.{self.formula}"

    def __hash__(self) -> int:
        return self.hash


class ExistsEq(Formula):
    def __init__(self, count: int, var: Var, formula: Formula) -> None:
        self.var = var
        self.count = count
        self.formula = formula
        self.hash = hash(str(self))

    def is_atomic(self) -> bool:
        return False

    def free_variables(self) -> set[Var]:
        return self.formula.free_variables() - {self.var}

    def is_gc2(self) -> bool:
        return False

    def _evaluate(self, graph: Graph, x, y):
        if self.var == Var.x:
            if (self, y) in graph.evaluations:
                return graph.evaluations[self, y]
            count = 0
            for x in graph.nodes():
                if self.formula._evaluate(graph, x, y):
                    count += 1
                if count > self.count:
                    graph.evaluations[self, y] = False
                    return False
            else:
                if count < self.count:
                    graph.evaluations[self, y] = False
                    return False
                graph.evaluations[self, y] = True
                return True
        else:
            if (self, x) in graph.evaluations:
                return graph.evaluations[self, x]
            count = 0
            for y in graph.nodes():
                if self.formula._evaluate(graph, x, y):
                    count += 1
                if count > self.count:
                    graph.evaluations[self, x] = False
                    return False
            else:
                if count < self.count:
                    graph.evaluations[self, x] = False
                    return False
                graph.evaluations[self, x] = True
                return True

    def __str__(self) -> str:
        return f"∃={self.count}{self.var.name}.{self.formula}"

    def __hash__(self) -> int:
        return self.hash


class ExistsLeq(Formula):
    def __init__(self, count: int, var: Var, formula: Formula) -> None:
        self.var = var
        self.count = count
        self.formula = formula
        self.hash = hash(str(self))

    def is_atomic(self) -> bool:
        return False

    def free_variables(self) -> set[Var]:
        return self.formula.free_variables() - {self.var}

    def is_gc2(self) -> bool:
        return False

    def _evaluate(self, graph: Graph, x, y):
        self.formula.evaluations = graph.evaluations
        if self.var == Var.x:
            if (self, y) in graph.evaluations:
                return graph.evaluations[self, y]
            count = 0
            for x in graph.nodes():
                if self.formula._evaluate(graph, x, y):
                    count += 1
                if count > self.count:
                    graph.evaluations[self, y] = False
                    return False
            else:
                graph.evaluations[self, y] = True
                return True
        else:
            if (self, x) in graph.evaluations:
                return graph.evaluations[self, x]
            count = 0
            for y in graph.nodes():
                if self.formula._evaluate(graph, x, y):
                    count += 1
                if count > self.count:
                    graph.evaluations[self, x] = False
                    return False
            else:
                graph.evaluations[self, x] = True
                return True

    def __str__(self) -> str:
        return f"∃≤{self.count}{self.var.name}.{self.formula}"

    def __hash__(self) -> int:
        return self.hash


class GuardedExistsGeq(Formula):
    def __init__(self, count: int, var: Var, formula: Formula) -> None:
        self.var = var
        self.count = count
        self.formula = formula
        self.hash = hash(str(self))

    def is_atomic(self) -> bool:
        return False

    def free_variables(self) -> set[Var]:
        return self.formula.free_variables() - {self.var}

    def is_gc2(self) -> bool:
        return self.formula.is_gc2()

    def _evaluate(self, graph: Graph, x, y):
        if self.var == Var.x:
            if (self, y) in graph.evaluations:
                return graph.evaluations[self, y]
            count = 0
            for x in graph.neighbors(y):
                if self.formula._evaluate(graph, x, y):
                    count += 1
                if count >= self.count:
                    graph.evaluations[self, y] = True
                    return True
            else:
                graph.evaluations[self, y] = False
                return False
        else:
            if (self, x) in graph.evaluations:
                return graph.evaluations[self, x]
            count = 0
            for y in graph.neighbors(x):
                if self.formula._evaluate(graph, x, y):
                    count += 1
                if count >= self.count:
                    graph.evaluations[self, x] = True
                    return True
            else:
                graph.evaluations[self, x] = False
                return False

    def __str__(self) -> str:
        return f"∃≥{self.count}{self.var.name}.(E(x, y) ∧ {self.formula})"


class GuardedExistsEq(Formula):
    def __init__(self, count: int, var: Var, formula: Formula) -> None:
        self.var = var
        self.count = count
        self.formula = formula
        self.hash = hash(str(self))

    def is_atomic(self) -> bool:
        return False

    def free_variables(self) -> set[Var]:
        return self.formula.free_variables() - {self.var}

    def is_gc2(self) -> bool:
        return self.formula.is_gc2()

    def _evaluate(self, graph: Graph, x, y):
        if self.var == Var.x:
            if (self, y) in graph.evaluations:
                return graph.evaluations[self, y]
            count = 0
            for x in graph.neighbors(y):
                if self.formula._evaluate(graph, x, y):
                    count += 1
                if count > self.count:
                    graph.evaluations[self, y] = False
                    return False
            else:
                if count < self.count:
                    graph.evaluations[self, y] = False
                    return False
                graph.evaluations[self, y] = True
                return True
        else:
            if (self, x) in graph.evaluations:
                return graph.evaluations[self, x]
            count = 0
            for y in graph.neighbors(x):
                if self.formula._evaluate(graph, x, y):
                    count += 1
                if count > self.count:
                    graph.evaluations[self, x] = False
                    return False
            else:
                if count < self.count:
                    graph.evaluations[self, x] = False
                    return False
                graph.evaluations[self, x] = True
                return True

    def __str__(self) -> str:
        return f"∃={self.count}{self.var.name}.(E(x, y) ∧ {self.formula})"


class GuardedExistsLeq(Formula):
    def __init__(self, count: int, var: Var, formula: Formula) -> None:
        self.var = var
        self.count = count
        self.formula = formula
        self.hash = hash(str(self))

    def is_atomic(self) -> bool:
        return False

    def free_variables(self) -> set[Var]:
        return self.formula.free_variables() - {self.var}

    def is_gc2(self) -> bool:
        return self.formula.is_gc2()

    def _evaluate(self, graph: Graph, x, y):
        self.formula.evaluations = graph.evaluations
        if self.var == Var.x:
            if (self, y) in graph.evaluations:
                return graph.evaluations[self, y]
            count = 0
            for x in graph.neighbors(y):
                if self.formula._evaluate(graph, x, y):
                    count += 1
                if count > self.count:
                    graph.evaluations[self, y] = False
                    return False
            else:
                graph.evaluations[self, y] = True
                return True
        else:
            if (self, x) in graph.evaluations:
                return graph.evaluations[self, x]
            count = 0
            for y in graph.neighbors(x):
                if self.formula._evaluate(graph, x, y):
                    count += 1
                if count > self.count:
                    graph.evaluations[self, x] = False
                    return False
            else:
                graph.evaluations[self, x] = True
                return True

    def __str__(self) -> str:
        return f"∃≤{self.count}{self.var.name}.(E(x, y) ∧ {self.formula})"


class Forall(Formula):
    def __init__(self, var: Var, formula: Formula) -> None:
        self.var = var
        self.formula = formula
        self.hash = hash(str(self))

    def is_atomic(self) -> bool:
        return False

    def free_variables(self) -> set[Var]:
        return self.formula.free_variables() - {self.var}

    def is_gc2(self) -> bool:
        return False

    def _evaluate(self, graph: Graph, x, y):
        self.formula.evaluations = graph.evaluations
        if self.var == Var.x:
            if (self, y) in graph.evaluations:
                return graph.evaluations[self, y]
            else:
                for x in graph.nodes():
                    if not self.formula._evaluate(graph, x, y):
                        graph.evaluations[self, y] = False
                        return False
                else:
                    graph.evaluations[self, y] = True
                    return True
        else:
            if (self, x) in graph.evaluations:
                return graph.evaluations[self, x]
            else:
                for y in graph.nodes():
                    if not self.formula._evaluate(graph, x, y):
                        graph.evaluations[self, x] = False
                        return False
                else:
                    graph.evaluations[self, x] = True
                    return True

    def __str__(self) -> str:
        return f"∀{self.var.name}.{self.formula}"

    def __hash__(self) -> int:
        return self.hash


class GuardedForall(Formula):
    def __init__(self, var: Var, formula: Formula) -> None:
        self.var = var
        self.formula = formula
        self.hash = hash(str(self))

    def is_atomic(self) -> bool:
        return False

    def free_variables(self) -> set[Var]:
        return self.formula.free_variables() - {self.var}

    def is_gc2(self) -> bool:
        return self.formula.is_gc2()

    def _evaluate(self, graph: Graph, x, y):
        self.formula.evaluations = graph.evaluations
        if self.var == Var.x:
            if (self, y) in graph.evaluations:
                return graph.evaluations[self, y]
            else:
                for x in graph.neighbors(y):
                    if not self.formula._evaluate(graph, x, y):
                        graph.evaluations[self, y] = False
                        return False
                else:
                    graph.evaluations[self, y] = True
                    return True
        else:
            if (self, x) in graph.evaluations:
                return graph.evaluations[self, x]
            else:
                for y in graph.neighbors(x):
                    if not self.formula._evaluate(graph, x, y):
                        graph.evaluations[self, x] = False
                        return False
                else:
                    graph.evaluations[self, x] = True
                    return True

    def __str__(self) -> str:
        return f"∀{self.var.name}.(E(x, y) → {self.formula})"


class E(Formula):
    def __init__(self, v: Var, w: Var) -> None:
        self.v = v
        self.w = w
        self.hash = hash("E(x, y)") if v != w else hash("E(x, x)")

    def is_atomic(self) -> bool:
        return True

    def free_variables(self) -> set[Var]:
        return {self.v, self.w}

    def is_gc2(self) -> bool:
        return False

    def _evaluate(self, graph: Graph, x, y):
        if (x, y) in graph.edges():
            return True
        if (y, x) in graph.edges():
            return True
        return False

    def __str__(self) -> str:
        return f"E({self.v.name}, {self.w.name})"

    def __hash__(self) -> int:
        return self.hash


class Atom(Formula):
    def __init__(self, name: str, v: Var) -> None:
        self.name = name
        self.v = v
        self.hash = hash(str(self))

    def is_atomic(self) -> bool:
        return True

    def free_variables(self) -> set[Var]:
        return {self.v}

    def is_gc2(self) -> bool:
        return True

    def _evaluate(self, graph: Graph, x, y):
        if self.v == Var.x:
            return bool(graph.nodes[x][self.name])
        else:
            return bool(graph.nodes[y][self.name])

    def __str__(self) -> str:
        return f"{self.name}({self.v.name})"

    def __hash__(self) -> int:
        return self.hash


class Top(Formula):
    def __init__(self) -> None:
        self.hash = hash(str(self))

    def is_atomic(self) -> bool:
        return True

    def free_variables(self) -> set[Var]:
        return set()

    def is_gc2(self) -> bool:
        return True

    def _evaluate(self, graph: Graph, x, y):
        return True

    def __str__(self) -> str:
        return "⊤"


class Var(Enum):
    x = 0
    y = 1
