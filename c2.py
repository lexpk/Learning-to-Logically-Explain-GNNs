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

    def evaluate(self, graph: Graph, x: int = None, y: int = None) -> bool:
        if not hasattr(graph, "evaluations"):
            graph.evaluations = {}
        if x is None and y is None:
            return Forall(
                Var.x,
                Forall(
                    Var.y,
                    self
                )
            )._evaluate(graph, Var.x, Var.y)
        elif x is None:
            return Forall(Var.y, self)._evaluate(graph, x, Var.y)
        elif y is None:
            return Forall(Var.x, self)._evaluate(graph, Var.x, y)
        else:
            return self._evaluate(graph, x, y)

    @abstractmethod
    def _evaluate(self, graph: Graph, x, y) -> bool:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


class And(Formula):
    def __init__(self, *conjunts: Formula) -> None:
        self.children = conjunts
        self.hash = hash(str(self))

    def is_atomic(self) -> bool:
        return bool(self.children)

    def free_variables(self) -> set[Var]:
        return set.union(*[c.free_variables() for c in self.children])

    def is_gc2(self) -> bool:
        for c in self.children:
            if not c.is_gc2():
                return False
        else:
            return True

    def _evaluate(self, graph: Graph, x, y) -> bool:
        return all([c._evaluate(graph, x, y) for c in self.children])

    def __repr__(self) -> str:
        if self.children:
            return f"({' ∧ '.join([str(c) for c in self.children])})"
        else:
            return "⊤"

    def __hash__(self) -> int:
        return self.hash


class Or(Formula):
    def __init__(self, *disjuncts: Formula) -> None:
        self.children = disjuncts
        self.hash = hash(str(self))

    def is_atomic(self) -> bool:
        return bool(self.children)

    def free_variables(self) -> set[Var]:
        return set.union(*[c.free_variables() for c in self.children])

    def is_gc2(self) -> bool:
        return all([c.is_gc2() for c in self.children])

    def _evaluate(self, graph: Graph, x, y):
        for c in self.children:
            if c._evaluate(graph, x, y):
                return True
        else:
            return False

    def __repr__(self) -> str:
        if self.children:
            return f"({' ∨ '.join([str(c) for c in self.children])})"
        else:
            return "⊥"

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

    def __repr__(self) -> str:
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

    def __repr__(self) -> str:
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

    def _formula_free_variables(self) -> set[Var]:
        if not hasattr(self, "_formula_free_variables_cache"):
            self._formula_free_variables_cache = self.formula.free_variables()
        return self._formula_free_variables_cache

    def _evaluate(self, graph: Graph, x, y):
        if self.var == Var.x:
            if (self, y) in graph.evaluations:
                return graph.evaluations[self, y]
            if Var.x not in self._formula_free_variables():
                return self.formula._evaluate(graph, x, y)
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
            if Var.y not in self._formula_free_variables():
                return self.formula._evaluate(graph, x, y)
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

    def __repr__(self) -> str:
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

    def _formula_free_variables(self) -> set[Var]:
        if not hasattr(self, "_formula_free_variables_cache"):
            self._formula_free_variables_cache = self.formula.free_variables()
        return self._formula_free_variables_cache

    def _evaluate(self, graph: Graph, x, y):
        if self.var == Var.x:
            if (self, y) in graph.evaluations:
                return graph.evaluations[self, y]
            if Var.x not in self._formula_free_variables():
                return self.formula._evaluate(graph, x, y)
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
            if Var.y not in self._formula_free_variables():
                return self.formula._evaluate(graph, x, y)
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

    def __repr__(self) -> str:
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

    def _formula_free_variables(self) -> set[Var]:
        if not hasattr(self, "_formula_free_variables_cache"):
            self._formula_free_variables_cache = self.formula.free_variables()
        return self._formula_free_variables_cache

    def _evaluate(self, graph: Graph, x, y):
        self.formula.evaluations = graph.evaluations
        if self.var == Var.x:
            if (self, y) in graph.evaluations:
                return graph.evaluations[self, y]
            if Var.x not in self._formula_free_variables():
                return self.formula._evaluate(graph, x, y)
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
            if Var.y not in self._formula_free_variables():
                return self.formula._evaluate(graph, x, y)
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

    def __repr__(self) -> str:
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

    def _formula_free_variables(self) -> set[Var]:
        if not hasattr(self, "_formula_free_variables_cache"):
            self._formula_free_variables_cache = self.formula.free_variables()
        return self._formula_free_variables_cache

    def _evaluate(self, graph: Graph, x, y):
        if self.var == Var.x:
            if (self, y) in graph.evaluations:
                return graph.evaluations[self, y]
            if Var.x not in self._formula_free_variables():
                return self.formula._evaluate(graph, x, y)
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
            if Var.y not in self._formula_free_variables():
                return self.formula._evaluate(graph, x, y)
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

    def __repr__(self) -> str:
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

    def _formula_free_variables(self) -> set[Var]:
        if not hasattr(self, "_formula_free_variables_cache"):
            self._formula_free_variables_cache = self.formula.free_variables()
        return self._formula_free_variables_cache

    def _evaluate(self, graph: Graph, x, y):
        if self.var == Var.x:
            if (self, y) in graph.evaluations:
                return graph.evaluations[self, y]
            if Var.x not in self._formula_free_variables():
                return self.formula._evaluate(graph, x, y)
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
            if Var.y not in self._formula_free_variables():
                return self.formula._evaluate(graph, x, y)
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

    def __repr__(self) -> str:
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

    def __repr__(self) -> str:
        return f"∃≤{self.count}{self.var.name}.(E(x, y) ∧ {self.formula})"


class Exists(Formula):
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
        if self.var == Var.x:
            if (self, y) in graph.evaluations:
                return graph.evaluations[self, y]
            for x in graph.nodes():
                if self.formula._evaluate(graph, x, y):
                    graph.evaluations[self, y] = True
                    return True
            else:
                graph.evaluations[self, y] = False
                return False
        else:
            if (self, x) in graph.evaluations:
                return graph.evaluations[self, x]
            for y in graph.nodes():
                if self.formula._evaluate(graph, x, y):
                    graph.evaluations[self, x] = True
                    return True
            else:
                graph.evaluations[self, x] = False
                return False

    def __repr__(self) -> str:
        return f"∃{self.var.name}.{self.formula}"

    def __hash__(self) -> int:
        return self.hash


class GuardedExists(Formula):
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

    def _formula_free_variables(self) -> set[Var]:
        if not hasattr(self, "_formula_free_variables_cache"):
            self._formula_free_variables_cache = self.formula.free_variables()
        return self._formula_free_variables_cache

    def _evaluate(self, graph: Graph, x, y):
        if self.var == Var.x:
            if (self, y) in graph.evaluations:
                return graph.evaluations[self, y]
            if Var.x not in self._formula_free_variables():
                return self.formula._evaluate(graph, x, y)
            for x in graph.neighbors(y):
                if self.formula._evaluate(graph, x, y):
                    graph.evaluations[self, y] = True
                    return True
            else:
                graph.evaluations[self, y] = False
                return False
        else:
            if (self, x) in graph.evaluations:
                return graph.evaluations[self, x]
            if Var.y not in self._formula_free_variables():
                return self.formula._evaluate(graph, x, y)
            for y in graph.neighbors(x):
                if self.formula._evaluate(graph, x, y):
                    graph.evaluations[self, x] = True
                    return True
            else:
                graph.evaluations[self, x] = False
                return False

    def __repr__(self) -> str:
        return f"∃{self.var.name}.(E(x, y) ∧ {self.formula})"


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

    def _formula_free_variables(self) -> set[Var]:
        if not hasattr(self, "_formula_free_variables_cache"):
            self._formula_free_variables_cache = self.formula.free_variables()
        return self._formula_free_variables_cache

    def _evaluate(self, graph: Graph, x, y):
        self.formula.evaluations = graph.evaluations
        if self.var == Var.x:
            if (self, y) in graph.evaluations:
                return graph.evaluations[self, y]
            if Var.x not in self._formula_free_variables():
                return self.formula._evaluate(graph, x, y)
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
            if Var.y not in self._formula_free_variables():
                return self.formula._evaluate(graph, x, y)
            for y in graph.nodes():
                if not self.formula._evaluate(graph, x, y):
                    graph.evaluations[self, x] = False
                    return False
            else:
                graph.evaluations[self, x] = True
                return True

    def __repr__(self) -> str:
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

    def _formula_free_variables(self) -> set[Var]:
        if not hasattr(self, "_formula_free_variables_cache"):
            self._formula_free_variables_cache = self.formula.free_variables()
        return self._formula_free_variables_cache

    def _evaluate(self, graph: Graph, x, y):
        self.formula.evaluations = graph.evaluations
        if self.var == Var.x:
            if (self, y) in graph.evaluations:
                return graph.evaluations[self, y]
            if Var.x not in self._formula_free_variables():
                return self.formula._evaluate(graph, x, y)
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
            if Var.y not in self._formula_free_variables():
                return self.formula._evaluate(graph, x, y)
            for y in graph.neighbors(x):
                if not self.formula._evaluate(graph, x, y):
                    graph.evaluations[self, x] = False
                    return False
            else:
                graph.evaluations[self, x] = True
                return True

    def __repr__(self) -> str:
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

    def __repr__(self) -> str:
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

    def __repr__(self) -> str:
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

    def __repr__(self) -> str:
        return "⊤"


class Bottom(Formula):
    def __init__(self) -> None:
        self.hash = hash(str(self))

    def is_atomic(self) -> bool:
        return True

    def free_variables(self) -> set[Var]:
        return set()

    def is_gc2(self) -> bool:
        return True

    def _evaluate(self, graph: Graph, x, y):
        return False

    def __repr__(self) -> str:
        return "⊥"
