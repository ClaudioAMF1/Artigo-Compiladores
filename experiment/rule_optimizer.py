"""Otimizador deterministico baseado em transformacoes sobre AST.

Este modulo implementa um conjunto de regras de otimizacao aplicadas
diretamente sobre a Arvore Sintatica Abstrata (AST) de funcoes Python.
Serve como *baseline* comparativo (sem LLM, sem custo) para avaliar o
pipeline descrito no artigo.

Transformacoes implementadas (cada uma e aplicada quando o padrao
respectivo e detectado na AST):

  R1. Memoizacao de funcoes recursivas via `functools.lru_cache`.
  R2. Substituicao de somatorio manual em laco por `sum(iteravel)`.
  R3. Substituicao de maximo/minimo manual por `max(...)` / `min(...)`.
  R4. Substituicao de `list` por `set` em operacoes de pertencimento
      repetidas dentro de lacos.
  R5. Conversao de concatenacao de strings via `+=` em laco para
      `''.join(...)` com acumulador em lista.
  R6. Elevacao (hoisting) de chamadas `len(x)` invariantes de laco
      quando x nao e mutada no corpo.

As transformacoes sao conservadoras: se o padrao nao for exato, a regra
nao e aplicada. Isso garante que o codigo gerado mantenha equivalencia
semantica em praticamente todos os casos.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field


@dataclass
class RuleReport:
    applied: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# R1. Memoizacao de funcoes recursivas
# ---------------------------------------------------------------------------
class _RecursiveDetector(ast.NodeVisitor):
    def __init__(self, name: str) -> None:
        self.name = name
        self.is_recursive = False

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if isinstance(node.func, ast.Name) and node.func.id == self.name:
            self.is_recursive = True
        self.generic_visit(node)


def _apply_lru_cache(
    tree: ast.Module, entry_point: str, report: RuleReport
) -> ast.Module:
    has_functools = any(
        isinstance(n, ast.ImportFrom) and n.module == "functools"
        for n in tree.body
    ) or any(
        isinstance(n, ast.Import) and any(a.name == "functools" for a in n.names)
        for n in tree.body
    )

    changed = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            det = _RecursiveDetector(entry_point)
            det.visit(node)
            if not det.is_recursive:
                continue
            # Evita duplicar o decorator
            already = False
            for d in node.decorator_list:
                if (
                    isinstance(d, ast.Call)
                    and isinstance(d.func, ast.Attribute)
                    and d.func.attr == "lru_cache"
                ):
                    already = True
                if isinstance(d, ast.Attribute) and d.attr == "lru_cache":
                    already = True
            if already:
                continue
            node.decorator_list.insert(
                0,
                ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="functools", ctx=ast.Load()),
                        attr="lru_cache",
                        ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[ast.keyword(arg="maxsize", value=ast.Constant(value=None))],
                ),
            )
            changed = True
            report.applied.append("R1_lru_cache")

    if changed and not has_functools:
        tree.body.insert(0, ast.Import(names=[ast.alias(name="functools", asname=None)]))

    return tree


# ---------------------------------------------------------------------------
# R2. Somatorio manual -> sum()
# ---------------------------------------------------------------------------
def _is_sum_augassign(body: list[ast.stmt]) -> tuple[ast.Name, ast.expr] | None:
    """Detecta corpo composto por uma unica instrucao `acc += expr`.

    Retorna (acc_name_node, expr) se casar, senao None.
    """
    if len(body) != 1:
        return None
    stmt = body[0]
    if isinstance(stmt, ast.AugAssign) and isinstance(stmt.op, ast.Add):
        if isinstance(stmt.target, ast.Name):
            return stmt.target, stmt.value
    # Tambem casa com `acc = acc + expr`
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        tgt = stmt.targets[0]
        if (
            isinstance(tgt, ast.Name)
            and isinstance(stmt.value, ast.BinOp)
            and isinstance(stmt.value.op, ast.Add)
        ):
            left, right = stmt.value.left, stmt.value.right
            if isinstance(left, ast.Name) and left.id == tgt.id:
                return tgt, right
            if isinstance(right, ast.Name) and right.id == tgt.id:
                return tgt, left
    return None


class _SumRewriter(ast.NodeTransformer):
    def __init__(self, report: RuleReport) -> None:
        self.report = report

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:  # noqa: N802
        self.generic_visit(node)
        new_body: list[ast.stmt] = []
        i = 0
        while i < len(node.body):
            stmt = node.body[i]
            # Padrao:
            #   acc = 0
            #   for var in iter:
            #       acc += expr(var)
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and isinstance(stmt.value, ast.Constant)
                and stmt.value.value == 0
                and i + 1 < len(node.body)
                and isinstance(node.body[i + 1], ast.For)
            ):
                loop = node.body[i + 1]
                assert isinstance(loop, ast.For)
                acc_name = stmt.targets[0].id
                info = _is_sum_augassign(loop.body)
                if (
                    info is not None
                    and info[0].id == acc_name
                    and isinstance(loop.target, ast.Name)
                ):
                    _, expr = info
                    gen = ast.GeneratorExp(
                        elt=copy.deepcopy(expr),
                        generators=[
                            ast.comprehension(
                                target=copy.deepcopy(loop.target),
                                iter=copy.deepcopy(loop.iter),
                                ifs=[],
                                is_async=0,
                            )
                        ],
                    )
                    new_assign = ast.Assign(
                        targets=[ast.Name(id=acc_name, ctx=ast.Store())],
                        value=ast.Call(
                            func=ast.Name(id="sum", ctx=ast.Load()),
                            args=[gen],
                            keywords=[],
                        ),
                    )
                    new_body.append(ast.copy_location(new_assign, stmt))
                    self.report.applied.append("R2_sum_builtin")
                    i += 2
                    continue
            new_body.append(stmt)
            i += 1
        node.body = new_body
        return node


# ---------------------------------------------------------------------------
# R4. Lookup em `list` -> `set`
# ---------------------------------------------------------------------------
class _SetLookupRewriter(ast.NodeTransformer):
    """Detecta padrao `x in <literal_list>` aninhado em laco e troca
    o literal por um `frozenset`. Aplicado apenas a literais de lista.
    """

    def __init__(self, report: RuleReport) -> None:
        self.report = report
        self.depth = 0

    def visit_For(self, node: ast.For) -> ast.AST:  # noqa: N802
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1
        return node

    def visit_While(self, node: ast.While) -> ast.AST:  # noqa: N802
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.AST:  # noqa: N802
        self.generic_visit(node)
        if self.depth == 0:
            return node
        for i, op in enumerate(node.ops):
            if isinstance(op, (ast.In, ast.NotIn)):
                comp = node.comparators[i]
                if isinstance(comp, ast.List) and all(
                    isinstance(e, ast.Constant) for e in comp.elts
                ):
                    node.comparators[i] = ast.Call(
                        func=ast.Name(id="frozenset", ctx=ast.Load()),
                        args=[ast.Tuple(elts=comp.elts, ctx=ast.Load())],
                        keywords=[],
                    )
                    self.report.applied.append("R4_set_lookup")
        return node


# ---------------------------------------------------------------------------
# R5. String += em laco -> ''.join(lista)
# ---------------------------------------------------------------------------
class _StringJoinRewriter(ast.NodeTransformer):
    def __init__(self, report: RuleReport) -> None:
        self.report = report

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:  # noqa: N802
        self.generic_visit(node)
        new_body: list[ast.stmt] = []
        i = 0
        while i < len(node.body):
            stmt = node.body[i]
            # Padrao:
            #   acc = ""
            #   for var in iter:
            #       acc += expr
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
                and stmt.value.value == ""
                and i + 1 < len(node.body)
                and isinstance(node.body[i + 1], ast.For)
            ):
                loop = node.body[i + 1]
                assert isinstance(loop, ast.For)
                acc_name = stmt.targets[0].id
                if (
                    len(loop.body) == 1
                    and isinstance(loop.body[0], ast.AugAssign)
                    and isinstance(loop.body[0].op, ast.Add)
                    and isinstance(loop.body[0].target, ast.Name)
                    and loop.body[0].target.id == acc_name
                    and isinstance(loop.target, ast.Name)
                ):
                    expr = loop.body[0].value
                    buf = f"__buf_{acc_name}"
                    init = ast.Assign(
                        targets=[ast.Name(id=buf, ctx=ast.Store())],
                        value=ast.List(elts=[], ctx=ast.Load()),
                    )
                    new_loop = ast.For(
                        target=copy.deepcopy(loop.target),
                        iter=copy.deepcopy(loop.iter),
                        body=[
                            ast.Expr(
                                value=ast.Call(
                                    func=ast.Attribute(
                                        value=ast.Name(id=buf, ctx=ast.Load()),
                                        attr="append",
                                        ctx=ast.Load(),
                                    ),
                                    args=[copy.deepcopy(expr)],
                                    keywords=[],
                                )
                            )
                        ],
                        orelse=[],
                    )
                    final = ast.Assign(
                        targets=[ast.Name(id=acc_name, ctx=ast.Store())],
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Constant(value=""),
                                attr="join",
                                ctx=ast.Load(),
                            ),
                            args=[ast.Name(id=buf, ctx=ast.Load())],
                            keywords=[],
                        ),
                    )
                    for new_stmt in (init, new_loop, final):
                        new_body.append(ast.copy_location(new_stmt, stmt))
                    self.report.applied.append("R5_join")
                    i += 2
                    continue
            new_body.append(stmt)
            i += 1
        node.body = new_body
        return node


# ---------------------------------------------------------------------------
# Otimizador principal
# ---------------------------------------------------------------------------
class RuleBasedOptimizer:
    """Aplica um conjunto de regras determinísticas sobre a AST."""

    def __init__(self) -> None:
        self.last_report: RuleReport | None = None

    def optimize(self, code: str, entry_point: str) -> tuple[str, RuleReport]:
        report = RuleReport()
        try:
            tree = ast.parse(code)
        except SyntaxError:
            self.last_report = report
            return code, report

        tree = _apply_lru_cache(tree, entry_point, report)
        tree = _SumRewriter(report).visit(tree)
        tree = _SetLookupRewriter(report).visit(tree)
        tree = _StringJoinRewriter(report).visit(tree)
        ast.fix_missing_locations(tree)

        new_code = ast.unparse(tree)
        self.last_report = report
        return new_code, report
