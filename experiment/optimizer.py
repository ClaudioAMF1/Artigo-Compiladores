"""LLM-based optimizer for Python code.

Usa GPT-4o via OpenAI API para sugerir uma versao otimizada de uma
funcao Python, preservando sua assinatura e semantica. Suporta
Retrieval-Augmented Generation trivial baseado em features extraidas
via AST (numero de lacos aninhados, chamadas recursivas, etc.).
"""
from __future__ import annotations

import ast
import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StaticFeatures:
    """Features estaticas extraidas via analise de AST."""

    num_functions: int = 0
    num_loops: int = 0
    max_loop_depth: int = 0
    num_calls: int = 0
    num_recursive_calls: int = 0
    num_comprehensions: int = 0
    has_typing_hints: bool = False
    imports: list[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        return (
            f"- funcoes definidas: {self.num_functions}\n"
            f"- lacos (for/while): {self.num_loops}\n"
            f"- profundidade maxima de aninhamento: {self.max_loop_depth}\n"
            f"- chamadas de funcao: {self.num_calls}\n"
            f"- chamadas recursivas detectadas: {self.num_recursive_calls}\n"
            f"- comprehensions: {self.num_comprehensions}\n"
            f"- anotacoes de tipo presentes: {self.has_typing_hints}\n"
            f"- imports: {', '.join(self.imports) or 'nenhum'}"
        )


def extract_features(code: str, entry_point: str) -> StaticFeatures:
    """Extrai features estaticas via AST."""
    feats = StaticFeatures()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return feats

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.loop_depth = 0

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            feats.num_functions += 1
            if node.returns is not None or any(
                a.annotation is not None for a in node.args.args
            ):
                feats.has_typing_hints = True
            self.generic_visit(node)

        def visit_For(self, node: ast.For) -> None:  # noqa: N802
            feats.num_loops += 1
            self.loop_depth += 1
            feats.max_loop_depth = max(feats.max_loop_depth, self.loop_depth)
            self.generic_visit(node)
            self.loop_depth -= 1

        def visit_While(self, node: ast.While) -> None:  # noqa: N802
            feats.num_loops += 1
            self.loop_depth += 1
            feats.max_loop_depth = max(feats.max_loop_depth, self.loop_depth)
            self.generic_visit(node)
            self.loop_depth -= 1

        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            feats.num_calls += 1
            if isinstance(node.func, ast.Name) and node.func.id == entry_point:
                feats.num_recursive_calls += 1
            self.generic_visit(node)

        def visit_ListComp(self, node: ast.ListComp) -> None:  # noqa: N802
            feats.num_comprehensions += 1
            self.generic_visit(node)

        def visit_DictComp(self, node: ast.DictComp) -> None:  # noqa: N802
            feats.num_comprehensions += 1
            self.generic_visit(node)

        def visit_SetComp(self, node: ast.SetComp) -> None:  # noqa: N802
            feats.num_comprehensions += 1
            self.generic_visit(node)

        def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:  # noqa: N802
            feats.num_comprehensions += 1
            self.generic_visit(node)

        def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
            for n in node.names:
                feats.imports.append(n.name)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
            if node.module:
                feats.imports.append(node.module)

    Visitor().visit(tree)
    return feats


SYSTEM_PROMPT = textwrap.dedent(
    """
    Voce e um assistente especialista em compiladores e otimizacao de
    codigo Python. Dada uma funcao Python, voce deve retornar uma versao
    semanticamente equivalente e mais eficiente (menor tempo de execucao
    e/ou menor uso de memoria). Siga estas regras rigorosamente:

    1. Preserve a assinatura da funcao exatamente (nome, parametros,
       anotacoes de tipo, ordem dos parametros).
    2. Preserve o comportamento observavel para qualquer entrada valida.
    3. NAO remova imports necessarios; adicione apenas os que estritamente
       forem necessarios para a versao otimizada (preferencialmente da
       biblioteca padrao).
    4. Responda APENAS com o codigo Python completo da funcao otimizada
       (e imports necessarios, se houver), dentro de um unico bloco de
       codigo delimitado por crases triplas ```python ... ```.
    5. NAO inclua explicacoes, testes ou exemplos de uso. Apenas o codigo.
    """
).strip()


USER_TEMPLATE = textwrap.dedent(
    """
    Otimize a funcao Python a seguir, preservando sua assinatura e
    semantica. A funcao de entrada (entry point) a ser otimizada se chama
    `{entry_point}`.

    Features estaticas extraidas via AST:
    {features}

    Codigo original:
    ```python
    {code}
    ```
    """
).strip()


def _extract_code(text: str) -> str:
    """Extrai o conteudo do primeiro bloco ```python ... ```."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


class LLMOptimizer:
    """Otimizador baseado em GPT-4o via OpenAI API."""

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.2) -> None:
        self.model = model
        self.temperature = temperature
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import openai

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY nao definida. Exporte a chave antes "
                    "de executar o pipeline."
                )
            self._client = openai.OpenAI(api_key=api_key)
        return self._client

    def optimize(self, code: str, entry_point: str) -> tuple[str, StaticFeatures]:
        """Pede ao LLM uma versao otimizada do codigo.

        Retorna (codigo_otimizado, features_estaticas).
        """
        feats = extract_features(code, entry_point)
        user = USER_TEMPLATE.format(
            entry_point=entry_point,
            features=feats.to_prompt(),
            code=code,
        )
        client = self._get_client()
        resp = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
        )
        raw = resp.choices[0].message.content or ""
        return _extract_code(raw), feats
