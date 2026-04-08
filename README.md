# Artigo - Compiladores

Artigo para a disciplina de Compiladores:

**Uso de Modelos de Linguagem para Otimização de Código Python: Limites e Possibilidades em Linguagens Dinâmicas**

## Estrutura

- `main.tex` — arquivo principal do artigo (template IEEE Conference, `IEEEtran`).
- `references.bib` — base bibliográfica em BibTeX.

## Como compilar no Overleaf

1. Crie um projeto em branco no [Overleaf](https://www.overleaf.com/).
2. Faça upload dos arquivos `main.tex` e `references.bib`.
3. Em *Menu* > *Compiler*, selecione **pdfLaTeX**.
4. Em *Menu* > *Main document*, selecione `main.tex`.
5. Compile. O Overleaf já traz a classe `IEEEtran` pré-instalada.

## Escopo do artigo

- **Problema:** quais são os limites da compilação de código Python e quais possibilidades podem ser exploradas dinamicamente para otimizar a linguagem na compilação.
- **Foco:** compiladores (CPython, PyPy, Numba, Cython, mypyc e PEP 659).
- **Foundation model:** GPT-4o (com comparativo previsto com Claude 3.5 Sonnet e DeepSeek-Coder).
- **Dataset público (objeto de pesquisa):** `Project_CodeNet_Python800` (IBM Project CodeNet) + `PIE` (*Performance Improving Edits*).
- **Objetivo:** propor uma arquitetura baseada em LLM que otimize estruturas de compilação de Python a partir de um dataset público estático.

## Seções do artigo

1. Introdução
2. Fundamentação Teórica
3. Trabalhos Relacionados
4. Metodologia e Arquitetura Proposta
5. Discussão: Limites e Possibilidades
6. Considerações Finais

## Sugestão de busca no Portal CAPES

Termos recomendados para consulta no Portal de Periódicos da CAPES:

- `"large language models" AND "compiler optimization"`
- `"LLM" AND "Python" AND "code optimization"`
- `"code optimization" AND "dynamic language"`
- `"Project CodeNet"` e `"Performance Improving Edits"`

Filtrar por anos 2023–2026 para recuperar os trabalhos mais recentes.
