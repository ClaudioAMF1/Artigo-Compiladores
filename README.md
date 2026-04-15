# Artigo - Compiladores

Artigo para a disciplina de Compiladores:

**Uso de Modelos de Linguagem para Otimização de Código Python: Limites e Possibilidades em Linguagens Dinâmicas**

## Estrutura do repositório

```
├── main.tex                      # Artigo IEEE em LaTeX (português)
├── references.bib                # Referências BibTeX
├── experiment/
│   ├── data/
│   │   └── HumanEval.jsonl.gz    # Dataset HumanEval (164 problemas)
│   ├── optimizer.py              # Cliente LLM multi-backend (Gemini, Groq, OpenAI)
│   ├── rule_optimizer.py         # Otimizador determinístico via AST (baseline)
│   ├── bench_utils.py            # Validação e benchmark
│   ├── pipeline.py               # Orquestrador do pipeline
│   ├── generate_table.py         # Gera tabelas LaTeX a partir dos resultados
│   └── results/
│       ├── results_rule_based.json  # Resultados do baseline (164 problemas)
│       └── tables.tex               # Tabelas LaTeX geradas
└── README.md
```

## Como compilar o artigo no Overleaf

1. Crie um projeto em branco no [Overleaf](https://www.overleaf.com/).
2. Faça upload dos arquivos `main.tex` e `references.bib`.
3. Em *Menu* > *Compiler*, selecione **pdfLaTeX**.
4. Em *Menu* > *Main document*, selecione `main.tex`.
5. Compile. O Overleaf já traz a classe `IEEEtran` pré-instalada.

## Escopo do artigo

- **Problema:** quais são os limites da compilação de código Python e quais possibilidades podem ser exploradas dinamicamente para otimizar a linguagem na compilação.
- **Foco:** compiladores (CPython, PyPy, Numba, Cython, mypyc e PEP 659).
- **Foundation model:** Gemini 2.0 Flash (Google, tier gratuito).
- **Dataset público:** HumanEval (OpenAI) — 164 problemas Python com testes unitários.
- **Objetivo:** propor e avaliar uma arquitetura baseada em LLM que otimize código Python.

## Seções do artigo

1. Introdução
2. Fundamentação Teórica
3. Trabalhos Relacionados
4. Metodologia e Arquitetura Proposta
5. **Resultados Experimentais** (com dados reais do baseline)
6. Discussão: Limites e Possibilidades
7. Considerações Finais

## Como rodar o experimento

### Pré-requisitos

```bash
pip install openai requests
```

### 1. Baseline determinístico (sem chave de API)

```bash
python experiment/pipeline.py \
    --backend rule_based \
    --n 164 \
    --seed 42 \
    --out experiment/results/results_rule_based.json
```

### 2. Com Gemini (gratuito)

1. Crie uma chave gratuita em https://aistudio.google.com/apikey (não pede cartão de crédito).
2. Exporte a chave e rode:

```bash
export GEMINI_API_KEY='sua-chave-aqui'

python experiment/pipeline.py \
    --backend gemini \
    --n 164 \
    --seed 42 \
    --out experiment/results/results_gemini.json
```

### 3. Com Groq (gratuito)

1. Crie uma chave gratuita em https://console.groq.com/keys (não pede cartão).
2. Exporte a chave e rode:

```bash
export GROQ_API_KEY='sua-chave-aqui'

python experiment/pipeline.py \
    --backend groq \
    --n 164 \
    --seed 42 \
    --out experiment/results/results_groq.json
```

### 4. Gerar tabelas LaTeX dos resultados

```bash
python experiment/generate_table.py \
    --results experiment/results/results_gemini.json \
    --out experiment/results/tables_gemini.tex
```

## Resultados do baseline (já incluídos no artigo)

| Métrica | Valor |
|---|---|
| Problemas avaliados | 164 |
| Preservação semântica | 164/164 (100.0%) |
| Problemas melhorados (>1.05x) | 15 |
| Problemas neutros | 126 |
| Regressões (<0.95x) | 14 |
| Regras disparadas | 8 |
| Speedup médio | 5.927x |
| Speedup mediano | 1.003x |
| Speedup máximo | 615.378x (fibfib, lru_cache) |

## Sugestão de busca no Portal CAPES

Termos recomendados para consulta no Portal de Periódicos da CAPES:

- `"large language models" AND "compiler optimization"`
- `"LLM" AND "Python" AND "code optimization"`
- `"code optimization" AND "dynamic language"`
- `"HumanEval" AND "performance"`
- `"Performance Improving Edits"`

Filtrar por anos 2023-2026 para recuperar os trabalhos mais recentes.
