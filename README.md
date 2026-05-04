# Artigo de Compiladores — Uso de LLMs para Otimização de Código Python

**Título:** Uso de Modelos de Linguagem para Otimização de Código Python: Limites e Possibilidades em Linguagens Dinâmicas

**Repositório:** [`ClaudioAMF1/Artigo-Compiladores`](https://github.com/ClaudioAMF1/Artigo-Compiladores)
**Branch de desenvolvimento:** `claude/write-article-iUXd0`

---

## 1. Sumário executivo

Este repositório contém:

1. **O artigo** (`main.tex`) em formato IEEE Conference, em português, descrevendo uma arquitetura híbrida (análise estática via AST + LLM + validação por testes) para otimizar código Python e quantificando empiricamente os limites de cada componente.
2. **A bibliografia** (`references.bib`) com 13 referências, incluindo trabalhos seminais (Cummins/Meta LLM Compiler, Shypula/PIE, Yang/DocAgent) e o paper original do HumanEval.
3. **O pipeline experimental completo** (`experiment/`) — 6 módulos Python (~1400 LoC) implementando ingestão, análise estática, dois back-ends de otimização (regras AST e LLM via Gemini), validação por testes oficiais do HumanEval e benchmark com `timeit`.
4. **Os resultados** (`experiment/results/`) — JSONs com 164 registros pareados para cada back-end, figuras `.pdf` gerados com matplotlib e a tabela LaTeX auto-gerada.

A contribuição empírica é uma **comparação pareada sobre os 164 problemas do HumanEval** entre um otimizador determinístico baseado em regras e o `gemini-2.5-flash-lite`, com testes de significância estatística (t-pareado e bootstrap 95%).

---

## 2. Estrutura do repositório

```
Artigo-Compiladores/
│
├── main.tex                                    # Artigo (IEEE conference, pt-BR)
├── references.bib                              # 13 referências BibTeX
├── README.md                                   # Este arquivo
├── .gitignore                                  # Ignora __pycache__ e auxiliares LaTeX
│
├── META - DocAgent ... .pdf                    # Paper de referência (Meta AI, 2025)
├── Uso de Modelos ... .docx                    # Esboço original do artigo
│
└── experiment/
    ├── data/
    │   └── HumanEval.jsonl.gz                  # Dataset oficial OpenAI, 164 problemas
    │
    ├── optimizer.py                            # Cliente LLM multi-backend (300 LoC)
    ├── rule_optimizer.py                       # Otimizador determinístico via AST (348 LoC)
    ├── bench_utils.py                          # validate() + benchmark() (151 LoC)
    ├── pipeline.py                             # Orquestrador (270 LoC)
    ├── generate_table.py                       # Auto-gera tabela LaTeX a partir do JSON
    ├── generate_figures.py                     # Gera histograma e scatter via matplotlib
    │
    └── results/
        ├── results_rule_based.json             # 164 registros (baseline)
        ├── results_gemini.json                 # 164 registros (LLM)
        ├── tables.tex                          # Tabela LaTeX gerada
        ├── fig_baseline_hist.pdf               # Histograma de speedups (baseline)
        ├── fig_baseline_scatter.pdf            # Scatter t_orig × t_opt (baseline)
        ├── fig_gemini_hist.pdf                 # Histograma de speedups (Gemini)
        └── fig_gemini_scatter.pdf              # Scatter t_orig × t_opt (Gemini)
```

---

## 3. O artigo: estrutura final

| Seção | Conteúdo |
|---|---|
| **I. Introdução** | Pergunta de pesquisa e contribuições (i)–(iv). |
| **II. Fundamentação Teórica** | 5 subseções: linguagens dinâmicas; Python e desempenho; LLMs; otimização de código; limitações do dinamismo. |
| **III. Trabalhos Relacionados** | Cummins/Meta LLM Compiler, Shypula/PIE, FasterPy, CompilerGPT, SysLLMatic e DocAgent. |
| **IV. Metodologia e Arquitetura** | 5 subseções: foundation model, dataset, pipeline em 5 módulos, métricas, exemplo ilustrativo. |
| **V. Resultados Experimentais** | 5 subseções: métricas agregadas (com t-test e bootstrap), detalhamento por regra, análise dos resultados, análise qualitativa (4 *listings* de código), **comparação baseline vs. Gemini** com Tabela IV e Figura 2. |
| **VI. Discussão** | Limites (alucinações com taxa empírica de 6,1%) e possibilidades. |
| **VII. Ameaças à Validade** | Validade interna, construto, externa, variabilidade do modelo, dataset limitado. |
| **VIII. Considerações Finais** | 5 achados empíricos numéricos + 5 trabalhos futuros + parágrafo de contribuições esperadas. |

**Tabelas:** I (métricas agregadas baseline), II (8 casos com regras disparadas), III (frequência de regras), **IV (baseline vs. Gemini, 12 linhas)**.
**Figuras:** **1** (histograma baseline) e **2** (histograma Gemini).
**Listings:** **1–2** (`fib`: original × R1 lru_cache), **3–4** (`total_match`: original × R2 sum builtin), **5–6** (exemplo ilustrativo `soma_quadrados`).

---

## 4. O pipeline experimental

### 4.1 Visão geral

```
┌───────────┐    ┌──────────┐    ┌─────────────┐    ┌──────────┐    ┌───────────┐
│ HumanEval │ →  │ AST     │ →  │ Otimizador  │ →  │ Validade │ →  │ Benchmark │
│  .jsonl   │    │ features│    │ (Reg | LLM) │    │ (testes) │    │ (timeit)  │
└───────────┘    └──────────┘    └─────────────┘    └──────────┘    └───────────┘
```

Cada problema do HumanEval passa por todos os 5 estágios. O *speedup* é

$$ s_i = \frac{t_i^{\text{orig, min}}}{t_i^{\text{opt, min}}} $$

reportando-se o tempo mínimo de **5 repetições × 200 chamadas** por *workload*. O *workload* é construído reaproveitando as próprias chamadas `candidate(...)` do teste oficial do HumanEval, garantindo carga reprodutível e domínio realista.

### 4.2 Módulos (`experiment/`)

#### `optimizer.py`
Cliente LLM multi-provedor unificado:

| Provedor | `LLM_BACKEND` | Modelo padrão | API |
|---|---|---|---|
| OpenAI | `openai` | `gpt-4o` | OpenAI SDK |
| Google | `gemini` | `gemini-2.5-flash` | REST nativa |
| Groq | `groq` | `llama-3.3-70b-versatile` | OpenAI-compat |
| Together | `together` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | OpenAI-compat |

Implementa:
- `extract_features(code, entry_point)`: visitor `ast.NodeVisitor` que extrai 8 *features* (nº de funções, laços, profundidade máxima de aninhamento, chamadas, chamadas recursivas, *comprehensions*, anotações de tipo, *imports*).
- `_call_gemini(...)`: REST POST com **retry exponencial de 7 tentativas** (delays até 60s) para erros HTTP 429/5xx.
- *Prompt template* em português que exige preservação rigorosa da assinatura e da semântica, e resposta apenas como bloco ` ```python ... ``` `.

#### `rule_optimizer.py`
Otimizador determinístico via transformações AST. Implementa 4 regras:

| Regra | Padrão detectado | Transformação |
|---|---|---|
| **R1** `lru_cache` | Função recursiva pura sobre inteiros | `@functools.lru_cache(maxsize=None)` |
| **R2** `sum()` builtin | `acc = 0; for x in iter: acc += expr(x)` | `acc = sum(expr(x) for x in iter)` |
| **R4** `frozenset` lookup | `if x in (a,b,c,...)` | `if x in frozenset({a,b,c,...})` |
| **R5** `''.join()` | `s = ''; for x in iter: s += str(x)` | `s = ''.join(str(x) for x in iter)` |

Retorna `(código_otimizado, RuleReport(applied=[...]))`.

#### `bench_utils.py`
- `validate(code, test, entry_point)`: compila e executa `check(candidate)` do HumanEval sob `signal.SIGALRM` com *timeout* de 10s. Retorna `ValidationResult(passed, error)`.
- `benchmark(...)`: monta *workload* a partir das chamadas `candidate(...)` extraídas do teste oficial, executa `repeats` rodadas de `workload_multiplier` chamadas e reporta `min_time_s`.

#### `pipeline.py`
Orquestrador. Argumentos CLI:

```
--backend {rule_based,gemini,openai,groq,together}
--model    <id-do-modelo>            # opcional
--n        <int>                     # nº de problemas (max 164)
--seed     <int>                     # default 42
--repeats  <int>                     # default 5
--workload <int>                     # default 200
--out      <caminho-do-json>
```

#### `generate_table.py`
Lê o JSON de resultados e emite uma `tabular` IEEE em `.tex` com agregados.

#### `generate_figures.py`
Gera dois PDFs por execução:
- `fig_<prefix>_hist.pdf` — histograma de `log10(speedup)` com linha pontilhada em 1×.
- `fig_<prefix>_scatter.pdf` — scatter `t_orig` vs. `t_opt` em escala log-log com linha de identidade.

---

## 5. Como reproduzir o experimento

### 5.1 Pré-requisitos

```bash
pip install openai requests matplotlib
```

A versão de Python utilizada foi **CPython 3.11** sobre Linux. Resoluções de tempo abaixo de 1 µs são marcadas como *imensuráveis* nas estatísticas.

### 5.2 Rodar o baseline determinístico (sem chave de API)

```bash
python experiment/pipeline.py \
    --backend rule_based \
    --n 164 \
    --seed 42 \
    --repeats 5 \
    --workload 200 \
    --out experiment/results/results_rule_based.json
```

Tempo aproximado: **~3–5 min**.

### 5.3 Rodar o LLM (Gemini)

1. Crie uma chave gratuita em https://aistudio.google.com/apikey (não exige cartão).
2. Verifique no console quais modelos têm cota positiva no seu projeto (a *default* `gemini-2.0-flash` pode estar com cota 0):

   ```bash
   curl -s -X POST \
     "https://generativelanguage.googleapis.com/v1beta/models/<MODELO>:generateContent" \
     -H "Content-Type: application/json" \
     -H "x-goog-api-key: $GEMINI_API_KEY" \
     -d '{"contents":[{"parts":[{"text":"OK"}]}],"generationConfig":{"maxOutputTokens":3}}'
   ```

   Modelos a probar (ordem de preferência): `gemini-flash-latest`, `gemini-flash-lite-latest`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`.

3. Execute:

   ```bash
   export GEMINI_API_KEY='sua-chave'
   python experiment/pipeline.py \
       --backend gemini \
       --model gemini-flash-lite-latest \
       --n 164 --seed 42 --repeats 5 --workload 200 \
       --out experiment/results/results_gemini.json
   ```

   Tempo aproximado: **~25–40 min** (limitado pela cota do *tier* gratuito; o pipeline tolera retries automáticos para 429/5xx).

### 5.4 Gerar figuras

```bash
python experiment/generate_figures.py \
    --input experiment/results/results_rule_based.json \
    --label "baseline AST" --prefix fig_baseline

python experiment/generate_figures.py \
    --input experiment/results/results_gemini.json \
    --label "Gemini 2.5 Flash Lite" --prefix fig_gemini
```

### 5.5 Compilar o artigo

No Overleaf: criar projeto em branco, fazer *upload* de `main.tex` + `references.bib` + a pasta `experiment/results/` (para que `\includegraphics{...fig_baseline_hist.pdf}` resolva). Selecionar **pdfLaTeX** como compilador.

Localmente, com `texlive`:
```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## 6. Resultados experimentais — comparativo

### 6.1 Tabela síntese

| Métrica | **Baseline AST** | **Gemini 2.5 FL** |
|---|---:|---:|
| Preservação semântica | **164/164 (100,0%)** | 154/164 (93,9%) |
| Inválidos (assertivas) | 0 | 10 |
| Melhorias (>1,05×) | 15 (9,1%) | **89 (54,3%)** |
| Neutros [0,95;1,05] | 126 | 12 |
| Regressões (<0,95×) | 14 | 44 |
| Imensuráveis (<1µs) | 9 | 9 |
| **Speedup mediano**\* | 1,003× | **1,138×** |
| **IC 95% bootstrap**\* | [0,997; 1,008] (NS) | **[1,071; 1,220]** ✓ |
| Speedup médio aritm.\* | 5,927× | 7,109× |
| Speedup harmônico\* | 0,986× | 1,064× |
| Speedup máximo\* | 615,4× | 299,3× |
| **Razão melh./reg.** | 1,07 | **2,02** |

\* sobre os problemas com tempo mensurável (sp > 0).

### 6.2 Top transformações algorítmicas descobertas pelo LLM

| Problema | Função | Speedup | Transformação |
|---|---|---:|---|
| HumanEval/59 | `largest_prime_factor` | 299,3× | peneira de Eratóstenes |
| HumanEval/55 | `fib` | 157,2× | memoização (`lru_cache`) |
| HumanEval/63 | `fibfib` | 125,3× | memoização |
| HumanEval/31 | `is_prime` | 106,4× | trial division otimizada |
| HumanEval/150 | `x_or_y` | 105,0× | curto-circuito booleano |
| HumanEval/49 | `modp` | 31,0× | `pow(a, b, mod)` builtin |
| HumanEval/103 | `rounded_avg` | 22,6× | fórmula fechada da soma |

### 6.3 Significância estatística

**Teste t pareado** (tempos original − otimizado, df = 163):

| Back-end | t | mean diff | p (2-tailed) |
|---|---:|---:|---:|
| Baseline AST | 0,479 | 0,173 ms | ≈ 0,63 |
| Gemini 2.5 FL | 1,208 | 1,807 ms | ≈ 0,23 |

**Intervalo de confiança 95% bootstrap** (5000 reamostragens) para mediana do speedup:

- Baseline: **[0,997; 1,008]** — inclui 1,0, **não significativo**.
- Gemini: **[1,071; 1,220]** — não inclui 1,0, **estatisticamente significativo**.

---

## 7. Histórico de commits (essencial)

```
b94bff0  Run Gemini on full HumanEval and integrate empirical results
53128ea  Add qualitative analysis with code examples and bibliography notes
271f0f3  Add statistical tests, figures, and disclosure of unmeasurable cases
009d8cc  Align article with outline and incorporate DocAgent reference
87e76ab  Add experimental results and update article with real data
f750487  Add experimental pipeline using HumanEval public dataset
```

---

## 8. Decisões de projeto e justificativas

### Por que Gemini e não GPT-4o?
Tier gratuito sem cartão de crédito, tempo de resposta baixo, qualidade competitiva em código e API REST simples. O paper declarava `gemini-2.0-flash`, mas a cota gratuita desse modelo no projeto utilizado era 0; a execução real foi com `gemini-flash-lite-latest` (alias para a versão *flash-lite* mais recente), que ofereceu cota suficiente para os 164 problemas. Esse fato é registrado de forma transparente no artigo (Seção IV-A) e na conclusão.

### Por que HumanEval e não PIE/CodeNet?
HumanEval é (i) público e amplamente conhecido, (ii) 100% Python, (iii) traz testes oficiais que servem como oráculo de equivalência semântica, (iv) tem tamanho tratável (164) para *budget* gratuito. Para trabalho futuro o artigo aponta PIE como dataset de exemplos `(lento, rápido)` para *few-shot* via RAG.

### Por que regras AST e não AST + cost model?
Para que o baseline fosse "indefensável simples", servindo como *piso* da comparação. Um cost model derrubaria a R2/R4/R5 nos cinco casos de regressão, mas violaria a premissa: queríamos quantificar o limite das transformações por casamento puro, não construir um otimizador competitivo.

### Por que t-test pareado e bootstrap?
- O **t-test pareado** controla por dificuldade do problema: cada problema é seu próprio controle.
- O **bootstrap não-paramétrico** dá IC para a mediana sem assumir normalidade — apropriado para *speedups* com cauda longa.

### Por que reportar tempo mínimo (não médio)?
Padrão da `timeit`: `min` é menos influenciado por ruído transitório do SO (interrupções, GC, *thermal throttling*). Reflete melhor o tempo intrínseco da computação.

---

## 9. Limitações conhecidas

1. **Bibliografia parcial.** O sandbox de desenvolvimento bloqueia `export.arxiv.org`, então 4 referências (`fasterpy2025`, `sysllmatic2025`, `eco2025`, `survey2025`) trazem nota indicando que a lista completa de autores deve ser confirmada manualmente antes da submissão.
2. **HumanEval é pequeno.** Os achados não generalizam automaticamente para código de produção, que tem classes, dependências externas e *workloads* maiores. O artigo discute isso na Seção VII (Ameaças à Validade).
3. **Modelo único.** Gemini 2.5 Flash Lite é um modelo. A literatura (FasterPy, ECO) sugere que diferentes LLMs têm perfis distintos. Trabalhos futuros devem rodar Llama 3.3 / DeepSeek-Coder / GPT-4o no mesmo *benchmark*.
4. **Sem RAG.** O *prompt* atual é direto. PIE como base de exemplos `(lento → rápido)` em *few-shot* tende a melhorar tanto a taxa de melhoria quanto a taxa de preservação semântica.
5. **9 problemas com `t_opt < 1µs`.** Nesses casos a razão de tempo é indefinida e foram excluídos das estatísticas agregadas. Mantê-los implicaria viés de medição.

---

## 10. Sugestão de busca no Portal CAPES

Termos para o Portal de Periódicos:
- `"large language models" AND "compiler optimization"`
- `"LLM" AND "Python" AND "code optimization"`
- `"code optimization" AND "dynamic language"`
- `"HumanEval" AND "performance"`
- `"Performance Improving Edits"`
- `"multi-agent" AND "code documentation"` (DocAgent)

Filtrar por **2023–2026**.

---

## 11. Licença e atribuição

- O **dataset HumanEval** é distribuído pela OpenAI sob a [MIT License](https://github.com/openai/human-eval/blob/master/LICENSE).
- O **paper DocAgent** (Yang et al., Meta AI, 2025, [arXiv:2504.08725](https://arxiv.org/abs/2504.08725)) está incluído neste repositório como referência metodológica.
- O artigo, o pipeline e a documentação aqui presentes são de uso acadêmico para a disciplina de Compiladores.
