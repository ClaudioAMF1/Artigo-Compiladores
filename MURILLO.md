# Handoff to Murillo

This document is a short, self-contained guide for reproducing the experiments and locating the artifacts that back the paper:

> **Using Language Models for Python Code Optimization: Limits and Possibilities in Dynamic Languages**

Branch: `claude/write-article-iUXd0`.
Main contact: Claudio Meireles / Lucas Fiche (IDP).

---

## 1. What's in the repository

```
Artigo-Compiladores/
├── main.tex                 # Article source (English, IEEE conference, 11 pages)
├── references.bib           # 40 references in DBLP format (zero orphans)
├── CITATIONS.md             # 40-reference verification checklist, numbered [1]-[40]
│                            # to match the IEEE-numeric labels in the compiled PDF
├── README.md                # Original project README
├── MURILLO.md               # This file
├── artigo-overleaf.zip      # Ready-to-upload Overleaf bundle (LaTeX + 5 PDFs)
│
└── experiment/
    ├── data/
    │   └── HumanEval.jsonl.gz       # OpenAI HumanEval (164 problems, MIT license)
    │
    ├── optimizer.py                 # LLM client (multi-provider OpenAI-compatible)
    ├── rule_optimizer.py            # Deterministic AST-rule baseline
    ├── bench_utils.py               # validate() + benchmark()
    ├── pipeline.py                  # Orchestrator (CLI)
    ├── generate_figures.py          # Per-backend histogram + scatter
    ├── generate_comparison.py       # 4-backend overlaid histogram
    ├── generate_table.py            # Aggregate table generator
    │
    └── results/
        ├── results_rule_based.json          # 164 records, AST baseline
        ├── results_gemini.json              # 164 records, Gemini 2.5 Flash Lite
        ├── results_groq_llama.json          # 164 records, Llama 3.3 70B
        ├── results_github_gpt4omini.json    # 164 records, GPT-4o-mini
        ├── fig_baseline_hist.pdf            # Figure 1 in the article
        ├── fig_comparison_hist.pdf          # Figure 2 in the article
        └── fig_{gemini,llama,gpt4omini}_*.pdf
```

---

## 2. How to reproduce the four experiments

### 2.1 Environment

```bash
python3 --version          # 3.11+ recommended (matches what we ran)
pip install openai requests matplotlib
git clone https://github.com/ClaudioAMF1/Artigo-Compiladores.git
cd Artigo-Compiladores
git checkout claude/write-article-iUXd0
```

### 2.2 Run the deterministic baseline (no API key needed)

```bash
python3 experiment/pipeline.py \
        --backend rule_based \
        --n 164 --seed 42 --repeats 5 --workload 200 \
        --out experiment/results/results_rule_based.json
```

Wall-clock: ~3–5 min on a typical laptop. Result is fully deterministic.

### 2.3 Run a Gemini back-end (Google AI Studio, free tier)

1. Create a key at https://aistudio.google.com/apikey (no card required).
2. Pick a model that has positive quota on your project (`gemini-flash-lite-latest` is the most stable on the free tier).

```bash
export GEMINI_API_KEY='AIza-your-key'
python3 experiment/pipeline.py \
        --backend gemini \
        --model gemini-flash-lite-latest \
        --n 164 --seed 42 --repeats 5 --workload 200 \
        --out experiment/results/results_gemini.json
```

Wall-clock: ~25–40 min, dominated by free-tier rate-limit backoff. The retry loop tolerates 429 / 5xx.

### 2.4 Run Llama 3.3 70B via Groq (free tier, no card)

1. Sign in at https://console.groq.com/keys and generate a key.

```bash
export GROQ_API_KEY='gsk_your-key'
python3 experiment/pipeline.py \
        --backend groq \
        --model llama-3.3-70b-versatile \
        --n 164 --seed 42 --repeats 5 --workload 200 \
        --out experiment/results/results_groq_llama.json
```

Wall-clock: ~30 min (30 RPM ceiling).

### 2.5 Run GPT-4o-mini via GitHub Models (free, no card)

1. Open https://github.com/marketplace/models, pick a model, click **Get API Key**.
   This generates a **fine-grained PAT** with scope `Models:Read`.

```bash
export GITHUB_TOKEN='github_pat_your-token'
python3 experiment/pipeline.py \
        --backend github_models \
        --model openai/gpt-4o-mini \
        --n 164 --seed 42 --repeats 5 --workload 200 \
        --out experiment/results/results_github_gpt4omini.json
```

Wall-clock: ~30–60 min, depending on per-model GitHub Models quota.

> **Tip:** the heavy LLM runs can be done in Google Colab (free, no card)
> using the same commands. The Colab cells we used during the original
> experiment are documented in the article-development chat history.

### 2.6 Regenerate the figures from the JSONs

```bash
python3 experiment/generate_figures.py \
        --input experiment/results/results_rule_based.json \
        --label "baseline AST" --prefix fig_baseline
python3 experiment/generate_figures.py \
        --input experiment/results/results_gemini.json \
        --label "Gemini 2.5 Flash Lite" --prefix fig_gemini
python3 experiment/generate_figures.py \
        --input experiment/results/results_groq_llama.json \
        --label "Llama 3.3 70B" --prefix fig_llama
python3 experiment/generate_figures.py \
        --input experiment/results/results_github_gpt4omini.json \
        --label "GPT-4o-mini" --prefix fig_gpt4omini
python3 experiment/generate_comparison.py
```

Output: `.pdf` files in `experiment/results/`.

### 2.7 Recompute the statistics that appear in the article

The article cites paired *t*-test, bootstrap 95% CI for the median speedup,
improvements / regressions / hallucinations counts, etc. The exact recipe
fits in one Python snippet:

```python
import json, statistics, math, random

def stats(path, label):
    r = json.load(open(path))['results']
    n = len(r)
    sps = [x['speedup'] for x in r if x['speedup'] > 0]
    valid = sum(1 for x in r if x['optimized_valid'])
    invalid = sum(1 for x in r if not x['optimized_valid'])
    imp = sum(1 for x in r if x['speedup'] > 1.05)
    reg = sum(1 for x in r if 0 < x['speedup'] < 0.95)
    # Paired t-test on raw times (sec)
    diffs = [x['original_time_s'] - x['optimized_time_s']
             for x in r if x['optimized_valid']]
    nd = len(diffs); md = sum(diffs) / nd
    sd = math.sqrt(sum((x - md) ** 2 for x in diffs) / (nd - 1))
    se = sd / math.sqrt(nd); t = md / se if se > 0 else 0
    # Bootstrap 95% CI for median speedup
    random.seed(0)
    meds = sorted(statistics.median(random.choices(sps, k=len(sps)))
                  for _ in range(5000))
    print(f"== {label} ==")
    print(f"  n={n} valid={valid} invalid={invalid} imp={imp} reg={reg}")
    print(f"  median sp = {statistics.median(sps):.4f}  "
          f"CI95 = [{meds[125]:.4f}; {meds[4874]:.4f}]")
    print(f"  paired t = {t:.3f}  mean diff = {md * 1000:.4f} ms")

for label, path in [
    ('AST baseline', 'experiment/results/results_rule_based.json'),
    ('Gemini 2.5 FL', 'experiment/results/results_gemini.json'),
    ('Llama 3.3 70B', 'experiment/results/results_groq_llama.json'),
    ('GPT-4o-mini',   'experiment/results/results_github_gpt4omini.json'),
]:
    stats(path, label)
```

Expected output (matches Table I and Table IV in the article):

```
== AST baseline ==
  n=164 valid=164 invalid=0 imp=15 reg=14
  median sp = 1.0034  CI95 = [0.9967; 1.0076]
  paired t = 0.479  mean diff = 0.1730 ms
== Gemini 2.5 FL ==
  n=164 valid=154 invalid=10 imp=89 reg=44
  median sp = 1.1381  CI95 = [1.0715; 1.2198]
== Llama 3.3 70B ==
  n=164 valid=126 invalid=38 imp=45 reg=50
  median sp = 0.9757  CI95 = [0.9493; 1.0151]
== GPT-4o-mini ==
  n=164 valid=135 invalid=29 imp=61 reg=48
  median sp = 1.0179  CI95 = [0.9785; 1.1326]
```

---

## 3. How to compile the article

### Overleaf (recommended)

1. Download `artigo-overleaf.zip` from the repo root.
2. https://www.overleaf.com → **New Project → Upload Project** → drop the zip.
3. **Menu → Compiler → pdfLaTeX**, **Main document → `main.tex`**.
4. **Recompile**.

### Local (TeX Live or MiKTeX)

```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## 4. Reference verification

`CITATIONS.md` is the canonical hand-off file for the bibliography.
It lists all 40 references in the exact `[1]`–`[40]` order produced by
`IEEEtran` in the compiled PDF, with a direct arXiv/DOI link plus a
DBLP search link for independent cross-checking.

How to obtain each PDF:
- **arXiv** links resolve to `arxiv.org/abs/...` — free, no login.
- **DOI** links resolve to IEEE Xplore / ACM Digital Library — if paywalled,
  open the same DOI through the **CAPES Portal** while logged in to your
  institution.
- **PEP 659** and the **Silva (2020) UFERSA monograph** are open-access on
  their respective official sites.

---

## 5. Repository access

- **GitHub URL**: https://github.com/ClaudioAMF1/Artigo-Compiladores
- **Branch with the latest work**: `claude/write-article-iUXd0`
- To clone: `git clone -b claude/write-article-iUXd0 https://github.com/ClaudioAMF1/Artigo-Compiladores.git`
- For collaborator access: **Settings → Collaborators → Add people**.

---

## 6. Open questions / future work

The article lists six future-work directions; the three with the highest
expected payoff for a follow-up project are:

1. **RAG with PIE pairs** as few-shot examples — expected to reduce the
   23.2% hallucination rate observed for Llama 70B.
2. **Self-refine loop**: feed validation failures back into the prompt
   for a second attempt.
3. **Generalization to MBPP and Project CodeNet** under the same paired
   protocol, to confirm whether the model-choice-matters finding
   transfers beyond HumanEval.

Each of these reuses the existing pipeline with a small modification to
`pipeline.py`; the JSON output schema does not change.
