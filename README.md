# FTB-Graph: Language-Identity Circuit Discovery via Edge Attribution Patching

Code and results for **First-Token Broadcasters**: identifying and validating the attention-head
circuits ("language identity heads") that determine which language a multilingual LM commits to
at its first generated token.

Pipeline: node-level attribution patching (fast head screen) → Edge Attribution Patching (EAP) →
graph construction → exact-activation-patching verification on the EAP shortlist → three-check
circuit validation (necessity / out-of-graph irrelevance / sufficiency) → DAG visualization.

## Repo layout

```
src/
├── dataset/
│   └── full_suite.json            # 29-item multilingual benchmark (see Dataset below)
├── ftb_graph_notebooks/
│   ├── ftb-graph-pythia-and-aya.ipynb                          # standalone-arm pipeline notebook
│   │                                                             # (currently configured to run
│   │                                                             #  bloom-560m + aya-expanse-8b —
│   │                                                             #  see Known Gaps)
│   ├── ftb_graph_qwen_pair_A/
│   │   └── ftb-graph-qwen-pair-a.ipynb                         # base-to-instruct arm: 1.5B & 7B
│   └── ftb_graph_qwen_pair_B/
│       ├── ftb-graph-qwen2-5-1.5b-base-to-7b-base.ipynb        # scaling arm: base 1.5B → base 7B
│       └── ftb-graph-qwen2-5-1.5b-instruct-to-7b-instruct.ipynb # scaling arm: instruct 1.5B → instruct 7B
├── eap/          # {tag}_node_attr.json, {tag}_eap.json   (Stages 4–5 output, one pair per model)
├── verify/       # {tag}_verify.json                       (Stage 7: exact-patching verification)
├── validation/   # {tag}_validation.json                   (Stage 9: three-check validation)
└── graphs/       # {tag}_dag.png                            (Stage 8: verified circuit DAG)
```

`{tag}` = the HF model name with `/` replaced by `_`, e.g. `Qwen_Qwen2.5-1.5B`, `gpt2`,
`EleutherAI_pythia-2.8b`.

## Models covered

| Model |
|---|:-:|:-:|:-:|:-:|---|
| gpt2 | 
| bigscience/bloom-560m |
| EleutherAI/pythia-1b | 
| EleutherAI/pythia-2.8b |
| Qwen/Qwen2.5-1.5B |
| Qwen/Qwen2.5-1.5B-Instruct |
| Qwen/Qwen2.5-7B |
| Qwen/Qwen2.5-7B-Instruct | 

Eight architectures total. Four have the complete pipeline (standalone arm); the base-to-instruct
and scaling arms are partially complete (see Known Gaps).

## Dataset

`src/dataset/full_suite.json` — 29 hand-authored prompt pairs, built entirely in-notebook (no
external files):
- 21 neutral single-language prompts across 7 languages (en, fr, es, vi, zh, ru, ja)
- 6 code-switched prompts (English → native script mid-sentence)
- 2 semantic congruence probes (French-language prompt, French-topic vs. non-French-topic fact)

Every prompt is a fill-in-the-blank sentence ending right before the language-diagnostic word, so
the first generated token is the thing being scored. Only the `neutral` split is used by the
pipeline's sampling calls downstream (`select_balanced_samples(..., "neutral", ...)`); the
code-switch and semantic-congruence items are built but not yet consumed by `run_pipeline`.

This is a small, hand-authored seed set, not the 200–500-per-language-pair scale the original
project plan called for — noted here since it drives the wide standard deviations in the
validation results.

## Pipeline stages (per model)

1. **Setup** — installs, device check, architecture registry (gpt2 / bloom / qwen / pythia / aya).
2. **Node attribution** (`eap/{tag}_node_attr.json`) — one gradient pass per sample scores every
   attention head's contribution to the language-identity metric.
3. **Edge Attribution Patching** (`eap/{tag}_eap.json`) — scores every candidate
   `(src head → dst layer)` edge in one backward pass.
4. **Graph construction** — top-K EAP edges (K scaled to ~8% of the model's total heads, floor 15 /
   ceiling 60) become the verification shortlist.
5. **Exact-patching verification** (`verify/{tag}_verify.json`) — each shortlisted edge is
   re-checked with real activation patching (not gradient-based) to confirm sign and magnitude.
6. **Verified graph + DAG plot** (`graphs/{tag}_dag.png`) — keeps only edges EAP flagged *and*
   exact patching confirmed.
7. **Three-check validation** (`validation/{tag}_validation.json`) — ablates in-graph heads
   (necessity/completeness) vs. an equal-sized out-of-graph set (irrelevance) and reports the
   drop in the language-identity metric for each.

Output JSON schemas are documented inline in each notebook's corresponding stage; see the pipeline
cells in `ftb-graph-qwen-pair-a.ipynb` for the canonical field names (`completeness_drop_mean`,
`n_in_graph_heads`, etc.) — all notebooks in this repo share the same schema.

## Environment

`requirements.txt` is a full `pip freeze` of the Kaggle/Colab notebook environment (600+ packages),
not a curated dependency list — most of it (fastai, tensorflow, bigquery, etc.) is platform
boilerplate unrelated to this project. The packages that actually matter are: `torch`,
`transformers`, `accelerate`, `bitsandbytes`, `networkx`, `langdetect`, `langid`. Worth trimming
this down to a minimal `requirements.txt` before anyone tries to reproduce outside Kaggle.

Notebooks assume a Kaggle T4×2 GPU runtime and load models in fp32 (recommended for EAP — fp16
gradients are noisy/underflow-prone near zero). Larger models (Qwen2.5-7B pair, aya-expanse-8b)
are loaded in 4-bit via `bitsandbytes` to fit in memory; the 1.5B pair and all standalone models
load in full precision.
