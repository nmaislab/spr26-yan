# FedLLM-Guard

Yan Wu · Eyhab Al-Masri · https://github.com/nmaislab/spr26-yan

## Project description

**Problem.** Federated learning lets many clients train a shared model without sharing raw data, but a fraction of clients can poison updates (label flipping, scaled gradients, or updates crafted to look normal). We study whether a server-side guard can catch malicious clients from **update behavior records (UBRs)**—compact stats such as update norm, cosine similarity to the global model, and round-relative cohort deviations.

**Approach.** FedLLM-Guard profiles benign UBRs in a clean run (Task 2), then each round builds a UBR per client and sends it to an **LLM semantic auditor** (Ollama, Qwen2.5-7B, temperature 0) with a structured JSON prompt. The auditor labels each client benign or malicious; malicious clients receive **zero FedAvg weight** (partial repair by rejection). A rule-based **MockAuditor** is also implemented for dry runs without Ollama—it uses the same numeric thresholds but no LLM calls.

**Key finding (this quarter, MNIST, 5 reps, rounds 11–100).** Under the standard cohort-primary guard, the **UBR evasion attack (C)** reaches the highest attack-phase evasion (~14%). The same evasion attack under a **profile-hybrid** guard (D)—LLM on cohort rules 1–3, fixed profile rules 4–5 when the cohort is clean—drops evasion to **0%** while final accuracy stays ~98%. Condition **E** shows that **model-only poisoning without label flip**, still projected with Task-3 UBR evasion, evades less (~4%) than flip+evasion (C).

**What is in this repo.** The committed `data/` and `results/` logs come from **`--auditor ollama`** runs (five repetitions per condition A–E). Mock is supported in code but was not used to produce those reported numbers.

---

## Layout

```
spr26-yan/
├── README.md
├── setup_env.sh
├── src/
│   ├── fedllm_guard.py
│   └── backfill_summaries.py
├── data/              # benign_stats.json, task2 log, condition_* JSONL
├── results/           # summary JSON, all_summaries.json, RESULTS.md, CSV
├── figures/
├── notebooks/
└── artifacts/         # local .pkl from reruns (gitignored)
```

Do not commit MNIST binaries or `.pkl` files.

---

## Dataset access

MNIST is loaded automatically via **flwr-datasets** (Hugging Face cache on first run).

- Official source: http://yann.lecun.com/exdb/mnist/
- Torchvision equivalent: `torchvision.datasets.MNIST(..., download=True)`

---

## Required libraries

Python **3.12**. Tested versions:

| Package | Version |
|---------|---------|
| torch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| flwr | 1.9.0 |
| flwr-datasets | 0.1.0 |
| numpy | 1.26.4 |
| matplotlib | 3.10.9 |
| scipy | 1.17.1 |
| requests | 2.34.2 |
| ray | (with flwr simulation) |

```bash
bash setup_env.sh
conda activate fedllm
```

Manual install:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install "flwr[simulation]==1.9.0" "flwr-datasets[vision]==0.1.0" ray matplotlib scipy requests
```

---

## Environment setup

```bash
bash setup_env.sh
source /opt/miniconda3/bin/activate fedllm   # or: conda activate fedllm
cd src
```

---

## Running Task 2 (profiling)

Builds `data/benign_stats.json` and `data/task2_ubr_profiling.jsonl`. Profiling mode does not call the LLM for labels; `--auditor mock` is enough:

```bash
python fedllm_guard.py --task profile --auditor mock
```

---

## Running conditions A–E

| Cond | Client attack | Server guard |
|------|---------------|--------------|
| A | Label flip 1→7 | Standard LLM |
| B | Flip + 5× update scale | Standard LLM |
| C | Flip + Task-3 UBR evasion | Standard LLM |
| D | **Same as C** | Profile-hybrid (default for D) |
| E | **Model-only** backdoor grad steps (no label flip), then **same UBR projection as C** | Standard LLM |

Warmup: in C/D/E, malicious clients send honest updates for rounds 1–10. Primary metric: **`evasion_rate_aligned`** (rounds 11–100 only).

**Mock auditor** (no Ollama; rule-based thresholds only):

```bash
python fedllm_guard.py --task A --rep 0 --auditor mock
python fedllm_guard.py --task D --rep 0 --auditor mock
python fedllm_guard.py --task all --reps 5 --auditor mock
```

**Ollama** (used for all committed experiment logs):

```bash
ollama pull qwen2.5:7b
ollama serve   # separate terminal

python fedllm_guard.py --task C --rep 0 --auditor ollama
python fedllm_guard.py --task D --rep 0 --auditor ollama
python fedllm_guard.py --task E --rep 0 --auditor ollama
```

Refresh summaries from existing JSONL:

```bash
python backfill_summaries.py
```

---

## Running with Ollama LLM

Install: https://ollama.com/download  

Model: `qwen2.5:7b` · Temperature fixed at **0.0** in code · JSON output format enabled.

---

## Output location

| Output | Path |
|--------|------|
| Benign stats, JSONL logs | `data/` |
| Per-rep summaries, table | `results/` |
| Figures | `figures/` |
| Pickles from new runs | `artifacts/` (gitignored) |

---

## Reproducing key results

**Summaries only** (fast; uses committed logs):

```bash
cd src && python backfill_summaries.py
```

See `results/results_table.csv` and `results/RESULTS.md`.

**Full rerun** (slow; needs GPU + Ollama for LLM conditions):

```bash
cd src
python fedllm_guard.py --task profile --auditor mock
for COND in A B C E; do
  for REP in 0 1 2 3 4; do
    python fedllm_guard.py --task $COND --rep $REP --auditor ollama
  done
done
for REP in 0 1 2 3 4; do
  python fedllm_guard.py --task D --rep $REP --auditor ollama
done
python backfill_summaries.py
```

---
