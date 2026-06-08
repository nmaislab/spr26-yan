<<<<<<< HEAD

=======
# FedLLM-Guard

Yan Wu · Eyhab Al-Masri · https://github.com/nmaislab/spr26-yan

Flower simulation on MNIST: ten clients, three fixed malicious (30%). Each round the server builds an **update behavior record (UBR)** per client and runs an auditor. Malicious clients get weight zero and do not enter the FedAvg sum.

Standard guard (A/B/C/E): Ollama **qwen2.5:7b** with a cohort-first prompt. Condition **C** adds a UBR evasion attack (~14% bypass in rounds 11–100). Condition **D** repeats the same attack but uses a **profile-hybrid** guard (LLM on cohort rules 1–3, fixed rules on profile when the cohort looks clean) and reaches **0%** attack-phase evasion in our five runs.

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
├── figures/           # plots for the report
├── notebooks/         # optional Jupyter work
└── artifacts/         # local pickles from runs (gitignored)
```

Do not commit MNIST binaries or `.pkl` files. MNIST is fetched automatically on first run.

## Dataset

MNIST via `flwr-datasets` (Hugging Face cache). Official page: http://yann.lecun.com/exdb/mnist/

## Dependencies

Tested on Python 3.12:

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

```bash
bash setup_env.sh
conda activate fedllm
```

Or install manually:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install "flwr[simulation]==1.9.0" "flwr-datasets[vision]==0.1.0" ray matplotlib scipy requests
```

## Commands

From the repo root:

```bash
conda activate fedllm
cd src
```

**Task 2 — benign profiling** (writes `data/benign_stats.json`):

```bash
python fedllm_guard.py --task profile --auditor mock
```

**One condition, mock auditor** (no Ollama):

```bash
python fedllm_guard.py --task A --rep 0 --auditor mock
python fedllm_guard.py --task D --rep 0 --auditor mock   # D defaults to profile-hybrid
```

**Five reps with Ollama** (LLM conditions):

```bash
# install Ollama: https://ollama.com/download
ollama pull qwen2.5:7b
ollama serve   # separate terminal

python fedllm_guard.py --task C --rep 0 --auditor ollama
python fedllm_guard.py --task D --rep 0 --auditor ollama   # profile-hybrid by default
```

Refresh metrics from existing JSONL:

```bash
python backfill_summaries.py
```

Outputs: `results/summary_*_rep*.json`, `results/all_summaries.json`, `results/results_table.csv`.

## Conditions

| | Attack | Guard |
|---|--------|-------|
| A | Label flip 1→7 | Standard LLM |
| B | Flip + 5× scale | Standard LLM |
| C | Flip + UBR evasion | Standard LLM |
| D | Same as C | Profile-hybrid |
| E | Model-only poison + UBR evasion | Standard LLM |

Warmup: rounds 1–10 malicious clients in C/D/E send honest updates. Report evasion uses **rounds 11–100** (`evasion_rate_aligned` in summary JSON).

## Reproduce the numbers in the report

The committed `data/` and `results/` folders already contain the five-rep MNIST runs used in the final report. To recompute summaries only:

```bash
cd src && python backfill_summaries.py
```

To rerun from scratch you need Ollama for LLM conditions and several hours of GPU time; see commands above for each condition.
>>>>>>> c3858ea (MNIST FedLLM-Guard experiments A-E, hybrid defense, logs and results)
