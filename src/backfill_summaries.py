#!/usr/bin/env python3
"""Refresh summary_*.json from data/condition_*.jsonl; write all_summaries.json + RESULTS.md + CSV."""
import csv
import json
import glob
import os
import re
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
T_WARMUP = 10
NUM_ROUNDS = 100
MAL_PER_ROUND = 3

SUMMARY_FIELDS = (
    "condition", "rep", "num_rounds", "max_round",
    "evasion_rate", "evasion_rate_aligned",
    "n_mal_entries", "n_mal_aligned", "n_ben_entries",
    "final_acc", "final_asr", "final_loss", "kl_norm",
)


def _ordered_summary(d: dict) -> dict:
    out = {k: d[k] for k in SUMMARY_FIELDS if k in d}
    for k, v in d.items():
        if k not in out:
            out[k] = v
    return out


def _rate(entries):
    if not entries:
        return 0.0
    return sum(1 for x in entries if x["audit_label"] == "benign") / len(entries)


def metrics_from_jsonl(path: str) -> dict:
    mal_all, mal_aligned, ben = [], [], []
    max_round = 0
    with open(path) as fh:
        for line in fh:
            e = json.loads(line)
            max_round = max(max_round, e["round"])
            if e["ground_truth"] == "malicious":
                mal_all.append(e)
                if e["round"] > T_WARMUP:
                    mal_aligned.append(e)
            else:
                ben.append(e)

    last = [e for e in mal_all + ben if e["round"] == max_round]
    return {
        "evasion_rate": _rate(mal_all),
        "evasion_rate_aligned": _rate(mal_aligned),
        "n_mal_entries": len(mal_all),
        "n_mal_aligned": len(mal_aligned),
        "n_ben_entries": len(ben),
        "max_round": max_round,
        "num_rounds": max_round,
        "final_acc": last[0]["global_acc"] if last else 0.0,
        "final_asr": last[0]["global_asr"] if last else 0.0,
        "final_loss": last[0]["global_loss"] if last else 0.0,
    }


def is_complete(m: dict) -> bool:
    return (
        m["max_round"] >= NUM_ROUNDS
        and m["n_mal_entries"] >= NUM_ROUNDS * MAL_PER_ROUND - 2
        and m["n_mal_aligned"] >= (NUM_ROUNDS - T_WARMUP) * MAL_PER_ROUND - 2
    )


def parse_summary_name(base: str):
    if "_pilot" in base:
        return None
    m = re.match(r"summary_([A-E])_rep(\d+)\.json$", base)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def write_results_table(by_cond: dict):
    rows = []
    for cond in "ABCDE":
        items = by_cond.get(cond, [])
        if not items:
            continue
        eva = [x["evasion_rate_aligned"] * 100 for x in items]
        acc = [x["final_acc"] * 100 for x in items]
        asr = [x["final_asr"] * 100 for x in items]
        kl = [x["kl_norm"] for x in items]
        rows.append({
            "condition": cond,
            "evasion_rate_aligned_mean_pct": round(st.mean(eva), 2),
            "evasion_rate_aligned_std_pct": round(st.pstdev(eva) if len(eva) > 1 else 0, 2),
            "final_acc_mean_pct": round(st.mean(acc), 2),
            "final_acc_std_pct": round(st.pstdev(acc) if len(acc) > 1 else 0, 2),
            "final_asr_mean_pct": round(st.mean(asr), 3),
            "final_asr_std_pct": round(st.pstdev(asr) if len(asr) > 1 else 0, 3),
            "kl_norm_mean": round(st.mean(kl), 2),
            "kl_norm_std": round(st.pstdev(kl) if len(kl) > 1 else 0, 2),
            "n_reps": len(items),
        })
    csv_path = os.path.join(RESULTS_DIR, "results_table.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return csv_path


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    updated, skipped = [], []
    for spath in sorted(glob.glob(os.path.join(RESULTS_DIR, "summary_*_rep*.json"))):
        base = os.path.basename(spath)
        parsed = parse_summary_name(base)
        if parsed is None:
            skipped.append((base, "pilot or unrecognized name"))
            continue
        cond, rep = parsed
        jpath = os.path.join(DATA_DIR, f"condition_{cond}_rep{rep}.jsonl")
        if not os.path.isfile(jpath):
            skipped.append((base, "no jsonl in data/"))
            continue

        m = metrics_from_jsonl(jpath)
        if not is_complete(m):
            skipped.append((
                base,
                f"incomplete (max_round={m['max_round']}, n_mal={m['n_mal_entries']})",
            ))
            continue

        with open(spath) as fh:
            summary = json.load(fh)
        kl = summary.get("kl_norm")
        summary.update(m)
        if kl is not None:
            summary["kl_norm"] = kl
        summary["condition"] = cond
        summary["rep"] = rep
        for obsolete in (
            "evasion_rate_attack", "n_mal_attack",
            "attack_rounds_note", "metric_note",
        ):
            summary.pop(obsolete, None)
        with open(spath, "w") as fh:
            json.dump(_ordered_summary(summary), fh, indent=2)
        updated.append(summary)

    with open(os.path.join(RESULTS_DIR, "all_summaries.json"), "w") as fh:
        json.dump([_ordered_summary(s) for s in updated], fh, indent=2)

    by_cond = {}
    for s in updated:
        by_cond.setdefault(s["condition"], []).append(s)

    csv_path = write_results_table(by_cond) if by_cond else None

    lines = [
        "# MNIST experiment results",
        "",
        "Primary evasion metric: `evasion_rate_aligned` (rounds 11–100, 270 malicious audits per rep).",
        "",
        "| Cond | Evasion all 100r | Evasion r11–100 | reps |",
        "|------|------------------|-----------------|------|",
    ]
    for cond in "ABCDE":
        items = by_cond.get(cond, [])
        if not items:
            continue
        ev = [x["evasion_rate"] * 100 for x in items]
        eva = [x["evasion_rate_aligned"] * 100 for x in items]
        lines.append(
            f"| {cond} | {st.mean(ev):.2f}±{st.pstdev(ev) if len(ev)>1 else 0:.2f}% | "
            f"{st.mean(eva):.2f}±{st.pstdev(eva) if len(ev)>1 else 0:.2f}% | {len(items)} |"
        )
    if csv_path:
        lines += ["", f"Full table: `{os.path.basename(csv_path)}`", ""]
    if skipped:
        lines += ["## Skipped", ""]
        for name, why in skipped:
            lines.append(f"- {name}: {why}")
        lines.append("")

    md_path = os.path.join(RESULTS_DIR, "RESULTS.md")
    with open(md_path, "w") as fh:
        fh.write("\n".join(lines))

    print(f"Updated {len(updated)} summaries → {RESULTS_DIR}")
    if csv_path:
        print(f"Wrote {csv_path}")
    print(open(md_path).read())


if __name__ == "__main__":
    main()
