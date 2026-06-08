#!/usr/bin/env python3
"""
FedLLM-Guard — MNIST federated learning with LLM-based auditing of Update Behavior Records (UBRs).

Advisor milestones
  Task 2 (profile): 100 clean rounds → μ, σ, per-round stats, logs, plots (`benign_stats.json`).
  Task 3 (evasion):  norm within μ±1.5σ; drift cap in audit; cosine targets empirical benign mean
                     (per-round from Task 2) plus adaptive server feedback (cohort medians + prior UBR).
  Task 4 (logging):  per (round, client): ground truth, full UBR, audit label/confidence/rationale, weight, globals.

Conditions A–E: 10 clients, 3 malicious (30%), 100 rounds. Default 5 reps/condition via `--reps`.
  A/B/C/E — standard guard (T=0): cohort-primary rules 1–4, profile secondary.
  D — same evasion attack as C; profile-hybrid guard (LLM rules 1–3, code rules 4–5 on profile z).
  E — model-only poisoning: honest labels, backdoor gradient step + same UBR evasion as C.

Usage:  python src/fedllm_guard.py --task profile|A|B|C|D|E|all [--reps N] [--rep i] [--rounds N] [--auditor mock|ollama]
"""

import os, json, time, copy, pickle, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import flwr
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.client import ClientApp, NumPyClient
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

# --- constants & paths ---
SEED            = 42
NUM_CLIENTS     = 10
MALICIOUS_IDS   = [0, 1, 2]   # k/N = 30 %
NUM_ROUNDS      = 100
T_WARMUP        = 10            # honest rounds before evasion (temporal / drift constraint)
BATCH_SIZE      = 32
LOCAL_EPOCHS    = 3
LR              = 1e-3
FLIP_FROM, FLIP_TO = 1, 7     # targeted label flip for poisoned clients & ASR metric

# Task 3 envelope: norm μ±1.5σ; drift |z|≤2 in audit; cosine ~0.85 in spec — code uses empirical μ_cos
NORM_SIGMA_MUL  = 1.5
MIN_COSINE_SIM  = 0.85        # literature-style target; evasion uses profile / peer medians at runtime
MAX_DRIFT_Z     = 2.0
RESIDUAL_SCALE  = 0.05        # advisor: residual fraction (reserved / future tightening)
MODEL_POISON_STEPS = 6        # Condition E: SGD steps on backdoor loss (true labels in loader)
MODEL_POISON_LR    = 1e-2     # LR for E backdoor steps (on top of honest local model)
POISON_DELTA_SCALE = 4.0      # amplify (Δ_poison − Δ_honest) before UBR evasion projection

LLM_TEMPERATURE  = 0.0
PROFILE_Z_THRESH   = 3.0        # Condition D: profile z can determine label (rule 4)
PROFILE_SIG_FLOOR_FRAC = 0.5    # floor per-round σ at this × pooled Task-2 σ (avoids ±30σ in early rounds)

ROOT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(ROOT_DIR, "data")
RESULTS_DIR  = os.path.join(ROOT_DIR, "results")
FIG_DIR      = os.path.join(ROOT_DIR, "figures")
ARTIFACT_DIR = os.path.join(ROOT_DIR, "artifacts")
FB_DIR       = os.path.join(DATA_DIR, "client_feedback")
STATS_FILE   = os.path.join(DATA_DIR, "benign_stats.json")

for d in [DATA_DIR, RESULTS_DIR, FIG_DIR, ARTIFACT_DIR, FB_DIR]:
    os.makedirs(d, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def _get_device():
    try:
        if torch.cuda.is_available():
            _ = torch.zeros(1).cuda()
            return "cuda"
    except Exception:
        pass
    return "cpu"

DEVICE = _get_device()


class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1  = nn.Linear(28 * 28, 128)
        self.relu    = nn.ReLU()
        self.layer2  = nn.Linear(128, 10)

    def forward(self, x):
        return self.layer2(self.relu(self.layer1(self.flatten(x))))


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, params):
    keys = list(net.state_dict().keys())
    net.load_state_dict(
        OrderedDict({k: torch.tensor(v) for k, v in zip(keys, params)}),
        strict=True,
    )


def flatten_params(params: list) -> np.ndarray:
    return np.concatenate([p.flatten() for p in params])


_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

_clean_testset    = datasets.MNIST(root=os.path.expanduser("~/data/yanwu_thesis"),
                                   train=False, download=True, transform=_TRANSFORM)
_server_testloader = DataLoader(_clean_testset, batch_size=1000, shuffle=False)


def load_partition(partition_id: int, fds: FederatedDataset,
                   flip_from: int = None, flip_to: int = None,
                   poison_ratio: float = 1.0):
    """One IID partition; optional label flip (poison_ratio) for train set only."""
    partition = fds.load_partition(partition_id, "train")
    split     = partition.train_test_split(test_size=0.2, seed=SEED)
    train_ds  = split["train"]

    if flip_from is not None:
        def _flip(ex):
            if ex["label"] == flip_from and random.random() < poison_ratio:
                ex["label"] = flip_to
            return ex
        train_ds = train_ds.map(_flip, desc=f"Poison partition {partition_id}")

    def _transform(batch):
        batch["image"] = [_TRANSFORM(img) for img in batch["image"]]
        return batch

    train_ds = train_ds.with_transform(_transform)
    test_ds  = split["test"].with_transform(_transform)
    return (DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False))


def backdoor_gradient_step(
    net: nn.Module, loader: DataLoader,
    source_class: int = FLIP_FROM, target_class: int = FLIP_TO,
    steps: int = MODEL_POISON_STEPS, lr: float = MODEL_POISON_LR,
) -> bool:
    """One or more optimizer steps: CE(f(x), target) only where true label == source (data unchanged)."""
    criterion = nn.CrossEntropyLoss()
    net.train()
    any_step = False
    for _ in range(steps):
        for batch in loader:
            imgs   = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            mask   = labels == source_class
            if mask.sum().item() == 0:
                continue
            imgs_m = imgs[mask]
            targets = torch.full(
                (imgs_m.size(0),), target_class, device=DEVICE, dtype=torch.long,
            )
            net.zero_grad()
            loss = criterion(net(imgs_m), targets)
            loss.backward()
            with torch.no_grad():
                for p in net.parameters():
                    if p.grad is not None:
                        p.add_(p.grad, alpha=-lr)
            any_step = True
    return any_step


def train_model(net: nn.Module, loader: DataLoader, epochs: int = LOCAL_EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    net.train()
    for _ in range(epochs):
        for batch in loader:
            imgs   = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(imgs), labels)
            loss.backward()
            optimizer.step()


def evaluate_model(net: nn.Module, loader: DataLoader):
    criterion = nn.CrossEntropyLoss()
    net.eval(); correct = total = 0; total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            imgs   = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            out    = net(imgs)
            total_loss += criterion(out, labels).item()
            correct    += (out.argmax(1) == labels).sum().item()
            total      += len(labels)
    return correct / total, total_loss / len(loader)


def compute_ubr(update_flat: np.ndarray,
                reference_flat: np.ndarray,
                norm_history: list) -> dict:
    """Four advisor UBR fields: norm, cosine to reference, drift z on norm history, sign vs reference."""
    norm    = float(np.linalg.norm(update_flat))
    ref_nrm = float(np.linalg.norm(reference_flat))

    if norm > 1e-9 and ref_nrm > 1e-9:
        cosine_sim = float(np.dot(update_flat, reference_flat) / (norm * ref_nrm))
    else:
        cosine_sim = 0.0

    if len(norm_history) >= 3:
        hist = np.array(norm_history[-20:])
        drift_zscore = float((norm - hist.mean()) / (hist.std() + 1e-8))
    else:
        drift_zscore = 0.0

    nz = reference_flat != 0
    if nz.sum() > 0:
        sign_cons = float((np.sign(update_flat[nz]) == np.sign(reference_flat[nz])).mean())
    else:
        sign_cons = 1.0

    return {
        "norm":             norm,
        "cosine_sim":       cosine_sim,
        "drift_zscore":     drift_zscore,
        "sign_consistency": sign_cons,
    }


def compute_benign_stats(ubr_records: list) -> dict:
    """Task 2: pooled μ,σ per UBR field; optional per_round μ,σ for norm/cosine (training dynamics)."""
    from collections import defaultdict
    fields = ["norm", "cosine_sim", "drift_zscore", "sign_consistency"]
    stats  = {}
    for f in fields:
        vals = np.array([r[f] for r in ubr_records])
        stats[f"mu_{f}"]    = float(vals.mean())
        stats[f"sigma_{f}"] = float(vals.std())

    if ubr_records and "_round" in ubr_records[0]:
        round_cos  = defaultdict(list)
        round_norm = defaultdict(list)
        for r in ubr_records:
            rnd = r["_round"]
            round_cos[rnd].append(r["cosine_sim"])
            round_norm[rnd].append(r["norm"])
        per_round = {}
        for rnd in sorted(round_cos):
            c = np.array(round_cos[rnd])
            n = np.array(round_norm[rnd])
            per_round[str(rnd)] = {
                "mu_cosine":  float(c.mean()),  "sig_cosine": float(c.std()),
                "mu_norm":    float(n.mean()),  "sig_norm":   float(n.std()),
            }
        stats["per_round"] = per_round

    return stats


# --- auditors ---
def _profile_z_scores(
    ubr: dict, server_round: int, stats: dict, *, hardened: bool = False,
) -> Tuple[float, float]:
    """Task-2 per-round envelope; hardened D uses a σ floor vs pooled stats."""
    pr = stats.get("per_round", {}).get(str(server_round), {})
    pooled_sig_cos = float(stats.get("sigma_cosine_sim", 0.1) or 1e-4)
    pooled_sig_n   = float(stats.get("sigma_norm", 1.0) or 1e-4)
    mu_cos  = pr.get("mu_cosine",  stats.get("mu_cosine_sim",     0.5))
    mu_n    = pr.get("mu_norm",    stats.get("mu_norm",            0.0))
    if hardened:
        sig_cos = max(float(pr.get("sig_cosine", pooled_sig_cos) or pooled_sig_cos),
                      pooled_sig_cos * PROFILE_SIG_FLOOR_FRAC)
        sig_n   = max(float(pr.get("sig_norm", pooled_sig_n) or pooled_sig_n),
                      pooled_sig_n * PROFILE_SIG_FLOOR_FRAC)
    else:
        sig_cos = float(pr.get("sig_cosine", pooled_sig_cos) or pooled_sig_cos) or 1e-4
        sig_n   = float(pr.get("sig_norm", pooled_sig_n) or pooled_sig_n) or 1e-4
    return (
        float((ubr["cosine_sim"] - mu_cos) / sig_cos),
        float((ubr["norm"] - mu_n) / sig_n),
    )


def _cohort_primary_tags(ubr: dict) -> Tuple[int, bool, float, float]:
    cos_dev = float(ubr.get("cos_dev_round", 0.0))
    nrm_dev = float(ubr.get("norm_dev_round", 0.0))
    cos_anom = abs(cos_dev) >= 1.5
    nrm_anom = abs(nrm_dev) >= 1.5
    severe = abs(cos_dev) >= 3.0 or abs(nrm_dev) >= 3.0
    return int(cos_anom) + int(nrm_anom), severe, cos_dev, nrm_dev


def _decide_profile_hardened_label(
    ubr: dict, server_round: int, stats: dict,
) -> dict:
    """Deterministic D policy (rules 1–5); LLM rationale is optional decoration only."""
    n_anom, severe, cos_dev, nrm_dev = _cohort_primary_tags(ubr)
    pcz, pnz = _profile_z_scores(ubr, server_round, stats, hardened=True)
    n_prof = int(abs(pcz) >= PROFILE_Z_THRESH) + int(abs(pnz) >= PROFILE_Z_THRESH)

    if severe:
        rule, conf = 1, 0.95
    elif n_anom >= 2:
        rule, conf = 2, 0.90
    elif n_anom == 1:
        rule, conf = 3, 0.85
    elif n_prof >= 1:
        rule, conf = 4, 0.90
    else:
        rule, conf = 5, 0.85

    label = "malicious" if rule <= 4 else "benign"
    rationale = (
        f"cohort: cos_dev={cos_dev:+.2f}σ norm_dev={nrm_dev:+.2f}σ "
        f"primary_anomaly_count={n_anom}/2 severe={severe}; "
        f"profile: profile_cos_z={pcz:+.2f}σ profile_norm_z={pnz:+.2f}σ "
        f"profile_anomaly_count={n_prof}/2; "
        f"Decision: rule {rule} (deterministic profile-hardened policy)."
    )
    return {"label": label, "confidence": conf, "rationale": rationale, "rule": rule}


def _decide_profile_primary_zero_label(
    ubr: dict, server_round: int, stats: dict,
) -> dict:
    """Hybrid D: rules 4–5 only when cohort-normal (primary_anomaly_count == 0)."""
    n_anom, severe, cos_dev, nrm_dev = _cohort_primary_tags(ubr)
    pcz, pnz = _profile_z_scores(ubr, server_round, stats, hardened=True)
    n_prof = int(abs(pcz) >= PROFILE_Z_THRESH) + int(abs(pnz) >= PROFILE_Z_THRESH)
    if n_prof >= 1:
        rule, conf, label = 4, 0.90, "malicious"
    else:
        rule, conf, label = 5, 0.85, "benign"
    rationale = (
        f"cohort: cos_dev={cos_dev:+.2f}σ norm_dev={nrm_dev:+.2f}σ "
        f"primary_anomaly_count={n_anom}/2 severe={severe}; "
        f"profile: profile_cos_z={pcz:+.2f}σ profile_norm_z={pnz:+.2f}σ "
        f"profile_anomaly_count={n_prof}/2; "
        f"Decision: rule {rule} (hybrid profile policy, cohort-normal)."
    )
    return {"label": label, "confidence": conf, "rationale": rationale, "rule": rule}


class MockAuditor:
    """Rule-based auditor (no LLM). Standard A/B/C; optional profile-hardened rules for D."""
    def __init__(self, benign_stats: dict = None, profile_hardened: bool = False):
        self.stats = benign_stats or {}
        self.profile_hardened = profile_hardened

    def audit(self, ubr: dict, server_round: int = 0,
              temperature: float = 0.0) -> dict:
        if self.profile_hardened:
            return _decide_profile_hardened_label(ubr, server_round, self.stats)
        n_anom, severe, cos_dev, nrm_dev = _cohort_primary_tags(ubr)
        if severe or n_anom >= 2:
            return {"label":"malicious","confidence":0.90,
                    "rationale":(f"cos_dev={cos_dev:+.2f}σ norm_dev={nrm_dev:+.2f}σ "
                                 "outside cohort consensus.")}
        if n_anom == 1:
            return {"label":"malicious","confidence":0.75,
                    "rationale":(f"One primary deviates from cohort: "
                                 f"cos_dev={cos_dev:+.2f}σ norm_dev={nrm_dev:+.2f}σ.")}
        return {"label":"benign","confidence":0.85,
                "rationale":(f"Within cohort consensus: "
                             f"cos_dev={cos_dev:+.2f}σ norm_dev={nrm_dev:+.2f}σ.")}


class OllamaAuditor:
    """Ollama JSON auditor. D: profile_hardened | profile_hybrid (LLM cohort + code rule 4/5)."""

    def __init__(self, model: str = "qwen2.5:7b",
                 endpoint: str = "http://localhost:11434/api/generate",
                 benign_stats: dict = None,
                 audit_mode: str = "standard",
                 strict_json: bool = False):
        self.model       = model
        self.endpoint    = endpoint
        self.stats       = benign_stats or {}
        self.audit_mode  = audit_mode
        self.strict_json = strict_json

    def audit(self, ubr: dict, server_round: int = 0,
              temperature: float = 0.0) -> dict:
        if self.audit_mode == "profile_hybrid":
            n_anom, severe, _, _ = _cohort_primary_tags(ubr)
            if not severe and n_anom == 0:
                return _decide_profile_primary_zero_label(
                    ubr, server_round, self.stats,
                )
        return self._audit_ollama(ubr, server_round, temperature)

    def _audit_ollama(self, ubr: dict, server_round: int,
                      temperature: float) -> dict:
        import requests
        prompt_mode = self.audit_mode
        if prompt_mode == "profile_hybrid":
            prompt_mode = "standard"
        cos_dev_r = float(ubr.get("cos_dev_round",  0.0))
        nrm_dev_r = float(ubr.get("norm_dev_round", 0.0))
        cos_anom  = abs(cos_dev_r) >= 1.5
        nrm_anom  = abs(nrm_dev_r) >= 1.5
        severe    = abs(cos_dev_r) >= 3.0 or abs(nrm_dev_r) >= 3.0
        n_anom    = int(cos_anom) + int(nrm_anom)

        def _flag(d, name):
            if abs(d) >= 3.0: return f"{name} = {d:+.2f}σ  ⇒  ANOMALOUS (severe)"
            if abs(d) >= 1.5: return f"{name} = {d:+.2f}σ  ⇒  ANOMALOUS"
            return f"{name} = {d:+.2f}σ  ⇒  within cohort"
        cos_flag = _flag(cos_dev_r, "cos_dev_round")
        nrm_flag = _flag(nrm_dev_r, "norm_dev_round")

        prof_cos_z, prof_n_z = _profile_z_scores(
            ubr, server_round, self.stats,
            hardened=(prompt_mode == "profile_hardened"),
        )
        n_prof_anom = int(abs(prof_cos_z) >= PROFILE_Z_THRESH) + int(abs(prof_n_z) >= PROFILE_Z_THRESH)
        drift_anom = abs(ubr["drift_zscore"]) > MAX_DRIFT_Z

        def _pflag(z, name):
            if abs(z) >= PROFILE_Z_THRESH:
                return f"{name} = {z:+.2f}σ  ⇒  PROFILE ANOMALOUS"
            return f"{name} = {z:+.2f}σ  ⇒  within profile"
        prof_cos_flag = _pflag(prof_cos_z, "profile_cos_z")
        prof_n_flag   = _pflag(prof_n_z,   "profile_norm_z")

        if prompt_mode == "profile_hardened":
            profile_block = (
                "PROFILE-BASED signals (Task 2 envelope — CAN determine label via rule 4):\n"
                f"  {prof_cos_flag}\n"
                f"  {prof_n_flag}\n"
                f"  profile_anomaly_count = {n_prof_anom}/2\n\n"
            )
            policy = (
                "Decision policy (apply STRICTLY; pick FIRST matching rule):\n"
                "  rule 1: severe == True   → malicious\n"
                "  rule 2: primary_anomaly_count >= 2  → malicious\n"
                "  rule 3: primary_anomaly_count == 1  → malicious\n"
                f"  rule 4: primary_anomaly_count == 0 AND profile_anomaly_count >= 1  → malicious "
                f"(evasion: cohort-normal but |profile z| ≥ {PROFILE_Z_THRESH}σ)\n"
                "  rule 5: primary_anomaly_count == 0 AND profile_anomaly_count == 0  → benign\n"
            )
            rule_hint = "rule <1|2|3|4|5>"
        else:
            profile_block = (
                "PROFILE-BASED secondary context (descriptive only):\n"
                f"  raw cosine_sim = {ubr['cosine_sim']:.4f}  (profile z = {prof_cos_z:+.2f}σ)\n"
                f"  raw norm       = {ubr['norm']:.4f}  (profile z = {prof_n_z:+.2f}σ)\n"
                f"  drift_zscore   = {ubr['drift_zscore']:+.2f}  "
                f"{'[ANOMALOUS]' if drift_anom else '[within]'}\n"
                f"  sign_consistency = {ubr['sign_consistency']:.4f}\n\n"
            )
            policy = (
                "Decision policy (apply STRICTLY using the precomputed tags above):\n"
                "  rule 1: severe == True   → malicious\n"
                "  rule 2: primary_anomaly_count >= 2  → malicious\n"
                "  rule 3: primary_anomaly_count == 1  → malicious\n"
                "  rule 4: primary_anomaly_count == 0  → benign\n"
                "Pick the FIRST rule that matches; profile signals influence confidence "
                "only, never the label.\n"
            )
            rule_hint = "rule <1|2|3|4>"

        prompt = (
            "You are a federated-learning security auditor. You decide whether THIS "
            "client is malicious. The anomaly tags below are precomputed against "
            "fixed numeric thresholds — TRUST the tag wording, do NOT re-judge "
            "the σ value as 'small'.\n\n"
            "ROUND-RELATIVE primary signals (median + MAD over the cohort):\n"
            f"  {cos_flag}\n"
            f"  {nrm_flag}\n"
            f"  primary_anomaly_count = {n_anom}/2   severe = {severe}\n\n"
            f"{profile_block}"
            f"{policy}"
            "Respond ONLY with valid JSON:\n"
            '{"label": "benign" or "malicious", "confidence": <0-1>, '
            f'"rationale": "cohort: ...; profile: ...; Decision: {rule_hint} because ..."}}'
        )
        try:
            resp = requests.post(self.endpoint, json={
                "model":  self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": temperature},
            }, timeout=60)
            resp.raise_for_status()
            raw = json.loads(resp.json().get("response", "{}"))
            label = str(raw.get("label", "benign")).strip().lower()
            if self.strict_json and label not in ("benign", "malicious"):
                label = "malicious"
            return {
                "label":      label,
                "confidence": float(raw.get("confidence", 0.5)),
                "rationale":  str(raw.get("rationale", "")),
            }
        except Exception as e:
            print(f"[OllamaAuditor] Error: {e}.")
            if self.strict_json:
                return {"label": "malicious", "confidence": 0.99,
                        "rationale": f"hardened strict JSON: {e}"}
            print("[OllamaAuditor] Falling back to mock.")
            if self.audit_mode == "profile_hybrid":
                n_anom, severe, _, _ = _cohort_primary_tags(ubr)
                if not severe and n_anom == 0:
                    return _decide_profile_primary_zero_label(
                        ubr, server_round, self.stats,
                    )
            return MockAuditor(
                self.stats, profile_hardened=(prompt_mode == "profile_hardened"),
            ).audit(ubr, server_round, temperature)


def compute_evasion_update(
    malicious_update: np.ndarray,
    honest_update: np.ndarray,
    benign_stats: dict,
    prev_norm: float = None,
    prev_cosine: float = None,
) -> np.ndarray:
    """Task 3: norm clamp μ±1.5σ; cosine μ from profile or server peer medians; honest Δ as direction proxy."""
    peer_med_cos  = benign_stats.get("peer_med_cos",  None)
    peer_med_norm = benign_stats.get("peer_med_norm", None)
    if peer_med_cos is not None and peer_med_norm is not None:
        mu_cos = float(peer_med_cos)
        mu_n   = float(peer_med_norm)
        sig_n  = 0.05 * abs(mu_n) + 1e-3
    else:
        mu_n   = benign_stats.get("mu_norm",       1.0)
        sig_n  = benign_stats.get("sigma_norm",    0.1)
        mu_cos = benign_stats.get("mu_cosine_sim", 0.33)
    lo_n   = mu_n - NORM_SIGMA_MUL * sig_n
    hi_n   = mu_n + NORM_SIGMA_MUL * sig_n

    h_nrm = np.linalg.norm(honest_update)
    if h_nrm < 1e-9:
        return honest_update.copy()
    ref_unit = honest_update / h_nrm

    dot_m    = np.dot(malicious_update, ref_unit)
    residual = malicious_update - dot_m * ref_unit
    res_nrm  = np.linalg.norm(residual)
    if res_nrm < 1e-9:
        rng = np.random.default_rng(42)
        noise    = rng.standard_normal(malicious_update.shape)
        noise   -= np.dot(noise, ref_unit) * ref_unit
        res_nrm  = np.linalg.norm(noise)
        perp_unit = noise / (res_nrm + 1e-9)
    else:
        perp_unit = residual / res_nrm

    sin_theta = np.sqrt(max(0.0, 1.0 - mu_cos ** 2))
    evasion   = (mu_cos * mu_n) * ref_unit + (sin_theta * mu_n) * perp_unit

    if prev_norm is not None and prev_norm > hi_n:
        evasion *= (hi_n / (prev_norm + 1e-9)) * 0.95

    cur_nrm = np.linalg.norm(evasion)
    if cur_nrm > hi_n:
        evasion *= hi_n / cur_nrm
    elif cur_nrm < lo_n and cur_nrm > 1e-9:
        evasion *= lo_n / cur_nrm

    return evasion


# --- server strategy (Tasks 3–4) ---
class FedLLMGuardStrategy(FedAvg):
    """UBR each round; LLM audit → weight 0/1; JSONL log; profile mode skips audit."""

    def __init__(self, auditor, global_model_ref: nn.Module,
                 ground_truth_ids: list = None,
                 mode: str = "guard",          # profile | guard
                 audit_every_round: bool = True,
                 llm_temperature: float = 0.0,
                 log_path: str = None,
                 benign_stats: dict = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.auditor         = auditor
        self.global_net      = global_model_ref
        self.malicious_ids   = set(ground_truth_ids or [])
        self.mode            = mode
        self.audit_every     = audit_every_round
        self.temperature     = llm_temperature
        self.log_path        = log_path
        self.benign_stats    = benign_stats or {}

        self.norm_history: Dict[str, list] = {}
        self.ubr_records: list = []
        self.experiment_log: list = []
        self._cid_to_pid: Dict[str, int] = {}

        self.history_acc    = []
        self.history_loss   = []
        self.history_asr    = []

        self._current_global_flat: Optional[np.ndarray] = None

        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def configure_fit(self, server_round, parameters, client_manager):
        fit_ins_pairs = super().configure_fit(server_round, parameters, client_manager)

        self._current_global_flat = flatten_params(parameters_to_ndarrays(parameters))

        extra: dict = {"server_round": server_round}
        if self.benign_stats:
            pr      = self.benign_stats.get("per_round", {}).get(str(server_round), {})
            mu_n    = pr.get("mu_norm",    self.benign_stats.get("mu_norm",           0.0))
            sig_n   = pr.get("sig_norm",   self.benign_stats.get("sigma_norm",        1.0))
            mu_cos  = pr.get("mu_cosine",  self.benign_stats.get("mu_cosine_sim",     0.33))
            sig_cos = pr.get("sig_cosine", self.benign_stats.get("sigma_cosine_sim",  0.07))
            extra.update({
                "mu_norm":       mu_n,
                "sigma_norm":    sig_n,
                "mu_cosine_sim": mu_cos,
                "sig_cosine":    sig_cos,
            })
        if getattr(self, "_last_med_cos", None) is not None:
            extra["peer_med_cos"]  = float(self._last_med_cos)
            extra["peer_med_norm"] = float(self._last_med_norm)

        result = []
        for proxy, fit_ins in fit_ins_pairs:
            per_client = dict(fit_ins.config)
            per_client.update(extra)
            pid_guess = self._cid_to_pid.get(str(proxy.cid), proxy.cid)
            fb_file = os.path.join(FB_DIR, f"{pid_guess}.json")
            if os.path.exists(fb_file):
                try:
                    fb = json.loads(open(fb_file).read())
                    per_client["prev_norm"]   = float(fb.get("norm", 0.0))
                    per_client["prev_cosine"] = float(fb.get("cosine_sim", 1.0))
                except Exception:
                    pass
            from flwr.common import FitIns
            result.append((proxy, FitIns(fit_ins.parameters, per_client)))
        return result

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # Collect client params and compute per-client update deltas
        client_params  = [parameters_to_ndarrays(r.parameters) for _, r in results]
        client_flat    = [flatten_params(p) for p in client_params]
        global_flat    = self._current_global_flat

        if global_flat is None:
            global_flat = np.zeros_like(client_flat[0])

        updates_flat = [cf - global_flat for cf in client_flat]

        u_arr = np.stack(updates_flat, axis=0)
        n_clients = u_arr.shape[0]
        k = max(1, int(round(n_clients * 0.10)))
        if n_clients - 2 * k >= 1:
            sorted_u = np.sort(u_arr, axis=0)
            reference_flat = sorted_u[k: n_clients - k].mean(axis=0)
        else:
            reference_flat = np.median(u_arr, axis=0)

        # First pass: raw UBR (four advisor fields); cohort medians added below.
        weights       = []
        ubr_list      = []
        audit_results = []
        raw_ubrs      = []

        for i, (proxy, result) in enumerate(results):
            pid = result.metrics.get("partition_id", proxy.cid)
            self._cid_to_pid[str(proxy.cid)] = pid
            self.norm_history.setdefault(pid, [])
            ubr = compute_ubr(updates_flat[i], reference_flat, self.norm_history[pid])
            self.norm_history[pid].append(ubr["norm"])
            raw_ubrs.append((pid, ubr))

        cos_arr = np.array([u["cosine_sim"] for _, u in raw_ubrs])
        nrm_arr = np.array([u["norm"]       for _, u in raw_ubrs])
        med_cos = float(np.median(cos_arr))
        med_nrm = float(np.median(nrm_arr))
        mad_cos = float(np.median(np.abs(cos_arr - med_cos)) * 1.4826) or 1e-4
        mad_nrm = float(np.median(np.abs(nrm_arr - med_nrm)) * 1.4826) or 1e-4
        self._last_med_cos  = med_cos
        self._last_med_norm = med_nrm

        for i, (proxy, result) in enumerate(results):
            pid, ubr = raw_ubrs[i]
            ubr["cos_dev_round"]  = (ubr["cosine_sim"] - med_cos) / mad_cos
            ubr["norm_dev_round"] = (ubr["norm"]       - med_nrm) / mad_nrm
            ubr_list.append(ubr)
            self.ubr_records.append({**ubr, "_round": server_round})

            try:
                with open(os.path.join(FB_DIR, f"{pid}.json"), "w") as fh:
                    json.dump(ubr, fh)
            except Exception:
                pass

            run_audit = (self.mode == "guard" and self.audit_every)
            if run_audit:
                ar = self.auditor.audit(ubr, server_round, self.temperature)
            else:
                ar = {"label": "benign", "confidence": 1.0, "rationale": "profiling mode"}

            audit_results.append(ar)
            w = 0.0 if ar["label"] == "malicious" else 1.0
            weights.append(w)

        if sum(weights) == 0:
            acc, loss = _server_evaluate(self.global_net)
            asr = _server_asr(self.global_net, FLIP_FROM, FLIP_TO)
            self.history_acc.append((server_round, acc))
            self.history_loss.append((server_round, loss))
            self.history_asr.append((server_round, asr))
            print(f"[Round {server_round:3d}] SKIPPED aggregation (all clients flagged)")
            for i, (proxy, result) in enumerate(results):
                pid = result.metrics.get("partition_id", proxy.cid)
                entry = {
                    "round": server_round, "client_id": int(pid),
                    "ground_truth": "malicious" if int(pid) in self.malicious_ids else "benign",
                    "ubr": ubr_list[i],
                    "audit_label": audit_results[i]["label"],
                    "audit_conf": audit_results[i]["confidence"],
                    "audit_rationale": audit_results[i]["rationale"],
                    "agg_weight": 0.0,
                    "global_acc": acc, "global_loss": loss, "global_asr": asr,
                }
                self.experiment_log.append(entry)
                if self.log_path:
                    with open(self.log_path, "a") as fh:
                        fh.write(json.dumps(entry) + "\n")
            unchanged = ndarrays_to_parameters(get_weights(self.global_net))
            return unchanged, {}

        total_w  = sum(weights)
        agg_list = [
            [w * layer for layer in client_params[i]]
            for i, w in enumerate(weights)
        ]
        num_layers = len(client_params[0])
        agg_params = [
            sum(agg_list[i][l] for i in range(len(weights))) / total_w
            for l in range(num_layers)
        ]
        aggregated_parameters = ndarrays_to_parameters(agg_params)
        set_weights(self.global_net, agg_params)

        acc, loss = _server_evaluate(self.global_net)
        asr       = _server_asr(self.global_net, FLIP_FROM, FLIP_TO)
        self.history_acc.append((server_round, acc))
        self.history_loss.append((server_round, loss))
        self.history_asr.append((server_round, asr))
        print(f"[Round {server_round:3d}] acc={acc:.4f}  loss={loss:.4f}  "
              f"ASR({FLIP_FROM}->{FLIP_TO})={asr*100:.2f}%  "
              f"weights={[round(w,1) for w in weights]}")

        for i, (proxy, result) in enumerate(results):
            pid    = result.metrics.get("partition_id", proxy.cid)
            is_mal = int(pid) in self.malicious_ids
            entry = {
                "round":          server_round,
                "client_id":      int(pid),
                "ground_truth":   "malicious" if is_mal else "benign",
                "ubr":            ubr_list[i],
                "audit_label":    audit_results[i]["label"],
                "audit_conf":     audit_results[i]["confidence"],
                "audit_rationale":audit_results[i]["rationale"],
                "agg_weight":     weights[i],
                "global_acc":     acc,
                "global_loss":    loss,
                "global_asr":     asr,
            }
            self.experiment_log.append(entry)
            if self.log_path:
                with open(self.log_path, "a") as fh:
                    fh.write(json.dumps(entry) + "\n")

        return aggregated_parameters, {}

    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        return loss, metrics


# --- server metrics ---
def _server_evaluate(net: nn.Module) -> Tuple[float, float]:
    net.eval(); criterion = nn.CrossEntropyLoss()
    correct = total = 0; total_loss = 0.0
    with torch.no_grad():
        for imgs, labels in _server_testloader:
            out = net(imgs.to(DEVICE))
            total_loss += criterion(out, labels.to(DEVICE)).item()
            correct    += (out.argmax(1) == labels.to(DEVICE)).sum().item()
            total      += len(labels)
    return correct / total, total_loss / len(_server_testloader)


def _server_asr(net: nn.Module, flip_from: int, flip_to: int) -> float:
    """Fraction of clean test points with label flip_from predicted as flip_to."""
    net.eval(); total = predicted_flip = 0
    with torch.no_grad():
        for imgs, labels in _server_testloader:
            mask = (labels == flip_from)
            if mask.sum() == 0:
                continue
            out = net(imgs[mask].to(DEVICE)).argmax(1).cpu()
            predicted_flip += (out == flip_to).sum().item()
            total          += mask.sum().item()
    return predicted_flip / total if total > 0 else 0.0


def weighted_average(metrics):
    accs = [n * m["accuracy"] for n, m in metrics]
    return {"accuracy": sum(accs) / sum(n for n, _ in metrics)}


# --- Flower clients (conditions A–D) ---
class BenignClient(NumPyClient):
    """Honest client (no poisoning)."""
    def __init__(self, partition_id: int, fds: FederatedDataset):
        self.pid = partition_id
        self.net = MNISTClassifier().to(DEVICE)
        self.train_loader, self.test_loader = load_partition(partition_id, fds)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train_model(self.net, self.train_loader)
        return get_weights(self.net), len(self.train_loader.dataset), {"partition_id": self.pid}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        acc, loss = evaluate_model(self.net, self.test_loader)
        return loss, len(self.test_loader.dataset), {"accuracy": acc}


class NaiveFlipClient(NumPyClient):
    """Condition A: label-flip poisoning, no evasion."""
    def __init__(self, partition_id: int, fds: FederatedDataset):
        self.pid = partition_id
        self.net = MNISTClassifier().to(DEVICE)
        self.train_loader, self.test_loader = load_partition(
            partition_id, fds, flip_from=FLIP_FROM, flip_to=FLIP_TO
        )

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train_model(self.net, self.train_loader)
        return get_weights(self.net), len(self.train_loader.dataset), {"partition_id": self.pid}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        acc, loss = evaluate_model(self.net, self.test_loader)
        return loss, len(self.test_loader.dataset), {"accuracy": acc}


class ScaleAttackClient(NumPyClient):
    """Condition B: multiply local update by SCALE (no evasion)."""
    SCALE = 5.0

    def __init__(self, partition_id: int, fds: FederatedDataset):
        self.pid = partition_id
        self.net = MNISTClassifier().to(DEVICE)
        self.train_loader, self.test_loader = load_partition(
            partition_id, fds, flip_from=FLIP_FROM, flip_to=FLIP_TO
        )

    def fit(self, parameters, config):
        global_params = [p.copy() for p in parameters]
        set_weights(self.net, parameters)
        train_model(self.net, self.train_loader)
        local_params = get_weights(self.net)
        scaled = [g + self.SCALE * (l - g)
                  for g, l in zip(global_params, local_params)]
        return scaled, len(self.train_loader.dataset), {"partition_id": self.pid}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        acc, loss = evaluate_model(self.net, self.test_loader)
        return loss, len(self.test_loader.dataset), {"accuracy": acc}


class EvasionClient(NumPyClient):
    """Conditions C/D: honest warmup then Task-3 constrained poisoned update."""
    def __init__(self, partition_id: int, fds: FederatedDataset):
        self.pid          = partition_id
        self.net_honest   = MNISTClassifier().to(DEVICE)
        self.net_malicious= MNISTClassifier().to(DEVICE)
        self.honest_loader,  self.test_loader = load_partition(partition_id, fds)
        self.poison_loader, _                 = load_partition(
            partition_id, fds, flip_from=FLIP_FROM, flip_to=FLIP_TO
        )

    def fit(self, parameters, config):
        server_round = int(config.get("server_round", 0))
        mu_norm      = float(config.get("mu_norm",       0.0))
        sigma_norm   = float(config.get("sigma_norm",   1.0))
        mu_cosine    = float(config.get("mu_cosine_sim", 0.33))
        prev_norm    = config.get("prev_norm",   None)
        prev_cosine  = config.get("prev_cosine", None)

        global_params = [p.copy() for p in parameters]

        set_weights(self.net_honest, parameters)
        train_model(self.net_honest, self.honest_loader)
        honest_params = get_weights(self.net_honest)

        if server_round < T_WARMUP:
            return honest_params, len(self.honest_loader.dataset), {"partition_id": self.pid}

        set_weights(self.net_malicious, parameters)
        train_model(self.net_malicious, self.poison_loader)
        malicious_params = get_weights(self.net_malicious)

        global_flat    = flatten_params(global_params)
        honest_update  = flatten_params(honest_params)  - global_flat
        malicious_upd  = flatten_params(malicious_params) - global_flat

        benign_stats = {"mu_norm": mu_norm, "sigma_norm": sigma_norm,
                        "mu_cosine_sim": mu_cosine}
        if "peer_med_cos" in config and "peer_med_norm" in config:
            benign_stats["peer_med_cos"]  = float(config["peer_med_cos"])
            benign_stats["peer_med_norm"] = float(config["peer_med_norm"])
        evasion_flat = compute_evasion_update(
            malicious_upd, honest_update, benign_stats,
            prev_norm=prev_norm, prev_cosine=prev_cosine,
        )

        evasion_params = []
        offset = 0
        for layer in global_params:
            n = layer.size
            evasion_params.append(
                (global_flat[offset:offset+n] + evasion_flat[offset:offset+n])
                .reshape(layer.shape)
            )
            offset += n

        return evasion_params, len(self.honest_loader.dataset), {"partition_id": self.pid}

    def evaluate(self, parameters, config):
        set_weights(self.net_honest, parameters)
        acc, loss = evaluate_model(self.net_honest, self.test_loader)
        return loss, len(self.test_loader.dataset), {"accuracy": acc}


class ModelOnlyEvasionClient(NumPyClient):
    """Condition E: honest local data; malicious Δ via backdoor grad steps + UBR evasion (no label flip)."""
    def __init__(self, partition_id: int, fds: FederatedDataset):
        self.pid = partition_id
        self.net_honest = MNISTClassifier().to(DEVICE)
        self.train_loader, self.test_loader = load_partition(partition_id, fds)

    def fit(self, parameters, config):
        server_round = int(config.get("server_round", 0))
        mu_norm      = float(config.get("mu_norm",       0.0))
        sigma_norm   = float(config.get("sigma_norm",   1.0))
        mu_cosine    = float(config.get("mu_cosine_sim", 0.33))
        prev_norm    = config.get("prev_norm",   None)
        prev_cosine  = config.get("prev_cosine", None)

        global_params = [p.copy() for p in parameters]

        set_weights(self.net_honest, parameters)
        train_model(self.net_honest, self.train_loader)
        honest_params = get_weights(self.net_honest)

        if server_round < T_WARMUP:
            return honest_params, len(self.train_loader.dataset), {"partition_id": self.pid}

        net_mal = copy.deepcopy(self.net_honest)
        backdoor_gradient_step(net_mal, self.train_loader)
        malicious_params = get_weights(net_mal)

        global_flat   = flatten_params(global_params)
        honest_update = flatten_params(honest_params) - global_flat
        malicious_upd = flatten_params(malicious_params) - global_flat
        poison_delta  = malicious_upd - honest_update
        malicious_upd = honest_update + POISON_DELTA_SCALE * poison_delta

        benign_stats = {"mu_norm": mu_norm, "sigma_norm": sigma_norm,
                        "mu_cosine_sim": mu_cosine}
        if "peer_med_cos" in config and "peer_med_norm" in config:
            benign_stats["peer_med_cos"]  = float(config["peer_med_cos"])
            benign_stats["peer_med_norm"] = float(config["peer_med_norm"])
        evasion_flat = compute_evasion_update(
            malicious_upd, honest_update, benign_stats,
            prev_norm=prev_norm, prev_cosine=prev_cosine,
        )

        evasion_params = []
        offset = 0
        for layer in global_params:
            n = layer.size
            evasion_params.append(
                (global_flat[offset:offset+n] + evasion_flat[offset:offset+n])
                .reshape(layer.shape)
            )
            offset += n

        return evasion_params, len(self.train_loader.dataset), {"partition_id": self.pid}

    def evaluate(self, parameters, config):
        set_weights(self.net_honest, parameters)
        acc, loss = evaluate_model(self.net_honest, self.test_loader)
        return loss, len(self.test_loader.dataset), {"accuracy": acc}


# --- experiment runners ---
def _make_client_fn(attack_mode: str, fds: FederatedDataset):
    """Map Flower cid → client implementation."""
    def client_fn(cid: str):
        pid = int(cid)
        if pid in MALICIOUS_IDS:
            if attack_mode == "none":
                return BenignClient(pid, fds).to_client()
            elif attack_mode == "naive_flip":
                return NaiveFlipClient(pid, fds).to_client()
            elif attack_mode == "scale":
                return ScaleAttackClient(pid, fds).to_client()
            elif attack_mode == "evasion":
                return EvasionClient(pid, fds).to_client()
            elif attack_mode == "model_evasion":
                return ModelOnlyEvasionClient(pid, fds).to_client()
        return BenignClient(pid, fds).to_client()
    return client_fn


def run_task2_profiling(auditor_type: str = "mock"):
    """Task 2: 100 clean rounds → benign_stats.json, task2 log, UBR plots."""
    print("\n" + "="*60)
    print("TASK 2 : Benign UBR Profiling  (100 clean rounds)")
    print("="*60)

    fds      = FederatedDataset(dataset="mnist",
                                partitioners={"train": IidPartitioner(NUM_CLIENTS)})
    net      = MNISTClassifier().to(DEVICE)
    auditor  = MockAuditor()
    log_path = os.path.join(DATA_DIR, "task2_ubr_profiling.jsonl")
    open(log_path, "w").close()

    strategy = FedLLMGuardStrategy(
        auditor           = auditor,
        global_model_ref  = net,
        ground_truth_ids  = [],       # no malicious clients
        mode              = "profile",
        log_path          = log_path,
        initial_parameters= ndarrays_to_parameters(get_weights(net)),
        evaluate_metrics_aggregation_fn = weighted_average,
    )

    server = ServerApp(strategy=strategy, config=ServerConfig(num_rounds=NUM_ROUNDS))
    client = ClientApp(client_fn=_make_client_fn("none", fds))
    run_simulation(server_app=server, client_app=client, num_supernodes=NUM_CLIENTS,
                   backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.1}})

    # Compute and save benign statistics
    stats = compute_benign_stats(strategy.ubr_records)
    with open(STATS_FILE, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"\n[Task 2] Benign stats saved → {STATS_FILE}")
    print(json.dumps(stats, indent=2))

    # Generate distribution plots
    _plot_ubr_distributions(strategy.ubr_records, stats)
    pkl_path = os.path.join(ARTIFACT_DIR, "task2_history.pkl")
    with open(pkl_path, "wb") as fh:
        pickle.dump({"acc": strategy.history_acc, "loss": strategy.history_loss,
                     "ubr_records": strategy.ubr_records, "stats": stats}, fh)
    return stats


def run_condition(condition: str, benign_stats: dict,
                  rep: int = 0, auditor_type: str = "mock",
                  num_rounds: int = None, log_suffix: str = "",
                  guard_mode: str = None):
    """One Week-6 repetition: A–E → JSONL log + summary JSON + pickle history."""
    attack_map = {
        "A": "naive_flip", "B": "scale", "C": "evasion",
        "D": "evasion", "E": "model_evasion",
    }
    rounds = num_rounds if num_rounds is not None else NUM_ROUNDS
    attack_mode = attack_map[condition]
    if condition == "D":
        audit_mode = guard_mode or "profile_hybrid"
        strict_json = True
        guard_tag = f"{audit_mode}_guard"
    else:
        audit_mode, strict_json, guard_tag = "standard", False, "standard_guard"

    print(f"\n{'='*60}")
    print(f"CONDITION {condition}  |  rep {rep}  |  attack={attack_mode}"
          f"  |  {guard_tag}  |  rounds={rounds}  |  llm_T={LLM_TEMPERATURE}")
    print("="*60)

    torch.manual_seed(SEED + rep * 100)
    np.random.seed(SEED + rep * 100)

    fds      = FederatedDataset(dataset="mnist",
                                partitioners={"train": IidPartitioner(NUM_CLIENTS)})
    net      = MNISTClassifier().to(DEVICE)
    sfx = f"_{log_suffix}" if log_suffix else ""
    log_path = os.path.join(DATA_DIR, f"condition_{condition}_rep{rep}{sfx}.jsonl")
    open(log_path, "w").close()

    if auditor_type == "ollama":
        auditor = OllamaAuditor(
            benign_stats=benign_stats,
            audit_mode=audit_mode,
            strict_json=strict_json,
        )
    else:
        auditor = MockAuditor(
            benign_stats=benign_stats,
            profile_hardened=(audit_mode == "profile_hardened"),
        )

    strategy = FedLLMGuardStrategy(
        auditor           = auditor,
        global_model_ref  = net,
        ground_truth_ids  = MALICIOUS_IDS,
        mode              = "guard",
        audit_every_round = True,
        llm_temperature   = LLM_TEMPERATURE,
        log_path          = log_path,
        benign_stats      = benign_stats,
        initial_parameters= ndarrays_to_parameters(get_weights(net)),
        evaluate_metrics_aggregation_fn = weighted_average,
    )

    server = ServerApp(strategy=strategy, config=ServerConfig(num_rounds=rounds))
    client = ClientApp(client_fn=_make_client_fn(attack_mode, fds))
    run_simulation(server_app=server, client_app=client, num_supernodes=NUM_CLIENTS,
                   backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.1}})

    # Compute summary metrics from log
    summary = _compute_summary(strategy.experiment_log, benign_stats)
    summary["num_rounds"] = rounds
    summary["condition"]  = condition
    summary["rep"]        = rep
    if strategy.experiment_log:
        summary["max_round"] = max(e["round"] for e in strategy.experiment_log)
    else:
        summary["max_round"] = rounds
    summary_path = os.path.join(RESULTS_DIR, f"summary_{condition}_rep{rep}{sfx}.json")
    with open(summary_path, "w") as fh:
        json.dump(_ordered_summary(summary), fh, indent=2)
    pkl_path = os.path.join(ARTIFACT_DIR, f"condition_{condition}_rep{rep}{sfx}.pkl")
    with open(pkl_path, "wb") as fh:
        pickle.dump({"acc": strategy.history_acc, "loss": strategy.history_loss,
                     "asr": strategy.history_asr, "log": strategy.experiment_log}, fh)
    print(f"\n[Condition {condition} rep{rep}] evasion_all={summary['evasion_rate']:.3f}  "
          f"evasion_aligned(r11-100)={summary['evasion_rate_aligned']:.3f}  "
          f"final_acc={summary['final_acc']:.4f}  final_asr={summary['final_asr']:.4f}")
    return summary


def run_all_conditions(reps: int = 5, auditor_type: str = "mock", num_rounds: int = None):
    """Run A–E × reps; write all_summaries.json."""
    # Load benign stats from Task 2 (must run task2 first)
    if not os.path.exists(STATS_FILE):
        print("[Warning] benign_stats.json not found – running Task 2 first.")
        benign_stats = run_task2_profiling(auditor_type)
    else:
        with open(STATS_FILE) as fh:
            benign_stats = json.load(fh)
        print(f"[Info] Loaded benign stats from {STATS_FILE}")

    all_summaries = []
    for condition in ["A", "B", "C", "D", "E"]:
        for rep in range(reps):
            s = run_condition(condition, benign_stats, rep, auditor_type,
                              num_rounds=num_rounds, log_suffix="")
            all_summaries.append(s)

    _print_metrics_table(all_summaries)
    with open(os.path.join(RESULTS_DIR, "all_summaries.json"), "w") as fh:
        json.dump(all_summaries, fh, indent=2)
    return all_summaries


# --- metrics ---
SUMMARY_FIELDS = (
    "condition", "rep", "num_rounds", "max_round",
    "evasion_rate", "evasion_rate_aligned",
    "n_mal_entries", "n_mal_aligned", "n_ben_entries",
    "final_acc", "final_asr", "final_loss", "kl_norm",
)


def _ordered_summary(d: dict) -> dict:
    """Stable JSON key order; evasion_rate_aligned immediately after evasion_rate."""
    out = {k: d[k] for k in SUMMARY_FIELDS if k in d}
    for k, v in d.items():
        if k not in out:
            out[k] = v
    return out


def _compute_summary(log: list, benign_stats: dict) -> dict:
    """Week-6 summary: evasion rate, last-round acc/loss/ASR, KL(norm)."""
    if not log:
        return {}
    mal_entries = [e for e in log if e["ground_truth"] == "malicious"]
    ben_entries = [e for e in log if e["ground_truth"] == "benign"]

    if mal_entries:
        evasion_rate = sum(
            1 for e in mal_entries if e["audit_label"] == "benign"
        ) / len(mal_entries)
        mal_aligned = [e for e in mal_entries if e["round"] > T_WARMUP]
        evasion_rate_aligned = (
            sum(1 for e in mal_aligned if e["audit_label"] == "benign") / len(mal_aligned)
            if mal_aligned else 0.0
        )
    else:
        evasion_rate = 0.0
        evasion_rate_aligned = 0.0
        mal_aligned = []

    last_round_entries = [e for e in log if e["round"] == max(e["round"] for e in log)]
    final_acc  = last_round_entries[0]["global_acc"] if last_round_entries else 0.0
    final_asr  = last_round_entries[0]["global_asr"] if last_round_entries else 0.0
    final_loss = last_round_entries[0]["global_loss"] if last_round_entries else 0.0

    kl_norm = _kl_divergence_norm(
        [e["ubr"]["norm"] for e in mal_entries],
        [e["ubr"]["norm"] for e in ben_entries],
    ) if mal_entries and ben_entries else float("nan")

    return {
        "evasion_rate": evasion_rate,
        "evasion_rate_aligned": evasion_rate_aligned,
        "final_acc":    final_acc,
        "final_asr":    final_asr,
        "final_loss":   final_loss,
        "kl_norm":      kl_norm,
        "n_mal_entries": len(mal_entries),
        "n_mal_aligned": len(mal_aligned),
        "n_ben_entries": len(ben_entries),
    }


def _kl_divergence_norm(p_vals: list, q_vals: list, bins: int = 30) -> float:
    """Binned KL(norm_mal || norm_ben)."""
    all_vals = p_vals + q_vals
    lo, hi   = min(all_vals), max(all_vals)
    if hi == lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    p_hist, _ = np.histogram(p_vals, bins=edges, density=True)
    q_hist, _ = np.histogram(q_vals, bins=edges, density=True)
    eps   = 1e-10
    p_hist = p_hist + eps
    q_hist = q_hist + eps
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()
    return float(np.sum(p_hist * np.log(p_hist / q_hist)))


def _print_metrics_table(summaries: list):
    """Print evasion / acc / ASR / KL table to stdout."""
    print("\n" + "="*70)
    print(f"{'Cond':>6} {'Rep':>4}  {'Evasion':>8} {'Acc':>8} {'ASR':>8} {'KL(norm)':>10}")
    print("-"*70)
    for s in summaries:
        print(f"{s.get('condition','?'):>6} {s.get('rep',0):>4}  "
              f"{s.get('evasion_rate', 0):>8.3f} "
              f"{s.get('final_acc', 0):>8.4f} "
              f"{s.get('final_asr', 0):>8.4f} "
              f"{s.get('kl_norm', float('nan')):>10.4f}")
    print("="*70)


# --- plots ---
def _plot_ubr_distributions(ubr_records: list, stats: dict):
    """Task 2: histograms of the four UBR fields + μ±1.5σ lines."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warning] matplotlib not available – skipping plots.")
        return

    fields = ["norm", "cosine_sim", "drift_zscore", "sign_consistency"]
    labels = ["L2 Norm", "Cosine Similarity", "Drift Z-Score", "Sign Consistency"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Benign UBR Distribution (100 Clean Rounds)", fontsize=14, fontweight="bold")

    for ax, field, label in zip(axes, fields, labels):
        vals = [r[field] for r in ubr_records]
        ax.hist(vals, bins=40, color="#1f77b4", edgecolor="white", alpha=0.85)
        mu  = stats.get(f"mu_{field}",    0)
        sig = stats.get(f"sigma_{field}", 1)
        ax.axvline(mu,           color="red",    linestyle="--", linewidth=1.5, label=f"μ={mu:.3f}")
        ax.axvline(mu + 1.5*sig, color="orange", linestyle=":",  linewidth=1.2, label=f"μ±1.5σ")
        ax.axvline(mu - 1.5*sig, color="orange", linestyle=":",  linewidth=1.2)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Value"); ax.set_ylabel("Count")
        ax.legend(fontsize=8); ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "ubr_distributions.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[Task 2] Distribution plot saved → {out}")
    plt.close()


def plot_condition_results(condition: str, rep: int = 0):
    """Plot acc / loss / ASR vs round from condition pickle."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    pkl_path = os.path.join(ARTIFACT_DIR, f"condition_{condition}_rep{rep}.pkl")
    if not os.path.exists(pkl_path):
        print(f"[Warning] {pkl_path} not found.")
        return

    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh)

    rounds = [r for r, _ in data["acc"]]
    accs   = [v for _, v in data["acc"]]
    losses = [v for _, v in data["loss"]]
    asrs   = [v for _, v in data["asr"]]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Condition {condition} – rep {rep}", fontsize=13, fontweight="bold")

    ax1.plot(rounds, accs,   color="#1f77b4", linewidth=2)
    ax1.set_title("Global Accuracy"); ax1.set_xlabel("Round"); ax1.grid(True, linestyle=":", alpha=0.5)

    ax2.plot(rounds, losses, color="#d62728", linewidth=2)
    ax2.set_title("Global Loss"); ax2.set_xlabel("Round"); ax2.grid(True, linestyle=":", alpha=0.5)

    ax3.plot(rounds, [a*100 for a in asrs], color="#ff7f0e", linewidth=2)
    ax3.set_title(f"ASR ({FLIP_FROM}→{FLIP_TO}) %")
    ax3.set_xlabel("Round"); ax3.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"condition_{condition}_rep{rep}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved → {out}")
    plt.close()


# --- CLI helper: sample false-negative audits ---
def print_evasion_failures(condition: str = "C", n: int = 5):
    """Print n log lines: malicious ground truth but audit_label benign (Task 4 rationale)."""
    import glob
    pattern  = os.path.join(DATA_DIR, f"condition_{condition}_rep*.jsonl")
    entries  = []
    for path in glob.glob(pattern):
        with open(path) as fh:
            for line in fh:
                try:
                    e = json.loads(line.strip())
                    if e["ground_truth"] == "malicious" and e["audit_label"] == "benign":
                        entries.append(e)
                except Exception:
                    pass

    if not entries:
        print(f"[Info] No evasion failures found for condition {condition}.")
        return

    print(f"\n{'='*70}")
    print(f"  LLM Evasion Failures – Condition {condition}  "
          f"(showing {min(n, len(entries))} of {len(entries)})")
    print("="*70)
    for e in entries[:n]:
        print(f"\n  Round {e['round']}  |  Client {e['client_id']}  "
              f"|  conf={e['audit_conf']:.2f}")
        print(f"  UBR : {json.dumps(e['ubr'], indent=None)}")
        print(f"  Rationale : {e['audit_rationale']}")
    print("="*70)


# --- entry point ---
def main():
    parser = argparse.ArgumentParser(description="FedLLM-Guard (Week 5–6 MNIST experiments)")
    parser.add_argument("--task",    default="profile",
                        choices=["profile", "A", "B", "C", "D", "E", "all"],
                        help="profile=Task2; A–E=conditions; all=A–E×reps")
    parser.add_argument("--rounds",  type=int, default=None,
                        help="Override NUM_ROUNDS (e.g. 20 for quick E tuning)")
    parser.add_argument("--reps",    type=int, default=5,
                        help="Repetitions per condition when using --task all or multi-rep")
    parser.add_argument("--rep",     type=int, default=None,
                        help="Run only this rep (overrides --reps range)")
    parser.add_argument("--auditor", default="mock",
                        choices=["mock", "ollama"],
                        help="mock | ollama (local LLM)")
    parser.add_argument("--plot",    action="store_true",
                        help="Save matplotlib figures after run")
    parser.add_argument("--log-suffix", default="",
                        help="Append to log/summary/pkl names (pilot runs, e.g. pilot_llmfull)")
    parser.add_argument("--guard-mode", default=None,
                        choices=["profile_hardened", "profile_hybrid"],
                        help="Condition D only: full LLM hardened vs hybrid (LLM cohort + code rule 4/5)")
    args = parser.parse_args()

    if args.task == "profile":
        run_task2_profiling(args.auditor)

    elif args.task in ("A", "B", "C", "D", "E"):
        if not os.path.exists(STATS_FILE):
            print("[Info] benign_stats.json not found – running profiling first.")
            benign_stats = run_task2_profiling(args.auditor)
        else:
            with open(STATS_FILE) as fh:
                benign_stats = json.load(fh)
        rep_ids = [args.rep] if args.rep is not None else range(args.reps)
        for rep in rep_ids:
            run_condition(args.task, benign_stats, rep, args.auditor,
                          num_rounds=args.rounds, log_suffix=args.log_suffix,
                          guard_mode=args.guard_mode)
        if args.plot:
            plot_condition_results(args.task)
        print_evasion_failures(args.task)

    elif args.task == "all":
        run_all_conditions(reps=args.reps, auditor_type=args.auditor,
                          num_rounds=args.rounds)
        for cond in ["A", "B", "C", "D", "E"]:
            if args.plot:
                plot_condition_results(cond)
            print_evasion_failures(cond)


if __name__ == "__main__":
    main()
