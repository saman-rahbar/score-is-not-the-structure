"""Brain-geometry alignment for syntactic generalization.

Fine-tunes a language model (Pythia) with an auxiliary linear-CKA loss that
pulls a chosen hidden layer toward fMRI language-network response geometry,
under four matched conditions (LM-only, aligned, shuffled-target, and
rank-matched random-target), and evaluates out-of-distribution syntactic
generalization on BLiMP. See README.md for usage and prerequisites.
"""


import os
import sys
import json
import math
import random
import hashlib
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------------------------------------------
# Global config
# ----------------------------------------------------------------------------
CONFIG = {
    # EXP_MODEL / EXP_SEEDS let us launch one job per model scale in parallel
    # (e.g. pythia-160m at 20 seeds for the headline TOST, pythia-410m/1.4b at
    # fewer seeds for the scale-generality check). Align layers are resolved
    # from the model's depth in main() so the early/mid/late sweep is
    # comparable across sizes (defaults below are the pythia-160m indices).
    "model_name": os.environ.get("EXP_MODEL", "EleutherAI/pythia-160m"),
    "seeds": list(range(int(os.environ.get("EXP_SEEDS", "5")))),
    "conditions": ["lm_only", "brain_align", "shuffled_brain", "random_matrix"],
    "align_layers_ablation": [2, 6, 10],   # hidden-state indices (resolved in main)
    "default_align_layer": 6,
    "align_layer_fracs": [0.15, 0.5, 0.85],  # early/mid/late as depth fractions
    "lambda_align": 1.0,                   # weight on alignment loss
    "lambda_sweep": [0.1, 1.0, 10.0],      # alignment-weight sweep (brain_align)
    "epochs": 6,
    "batch_size": 16,
    "lr": 5e-5,
    "weight_decay": 0.01,
    "max_len": 48,
    "brain_target_rank": 128,              # PCA rank for fMRI / random targets
    "blimp_configs": [
        "wh_questions_subject_gap",
        "determiner_noun_agreement_1",
        "anaphor_gender_agreement",
        "irregular_plural_subject_verb_agreement_1",
        "npi_present_1",
        "regular_plural_subject_verb_agreement_1",
        "ellipsis_n_bar_1",
        "left_branch_island_simple_question",
    ],
    "blimp_max_pairs_per_config": 200,     # cap for tractable eval
    "warmup_frac": 0.1,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OFFLINE = os.environ.get("HF_DATASETS_OFFLINE", "0") == "1" or \
          os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"


# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ----------------------------------------------------------------------------
# Pereira fMRI cache loading (login-node prep helper + offline-safe loader)
# ----------------------------------------------------------------------------
def build_pereira_cache(out_npz):
    """
    LOGIN-NODE ONLY (needs internet). Build a cache .npz with:
      'sentences' : object array of stimulus sentence strings
      'responses' : float32 [N, V] language-network fMRI responses
    from the public Pereira et al. (2018) release. This function is a thin
    wrapper; adapt the loader to your local mirror of the dataset. It is NOT
    called automatically on compute nodes.
    """
    raise NotImplementedError(
        "Run the Pereira download/prep on the login node and save a .npz with "
        "keys 'sentences' and 'responses'. Point PEREIRA_NPZ at it."
    )


def load_pereira():
    """
    Offline-safe loader. Requires a pre-built cache at PEREIRA_NPZ; raises if
    the cache is absent.
    """
    npz_path = os.environ.get("PEREIRA_NPZ", "pereira_cache.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"Pereira fMRI cache not found at '{npz_path}'. On the login node "
            f"build it (build_pereira_cache) and set PEREIRA_NPZ."
        )
    dat = np.load(npz_path, allow_pickle=True)
    sentences = [str(s) for s in dat["sentences"].tolist()]
    responses = np.asarray(dat["responses"], dtype=np.float32)
    assert responses.shape[0] == len(sentences), "sentence/response mismatch"
    if responses.shape[0] < 64:
        raise ValueError("Pereira cache has too few sentences to fine-tune on.")
    return sentences, responses


# ----------------------------------------------------------------------------
# Brain target preprocessing: z-score voxels, PCA to matched rank.
# ----------------------------------------------------------------------------
def prepare_brain_targets(responses, rank, seed):
    resp = np.asarray(responses, dtype=np.float64)
    mu = resp.mean(axis=0, keepdims=True)
    sd = resp.std(axis=0, keepdims=True) + 1e-6
    resp = (resp - mu) / sd
    # PCA via SVD, keep top-`rank` components
    U, S, Vt = np.linalg.svd(resp, full_matrices=False)
    r = min(rank, S.shape[0])
    brain = U[:, :r] * S[:r]  # [N, r] projected coordinates
    # per-column standardize the target space
    brain = (brain - brain.mean(0, keepdims=True)) / (brain.std(0, keepdims=True) + 1e-6)
    return brain.astype(np.float32), r


def shuffled_targets(brain, seed):
    rng = np.random.RandomState(seed + 777)
    perm = rng.permutation(brain.shape[0])
    return brain[perm].copy()


def random_matrix_targets(brain, seed):
    rng = np.random.RandomState(seed + 999)
    rnd = rng.randn(*brain.shape).astype(np.float32)
    rnd = (rnd - rnd.mean(0, keepdims=True)) / (rnd.std(0, keepdims=True) + 1e-6)
    return rnd


# ----------------------------------------------------------------------------
# Differentiable linear CKA (batchwise) alignment objective.
# We MAXIMIZE CKA, i.e. minimize (1 - CKA).
# ----------------------------------------------------------------------------
def linear_cka(X, Y, eps=1e-6):
    # X: [B, dx], Y: [B, dy]; center rows
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    # HSIC-style: ||X^T Y||_F^2 / (||X^T X||_F ||Y^T Y||_F)
    xty = X.t() @ Y
    num = (xty * xty).sum()
    xtx = X.t() @ X
    yty = Y.t() @ Y
    # Clamp each factor away from 0 INSIDE the sqrt: sqrt'(0) is infinite, so a
    # degenerate batch (near-constant pooled reps) otherwise makes the gradient
    # NaN, which corrupts the weights and collapses BLiMP to exactly 0.0000.
    den = torch.sqrt((xtx * xtx).sum().clamp_min(eps) *
                     (yty * yty).sum().clamp_min(eps))
    cka = num / den
    return torch.nan_to_num(cka, nan=0.0, posinf=0.0, neginf=0.0)


# ----------------------------------------------------------------------------
# Model loader (offline-safe)
# ----------------------------------------------------------------------------
def load_model_and_tokenizer(seed):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = CONFIG["model_name"]
    try:
        tok = AutoTokenizer.from_pretrained(name)
        # Force fp32: Pythia's config declares torch_dtype=float16, and newer
        # transformers honours it -> the model would load in fp16 and full
        # fine-tuning without a gradient scaler overflows to NaN. fp32 is safe
        # (a 160M model is tiny) and matches the stable local run.
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    except Exception as e:
        raise RuntimeError(
            f"Could not load '{name}' from cache. Prefetch on login node "
            f"(see PREREQUISITES). Underlying error: {e}"
        )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.to(device)
    return model, tok


# ----------------------------------------------------------------------------
# Tokenize fine-tuning corpus
# ----------------------------------------------------------------------------
def tokenize_corpus(sentences, tok):
    enc = tok(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=CONFIG["max_len"],
        return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"]


def mean_pool(hidden, mask):
    m = mask.unsqueeze(-1).float()
    return (hidden * m).sum(1) / (m.sum(1) + 1e-6)


@torch.no_grad()
def layer_cka_to_target(model, tok, sentences, target, layer):
    """Manipulation check: linear-CKA between the trained model's aligned-layer
    sentence geometry and the true fMRI target, over all fine-tuning sentences.
    High for a genuinely brain-aligned model, near baseline otherwise; this is
    what tells a reader the alignment loss actually moved representations rather
    than being inert."""
    model.eval()
    pooled_all = []
    bs = 32
    for i in range(0, len(sentences), bs):
        chunk = sentences[i:i + bs]
        enc = tok(chunk, padding=True, truncation=True,
                  max_length=CONFIG["max_len"], return_tensors="pt")
        ids = enc["input_ids"].to(device)
        am = enc["attention_mask"].to(device)
        out = model(input_ids=ids, attention_mask=am, output_hidden_states=True)
        pooled_all.append(mean_pool(out.hidden_states[layer], am).float().cpu())
    pooled = torch.cat(pooled_all, 0)
    tgt = torch.tensor(np.asarray(target), dtype=torch.float32)
    return float(linear_cka(pooled, tgt).item())


# ----------------------------------------------------------------------------
# Fine-tuning loop for a single (condition, layer, seed)
# ----------------------------------------------------------------------------
def finetune(condition, align_layer, seed, sentences, brain_map, lam_override=None):
    set_all_seeds(seed)
    model, tok = load_model_and_tokenizer(seed)

    input_ids, attn = tokenize_corpus(sentences, tok)
    n = input_ids.shape[0]

    # select alignment target for this condition
    if condition == "lm_only":
        targets = None
    elif condition == "brain_align":
        targets = torch.tensor(brain_map["true"])
    elif condition == "shuffled_brain":
        targets = torch.tensor(brain_map["shuffled"])
    elif condition == "random_matrix":
        targets = torch.tensor(brain_map["random"])
    else:
        raise ValueError(condition)

    opt = torch.optim.AdamW(
        model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
    )
    steps_per_epoch = math.ceil(n / CONFIG["batch_size"])
    total_steps = steps_per_epoch * CONFIG["epochs"]
    warmup = int(CONFIG["warmup_frac"] * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, s / max(1, warmup))
        * max(0.0, (total_steps - s) / max(1, total_steps - warmup)),
    )

    lam = (lam_override if lam_override is not None
           else (CONFIG["lambda_align"] if condition != "lm_only" else 0.0))

    model.train()
    rng = np.random.RandomState(seed + 12345)
    step = 0
    for epoch in range(CONFIG["epochs"]):
        order = rng.permutation(n)
        for bi in range(0, n, CONFIG["batch_size"]):
            idx = order[bi:bi + CONFIG["batch_size"]]
            if len(idx) < 3:
                continue  # CKA needs a few samples
            ids = input_ids[idx].to(device)
            am = attn[idx].to(device)
            # Ignore PAD positions in the LM loss (labels were the padded ids,
            # so the model was being trained to emit <pad> on padding).
            labels = ids.clone()
            labels[am == 0] = -100
            out = model(
                input_ids=ids,
                attention_mask=am,
                labels=labels,
                output_hidden_states=True,
            )
            lm_loss = out.loss

            if lam > 0.0:
                hs = out.hidden_states[align_layer]  # [B, L, H]
                pooled = mean_pool(hs, am).float()   # [B, H]
                # Unit-normalize rows so CKA sees a well-scaled matrix.
                pooled = pooled / (pooled.norm(dim=1, keepdim=True) + 1e-6)
                tgt = targets[idx].to(device).float()  # [B, r]
                cka = linear_cka(pooled, tgt)
                align_loss = 1.0 - cka
                loss = lm_loss + lam * align_loss
            else:
                loss = lm_loss

            # Never backprop a non-finite loss (belt-and-suspenders vs NaN).
            if not torch.isfinite(loss):
                loss = lm_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1

    return model, tok


# ----------------------------------------------------------------------------
# BLiMP evaluation (OOD syntactic generalization)
# ----------------------------------------------------------------------------
def load_blimp():
    from datasets import load_dataset
    pairs = []  # list of (good, bad, config_name)
    for cfg in CONFIG["blimp_configs"]:
        try:
            ds = load_dataset("nyu-mll/blimp", cfg, split="train")
        except Exception as e:
            raise RuntimeError(
                f"Could not load BLiMP config '{cfg}' from cache. Prefetch on "
                f"login node (see PREREQUISITES). Error: {e}"
            )
        cap = min(CONFIG["blimp_max_pairs_per_config"], len(ds))
        for i in range(cap):
            ex = ds[i]
            pairs.append((ex["sentence_good"], ex["sentence_bad"], cfg))
    return pairs


@torch.no_grad()
def sentence_logprob(model, tok, sentences):
    model.eval()
    scores = []
    bs = 32
    for bi in range(0, len(sentences), bs):
        chunk = sentences[bi:bi + bs]
        enc = tok(chunk, padding=True, truncation=True,
                  max_length=CONFIG["max_len"], return_tensors="pt")
        ids = enc["input_ids"].to(device)
        am = enc["attention_mask"].to(device)
        out = model(input_ids=ids, attention_mask=am)
        logits = out.logits[:, :-1, :]
        tgt = ids[:, 1:]
        logp = torch.log_softmax(logits.float(), dim=-1)
        tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        mask = am[:, 1:].float()
        seq_lp = (tok_lp * mask).sum(1)  # total log-prob (length-summed)
        scores.extend(seq_lp.cpu().tolist())
    return np.array(scores)


def blimp_accuracy(model, tok, pairs):
    good = [p[0] for p in pairs]
    bad = [p[1] for p in pairs]
    lp_good = sentence_logprob(model, tok, good)
    lp_bad = sentence_logprob(model, tok, bad)
    # A diverged (NaN) model must not be reported as 0.0 accuracy; surface it
    # as NaN instead.
    if not (np.isfinite(lp_good).all() and np.isfinite(lp_bad).all()):
        print("[WARN] non-finite log-probs: model diverged this run; "
              "reporting NaN accuracy (not 0.0).")
        return float("nan"), {}
    correct = (lp_good > lp_bad).astype(np.float64)
    # per-config accuracy too
    by_cfg = {}
    for (g, b, cfg), c in zip(pairs, correct):
        by_cfg.setdefault(cfg, []).append(c)
    by_cfg_acc = {k: float(np.mean(v)) for k, v in by_cfg.items()}
    return float(correct.mean()), by_cfg_acc


# ----------------------------------------------------------------------------
# 95% CI helper
# ----------------------------------------------------------------------------
def ci95(vals):
    vals = np.asarray(vals, dtype=np.float64)
    n = len(vals)
    m = float(vals.mean())
    if n < 2:
        return m, 0.0
    se = vals.std(ddof=1) / math.sqrt(n)
    h = se * stats.t.ppf(0.975, n - 1)
    return m, float(h)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    os.makedirs("figures", exist_ok=True)
    print("Device:", device)
    # Scale-generality models set EXP_MAIN_ONLY=1: run only the main 4-condition
    # comparison (+ CKA manipulation check), skipping the lambda sweep and layer
    # ablation, which we only need on the headline model.
    MAIN_ONLY = os.environ.get("EXP_MAIN_ONLY", "0") == "1"

    # ---- Resolve align layers from THIS model's depth (comparable across
    #      sizes), so the early/mid/late sweep is depth-relative not absolute.
    from transformers import AutoConfig
    hf_cfg = AutoConfig.from_pretrained(CONFIG["model_name"])
    n_layers = getattr(hf_cfg, "num_hidden_layers", 12)
    fracs = CONFIG["align_layer_fracs"]
    layers = sorted(set(int(np.clip(round(f * n_layers), 1, n_layers - 1))
                        for f in fracs))
    CONFIG["align_layers_ablation"] = layers
    CONFIG["default_align_layer"] = int(np.clip(round(0.5 * n_layers),
                                                1, n_layers - 1))
    model_tag = CONFIG["model_name"].split("/")[-1]
    print(f"Model {CONFIG['model_name']}: {n_layers} layers -> ablation {layers}, "
          f"default {CONFIG['default_align_layer']}; {len(CONFIG['seeds'])} seeds")
    print("Config:", json.dumps(CONFIG, indent=2))

    # ---- Optional: fMRI-target noise ceiling (from build_pereira_noiseceiling
    #      .py, run on the login node). Reported next to brain_align's CKA so a
    #      reader can judge how much reliable signal the target could carry.
    ceiling = None
    nc_path = os.environ.get("NOISE_CEILING_JSON", "noise_ceiling.json")
    if os.path.exists(nc_path):
        with open(nc_path) as f:
            ceiling = json.load(f)
        print(f"Loaded noise ceiling: {ceiling}")

    # ---- Load real data ----
    sentences, responses = load_pereira()
    print(f"Loaded Pereira: {len(sentences)} sentences, "
          f"responses shape {responses.shape}")

    blimp_pairs = load_blimp()
    print(f"Loaded BLiMP: {len(blimp_pairs)} minimal pairs across "
          f"{len(CONFIG['blimp_configs'])} configs")

    results = {"config": CONFIG}
    seeds = CONFIG["seeds"]

    # ============================================================
    # PART 1: main condition comparison at default alignment layer
    # ============================================================
    per_seed = {c: [] for c in CONFIG["conditions"]}
    per_seed_bycfg = {c: [] for c in CONFIG["conditions"]}
    per_seed_cka = {c: [] for c in CONFIG["conditions"]}   # manipulation check

    for seed in seeds:
        # brain targets depend on seed only for shuffle/random controls;
        # true targets are fixed (PCA is deterministic given data).
        brain_true, rank = prepare_brain_targets(
            responses, CONFIG["brain_target_rank"], seed
        )
        brain_map = {
            "true": brain_true,
            "shuffled": shuffled_targets(brain_true, seed),
            "random": random_matrix_targets(brain_true, seed),
        }
        for cond in CONFIG["conditions"]:
            print(f"\n=== seed={seed} condition={cond} "
                  f"layer={CONFIG['default_align_layer']} ===")
            model, tok = finetune(
                cond, CONFIG["default_align_layer"], seed, sentences, brain_map
            )
            acc, by_cfg = blimp_accuracy(model, tok, blimp_pairs)
            cka = layer_cka_to_target(
                model, tok, sentences, brain_map["true"],
                CONFIG["default_align_layer"])
            print(f"BLiMP acc = {acc:.4f}   CKA(to brain) = {cka:.4f}")
            per_seed[cond].append(acc)
            per_seed_bycfg[cond].append(by_cfg)
            per_seed_cka[cond].append(cka)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for cond in CONFIG["conditions"]:
        m, h = ci95(per_seed[cond])
        results[f"{cond}_blimp_per_seed"] = per_seed[cond]
        results[f"{cond}_blimp_mean"] = m
        results[f"{cond}_blimp_std"] = float(np.std(per_seed[cond], ddof=1))
        results[f"{cond}_blimp_ci95"] = h
        # Manipulation check: did the loss move representations toward the brain?
        results[f"{cond}_cka_per_seed"] = per_seed_cka[cond]
        results[f"{cond}_cka_mean"] = float(np.mean(per_seed_cka[cond]))
        results[f"{cond}_cka_std"] = float(np.std(per_seed_cka[cond], ddof=1))

    # ---- Statistical tests (paired across seeds) ----
    a = np.array(per_seed["brain_align"])
    b = np.array(per_seed["lm_only"])
    s = np.array(per_seed["shuffled_brain"])
    r = np.array(per_seed["random_matrix"])

    t_bl, p_bl = stats.ttest_rel(a, b)      # brain vs lm-only
    t_bs, p_bs = stats.ttest_rel(a, s)      # brain vs shuffled
    t_br, p_br = stats.ttest_rel(a, r)      # brain vs random
    w_bl = stats.wilcoxon(a, b) if len(a) >= 6 else None
    w_bs = stats.wilcoxon(a, s) if len(a) >= 6 else None

    def cohen_dz(x, y):
        d = x - y
        return float(d.mean() / (d.std(ddof=1) + 1e-12))

    results["ttest_brain_vs_lmonly_t"] = float(t_bl)
    results["ttest_brain_vs_lmonly_p"] = float(p_bl)
    results["cohen_dz_brain_vs_lmonly"] = cohen_dz(a, b)
    results["ttest_brain_vs_shuffled_t"] = float(t_bs)
    results["ttest_brain_vs_shuffled_p"] = float(p_bs)
    results["cohen_dz_brain_vs_shuffled"] = cohen_dz(a, s)
    results["ttest_brain_vs_random_t"] = float(t_br)
    results["ttest_brain_vs_random_p"] = float(p_br)
    results["cohen_dz_brain_vs_random"] = cohen_dz(a, r)
    if w_bl is not None:
        results["wilcoxon_brain_vs_lmonly_p"] = float(w_bl.pvalue)
    if w_bs is not None:
        results["wilcoxon_brain_vs_shuffled_p"] = float(w_bs.pvalue)

    # Aggregate per-config means for brain_align and lm_only
    def agg_bycfg(list_of_dicts):
        keys = list_of_dicts[0].keys()
        return {k: float(np.mean([d[k] for d in list_of_dicts])) for k in keys}

    results["brain_align_blimp_by_config"] = agg_bycfg(per_seed_bycfg["brain_align"])
    results["lm_only_blimp_by_config"] = agg_bycfg(per_seed_bycfg["lm_only"])

    # ============================================================
    # PART 1b: alignment-weight (lambda) sweep for brain_align
    # Separates "no benefit" from "wrong lambda", and reports the CKA reached
    # at each weight so a reader can see the loss was actually doing something.
    # Skipped for scale-generality models (EXP_MAIN_ONLY=1): those only need the
    # main 4-condition comparison to show the null holds at a larger size.
    # ============================================================
    lambda_sweep = {}
    for lam in ([] if MAIN_ONLY else CONFIG["lambda_sweep"]):
        accs, ckas = [], []
        for seed in seeds:
            brain_true, _ = prepare_brain_targets(
                responses, CONFIG["brain_target_rank"], seed)
            brain_map = {
                "true": brain_true,
                "shuffled": shuffled_targets(brain_true, seed),
                "random": random_matrix_targets(brain_true, seed),
            }
            print(f"\n=== lambda-sweep lam={lam} seed={seed} brain_align ===")
            model, tok = finetune("brain_align", CONFIG["default_align_layer"],
                                  seed, sentences, brain_map, lam_override=lam)
            acc, _ = blimp_accuracy(model, tok, blimp_pairs)
            cka = layer_cka_to_target(model, tok, sentences, brain_true,
                                      CONFIG["default_align_layer"])
            print(f"lam={lam} BLiMP={acc:.4f} CKA={cka:.4f}")
            accs.append(acc)
            ckas.append(cka)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        m, h = ci95(accs)
        lambda_sweep[str(lam)] = {
            "blimp_mean": m, "blimp_ci95": h, "blimp_per_seed": accs,
            "cka_mean": float(np.mean(ckas)), "cka_per_seed": ckas,
        }
    results["lambda_sweep"] = lambda_sweep

    # ============================================================
    # PART 2: layer ablation (brain_align only)
    # ============================================================
    layer_results = {}  # layer -> per-seed accs
    for layer in ([] if MAIN_ONLY else CONFIG["align_layers_ablation"]):
        if layer == CONFIG["default_align_layer"]:
            # reuse already-computed brain_align results
            layer_results[layer] = list(per_seed["brain_align"])
            continue
        accs = []
        for seed in seeds:
            brain_true, rank = prepare_brain_targets(
                responses, CONFIG["brain_target_rank"], seed
            )
            brain_map = {
                "true": brain_true,
                "shuffled": shuffled_targets(brain_true, seed),
                "random": random_matrix_targets(brain_true, seed),
            }
            print(f"\n=== ablation layer={layer} seed={seed} brain_align ===")
            model, tok = finetune("brain_align", layer, seed, sentences, brain_map)
            acc, _ = blimp_accuracy(model, tok, blimp_pairs)
            print(f"BLiMP acc = {acc:.4f}")
            accs.append(acc)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        layer_results[layer] = accs

    for layer, accs in layer_results.items():
        m, h = ci95(accs)
        results[f"layer{layer}_brain_blimp_per_seed"] = accs
        results[f"layer{layer}_brain_blimp_mean"] = m
        results[f"layer{layer}_brain_blimp_ci95"] = h

    # ============================================================
    # FIGURES
    # ============================================================
    # Fig 1: BLiMP by condition (bar + 95% CI)
    fig, ax = plt.subplots(figsize=(6, 4))
    conds = CONFIG["conditions"]
    means = [results[f"{c}_blimp_mean"] for c in conds]
    cis = [results[f"{c}_blimp_ci95"] for c in conds]
    labels = ["LM-only", "Brain-align", "Shuffled", "Random-mat"]
    x = np.arange(len(conds))
    ax.bar(x, means, yerr=cis, capsize=5,
           color=["#888888", "#2166ac", "#b2182b", "#f4a582"])
    for xi, c in zip(x, conds):
        for v in results[f"{c}_blimp_per_seed"]:
            ax.plot(xi, v, "k.", alpha=0.5)
    ax.axhline(0.5, ls="--", color="gray", lw=1, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("BLiMP accuracy (OOD syntax)")
    ax.set_title(f"BLiMP by condition (layer {CONFIG['default_align_layer']}, "
                 f"n={len(seeds)} seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/blimp_by_condition.pdf")
    plt.close(fig)

    # Fig 2: layer ablation (only when the ablation was run)
    if layer_results:
        fig, ax = plt.subplots(figsize=(6, 4))
        layers = sorted(layer_results.keys())
        lm = [np.mean(layer_results[l]) for l in layers]
        lc = [ci95(layer_results[l])[1] for l in layers]
        ax.errorbar(layers, lm, yerr=lc, marker="o", capsize=5, color="#2166ac",
                    label="brain-align")
        ax.axhline(results["lm_only_blimp_mean"], ls="--", color="#888888",
                   label="LM-only")
        ax.set_xlabel("Aligned hidden-layer index")
        ax.set_ylabel("BLiMP accuracy")
        ax.set_title("Layer ablation (brain-align)")
        ax.legend()
        fig.tight_layout()
        fig.savefig("figures/layer_ablation.pdf")
        plt.close(fig)

    # Fig 3: paired per-seed lines LM-only vs brain-align
    fig, ax = plt.subplots(figsize=(5, 4))
    for i in range(len(seeds)):
        ax.plot([0, 1], [b[i], a[i]], "-o", color="#2166ac", alpha=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["LM-only", "Brain-align"])
    ax.set_ylabel("BLiMP accuracy")
    ax.set_title(f"Paired seeds (p={results['ttest_brain_vs_lmonly_p']:.3g})")
    fig.tight_layout()
    fig.savefig("figures/blimp_paired_seeds.pdf")
    plt.close(fig)

    # ============================================================
    # Save results
    # ============================================================
    # verdict flags (for the paper narrative) — computed, not hardcoded
    results["hypothesis_supported_vs_lmonly"] = bool(
        (a.mean() > b.mean()) and (p_bl < 0.05)
    )
    results["control_vanishes_vs_shuffled"] = bool(
        (a.mean() > s.mean()) and (p_bs < 0.05)
    )
    results["ablation_vs_random"] = bool(
        (a.mean() > r.mean()) and (p_br < 0.05)
    )
    results["shuffled_minus_lmonly"] = float(s.mean() - b.mean())
    results["random_minus_lmonly"] = float(r.mean() - b.mean())

    # ---- TOST equivalence bound on brain_align vs lm_only (90% CI on the
    #      paired difference), computed from whatever seed count this run used.
    d_paired = a - b
    nseed = len(d_paired)
    if nseed >= 2:
        se = d_paired.std(ddof=1) / math.sqrt(nseed)
        tcrit90 = stats.t.ppf(0.95, nseed - 1)
        results["tost_diff_mean"] = float(d_paired.mean())
        results["tost_ci90_low"] = float(d_paired.mean() - tcrit90 * se)
        results["tost_ci90_high"] = float(d_paired.mean() + tcrit90 * se)
        # smallest equivalence margin at which TOST would pass (both one-sided)
        results["tost_upper_bound_on_benefit"] = float(d_paired.mean() + tcrit90 * se)

    # ---- Attach the fMRI-target noise ceiling if available ----
    if ceiling is not None:
        results["noise_ceiling"] = ceiling

    out_name = f"results_{model_tag}.json"
    with open(out_name, "w") as f:
        json.dump(results, f, indent=2)
    # also write the canonical results.json for the headline model
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_name}")

    print("\n" + json.dumps(results))


if __name__ == "__main__":
    main()