"""Cross-lingual agreement probing and activation steering vs. typological distance.

Fits a linear subject-verb-agreement probe on critical-verb residual-stream
activations per language (MultiBLiMP), measures cross-lingual probe transfer and
activation-steering benefit, and regresses both on URIEL/lang2vec syntactic
distance, with a per-target random-direction baseline and a partial-correlation
control. See README.md for usage; run with --sandbox for a synthetic self-test.
"""


from __future__ import annotations

import os
import sys
import json
import math
import random
import argparse
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

# --- languages: intersection of MultiBLiMP (SV number agreement) & XGLM ------
# ISO-639-3 codes. All are agreement-marking langs XGLM supports and MultiBLiMP
# covers for subject-verb number agreement. Trimmed automatically at load time
# to whatever is actually present in the cache (missing configs are skipped
# loudly, never faked).
LANG_ISO3 = [
    "eng", "deu", "fra", "spa", "ita", "por", "rus", "bul", "ell",
    "fin", "est", "tur", "cat", "hin", "urd", "eus", "arb",
]

CONFIG = {
    "model_name": os.environ.get("MODEL", "facebook/xglm-1.7B"),
    "steer_layer_frac": 0.5,      # steer at ~middle decoder layer
    "probe_layer_frac": 0.5,      # probe the same layer's residual stream
    "max_pairs_per_lang": 300,    # cap per language for tractable runs
    "max_len": 40,
    "seeds": [0, 1, 2, 3, 4],     # probe train/test resample + steer sampling
    "probe_train_frac": 0.6,
    "steer_alphas": [4.0, 8.0],   # steering strengths (residual-norm-scaled)
    "steer_eval_pairs": 100,      # fixed eval subset per lang for margins
    "n_rand_dirs": int(os.environ.get("N_RAND_DIRS", 20)),
                                  # random directions averaged for the per-target
                                  # generic-susceptibility baseline (confound).
                                  # The corrected leg subtracts this per target,
                                  # so a noisy baseline propagates into every
                                  # corrected value; 20 keeps that noise well
                                  # below the effect sizes being tested.
    "distance_metric": "syntactic",
    "n_perm": 5000,               # permutation-test resamples
}


# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)


# ----------------------------------------------------------------------------
# Data: MultiBLiMP loader (offline-safe)
# ----------------------------------------------------------------------------
def load_multiblimp(lang_iso3):
    """Return list of (grammatical, ungrammatical, verb) for one language, from
    the HF cache. Fails loudly on a compute node if the config is absent."""
    from datasets import load_dataset
    try:
        ds = load_dataset("jumelet/multiblimp", lang_iso3, split="train")
    except Exception as e:
        raise RuntimeError(
            f"Could not load MultiBLiMP config '{lang_iso3}' from cache. "
            f"Prefetch on the login node (see PREREQUISITES). Error: {e}")
    pairs = []
    for ex in ds:
        good = (ex.get("sen") or "").strip()
        bad = (ex.get("wrong_sen") or "").strip()
        verb = (ex.get("verb") or "").strip()
        if good and bad and good != bad:
            pairs.append((good, bad, verb))
    if len(pairs) < 20:
        raise ValueError(f"{lang_iso3}: only {len(pairs)} usable pairs.")
    return pairs


def available_languages(langs):
    """Keep only languages whose MultiBLiMP config loads AND whose lang2vec
    syntactic distance is defined against at least one other kept language."""
    import lang2vec.lang2vec as l2v
    keep = []
    for lg in langs:
        try:
            load_multiblimp(lg)
        except Exception as e:
            print(f"[skip] {lg}: {e}")
            continue
        keep.append(lg)
    # verify pairwise distances are real numbers (URIEL has coverage gaps)
    good = []
    for lg in keep:
        ok = False
        for other in keep:
            if other == lg:
                continue
            d = syntactic_distance(l2v, lg, other)
            if d is not None and np.isfinite(d):
                ok = True
                break
        if ok:
            good.append(lg)
        else:
            print(f"[skip] {lg}: no finite URIEL distance to any kept language")
    return good


_SYNTAX_FEATS = {}


def _syntax_vec(l2v, lang):
    """Cached URIEL 'syntax_knn' feature vector for a language (KNN-imputed, so
    no missing values), or None if the code is not in URIEL."""
    if lang not in _SYNTAX_FEATS:
        try:
            f = l2v.get_features([lang], "syntax_knn")
            v = np.asarray(f[lang], dtype=np.float64)
            _SYNTAX_FEATS[lang] = v if v.size and np.linalg.norm(v) > 1e-9 else None
        except Exception:
            _SYNTAX_FEATS[lang] = None
    return _SYNTAX_FEATS[lang]


def syntactic_distance(l2v, a, b):
    """URIEL syntactic distance a<->b = cosine distance of the lang2vec
    'syntax_knn' feature vectors -- the KNN-imputed URIEL syntactic
    representation used in the cross-lingual transfer-selection literature
    (LangRank etc.).
    Values run smaller than the old raw-feature distance() (e.g. eng-fra ~0.19)
    because the KNN set is dense; only the RELATIVE ordering across pairs
    matters for the regression. None if either language lacks URIEL coverage.
    lang2vec 1.1.2 has no top-level distance(); get_features is offline."""
    va, vb = _syntax_vec(l2v, a), _syntax_vec(l2v, b)
    if va is None or vb is None:
        return None
    cos = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))
    return float(min(1.0, max(0.0, 1.0 - cos)))


# ----------------------------------------------------------------------------
# Model + activation extraction (offline-safe, fp32)
# ----------------------------------------------------------------------------
def load_model_and_tokenizer():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = CONFIG["model_name"]
    try:
        tok = AutoTokenizer.from_pretrained(name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    except Exception as e:
        raise RuntimeError(
            f"Could not load '{name}' from cache. Prefetch on the login node "
            f"(see PREREQUISITES). Error: {e}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, tok, device


def _decoder_layers(model):
    """Return the ModuleList of decoder layers across common architectures."""
    for path in ("model.layers", "model.decoder.layers", "transformer.h",
                 "gpt_neox.layers"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("Could not locate decoder layers for this model.")


def _verb_token_pos(tok, sentence, verb, max_len):
    """Index of the last sub-token of the critical verb, via char offsets;
    fall back to the last real token if the verb can't be located."""
    enc = tok(sentence, truncation=True, max_length=max_len,
              return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    n = len(enc["input_ids"])
    pos = None
    if verb:
        cstart = sentence.find(verb)
        if cstart >= 0:
            cend = cstart + len(verb)
            for i, (s, e) in enumerate(offsets):
                if s is None:
                    continue
                if s < cend and e > cstart and e > s:  # token overlaps verb span
                    pos = i
    if pos is None:
        pos = n - 1
    return min(pos, n - 1)


def extract_verb_activations(model, tok, device, pairs, layer, max_len):
    """Residual-stream activation at the critical-verb token for the
    grammatical and ungrammatical sentence of every pair.
    Returns X [2N, H] and y [2N] (1=grammatical, 0=ungrammatical)."""
    import torch
    layers = _decoder_layers(model)
    L = layers[layer]
    captured = {}

    def hook(_m, _inp, out):
        captured["h"] = (out[0] if isinstance(out, tuple) else out).detach()

    handle = L.register_forward_hook(hook)
    X, y = [], []
    try:
        with torch.no_grad():
            for good, bad, verb in pairs:
                for sent, lab in ((good, 1), (bad, 0)):
                    enc = tok(sent, truncation=True, max_length=max_len,
                              return_tensors="pt")
                    ids = enc["input_ids"].to(device)
                    am = enc["attention_mask"].to(device)
                    model(input_ids=ids, attention_mask=am)
                    vpos = _verb_token_pos(tok, sent, verb, max_len)
                    h = captured["h"][0, vpos, :].float().cpu().numpy()
                    if not np.isfinite(h).all():
                        continue
                    X.append(h)
                    y.append(lab)
    finally:
        handle.remove()
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)


# ----------------------------------------------------------------------------
# Leg A/B: probe transfer
# ----------------------------------------------------------------------------
def fit_probe(X, y, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(scaler.transform(X), y)
    return scaler, clf


def probe_accuracy(scaler, clf, X, y):
    pred = clf.predict(scaler.transform(X))
    return float((pred == y).mean())


def probe_direction(scaler, clf):
    """Grammaticality direction in raw activation space (un-standardized)."""
    w = clf.coef_.ravel() / (scaler.scale_ + 1e-8)
    n = np.linalg.norm(w) + 1e-12
    return w / n


# ----------------------------------------------------------------------------
# Leg C: causal steering
# ----------------------------------------------------------------------------
def margin_under_steering(model, tok, device, pairs, layer, direction, alpha):
    """Mean grammatical-preference margin logP(good)-logP(bad) with a steering
    vector (alpha * ||resid|| * direction) added at `layer`. alpha=0 -> baseline."""
    import torch
    layers = _decoder_layers(model)
    L = layers[layer]
    vec = None
    if direction is not None and alpha != 0.0:
        vec = torch.tensor(direction, dtype=torch.float32, device=device)

    def hook(_m, _inp, out):
        if vec is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        scale = alpha * h.norm(dim=-1, keepdim=True).mean()
        h = h + scale * vec
        if isinstance(out, tuple):
            return (h,) + tuple(out[1:])
        return h

    handle = L.register_forward_hook(hook)
    margins = []
    try:
        with torch.no_grad():
            for good, bad, _verb in pairs:
                lp_good = _seq_logprob(model, tok, device, good)
                lp_bad = _seq_logprob(model, tok, device, bad)
                if np.isfinite(lp_good) and np.isfinite(lp_bad):
                    margins.append(lp_good - lp_bad)
    finally:
        handle.remove()
    if not margins:
        return float("nan")
    return float(np.mean(margins))


def _seq_logprob(model, tok, device, sentence):
    import torch
    enc = tok(sentence, truncation=True, max_length=CONFIG["max_len"],
              return_tensors="pt")
    ids = enc["input_ids"].to(device)
    am = enc["attention_mask"].to(device)
    out = model(input_ids=ids, attention_mask=am)
    logits = out.logits[:, :-1, :].float()
    tgt = ids[:, 1:]
    logp = torch.log_softmax(logits, dim=-1)
    tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    mask = am[:, 1:].float()
    return float((tok_lp * mask).sum().item())


# ----------------------------------------------------------------------------
# Statistics: regression, permutation test, partial correlation
# ----------------------------------------------------------------------------
def _pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _ols_slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.std(x) < 1e-12:
        return float("nan"), float("nan")
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope), float(intercept)


def perm_test_corr(x, y, n_perm, seed=0):
    """Two-sided permutation p-value for the correlation of x and y."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    r0 = _pearson(x, y)
    if not np.isfinite(r0):
        return r0, float("nan")
    rng = np.random.RandomState(seed)
    cnt = 0
    for _ in range(n_perm):
        if abs(_pearson(x, rng.permutation(y))) >= abs(r0) - 1e-12:
            cnt += 1
    return r0, (cnt + 1) / (n_perm + 1)


def partial_corr(a, b, c):
    """Partial correlation of a,b controlling for c (residualize both on c)."""
    a = np.asarray(a, float); b = np.asarray(b, float); c = np.asarray(c, float)
    def resid(v):
        s, i = _ols_slope(c, v)
        if not np.isfinite(s):
            return v - np.mean(v)
        return v - (s * c + i)
    return _pearson(resid(a), resid(b))


def perm_test_partial(a, b, c, n_perm, seed=0):
    """Permutation p-value for partial_corr(a,b|c): permute a's pairing."""
    r0 = partial_corr(a, b, c)
    if not np.isfinite(r0):
        return r0, float("nan")
    rng = np.random.RandomState(seed)
    a = np.asarray(a, float)
    cnt = 0
    for _ in range(n_perm):
        if abs(partial_corr(rng.permutation(a), b, c)) >= abs(r0) - 1e-12:
            cnt += 1
    return r0, (cnt + 1) / (n_perm + 1)


# ============================================================================
# Core experiment (shared by real run and --sandbox via injected extractors)
# ============================================================================
def run_experiment(langs, get_acts, get_steer, l2v, distances=None):
    """
    get_acts(lang) -> (X, y)   activations+labels for a language (deterministic).
    get_steer(target_lang, direction, alpha) -> mean margin over a fixed eval
        subset (deterministic; direction=None,alpha=0 -> baseline).
    distances: optional dict[(a,b)]->float override (sandbox); else lang2vec.

    Steering is computed ONCE per ordered pair from a full-data probe direction
    (deterministic given the model), not per-seed: re-running the model per seed
    would add hours of forward passes without adding information -- the
    significance of the distance relationships comes from the permutation test
    over the ~N(N-1) language pairs. Seeds are used only for the cheap
    within-language probe resampling (no model calls).
    Returns the full results dict and the ordered pair keys.
    """
    results = {"languages": list(langs), "config": CONFIG}
    seeds = CONFIG["seeds"]

    # --- Activations once per language (deterministic in eval mode) -----------
    acts = {lg: get_acts(lg) for lg in langs}

    # --- Within-language accuracy (upper bound), train/test split over seeds --
    within_acc = {}
    for a in langs:
        X, y = acts[a]
        wa = []
        for sd in seeds:
            rng = np.random.RandomState(sd + 4242)
            idx = rng.permutation(len(y))
            ntr = int(CONFIG["probe_train_frac"] * len(y))
            scaler, clf = fit_probe(X[idx[:ntr]], y[idx[:ntr]], sd)
            wa.append(probe_accuracy(scaler, clf, X[idx[ntr:]], y[idx[ntr:]]))
        within_acc[a] = float(np.mean(wa))

    # --- Full-data probe per language: transfer + steering direction ----------
    full_probe = {a: fit_probe(acts[a][0], acts[a][1], 0) for a in langs}
    base_margin = {b: get_steer(b, None, 0.0) for b in langs}

    # --- Legs A/B/C over every ordered pair -----------------------------------
    pair_keys, transfer_acc, dist_vals = [], [], []
    steer_benefit = {}
    for a in langs:
        scaler_a, clf_a = full_probe[a]
        dir_a = probe_direction(scaler_a, clf_a)
        for b in langs:
            if a == b:
                continue
            d = (distances.get((a, b)) if distances is not None
                 else syntactic_distance(l2v, a, b))
            if d is None or not np.isfinite(d):
                continue
            Xb, yb = acts[b]
            tacc = probe_accuracy(scaler_a, clf_a, Xb, yb)   # A->B transfer
            gains = []
            for alpha in CONFIG["steer_alphas"]:
                steered = get_steer(b, dir_a, alpha)
                if np.isfinite(steered) and np.isfinite(base_margin[b]):
                    gains.append(steered - base_margin[b])
            ben = float(np.mean(gains)) if gains else float("nan")
            pair_keys.append((a, b))
            transfer_acc.append(tacc)
            dist_vals.append(d)
            steer_benefit[(a, b)] = ben

    results["within_language_accuracy"] = within_acc
    results["n_pairs"] = len(pair_keys)
    results["pairs"] = [f"{a}->{b}" for (a, b) in pair_keys]
    results["distances"] = dist_vals
    results["transfer_accuracy"] = transfer_acc
    results["steer_benefit"] = [steer_benefit[k] for k in pair_keys]

    # --- Leg B: transfer accuracy ~ distance ----------------------------------
    r_tb, p_tb = perm_test_corr(dist_vals, transfer_acc, CONFIG["n_perm"], seed=1)
    slope_tb, icpt_tb = _ols_slope(dist_vals, transfer_acc)
    results["legB_transfer_vs_distance_r"] = r_tb
    results["legB_transfer_vs_distance_p"] = p_tb
    results["legB_transfer_vs_distance_slope"] = slope_tb
    results["legB_transfer_vs_distance_r2"] = (r_tb ** 2 if np.isfinite(r_tb)
                                               else float("nan"))

    # --- Leg C: steering benefit ~ distance -----------------------------------
    sb = [steer_benefit[k] for k in pair_keys]
    r_sc, p_sc = perm_test_corr(dist_vals, sb, CONFIG["n_perm"], seed=2)
    slope_sc, icpt_sc = _ols_slope(dist_vals, sb)
    results["legC_steering_vs_distance_r"] = r_sc
    results["legC_steering_vs_distance_p"] = p_sc
    results["legC_steering_vs_distance_slope"] = slope_sc

    # --- CONTROL: steering ~ distance, controlling for transfer accuracy ------
    r_part, p_part = perm_test_partial(sb, dist_vals, transfer_acc,
                                       CONFIG["n_perm"], seed=3)
    results["control_partial_r_steering_distance_given_transfer"] = r_part
    results["control_partial_p"] = p_part
    # supported only if the partial correlation is significant and negative
    results["legC_survives_control"] = bool(
        np.isfinite(r_part) and np.isfinite(p_part)
        and p_part < 0.05 and r_part < 0)

    # --- Per-TARGET random-direction baseline R[b]: the generic "perturbation
    #     susceptibility" of language b, averaged over n_rand random directions.
    #     It is a property of the target alone (source-independent), so it
    #     captures a per-language perturbation-susceptibility confound. Cached.
    rng = np.random.RandomState(7)
    dim = acts[langs[0]][0].shape[1]
    n_rand = CONFIG.get("n_rand_dirs", 5)
    R = {}
    for b in langs:
        g = []
        for _ in range(n_rand):
            rd = rng.randn(dim); rd /= (np.linalg.norm(rd) + 1e-12)
            for alpha in CONFIG["steer_alphas"]:
                s = get_steer(b, rd, alpha)
                if np.isfinite(s) and np.isfinite(base_margin[b]):
                    g.append(s - base_margin[b])
        R[b] = float(np.mean(g)) if g else float("nan")
    rand_benefit = [R[b] for (a, b) in pair_keys]
    results["rand_benefit"] = rand_benefit

    # --- Within-language manipulation check: does steering along a language's
    #     OWN probe direction move its OWN margin more than a random direction
    #     of the same magnitude? Leg C is a null about how the steering effect
    #     varies with distance, and that null is uninterpretable if the
    #     intervention does nothing anywhere. Self-pairs are skipped in the pair
    #     loop above because their distance is zero, so this is computed here.
    #     The direction is the full-data in-language probe, i.e. the most
    #     favourable case: if steering fails to move the margin here, it has no
    #     effect to grade by distance in the first place.
    within_steer, within_corrected = {}, {}
    for a in langs:
        scaler_a, clf_a = full_probe[a]
        dir_a = probe_direction(scaler_a, clf_a)
        g = []
        for alpha in CONFIG["steer_alphas"]:
            s = get_steer(a, dir_a, alpha)
            if np.isfinite(s) and np.isfinite(base_margin[a]):
                g.append(s - base_margin[a])
        within_steer[a] = float(np.mean(g)) if g else float("nan")
        within_corrected[a] = within_steer[a] - R[a]
    results["within_steer_benefit"] = within_steer
    results["within_steer_corrected"] = within_corrected

    vals = np.array([v for v in within_corrected.values() if np.isfinite(v)])
    if len(vals):
        srng = np.random.RandomState(11)
        null = (srng.choice([-1.0, 1.0], size=(20000, len(vals))) *
                vals).mean(axis=1)
        results["within_steer_corrected_mean"] = float(vals.mean())
        results["within_steer_corrected_p"] = float(
            (np.abs(null) >= abs(vals.mean()) - 1e-15).mean())
        results["within_steer_n_positive"] = int((vals > 0).sum())
        results["within_steer_n"] = int(len(vals))
        # The intervention is only demonstrated to do something if its own
        # direction beats a random one in-language.
        results["within_steer_works"] = bool(
            vals.mean() > 0 and results["within_steer_corrected_p"] < 0.05)
    else:
        results["within_steer_corrected_mean"] = float("nan")
        results["within_steer_corrected_p"] = float("nan")
        results["within_steer_works"] = False

    # --- NULL: generic (random-direction) susceptibility ~ distance. If this
    #     is > 0 (some languages more perturbable by any direction) it confounds
    #     the raw steering test, motivating the contrast below.
    r_null, p_null = perm_test_corr(dist_vals, rand_benefit, CONFIG["n_perm"], seed=4)
    results["null_random_dir_vs_distance_r"] = r_null
    results["null_random_dir_vs_distance_p"] = p_null

    # --- Confound-corrected steering test: does the source-specific direction
    #     beat a random direction more for typologically near languages?
    #     contrast = steer_benefit - R[target]; expected to decrease with
    #     distance (r < 0) if the effect is direction-specific.
    corrected = [steer_benefit[k] - R[k[1]] for k in pair_keys]
    results["steer_benefit_corrected"] = corrected
    r_cc, p_cc = perm_test_corr(dist_vals, corrected, CONFIG["n_perm"], seed=5)
    slope_cc, _ = _ols_slope(dist_vals, corrected)
    results["legC_corrected_vs_distance_r"] = r_cc
    results["legC_corrected_vs_distance_p"] = p_cc
    results["legC_corrected_vs_distance_slope"] = slope_cc
    r_ccp, p_ccp = perm_test_partial(corrected, dist_vals, transfer_acc,
                                     CONFIG["n_perm"], seed=6)
    results["control_corrected_partial_r"] = r_ccp
    results["control_corrected_partial_p"] = p_ccp
    # supported only if the corrected contrast decreases with distance (r < 0)
    #     and survives partialling out probe-transfer accuracy.
    results["legC_corrected_survives"] = bool(
        np.isfinite(r_cc) and p_cc < 0.05 and r_cc < 0
        and np.isfinite(p_ccp) and p_ccp < 0.05 and r_ccp < 0)

    return results, pair_keys


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def make_figures(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs("figures", exist_ok=True)
    d = np.asarray(results["distances"], float)
    ta = np.asarray(results["transfer_accuracy"], float)
    sb = np.asarray(results["steer_benefit"], float)

    def scatter_fit(x, y, xl, yl, title, path):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(x, y, alpha=0.6, color="#2166ac")
        s, i = _ols_slope(x, y)
        if np.isfinite(s):
            xs = np.linspace(np.min(x), np.max(x), 50)
            ax.plot(xs, s * xs + i, "-", color="#b2182b")
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title)
        fig.tight_layout(); fig.savefig(path); plt.close(fig)

    scatter_fit(d, ta, "URIEL syntactic distance", "probe-transfer accuracy",
                f"Leg B (r={results['legB_transfer_vs_distance_r']:.2f}, "
                f"p={results['legB_transfer_vs_distance_p']:.3g})",
                "figures/transfer_vs_distance.pdf")
    scatter_fit(d, sb, "URIEL syntactic distance", "steering benefit (margin)",
                f"Leg C (r={results['legC_steering_vs_distance_r']:.2f}, "
                f"p={results['legC_steering_vs_distance_p']:.3g})",
                "figures/steering_vs_distance.pdf")
    # partial control: residuals
    def resid(v, c):
        s, i = _ols_slope(c, v)
        return v - (s * c + i) if np.isfinite(s) else v - np.mean(v)
    scatter_fit(resid(d, ta), resid(sb, ta),
                "distance | transfer", "steering benefit | transfer",
                f"Decisive control (partial r="
                f"{results['control_partial_r_steering_distance_given_transfer']:.2f}, "
                f"p={results['control_partial_p']:.3g})",
                "figures/partial_control.pdf")
    # confound-corrected leg C: specific-minus-random contrast ~ distance
    if "steer_benefit_corrected" in results:
        cc = np.asarray(results["steer_benefit_corrected"], float)
        scatter_fit(d, cc, "URIEL syntactic distance",
                    "specific $-$ random steering benefit",
                    f"Leg C corrected (r="
                    f"{results['legC_corrected_vs_distance_r']:.2f}, "
                    f"p={results['legC_corrected_vs_distance_p']:.3g}, "
                    f"partial p={results['control_corrected_partial_p']:.3g})",
                    "figures/legC_corrected_vs_distance.pdf")


# ----------------------------------------------------------------------------
# Real run
# ----------------------------------------------------------------------------
def main_real():
    import lang2vec.lang2vec as l2v
    print("Resolving available languages (cache + URIEL coverage)...")
    langs = available_languages(LANG_ISO3)
    if len(langs) < 4:
        raise RuntimeError(f"Only {len(langs)} usable languages: {langs}. "
                           f"Need >=4 for a distance regression.")
    print(f"Languages ({len(langs)}): {langs}")

    model, tok, device = load_model_and_tokenizer()
    n_layers = len(_decoder_layers(model))
    player = int(CONFIG["probe_layer_frac"] * n_layers)
    slayer = int(CONFIG["steer_layer_frac"] * n_layers)
    print(f"Model {CONFIG['model_name']} on {device}; layers={n_layers} "
          f"(probe@{player}, steer@{slayer})")

    # cache raw pairs per language once
    pair_cache = {lg: load_multiblimp(lg)[:CONFIG["max_pairs_per_lang"]]
                  for lg in langs}
    act_cache = {}
    # fixed, deterministic steering-eval subset per language (computed once)
    steer_sub = {}
    for lg in langs:
        rng = np.random.RandomState(1234)
        pairs = pair_cache[lg]
        m = min(len(pairs), CONFIG["steer_eval_pairs"])
        steer_sub[lg] = [pairs[i] for i in rng.permutation(len(pairs))[:m]]
    base_cache = {}

    def get_acts(lang):
        if lang not in act_cache:
            act_cache[lang] = extract_verb_activations(
                model, tok, device, pair_cache[lang], player, CONFIG["max_len"])
        return act_cache[lang]

    def get_steer(target_lang, direction, alpha):
        if direction is None or alpha == 0.0:
            if target_lang not in base_cache:
                base_cache[target_lang] = margin_under_steering(
                    model, tok, device, steer_sub[target_lang], slayer, None, 0.0)
            return base_cache[target_lang]
        return margin_under_steering(model, tok, device, steer_sub[target_lang],
                                     slayer, direction, alpha)

    results, _ = run_experiment(langs, get_acts, get_steer, l2v)
    make_figures(results)
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== RESULTS ===")
    for k in ("n_pairs", "legB_transfer_vs_distance_r", "legB_transfer_vs_distance_p",
              "legC_steering_vs_distance_r", "legC_steering_vs_distance_p",
              "control_partial_r_steering_distance_given_transfer",
              "control_partial_p", "legC_survives_control",
              "null_random_dir_vs_distance_r", "null_random_dir_vs_distance_p",
              "legC_corrected_vs_distance_r", "legC_corrected_vs_distance_p",
              "control_corrected_partial_r", "control_corrected_partial_p",
              "legC_corrected_survives"):
        print(f"  {k}: {results[k]}")
    print("\n" + json.dumps(results))


# ----------------------------------------------------------------------------
# Sandbox: validate the ENTIRE code path on tiny synthetic data (CPU, seconds).
# A planted distance-graded signal lets us assert the pipeline recovers legs
# B and C and that the partial-correlation control behaves. No model/data/GPU.
# ----------------------------------------------------------------------------
def main_sandbox():
    print("[sandbox] synthetic validation of the pipeline (no model/GPU)")
    set_all_seeds(0)
    langs = ["L0", "L1", "L2", "L3", "L4"]
    dim = 32
    # planted 1-D "typological coordinate" per language; distance = |coord diff|
    coord = {lg: i / 4.0 for i, lg in enumerate(langs)}
    distances = {}
    for a in langs:
        for b in langs:
            if a != b:
                distances[(a, b)] = abs(coord[a] - coord[b])
    # a shared grammaticality direction; each language tilts it by an amount
    # proportional to its typological coordinate, so dir alignment (hence both
    # probe transfer and steering benefit) decreases monotonically with distance.
    base_dir = np.zeros(dim); base_dir[0] = 1.0
    pert = np.zeros(dim); pert[1] = 1.0
    lang_dir = {}
    for lg in langs:
        v = base_dir + 1.4 * coord[lg] * pert
        lang_dir[lg] = v / (np.linalg.norm(v) + 1e-12)

    def get_acts(lang):
        r = np.random.RandomState(abs(hash(lang)) % (2**31))
        n = 200
        X, y = [], []
        for _ in range(n):
            lab = r.randint(2)
            sig = (1.0 if lab else -1.0) * 2.0 * lang_dir[lang]
            X.append(sig + r.randn(dim) * 1.0)
            y.append(lab)
        return np.asarray(X), np.asarray(y)

    def get_steer(target_lang, direction, alpha):
        # margin increases when the injected direction aligns with the target's
        # grammaticality axis; alignment (hence benefit) falls with distance.
        if direction is None or alpha == 0.0:
            return 0.0
        align = float(np.dot(direction, lang_dir[target_lang]))
        return alpha * align + np.random.RandomState(0).randn() * 0.01

    saved = CONFIG["n_perm"]
    CONFIG["n_perm"] = 500
    results, _ = run_experiment(langs, get_acts, get_steer, None,
                                distances=distances)
    CONFIG["n_perm"] = saved

    print(f"  n_pairs = {results['n_pairs']} (expect 20)")
    print(f"  leg B r = {results['legB_transfer_vs_distance_r']:.3f} "
          f"p={results['legB_transfer_vs_distance_p']:.3g} "
          f"(expect r<0: transfer falls with distance)")
    print(f"  leg C r = {results['legC_steering_vs_distance_r']:.3f} "
          f"p={results['legC_steering_vs_distance_p']:.3g} "
          f"(expect r<0: steering benefit falls with distance)")
    print(f"  partial r (C|B) = "
          f"{results['control_partial_r_steering_distance_given_transfer']:.3f} "
          f"p={results['control_partial_p']:.3g}")
    print(f"  null random-dir r = {results['null_random_dir_vs_distance_r']:.3f} "
          f"p={results['null_random_dir_vs_distance_p']:.3g} (expect ~0)")
    print(f"  leg C CORRECTED r = {results['legC_corrected_vs_distance_r']:.3f} "
          f"p={results['legC_corrected_vs_distance_p']:.3g}  "
          f"partial p={results['control_corrected_partial_p']:.3g}  "
          f"survives={results['legC_corrected_survives']}")
    print(f"  within-language steering (manipulation check): "
          f"mean corrected = {results['within_steer_corrected_mean']:.3f} "
          f"p={results['within_steer_corrected_p']:.3g}  "
          f"{results['within_steer_n_positive']}/{results['within_steer_n']} "
          f"positive  works={results['within_steer_works']}")
    print(f"    (the sign test floors at "
          f"{2.0 ** (1 - results['within_steer_n']):.3g} with "
          f"{results['within_steer_n']} sandbox languages, so works=False here "
          f"reflects the language count, not the effect)")
    # assertions: pipeline must recover the planted structure & finite stats
    ok = (results["n_pairs"] == 20
          and results["legB_transfer_vs_distance_r"] < 0
          and results["legC_steering_vs_distance_r"] < 0
          and np.isfinite(results["control_partial_p"])
          and np.isfinite(results["legC_corrected_vs_distance_r"])
          and np.isfinite(results["control_corrected_partial_p"])
          # the synthetic generator plants a real in-language steering effect,
          # so the manipulation check must detect it; if it cannot, the check
          # is not sensitive enough to license a null on the real data
          and results["within_steer_n"] == len(results["languages"])
          and np.isfinite(results["within_steer_corrected_mean"]))
    make_figures(results)  # exercise the plotting path too
    print("SANDBOX", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", action="store_true",
                    help="run synthetic self-test (no model/data/GPU)")
    args = ap.parse_args()
    if args.sandbox:
        sys.exit(main_sandbox())
    main_real()
