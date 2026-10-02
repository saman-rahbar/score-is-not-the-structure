#!/usr/bin/env python
"""
build_pereira_noiseceiling.py  --  LOGIN-NODE ONLY (needs internet).

Estimate a NOISE CEILING for the fMRI alignment target, so the null in the paper
can be read as evidence about brain geometry rather than about a degraded target.

The metric is chosen to match what the experiment optimizes: linear CKA between
sentence-level representations. We split SUBJECTS into two independent halves,
build each half's [N_sentences, V_half] response matrix, and compute linear CKA
between the two halves' sentence-similarity (Gram) structure. Because linear CKA
compares Gram matrices in SENTENCE space, it is well-defined even though the two
halves contain different voxels. Averaged over many random subject-splits, this
upper-bounds how much CKA any model representation could share with the true
signal: the reliable, subject-general part of the target. We also report an RSA
reliability (correlation of the two halves' RDMs, Spearman-Brown corrected).

Interpretation for the paper: brain_align reaches CKA ~0.34 to the averaged
target; comparing that to this ceiling tells the reader whether the target
carried substantial reliable signal that the model partly captured (a strong,
interpretable null) or whether the target was too noisy to matter.

Usage (a cluster login node with internet access):
  pip install brainscore_language
  python build_pereira_noiseceiling.py --inspect        # check subject coord
  python build_pereira_noiseceiling.py \
      --out $SCRATCH/noise_ceiling.json
  export NOISE_CEILING_JSON=$SCRATCH/noise_ceiling.json
"""
import argparse
import json
import sys
import numpy as np
import pandas as pd

DATA_IDS = ["Pereira2018.243sentences", "Pereira2018.384sentences"]
N_SPLITS = 50   # 6 subjects -> only ~10 distinct 3v3 splits; 50 oversamples them
RANK = 128      # must match experiment_full.CONFIG['brain_target_rank']


def _load_assembly(data_id):
    errs = []
    try:
        from brainscore_language import load_dataset
        return load_dataset(data_id)
    except Exception as e:
        errs.append(f"load_dataset: {e}")
    try:
        from brainscore_language import load_benchmark
        return load_benchmark(f"{data_id}-linear").data
    except Exception as e:
        errs.append(f"load_benchmark(...).data: {e}")
    raise RuntimeError(f"could not load {data_id} -> " + " | ".join(errs))


def _presentation_dim(a):
    return "presentation" if "presentation" in a.dims else a.dims[0]


def _neuroid_dim(a):
    return "neuroid" if "neuroid" in a.dims else a.dims[-1]


def _subject_labels(a):
    """Per-voxel subject id array (length V). The Pereira neuroid axis is a
    pandas MultiIndex with an explicit 'subject' level, so pull that; fall back
    to a plausible small-cardinality level, then to flat coords."""
    neuro = _neuroid_dim(a)
    idx = a.indexes.get(neuro, None)
    if isinstance(idx, pd.MultiIndex):
        for name in idx.names:
            if name and "subject" in str(name).lower():
                return np.asarray(idx.get_level_values(name))
        for name in idx.names:                     # fallback: subject-like level
            vals = idx.get_level_values(name)
            if 2 <= len(pd.unique(vals)) <= 40:
                return np.asarray(vals)
        return None
    # flat-coord fallback (non-MultiIndex assemblies)
    cands = []
    for name, coord in a.coords.items():
        if coord.dims != (neuro,):
            continue
        vals = np.asarray(coord.values)
        cands.append((name, vals, len(np.unique(vals))))
    named = [(n, v, u) for (n, v, u) in cands if "subject" in n.lower() and u > 1]
    if named:
        named.sort(key=lambda t: t[2])
        return named[0][1]
    plaus = [(n, v, u) for (n, v, u) in cands if 2 <= u <= 40]
    if plaus:
        plaus.sort(key=lambda t: t[2])
        return plaus[0][1]
    return None


def _sentences(a):
    pres = _presentation_dim(a)
    idx = a.indexes.get(pres, None)
    if isinstance(idx, pd.MultiIndex):
        best, best_len = None, -1.0
        for name in idx.names:
            vals = idx.get_level_values(name)
            if vals.dtype == object or vals.dtype.kind in ("U", "S"):
                avg = float(np.mean([len(str(v)) for v in vals[:20]]))
                if avg > best_len:
                    best, best_len = name, avg
        if best is not None:
            return [str(s) for s in idx.get_level_values(best)]
    best, best_len = None, -1.0
    for name, coord in a.coords.items():
        if coord.dims == (pres,):
            vals = np.asarray(coord.values)
            if vals.dtype.kind in ("U", "S", "O"):
                avg = float(np.mean([len(str(v)) for v in vals[:20]]))
                if avg > best_len:
                    best, best_len = name, avg
    return [str(s) for s in np.asarray(a.coords[best].values)] if best else None


def linear_cka(X, Y):
    # Compute in SAMPLE space via Gram matrices (N x N), not feature space
    # (V x V): identical result, but O(N^2 V) instead of O(V^2 N) -- essential
    # when V is ~10k voxels. ||X^T Y||_F^2 = <XX^T, YY^T>_F, etc.
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    K = X @ X.T                      # [N, N]
    L = Y @ Y.T                      # [N, N]
    num = float(np.sum(K * L))
    den = float(np.linalg.norm(K) * np.linalg.norm(L))
    return num / den if den > 0 else np.nan


def pca_target(X, rank):
    """Mirror experiment_full.prepare_brain_targets: z-score voxels, PCA to
    `rank`, per-column standardize. Returns the [N, rank] target the model
    actually aligns to, so the split-half CKA of two halves' targets is a
    ceiling directly comparable to the model's brain_align CKA."""
    mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True) + 1e-6
    Xz = (X - mu) / sd
    U, S, _ = np.linalg.svd(Xz, full_matrices=False)
    r = min(rank, S.shape[0])
    T = U[:, :r] * S[:r]
    return (T - T.mean(0, keepdims=True)) / (T.std(0, keepdims=True) + 1e-6)


def rdm_upper(X):
    """Vectorized upper-triangular RDM (1 - Pearson correlation) of sentences."""
    Xz = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)
    C = (Xz @ Xz.T) / X.shape[1]
    iu = np.triu_indices(C.shape[0], k=1)
    return (1.0 - C)[iu]


def orient(a):
    pres = _presentation_dim(a)
    vals = a.values
    if a.dims[0] != pres:
        vals = vals.T
    return np.asarray(vals, dtype=np.float64)


def ceiling_for(a, rng):
    vals = orient(a)                      # [N, V]
    subj = _subject_labels(a)
    if subj is None:
        return None
    sents = _sentences(a)
    # collapse repeated sentences to one row (mean), keep voxel axis
    uniq = {}
    for i, s in enumerate(sents):
        uniq.setdefault(s, []).append(i)
    order = list(uniq.keys())
    rows = np.stack([vals[uniq[s]].mean(0) for s in order])  # [Nuniq, V]
    good = ~np.isnan(rows).any(0)
    rows = rows[:, good]
    subj = np.asarray(subj)[good]
    subjects = np.unique(subj)
    if len(subjects) < 2:
        return None
    ckas, tgt_ckas, rsas = [], [], []
    for _ in range(N_SPLITS):
        perm = rng.permutation(subjects)
        half1 = set(perm[: len(perm) // 2]); half2 = set(perm[len(perm) // 2:])
        if not half1 or not half2:
            continue
        c1 = np.array([s in half1 for s in subj])
        c2 = ~c1
        X1, X2 = rows[:, c1], rows[:, c2]
        if X1.shape[1] < RANK or X2.shape[1] < RANK:
            continue
        ckas.append(linear_cka(X1, X2))                    # raw voxel geometry
        tgt_ckas.append(linear_cka(pca_target(X1, RANK),   # PCA-128 target the
                                   pca_target(X2, RANK)))   # model aligns to
        r1, r2 = rdm_upper(X1), rdm_upper(X2)
        rsas.append(float(np.corrcoef(r1, r2)[0, 1]))
    ckas = [c for c in ckas if np.isfinite(c)]
    tgt_ckas = [c for c in tgt_ckas if np.isfinite(c)]
    rsas = [r for r in rsas if np.isfinite(r)]
    if not ckas:
        return None
    rsa = float(np.mean(rsas)) if rsas else np.nan
    rsa_sb = (2 * rsa / (1 + rsa)) if np.isfinite(rsa) else np.nan
    return {
        "n_subjects": int(len(subjects)),
        "n_sentences": int(rows.shape[0]),
        "n_voxels": int(rows.shape[1]),
        "raw_split_half_cka_mean": float(np.mean(ckas)),
        "raw_split_half_cka_std": float(np.std(ckas)),
        # the number comparable to the model's brain_align CKA:
        "target_split_half_cka_mean": float(np.mean(tgt_ckas)) if tgt_ckas else np.nan,
        "target_split_half_cka_std": float(np.std(tgt_ckas)) if tgt_ckas else np.nan,
        "rsa_split_half": rsa,
        "rsa_spearman_brown": rsa_sb,
    }


def main(out_path, inspect=False):
    rng = np.random.RandomState(0)
    per = {}
    for data_id in DATA_IDS:
        try:
            a = _load_assembly(data_id)
        except Exception as e:
            print(f"[warn] skip {data_id}: {e}")
            continue
        if inspect:
            neuro = _neuroid_dim(a)
            print(f"{data_id}: dims={a.dims} shape={tuple(a.shape)}")
            for n, c in a.coords.items():
                if c.dims == (neuro,):
                    v = np.asarray(c.values)
                    print(f"  neuroid coord {n!r}: n_unique={len(np.unique(v))} "
                          f"sample={v[:3]}")
            continue
        res = ceiling_for(a, rng)
        if res:
            per[data_id] = res
            print(f"[ok] {data_id}: target(PCA-{RANK}) CKA "
                  f"{res['target_split_half_cka_mean']:.3f}"
                  f"±{res['target_split_half_cka_std']:.3f} | "
                  f"raw CKA {res['raw_split_half_cka_mean']:.3f} | "
                  f"RSA(SB) {res['rsa_spearman_brown']:.3f} | "
                  f"{res['n_subjects']} subjects")
    if inspect:
        return
    if not per:
        sys.exit("No subject-resolved assembly; re-run --inspect and share coords.")
    # sentence-weighted means across the two experiments
    tot = sum(v["n_sentences"] for v in per.values())
    def wmean(key):
        return sum(v[key] * v["n_sentences"] for v in per.values()) / tot
    out = {
        "method": f"split-half over subjects; linear CKA in sentence space; "
                  f"{N_SPLITS} random splits; target = z-score + PCA-{RANK} "
                  f"(matches the model's alignment target); RSA Spearman-Brown.",
        # headline: ceiling on the ACTUAL PCA-128 target the model aligns to,
        # directly comparable to brain_align's CKA.
        "target_cka_ceiling": float(wmean("target_split_half_cka_mean")),
        "raw_cka_ceiling": float(wmean("raw_split_half_cka_mean")),
        "rsa_ceiling_spearman_brown": float(wmean("rsa_spearman_brown")),
        "per_experiment": per,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] wrote {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="noise_ceiling.json")
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()
    main(args.out, inspect=args.inspect)
