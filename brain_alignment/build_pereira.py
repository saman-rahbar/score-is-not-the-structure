#!/usr/bin/env python
"""
build_pereira.py  --  LOGIN-NODE ONLY (needs internet).

Produce the pereira_cache.npz that experiment_full.py expects:
  sentences : object array of stimulus sentence strings, shape [N]
  responses : float32 [N, V] language-network fMRI responses (one row/sentence)

Source: Pereira et al. (2018), via the Brain-Score language package, which
ships the assembly with language-network responses already extracted.

Usage (a cluster login node with internet access):
  pip install brainscore_language
  # 1) verify structure first (no file written):
  python build_pereira.py --inspect
  # 2) build the cache:
  python build_pereira.py --out $SCRATCH/pereira_cache.npz
  # 3) point the experiment at it:
  export PEREIRA_NPZ=$SCRATCH/pereira_cache.npz

If --inspect shows different coord names than this script guesses, paste its
output back and the detection below can be adjusted.
"""
import argparse
import sys
import numpy as np
import pandas as pd

# Pereira2018 has two sentence experiments (627 sentences total). Both assemblies
# are already the LANGUAGE-network file (assy_Pereira2018_language.nc), so every
# neuroid is a language voxel -- no ROI filtering needed.
DATA_IDS = ["Pereira2018.243sentences", "Pereira2018.384sentences"]


def _load_assembly(data_id):
    """Load a Brain-Score data assembly, tolerant of API differences."""
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


def _sentences(a):
    """Pull the sentence strings out of the presentation index (a pandas
    MultiIndex whose levels include the sentence text), or a flat string coord."""
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
    # flat coord fallback: longest-string presentation coord
    best, best_len = None, -1.0
    for name, coord in a.coords.items():
        if coord.dims == (pres,):
            vals = np.asarray(coord.values)
            if vals.dtype.kind in ("U", "S", "O"):
                avg = float(np.mean([len(str(v)) for v in vals[:20]]))
                if avg > best_len:
                    best, best_len = name, avg
    return [str(s) for s in np.asarray(a.coords[best].values)] if best else None


def _neuroid_ids(a):
    """A stable per-voxel id (full index tuple), used to intersect the voxel
    sets of the two experiments (same subjects -> overlapping voxels)."""
    neuro = _neuroid_dim(a)
    idx = a.indexes.get(neuro, None)
    if isinstance(idx, pd.MultiIndex):
        return np.array(["|".join(map(str, t)) for t in idx], dtype=object)
    return np.asarray(a.coords[neuro].values).astype(object)


def _inspect(a):
    print("  DIMS:", a.dims, "SHAPE:", tuple(a.shape))
    for n, c in a.coords.items():
        vals = np.asarray(c.values)
        print(f"    coord {n!r} dims={c.dims} dtype={vals.dtype} "
              f"n_unique={len(np.unique(vals))} sample={vals[:3]}")


def build(out_path, inspect=False):
    assemblies = []
    for data_id in DATA_IDS:
        try:
            a = _load_assembly(data_id)
            print(f"[ok] {data_id}: dims={a.dims} shape={tuple(a.shape)}")
            assemblies.append(a)
        except Exception as e:
            print(f"[warn] skipping {data_id}: {e}")
    if not assemblies:
        sys.exit("No Pereira assemblies loaded. Is brainscore_language installed "
                 "and does the login node have internet?")

    if inspect:
        for a in assemblies:
            _inspect(a)
        print("\nInspect only -- no file written. If the sentence/language coords "
              "above are not auto-detected below, share this output.")
        return

    per = []   # (sentences, values[N,V], neuroid_ids[V]) per experiment
    for a in assemblies:
        pres = _presentation_dim(a)
        sents = _sentences(a)
        if sents is None:
            sys.exit("Could not find sentence text. Re-run --inspect and share it.")
        vals = a.values
        if a.dims[0] != pres:                 # orient as [presentation, neuroid]
            vals = vals.T
        per.append((sents, np.asarray(vals, dtype=np.float32), _neuroid_ids(a)))
        print(f"  {len(sents)} sentences x {vals.shape[1]} language voxels")

    if len(per) == 1:
        sentences, responses = per[0][0], per[0][1]
    else:
        # Combine experiments on their COMMON voxels -> full 627 sentences.
        common = set(per[0][2])
        for _, _, ids in per[1:]:
            common &= set(ids)
        common = sorted(common)
        if len(common) >= 128:                # enough voxels to be useful
            sentences, blocks = [], []
            for sents, vals, ids in per:
                pos = {vid: i for i, vid in enumerate(ids)}
                cols = [pos[c] for c in common]
                blocks.append(vals[:, cols])
                sentences += sents
            responses = np.concatenate(blocks, axis=0)
            print(f"  combined on {len(common)} common voxels -> "
                  f"{len(sentences)} sentences")
        else:                                 # too little overlap: use the larger
            big = max(per, key=lambda p: p[1].shape[0])
            sentences, responses = big[0], big[1]
            print(f"  [warn] only {len(common)} common voxels; using the "
                  f"{responses.shape[0]}-sentence experiment alone")

    # one row per unique sentence (mean over repeats), drop NaN voxels
    uniq = {}
    for s, r in zip(sentences, responses):
        uniq.setdefault(s, []).append(r)
    sentences = list(uniq.keys())
    responses = np.stack([np.mean(uniq[s], axis=0) for s in sentences]).astype(np.float32)
    good = ~np.isnan(responses).any(axis=0)
    responses = responses[:, good]

    np.savez(out_path,
             sentences=np.array(sentences, dtype=object),
             responses=responses)
    print(f"[done] wrote {out_path}: {len(sentences)} sentences "
          f"x {responses.shape[1]} language voxels")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pereira_cache.npz")
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()
    build(args.out, inspect=args.inspect)
