"""Separates alignment-in-general from brain-specific structure in the brain-alignment
manipulation check.

The brain-aligned CKA to the fMRI target is not by itself the quantity the
downstream null concerns, because the geometry-broken controls raise CKA toward
the same target. This reports every condition at every scale, the brain-specific
increment over each control with a paired sign-flip test, and the share of the
brain-aligned CKA that the broken targets recover on their own. Conditions share
optimization, initialization and data ordering, so seeds are paired. Reads the
stored results only; no model or GPU needed. See README.md for usage.
"""


from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np

CONDS = ["lm_only", "brain_align", "shuffled_brain", "random_matrix"]
NICE = {
    "lm_only": "LM-only",
    "brain_align": "brain",
    "shuffled_brain": "shuffled",
    "random_matrix": "random",
}


def signflip(diff, draws=20000, seed=0):
    """Paired sign-flip permutation test on per-seed differences."""
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, float)
    obs = diff.mean()
    null = (rng.choice([-1.0, 1.0], size=(draws, len(diff))) * diff).mean(axis=1)
    return obs, float((np.abs(null) >= abs(obs) - 1e-15).mean())


def boot_ci(diff, draws=20000, seed=0):
    """Percentile bootstrap CI on the mean paired difference."""
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, float)
    m = diff[rng.integers(0, len(diff), size=(draws, len(diff)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--scales", nargs="+", default=["160m", "410m", "1.4b"])
    args = ap.parse_args()
    R = Path(args.results_dir)

    ceiling = json.loads(
        (R / "noise_ceiling.json").read_text())["target_cka_ceiling"]
    print(f"subject-split noise ceiling (target CKA): {ceiling:.3f}\n")

    # --- levels ---------------------------------------------------------------
    print("CKA by condition and scale (share of ceiling):")
    store = {}
    for m in args.scales:
        d = json.loads((R / f"results_pythia-{m}.json").read_text())
        store[m] = d
        cells = "  ".join(
            f"{NICE[c]} {d[f'{c}_cka_mean']:.3f} "
            f"({100 * d[f'{c}_cka_mean'] / ceiling:.0f}%)" for c in CONDS)
        print(f"  {m:6s} {cells}")

    # --- brain-specific increment ---------------------------------------------
    print("\nbrain-specific increment, paired over shared seeds:")
    for m in args.scales:
        d = store[m]
        b = np.array(d["brain_align_cka_per_seed"], float)
        print(f"  {m} (n={len(b)} seeds)")
        for c in ["shuffled_brain", "random_matrix"]:
            diff = b - np.array(d[f"{c}_cka_per_seed"], float)
            obs, p = signflip(diff)
            lo, hi = boot_ci(diff)
            print(f"    vs {NICE[c]:9s} {obs:+.4f} (95% CI {lo:+.4f}, "
                  f"{hi:+.4f})  p = {p:.4f}  "
                  f"= {100 * obs / ceiling:.1f}% of ceiling")

    # --- how much a broken target already buys --------------------------------
    print("\nshare of the brain-aligned CKA recovered by broken targets:")
    for m in args.scales:
        d = store[m]
        b, lm = d["brain_align_cka_mean"], d["lm_only_cka_mean"]
        for c in ["shuffled_brain", "random_matrix"]:
            v = d[f"{c}_cka_mean"]
            print(f"  {m:6s} {NICE[c]:9s} {100 * v / b:5.1f}% of the level, "
                  f"{100 * (v - lm) / (b - lm):5.1f}% of the gain over LM-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
