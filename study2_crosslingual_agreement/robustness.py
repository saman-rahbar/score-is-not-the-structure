"""Robustness checks for the Study 2 correlations: phylogenetic
non-independence, and probe validity.

Re-scores every leg against a language-label permutation (Mantel) instead of a
pair permutation, since the 272 ordered pairs come from 17 languages and URIEL
distances are phylogenetically structured. Also re-runs both legs restricted to
the languages whose within-language probe is above chance, since a probe that
never worked would produce transfer decay for the wrong reason. Reads
results.json only; no model or GPU needed. See README.md for usage.
"""


from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np

# --- guard values: point estimates as published ------------------------------
# Probe transfer and raw steering are deterministic given the model, so these
# must reproduce exactly; a mismatch means the definitions or the inputs moved
# and nothing below can be trusted.
PUBLISHED_EXACT = {
    "legB_r": -0.6573355858369012,
    "legC_raw_r": 0.08086588377207135,
}

# These depend on the per-target random-direction baseline, so they shift with
# N_RAND_DIRS and its seed. Reported for comparison, never used to fail the run.
PUBLISHED_BASELINE_DEPENDENT = {
    "legC_corrected_r": -0.1817944363959412,
    "partial_corrected_r": 0.12628258636153036,
    "null_random_r": 0.22971821603089151,
}


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def residualize(y, z):
    """Residuals of y regressed on [1, z]."""
    y, z = np.asarray(y, float), np.asarray(z, float)
    X = np.column_stack([np.ones(len(z)), z])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_r(x, y, z) -> float:
    """Correlation of x and y with z partialled out of both."""
    return pearson(residualize(x, z), residualize(y, z))


def mantel_p(langs, src, tgt, dist_lookup, y, stat_fn, draws=10000, seed=0):
    """Permutation p from relabelling languages rather than shuffling pairs.

    Relabelling moves a whole language at once, so the 32-pairs-per-language
    dependency that a pair-level permutation ignores is preserved. Returns the
    observed statistic and its two-sided p.
    """
    rng = np.random.default_rng(seed)
    obs = stat_fn(np.array([dist_lookup[(a, b)] for a, b in zip(src, tgt)]), y)
    idx = np.arange(len(langs))
    n_ge = 0
    for _ in range(draws):
        pi = rng.permutation(idx)
        m = {langs[i]: langs[pi[i]] for i in idx}
        d = np.array([dist_lookup[(m[a], m[b])] for a, b in zip(src, tgt)])
        if abs(stat_fn(d, y)) >= abs(obs) - 1e-12:
            n_ge += 1
    return obs, (n_ge + 1) / (draws + 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/results.json")
    ap.add_argument("--threshold", type=float, default=0.60,
                    help="within-language probe accuracy required to treat a "
                         "language's probe as working (default 0.60)")
    ap.add_argument("--draws", type=int, default=10000)
    args = ap.parse_args()

    d = json.loads(Path(args.results).read_text())
    src = np.array([p.split("->")[0] for p in d["pairs"]])
    tgt = np.array([p.split("->")[1] for p in d["pairs"]])
    dist = np.array(d["distances"], float)
    transfer = np.array(d["transfer_accuracy"], float)
    steer_raw = np.array(d["steer_benefit"], float)
    steer_corr = np.array(d["steer_benefit_corrected"], float)
    rand_ben = np.array(d["rand_benefit"], float)
    within = d["within_language_accuracy"]
    langs = sorted(within)
    lookup = {(a, b): v for a, b, v in zip(src, tgt, dist)}

    print(f"{len(d['pairs'])} ordered pairs over {len(langs)} languages\n")

    # --- reproduction guard ---------------------------------------------------
    checks = {
        "legB_r": pearson(dist, transfer),
        "legC_raw_r": pearson(dist, steer_raw),
        "legC_corrected_r": pearson(dist, steer_corr),
        "partial_corrected_r": partial_r(dist, steer_corr, transfer),
        "null_random_r": pearson(dist, rand_ben),
    }
    print("reproduction guard (deterministic quantities, must match):")
    bad = [k for k in PUBLISHED_EXACT
           if abs(checks[k] - PUBLISHED_EXACT[k]) > 1e-6]
    for k, ref in PUBLISHED_EXACT.items():
        print(f"  {k:22s} {checks[k]:+.4f} vs {ref:+.4f}  "
              f"{'OK' if k not in bad else 'MISMATCH'}")
    if bad:
        raise SystemExit("\nreproduction failed; nothing below is trustworthy")

    print("baseline-dependent (shifts with N_RAND_DIRS; informational):")
    for k, ref in PUBLISHED_BASELINE_DEPENDENT.items():
        delta = checks[k] - ref
        print(f"  {k:22s} {checks[k]:+.4f} vs {ref:+.4f}  "
              f"(delta {delta:+.4f})")

    # --- language-level null, full cohort -------------------------------------
    print("\nlanguage-level (Mantel) permutation, full cohort:")
    for name, y in [("leg B  transfer", transfer),
                    ("leg C  raw", steer_raw),
                    ("leg C  corrected", steer_corr),
                    ("null   random dir", rand_ben)]:
        obs, p = mantel_p(langs, src, tgt, lookup, y, pearson, args.draws)
        print(f"  {name:20s} r = {obs:+.4f}   p = {p:.4f}")
    obs, p = mantel_p(langs, src, tgt, lookup, steer_corr,
                      lambda a, b: partial_r(a, b, transfer), args.draws)
    print(f"  {'partial (corr|xfer)':20s} r = {obs:+.4f}   p = {p:.4f}")

    # --- probe validity -------------------------------------------------------
    good = [l for l in langs if within[l] >= args.threshold]
    poor = sorted((l for l in langs if within[l] < args.threshold),
                  key=lambda l: within[l])
    print(f"\nprobe validity (threshold {args.threshold}):")
    print("  below threshold: " + ", ".join(f"{l} {within[l]:.3f}"
                                            for l in poor))
    mask = np.isin(src, good) & np.isin(tgt, good)
    sub = sorted(set(src[mask]) | set(tgt[mask]))
    print(f"  both endpoints usable: {int(mask.sum())} pairs, "
          f"{len(sub)} languages")
    for name, y in [("leg B  transfer", transfer),
                    ("leg C  corrected", steer_corr)]:
        r = pearson(dist[mask], y[mask])
        _, p = mantel_p(sub, src[mask], tgt[mask], lookup, y[mask], pearson,
                        args.draws)
        print(f"    {name:20s} r = {r:+.4f}   p = {p:.4f}")

    # --- does cross-lingual steering do anything at all? ----------------------
    # The within-language manipulation check uses self-pairs, but the
    # distance-graded null concerns cross-pairs, so the intervention has to be
    # shown to work across languages too, not only within one.
    sb = np.array(d["steer_benefit"], float)
    rb = np.array(d["rand_benefit"], float)
    adv = sb - rb
    per_src = np.array([adv[src == l].mean() for l in langs])
    rng2 = np.random.default_rng(0)
    null = (rng2.choice([-1.0, 1.0], size=(20000, len(per_src))) *
            per_src).mean(axis=1)
    p_cross = float((np.abs(null) >= abs(per_src.mean()) - 1e-15).mean())
    print(f"\ncross-lingual steering vs the random baseline:")
    print(f"  mean advantage {adv.mean():+.3f} nats over {len(adv)} pairs, "
          f"beating random on {(adv > 0).sum()}")
    print(f"  per-source-language mean {per_src.mean():+.3f}, "
          f"{(per_src > 0).sum()}/{len(langs)} positive, sign-flip p = {p_cross:.4f}")

    wvals = list(within.values())
    print(f"\nwithin-language probe accuracy: mean {np.mean(wvals):.3f}, "
          f"range {min(wvals):.3f} to {max(wvals):.3f}")
    print(f"  transfer pairs above that maximum: "
          f"{int((transfer > max(wvals)).sum())} of {len(transfer)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
