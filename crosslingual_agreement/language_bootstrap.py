"""Confidence intervals that treat languages, not pairs, as the units.

The 272 ordered pairs come from 17 languages, so resampling pairs would
understate the uncertainty. Each bootstrap draw resamples the 17 languages
with replacement and recomputes three correlations:

  transfer, all languages     transfer accuracy against distance, over every
                              ordered pair of distinct languages in the draw
  transfer, valid languages   the same, keeping only languages whose
                              within-language probe clears 0.60
  reliability gradient        within-language accuracy against each
                              language's mean distance, one value per drawn
                              language (repeats included)

    python language_bootstrap.py            # 10,000 draws, seed 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent / "results" / "results.json"
THRESHOLD = 0.60


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = json.loads(R.read_text())
    langs = d["languages"]
    within = d["within_language_accuracy"]
    pairs = [tuple(p.split("->")) for p in d["pairs"]]
    dist = np.array(d["distances"], float)
    transfer = np.array(d["transfer_accuracy"], float)
    mean_d = {l: float(np.mean([v for (a, b), v in zip(pairs, dist) if l in (a, b)])) for l in langs}
    valid = {l for l in langs if within[l] >= THRESHOLD}

    def pair_r(keep):
        idx = [i for i, (a, b) in enumerate(pairs) if a in keep and b in keep]
        return np.corrcoef(dist[idx], transfer[idx])[0, 1] if len(idx) > 2 else np.nan

    rng = np.random.default_rng(args.seed)
    full, restricted, gradient = [], [], []
    for _ in range(args.draws):
        draw = list(rng.choice(langs, len(langs), replace=True))
        full.append(pair_r(set(draw)))
        restricted.append(pair_r(set(draw) & valid))
        gradient.append(np.corrcoef([within[l] for l in draw], [mean_d[l] for l in draw])[0, 1])

    def report(name, point, xs):
        xs = np.array([x for x in xs if np.isfinite(x)])
        lo, hi = np.percentile(xs, [2.5, 97.5])
        print(f"  {name:28s} r = {point:+.3f}   95% interval [{lo:+.2f}, {hi:+.2f}]")

    print(f"bootstrap over languages: {args.draws} draws, seed {args.seed}")
    report("transfer, all languages", pair_r(set(langs)), full)
    report(f"transfer, valid ({len(valid)} langs)", pair_r(valid), restricted)
    report("reliability gradient", np.corrcoef([within[l] for l in langs],
                                               [mean_d[l] for l in langs])[0, 1], gradient)


if __name__ == "__main__":
    main()
