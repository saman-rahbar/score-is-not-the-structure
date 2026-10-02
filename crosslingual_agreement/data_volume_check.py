"""Does MultiBLiMP data volume explain which within-language probes fail?

The obvious dismissal of the instrument result is that the chance-level probes
were simply undertrained. They were not: max_pairs_per_lang caps every language
at the same number of minimal pairs, so all but the smallest language fit their
probe on an identical training set. This reports availability, the capped amount
actually used, and the correlation of each with within-language accuracy.

Availability counts come from the MultiBLiMP cache and are recorded here rather
than reloaded, so the check runs without the datasets library or the cache; to
regenerate them, see the printed command in --show-source. See README.md.
"""


from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np

# Usable minimal pairs per language in the MultiBLiMP cache, i.e. rows with a
# distinct grammatical/ungrammatical pair, as counted by load_multiblimp().
AVAILABLE = {
    "eng": 770, "deu": 2298, "fra": 2548, "spa": 2541, "ita": 2999,
    "por": 3048, "rus": 3832, "bul": 2458, "ell": 1096, "fin": 2570,
    "est": 2575, "tur": 1742, "cat": 2284, "hin": 1447, "urd": 550,
    "eus": 273, "arb": 1215,
}

REGEN = ("python3 -c \"import sys; sys.path.insert(0,'.'); "
         "from experiment import load_multiblimp, LANG_ISO3; "
         "[print(l, len(load_multiblimp(l))) for l in LANG_ISO3]\"")


def pearson(x, y) -> float:
    return float(np.corrcoef(np.asarray(x, float), np.asarray(y, float))[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/results.json")
    ap.add_argument("--show-source", action="store_true",
                    help="print the command that regenerates the counts")
    args = ap.parse_args()
    if args.show_source:
        print(REGEN)
        return 0

    d = json.loads(Path(args.results).read_text())
    within = d["within_language_accuracy"]
    cap = d["config"]["max_pairs_per_lang"]
    frac = d["config"]["probe_train_frac"]
    langs = sorted(within)
    missing = [l for l in langs if l not in AVAILABLE]
    if missing:
        raise SystemExit(f"no availability recorded for {missing}; "
                         f"regenerate with:\n  {REGEN}")

    used = {l: min(AVAILABLE[l], cap) for l in langs}
    train = {l: int(used[l] * frac) for l in langs}

    print(f"cap {cap} pairs per language, probe_train_frac {frac}\n")
    print(f"{'lang':>5s} {'avail':>7s} {'used':>6s} {'train':>6s} {'within':>7s}")
    for l in sorted(langs, key=lambda x: -within[x]):
        flag = "  <- chance level" if within[l] < 0.60 else ""
        print(f"{l:>5s} {AVAILABLE[l]:7d} {used[l]:6d} {train[l]:6d} "
              f"{within[l]:7.3f}{flag}")

    r_avail = pearson([AVAILABLE[l] for l in langs], [within[l] for l in langs])
    r_used = pearson([used[l] for l in langs], [within[l] for l in langs])
    print(f"\nwithin-language accuracy vs raw availability : r = {r_avail:+.4f}")
    print(f"within-language accuracy vs amount used      : r = {r_used:+.4f}")
    print("  Availability is irrelevant past the cap, so the second is the "
          "one that bears on training size.")

    below = [l for l in langs if AVAILABLE[l] < cap]
    print(f"\nat the cap: {len(langs) - len(below)} of {len(langs)}; "
          f"below it: {below or 'none'}")
    if below:
        l = min(below, key=lambda x: AVAILABLE[x])
        worse = [b for b in langs if within[b] < within[l]]
        print(f"  {l} trains on the fewest pairs ({train[l]}) and still "
              f"outscores {len(worse)} language(s): {', '.join(sorted(worse))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
