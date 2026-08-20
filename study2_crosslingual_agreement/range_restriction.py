"""Is the leg-B attenuation probe INVALIDITY, or just range restriction?

The four chance-level languages are also among the most typologically distant,
so removing them truncates the predictor's range, which attenuates a correlation
mechanically whatever the probe was doing. The exhaustive 4-language enumeration
does not separate these: most random exclusion sets remove central languages and
so do not restrict range at all.

The needed control: drop four ABOVE-chance languages matched on mean typological
distance, and compare the attenuation. If dropping distance-matched valid
languages attenuates just as much, the probe-validity story is not established.
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np

R = Path("results/results.json")
d = json.loads(R.read_text())
src = np.array([p.split("->")[0] for p in d["pairs"]])
tgt = np.array([p.split("->")[1] for p in d["pairs"]])
dist = np.array(d["distances"], float)
transfer = np.array(d["transfer_accuracy"], float)
within = d["within_language_accuracy"]
langs = sorted(within)

mean_d = {l: float(np.mean([v for a, b, v in zip(src, tgt, dist)
                            if a == l or b == l])) for l in langs}
INVALID = [l for l in langs if within[l] < 0.60]
VALID = [l for l in langs if within[l] >= 0.60]


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def leg_b(keep):
    m = np.isin(src, keep) & np.isin(tgt, keep)
    if m.sum() < 10:
        return float("nan"), 0, float("nan")
    r = pearson(dist[m], transfer[m])
    return r, int(m.sum()), float(dist[m].max() - dist[m].min())


full_r, full_n, full_range = leg_b(langs)
inv_r, inv_n, inv_range = leg_b(VALID)

print("mean typological distance by language (chance-level marked):")
for l in sorted(langs, key=lambda x: -mean_d[x]):
    mark = "  <- chance-level probe" if l in INVALID else ""
    print(f"  {l:>5s}  mean_d={mean_d[l]:.3f}  within={within[l]:.3f}{mark}")

rank_of_invalid = sorted([sorted(langs, key=lambda x: -mean_d[x]).index(l) + 1
                          for l in INVALID])
print(f"\nthe four chance-level languages rank {rank_of_invalid} of "
      f"{len(langs)} by mean distance (1 = most distant)")

print(f"\nall 17          : r={full_r:+.4f}  n={full_n}  predictor range={full_range:.3f}")
print(f"drop the 4 invalid: r={inv_r:+.4f}  n={inv_n}  predictor range={inv_range:.3f}")

# --- distance-matched control: drop 4 VALID languages, as distant as possible
valid_by_dist = sorted(VALID, key=lambda x: -mean_d[x])
matched = valid_by_dist[:4]
m_r, m_n, m_range = leg_b([l for l in langs if l not in matched])
print(f"\ndrop the 4 most distant VALID languages ({', '.join(matched)}):")
print(f"                  : r={m_r:+.4f}  n={m_n}  predictor range={m_range:.3f}")

# --- null conditioned on the removed set's distance profile ------------------
# Compare against exclusion sets whose total mean-distance is at least as large
# as the invalid set's, i.e. sets that restrict range at least as much.
inv_profile = sum(mean_d[l] for l in INVALID)
print(f"\ninvalid set's summed mean distance: {inv_profile:.3f}")

all_sets, matched_sets = [], []
for combo in combinations(langs, 4):
    r, n, _ = leg_b([l for l in langs if l not in combo])
    if not np.isfinite(r):
        continue
    prof = sum(mean_d[l] for l in combo)
    all_sets.append(r)
    if prof >= inv_profile:
        matched_sets.append((r, combo, prof))

all_sets = np.array(all_sets)
print(f"unconditional null: {len(all_sets)} sets, "
      f"fraction with r >= {inv_r:+.4f} (as weak or weaker): "
      f"{(all_sets >= inv_r).mean():.4f}")

if matched_sets:
    mr = np.array([r for r, _, _ in matched_sets])
    print(f"distance-matched null: {len(mr)} sets with summed distance >= the "
          f"invalid set's")
    print(f"  their r: mean {mr.mean():+.4f}, min {mr.min():+.4f}, "
          f"max {mr.max():+.4f}")
    print(f"  fraction with r >= {inv_r:+.4f}: {(mr >= inv_r).mean():.4f}")
    print("\n  This is the number that matters. If it is large, dropping any")
    print("  equally-distant four attenuates as much, and the attenuation is")
    print("  range restriction, not probe validity.")
else:
    print("no exclusion set is as distance-extreme as the invalid one")
