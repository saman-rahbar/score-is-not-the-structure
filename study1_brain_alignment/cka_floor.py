"""What does linear CKA read between the fMRI target and a target whose
correspondence to it has been destroyed?

The content ablation shows a model trained on shuffled or random targets still
scores about 0.31 CKA against the true target. Two explanations are available
and they are very different. Either the alignment pressure drives any model
toward a geometry that happens to sit near the brain target, or the ablated
targets are themselves already about 0.31 from the true one under this measure,
in which case a model that fits them perfectly inherits that score and no
training dynamics are involved.

This distinguishes them without a GPU, by scoring the targets directly against
each other: CKA(true, shuffled) and CKA(true, random), using the same target
construction, the same shuffling, and the same random-matrix draw the experiment
uses. It also calibrates the measure's sensitivity, since CKA(true, true) is 1
by construction and the shuffled value is the floor below which a score carries
no correspondence information at all.

Runs on the Pereira cache alone. See README.md for usage.
"""


from __future__ import annotations

import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The experiment module is imported rather than reimplemented, so the target
# construction, the shuffle and the random draw are byte-identical to the ones
# the runs used. Its filename differs between this release and the working copy
# the experiments were run from, so try both before giving up.
_EXP = None
for _name in (os.environ.get("EXPERIMENT_MODULE"), "experiment",
              "experiment_full"):
    if not _name:
        continue
    try:
        _EXP = __import__(_name)
        break
    except ImportError:
        continue
if _EXP is None:
    raise SystemExit(
        "could not import the experiment module (tried 'experiment' and "
        "'experiment_full'). Run this from the directory holding it, or set "
        "EXPERIMENT_MODULE to its name.")

load_pereira = _EXP.load_pereira
prepare_brain_targets = _EXP.prepare_brain_targets
shuffled_targets = _EXP.shuffled_targets
random_matrix_targets = _EXP.random_matrix_targets


def cka(X, Y) -> float:
    """Linear CKA between two [N, d] matrices, rows centred.

    Numpy counterpart of the torch objective the experiment optimizes; the
    self-test below checks the two agree.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    xy = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    xx = np.linalg.norm(X.T @ X, ord="fro")
    yy = np.linalg.norm(Y.T @ Y, ord="fro")
    return float(xy / (xx * yy + 1e-12))


def selftest() -> int:
    """Agreement with the torch objective, and the two values CKA must give."""
    rng = np.random.RandomState(0)
    A = rng.randn(200, 32)
    B = rng.randn(200, 32)
    assert abs(cka(A, A) - 1.0) < 1e-9, "CKA(X, X) must be 1"
    assert 0.0 <= cka(A, B) < 0.3, f"independent draws should be low, got {cka(A, B)}"
    try:
        import torch
        torch_cka = _EXP.linear_cka
        t = float(torch_cka(torch.tensor(A), torch.tensor(B)))
        assert abs(t - cka(A, B)) < 1e-4, f"numpy {cka(A, B)} vs torch {t}"
        print(f"selftest OK (matches the torch objective to {abs(t - cka(A, B)):.2e})")
    except ImportError:
        print("selftest OK (torch absent; skipped the cross-check)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--draws", type=int, default=20,
                    help="shuffles and random matrices to average over")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    _, responses = load_pereira()
    true, r = prepare_brain_targets(responses, args.rank, 0)
    print(f"target: {true.shape[0]} sentences, rank {r}\n")

    sh = [cka(true, shuffled_targets(true, s)) for s in range(args.draws)]
    rd = [cka(true, random_matrix_targets(true, s)) for s in range(args.draws)]

    print(f"CKA(true, true)               = {cka(true, true):.4f}")
    print(f"CKA(true, shuffled)  mean     = {np.mean(sh):.4f} "
          f"(sd {np.std(sh):.4f}, min {np.min(sh):.4f}, max {np.max(sh):.4f})")
    print(f"CKA(true, random)    mean     = {np.mean(rd):.4f} "
          f"(sd {np.std(rd):.4f}, min {np.min(rd):.4f}, max {np.max(rd):.4f})")
    print()
    print("Compare these against what the trained models reach against the true")
    print("target. If the shuffled/random figures here are close to the trained")
    print("controls' scores, those controls are inheriting the similarity that")
    print("already exists between the targets, and no training dynamics are")
    print("needed to explain them. If these are much lower, the recovery seen in")
    print("training is produced by the alignment pressure itself.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
