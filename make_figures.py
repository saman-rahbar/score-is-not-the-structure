"""Figures for the measurement-audit framing.

fig_cka_decomposition.pdf : what a brain-alignment score reads when the brain
    is removed from the target, at three scales, against the reliability ceiling.
fig_probe_validity.pdf    : (a) probe validity itself declines with typological
    distance; (b) the transfer-distance gradient with and without the languages
    whose probe is at chance.

Reads only the stored results. No model, no GPU.
"""
from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parent))
S1 = REPO / "study1_brain_alignment/results"
S2 = REPO / "study2_crosslingual_agreement/results/results.json"
OUT = Path(os.environ.get("FIG_OUT", REPO / "figures"))
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
})

GREY, BLUE, RED, SAND = "#8c8c8c", "#2b6cb0", "#b03030", "#e2a355"


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1])


def fig_cka():
    ceiling = json.loads((S1 / "noise_ceiling.json").read_text())["target_cka_ceiling"]
    scales = ["160m", "410m", "1.4b"]
    names = {"160m": "Pythia-160M", "410m": "Pythia-410M", "1.4b": "Pythia-1.4B"}
    conds = [("lm_only", "LM-only", GREY), ("brain_align", "brain-aligned", BLUE),
             ("shuffled_brain", "shuffled", RED), ("random_matrix", "random", SAND)]

    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    width, gap = 0.19, 0.10
    xs = np.arange(len(scales))
    for i, (key, label, colour) in enumerate(conds):
        vals, errs = [], []
        for m in scales:
            d = json.loads((S1 / f"results_pythia-{m}.json").read_text())
            per = np.array(d[f"{key}_cka_per_seed"], float)
            vals.append(per.mean())
            errs.append(1.96 * per.std(ddof=1) / np.sqrt(len(per)))
        off = (i - 1.5) * (width + gap / 3)
        ax.bar(xs + off, vals, width, yerr=errs, capsize=2, color=colour,
               label=label, edgecolor="white", linewidth=0.4)

    ax.axhline(ceiling, ls="--", lw=1.0, color="black")
    ax.text(len(scales) - 0.42, ceiling + 0.012,
            f"reliability ceiling {ceiling:.2f}", fontsize=6.8, ha="right")
    ax.set_xticks(xs)
    ax.set_xticklabels([names[m] for m in scales])
    ax.set_ylabel("linear CKA to the fMRI target")
    ax.set_ylim(0, 0.60)
    ax.legend(ncol=4, frameon=False, loc="upper left",
              bbox_to_anchor=(0.0, 1.16), columnspacing=1.2, handlelength=1.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cka_decomposition.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_cka_decomposition.pdf")


def fig_validity():
    d = json.loads(S2.read_text())
    src = np.array([p.split("->")[0] for p in d["pairs"]])
    tgt = np.array([p.split("->")[1] for p in d["pairs"]])
    dist = np.array(d["distances"], float)
    transfer = np.array(d["transfer_accuracy"], float)
    within = d["within_language_accuracy"]
    langs = sorted(within)
    bad = [l for l in langs if within[l] < 0.60]
    good = [l for l in langs if within[l] >= 0.60]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(5.6, 2.7))

    # (a) probe validity vs mean typological distance to the rest of the sample.
    # Mean distance rather than distance from English: no language should be the
    # privileged reference, and it keeps all 17 rather than 16.
    mean_d = {l: float(np.mean([v for a, b, v in zip(src, tgt, dist)
                                if a == l or b == l])) for l in langs}
    xs = np.array([mean_d[l] for l in langs])
    ys = np.array([within[l] for l in langs])
    ls = list(langs)
    cols = [RED if l in bad else BLUE for l in ls]
    axL.scatter(xs, ys, c=cols, s=18, zorder=3)
    b1, b0 = np.polyfit(xs, ys, 1)
    gx = np.linspace(xs.min(), xs.max(), 50)
    axL.plot(gx, b0 + b1 * gx, color="black", lw=1.0, zorder=2)
    axL.axhline(0.5, ls=":", lw=0.9, color=RED)
    axL.text(xs.min(), 0.512, "chance", fontsize=6.5, ha="left", color=RED)
    for x, y, l in zip(xs, ys, ls):
        if l in bad:
            axL.annotate(l, (x, y), fontsize=6, xytext=(3.0, -1.0),
                         textcoords="offset points", color=RED)
    axL.set_xlabel("mean URIEL syntactic distance to the other 16")
    axL.set_ylabel("within-language probe accuracy")
    axL.set_title(f"(a) probe reliability falls with distance\n"
                  f"$r={pearson(xs, ys):+.2f}$", fontsize=7.6)
    axL.set_ylim(0.30, 1.0)

    # (b) leg B, all languages vs valid-probe languages
    m_all = np.ones(len(dist), bool)
    m_val = np.isin(src, good) & np.isin(tgt, good)
    axR.scatter(dist[~m_val], transfer[~m_val], s=7, color=RED, alpha=0.45,
                label="pair involves a chance-level probe", zorder=2)
    axR.scatter(dist[m_val], transfer[m_val], s=7, color=BLUE, alpha=0.55,
                label="both probes above chance", zorder=3)
    for mask, colour, lab in [(m_all, "black", f"all 17: $r={pearson(dist, transfer):+.2f}$"),
                              (m_val, BLUE, f"valid 13: $r={pearson(dist[m_val], transfer[m_val]):+.2f}$")]:
        b1, b0 = np.polyfit(dist[mask], transfer[mask], 1)
        gx = np.linspace(dist[mask].min(), dist[mask].max(), 50)
        axR.plot(gx, b0 + b1 * gx, color=colour, lw=1.3, label=lab, zorder=4)
    axR.set_xlabel("URIEL syntactic distance")
    axR.set_ylabel("probe-transfer accuracy")
    axR.set_title("(b) fitting with and without the failed probes", fontsize=7.6)
    axR.legend(frameon=False, fontsize=6.2, loc="upper center",
               bbox_to_anchor=(0.5, -0.30), ncol=2, handlelength=1.1,
               columnspacing=1.0, labelspacing=0.25)

    fig.tight_layout()
    fig.savefig(OUT / "fig_probe_validity.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_probe_validity.pdf")


if __name__ == "__main__":
    fig_cka()
    fig_validity()
