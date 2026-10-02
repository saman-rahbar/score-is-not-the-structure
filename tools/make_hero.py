#!/usr/bin/env python
"""Animated header for the README (assets/hero.gif).

Bars show the end-of-training CKA between Pythia-160M and the fMRI target
reported in the paper (20 seeds): no alignment, alignment to the real target,
and alignment to two targets whose correspondence to the brain was destroyed.
The dashed lines mark the reliability ceiling (0.540) and the CKA a scrambled
target already has with the real one, with no model (0.204).

    python tools/make_hero.py            # writes assets/hero.gif
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

OUT = Path(__file__).resolve().parents[1] / "assets" / "hero.gif"
ROWS = [("No alignment", 0.095, "#9aa5b1"),
        ("Aligned to real brain data", 0.337, "#0072B2"),
        ("Aligned to scrambled brain data", 0.309, "#E69F00"),
        ("Aligned to random noise", 0.310, "#E69F00")]
CEILING, FLOOR, XMAX = 0.540, 0.204, 0.62
INK, MUTED, BG, CARD, LINE = "#1b2430", "#5b6675", "#ffffff", "#f7f9fc", "#e3e8ef"

fams = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next((f for f in ("Helvetica Neue", "Helvetica", "Arial") if f in fams), "DejaVu Sans")
N_TITLE, N_GROW, N_CAPTION = 10, 34, 12


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


def frame(i):
    fig = plt.figure(figsize=(9.6, 3.6), dpi=150, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 37.5); ax.axis("off")
    ax.add_patch(plt.Rectangle((1.2, 1.2), 97.6, 35.1, fc=CARD, ec=LINE, lw=1.2, zorder=0))
    a = ease(i / N_TITLE)
    ax.text(4, 32.4, "The score is not the structure", fontsize=19, weight="bold", color=INK, alpha=a, va="center")
    ax.text(4, 29.0, "Similarity (CKA) between a language model and human brain responses, Pythia-160M, 20 seeds",
            fontsize=10.5, color=MUTED, alpha=a, va="center")
    g = ease((i - N_TITLE) / N_GROW)
    x0, x1 = 34, 92
    sx = lambda v: x0 + (x1 - x0) * v / XMAX
    top, bot = 24.6, 7.2
    for v, lab in ((CEILING, "ceiling 0.54"), (FLOOR, "scrambled vs real, no model 0.20")):
        ax.plot([sx(v)] * 2, [bot, top], color="#7b8794", lw=1.1, ls=(0, (3, 3)), alpha=a, zorder=1)
        ax.text(sx(v), top + 0.6, lab, fontsize=8.5, color=MUTED, ha="center", alpha=a)
    for k, (name, v, col) in enumerate(ROWS):
        y = 21.0 - k * 4.2
        ax.text(x0 - 1.5, y, name, ha="right", va="center", fontsize=10.5, color=INK,
                weight="bold" if k >= 2 else "normal")
        ax.add_patch(plt.Rectangle((x0, y - 1.1), x1 - x0, 2.2, fc=LINE, ec="none", zorder=0.5))
        ax.add_patch(plt.Rectangle((x0, y - 1.1), (sx(v) - x0) * g, 2.2, fc=col, ec="none", zorder=2))
        if g > 0:
            ax.text(x0 + (sx(v) - x0) * g + 0.8, y, f"{v * g:.3f}", va="center", fontsize=9.5,
                    color=INK, weight="bold", zorder=3)
    c = ease((i - N_TITLE - N_GROW) / N_CAPTION)
    ax.text(50, 3.6, "Scrambling the brain data keeps 92% of the score.",
            fontsize=11.5, color=INK, alpha=c, weight="bold", ha="center", va="center")
    fig.canvas.draw()
    img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
    plt.close(fig)
    return img


def main():
    n = N_TITLE + N_GROW + N_CAPTION
    frames = [frame(i) for i in range(n + 1)]
    pal = frames[-1].quantize(colors=96, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    q[0].save(OUT, save_all=True, append_images=q[1:], duration=[70] * n + [4200], loop=0,
              optimize=True, disposal=1)
    frames[-1].save(OUT.with_name("hero_still.png"))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
