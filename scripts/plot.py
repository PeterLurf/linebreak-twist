"""Figures from results/summary.csv and results/steering.csv -> figures/*.svg

    python scripts/plot.py
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
COLORS = {"Gemma-2": "#0072B2", "Qwen2.5": "#E69F00", "Pythia": "#009E73", "GPT-2": "#7F7F7F"}
MARKERS = {"Gemma-2": "o", "Qwen2.5": "s", "Pythia": "^", "GPT-2": "D"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})


def num(v):
    return float(v) if v not in ("", None) else np.nan


def scaling(rows):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.5, 3))
    for fam in COLORS:
        rs = sorted([r for r in rows if r["family"] == fam], key=lambda r: num(r["params_B"]))
        x = [num(r["params_B"]) for r in rs]
        kw = dict(color=COLORS[fam], marker=MARKERS[fam], ms=4, ls="-" if len(rs) > 1 else "none")
        a.plot(x, [num(r["effective_twist_heads"]) for r in rs], label=fam, **kw)
        b.plot(x, [-num(r["ablate_best_single"]) for r in rs], **kw)
        if len(rs) > 1:
            b.plot(x, [-num(r["ablate_top4"]) for r in rs], color=COLORS[fam], marker=MARKERS[fam], ms=4, ls="--", mfc="white")
    b.plot([num(r["params_B"]) for r in rows], [num(r["ablate_random_max_abs"]) for r in rows], "x", color="0.5", ms=4,
           label="random heads (max of 3 sets)")
    b.plot([], [], "k-", label="best single twist head"); b.plot([], [], "k--", label="top 4 twist heads")
    for ax in (a, b):
        ax.set_xscale("log"); ax.set_xlim(0.13, 40); ax.set_xlabel("parameters (billions)")
        ax.set_xticks([0.2, 0.5, 1, 2, 5, 10, 30]); ax.set_xticklabels(["0.2", "0.5", "1", "2", "5", "10", "30"])
        ax.xaxis.set_minor_formatter(NullFormatter())
    a.set_ylabel("effective twist heads"); a.legend(frameon=False, fontsize=7)
    b.set_ylim(0, 5.2); b.set_ylabel("break delay when ablated (characters)"); b.legend(frameon=False, fontsize=6.5)
    a.set_title("(a) one head when small, many when large", fontsize=8, loc="left")
    b.set_title("(b) each head matters less at scale", fontsize=8, loc="left")
    fig.tight_layout(); fig.savefig(FIG / "twist_scaling.svg"); plt.close(fig)


def steering(steer):
    curves, rand = defaultdict(dict), []
    fam = {}
    for r in steer:
        if num(r["clamp_auc_drop"]) > 0.1:  # clamping itself breaks the model (GPT-2 XL): not interpretable
            continue
        curves[r["model"]][int(r["delta"])] = num(r["shift"]); fam[r["model"]] = r["family"]
        rand.append(abs(num(r["random_shift"])))
    fig, ax = plt.subplots(figsize=(3.6, 3))
    xs = np.array([-16, -8, 0, 8, 16])
    ax.plot(xs, xs, ":", color="0.5", lw=0.8, label="perfect transfer")
    ax.axhspan(-max(rand), max(rand), color="0.85", lw=0, label=f"random subspace (|shift| ≤ {max(rand):.2f})")
    for m, c in curves.items():
        ax.plot(xs, [c.get(d, 0.0) for d in xs], "-o", ms=2.5, lw=0.9, alpha=0.8, color=COLORS[fam[m]])
    for f in sorted(set(fam.values())):
        ax.plot([], [], "-o", ms=2.5, color=COLORS[f], label=f)
    ax.axhline(0, color="k", lw=0.4); ax.axvline(0, color="k", lw=0.4)
    ax.set_xticks(xs); ax.set_ylim(-17, 17)
    ax.set_xlabel("count shift written into the model (characters)")
    ax.set_ylabel("change in break point (characters)")
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "steering.svg"); plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    rows = list(csv.DictReader(open(ROOT / "results" / "summary.csv")))
    steer = list(csv.DictReader(open(ROOT / "results" / "steering.csv")))
    scaling(rows); steering(steer)
    print("wrote", sorted(p.name for p in FIG.glob("*.svg")))


if __name__ == "__main__":
    main()
