"""Three NeurIPS-quality seaborn plots for the PATROL ablations.

Generates one standalone figure per hyperparameter:
    figures/ablation_check_tokens.{pdf,png}
    figures/ablation_backtrack_length.{pdf,png}
    figures/ablation_check_interval.{pdf,png}
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ── Paper-grade matplotlib defaults ──────────────────────────────────────────
sns.set_theme(
    context="paper",
    style="ticks",
    font="serif",
    font_scale=1.6,
    rc={
        "axes.titlesize":   16,
        "axes.labelsize":   16,
        "xtick.labelsize":  14,
        "ytick.labelsize":  14,
        "legend.fontsize":  13,
        "axes.linewidth":   1.1,
        "axes.edgecolor":   "#1a1a1a",
        "axes.grid":        True,
        "grid.linestyle":   ":",
        "grid.linewidth":   0.6,
        "grid.color":       "#b8b8b8",
        "grid.alpha":       0.7,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.direction":  "in",
        "ytick.direction":  "in",
        "lines.linewidth":  2.5,
        "lines.markersize": 9,
        "savefig.bbox":     "tight",
        "savefig.dpi":      300,
        "pdf.fonttype":     42,
        "ps.fonttype":      42,
        "mathtext.fontset": "stix",
        "font.family":      "serif",
    },
)

# ── Load ablation results ────────────────────────────────────────────────────
ROOT = Path(__file__).parent
D    = ROOT / "results_ablation"

rows = []
for f in sorted(D.glob("PATROL_*_RandomSearch_*_gpt41_judged.json")):
    s = json.load(open(f)).get("summary", {})
    m = re.search(r"_([a-z0-9]+)_RandomSearch_ci(\d+)_bl(\d+)_n", f.name)
    if not m: continue
    rows.append((m.group(1), int(m.group(2)), int(m.group(3)),
                 s.get("max_check_tokens"), s.get("asr_gpt")))

agg = defaultdict(list)
for model, ci, bl, mct, asr in rows:
    agg[(model, ci, bl, mct)].append(asr)


def stats(model, ci_fn, bl_fn, mct_fn, x_values):
    means, stds = [], []
    for x in x_values:
        vals = [v for k, v in agg.items()
                if k[0] == model and ci_fn(k[1], x) and bl_fn(k[2], x) and mct_fn(k[3], x)
                for v in v]
        if vals:
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
        else:
            means.append(np.nan); stds.append(0.0)
    return np.array(means), np.array(stds)


MODELS = [
    ("llama2", "LLaMA-2-7B",  "o", "-"),
    ("llama3", "LLaMA-3-8B",  "D", "--"),
    ("vicuna", "Vicuna-7B",   "^", "-."),
]
# Custom academic palette (deep navy, brick red, olive green) — chosen to print
# well on B&W projectors and to read clearly through line/marker shape alone.
COLORS = ["#1B3A6B", "#A8322D", "#5C7A29"]


# ── Plot helper ──────────────────────────────────────────────────────────────
def make_plot(title, xlabel, x_values, ci_fn, bl_fn, mct_fn,
              xscale=None, outname=None):
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for (key, label, marker, ls), color in zip(MODELS, COLORS):
        m, s = stats(key, ci_fn, bl_fn, mct_fn, x_values)
        ax.plot(x_values, m, marker=marker, linestyle=ls, color=color,
                label=label, markerfacecolor=color, markeredgecolor=color,
                markeredgewidth=0.0)
        ax.fill_between(x_values, m - s, m + s, color=color, alpha=0.12,
                        linewidth=0)

    if xscale == "log2":
        ax.set_xscale("log", base=2)
    ax.set_xticks(x_values)
    ax.set_xticklabels(x_values)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ASR (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    if title:
        ax.set_title(title, loc="left", pad=10)

    leg = ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        edgecolor="#1a1a1a",
        fancybox=False,
        borderpad=0.5,
        handlelength=2.2,
        labelspacing=0.35,
    )
    leg.get_frame().set_linewidth(0.7)
    sns.despine(ax=ax, trim=False)
    fig.tight_layout()

    out = ROOT / "figures" / f"{outname}.pdf"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  saved: {out}")


print("Generating ablation figures…")

# ── Check window: T_check ────────────────────────────────────────────────────
make_plot(
    title=None,
    xlabel=r"$T_{\mathrm{check}}$",
    x_values=[4, 8, 16, 32, 64, 128, 256],
    ci_fn=lambda ci, _: ci == 4,
    bl_fn=lambda bl, _: bl == 20,
    mct_fn=lambda mct, x: mct == x,
    xscale="log2",
    outname="ablation_check_tokens",
)

# ── Backtrack length: L_back ─────────────────────────────────────────────────
make_plot(
    title=None,
    xlabel=r"$L_{\mathrm{back}}$",
    x_values=[8, 12, 16, 20, 24, 28],
    ci_fn=lambda ci, _: ci == 4,
    bl_fn=lambda bl, x: bl == x,
    mct_fn=lambda mct, _: mct == 128,
    xscale=None,
    outname="ablation_backtrack_length",
)

# ── Check interval: Δ ────────────────────────────────────────────────────────
make_plot(
    title=None,
    xlabel=r"Check Interval $\Delta$",
    x_values=[2, 4, 6, 8, 16],
    ci_fn=lambda ci, x: ci == x,
    bl_fn=lambda bl, _: bl == 20,
    mct_fn=lambda mct, _: mct == 128,
    xscale=None,
    outname="ablation_check_interval",
)

print("\nAll three figures generated under figures/.")
