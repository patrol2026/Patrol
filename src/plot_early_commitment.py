"""
plot_early_commitment.py
────────────────────────
Two-panel figure:
  (left)  Detection CDF among successful jailbreaks: of the prompts whose
          full response is unsafe, what fraction has been flagged by prefix
          length t? Answers "are we able to detect early?"
  (right) Pr[final unsafe | unsafe at t] vs t.  Answers "is an early flag
          predictive of full-response harm?"

PATROL's monitoring window [c, m] is shaded on the left panel and a vertical
dashed line at t = m marks the default check horizon.

Usage
─────
    python plot_early_commitment.py
        [--in figures/early_commitment.json]
        [--out figures/early_commitment.pdf]
        [--models llama2 llama3 vicuna]
        [--horizon_m 128]   [--check_c 4]
        [--ymax_left 100]   [--ymax_right 102]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

REPO_ROOT = Path(__file__).parent.resolve()

# Default-tab10-style saturated palette + varied line styles in the spirit
# of common NeurIPS-published figures.
PALETTE = {
    "AdvBench":      "#7F7F7F",
    "GCG":           "#1F77B4",   # tab:blue
    "AutoDAN":       "#FF7F0E",   # tab:orange
    "PAIR":          "#2CA02C",   # tab:green
    "DeepInception": "#9467BD",   # tab:purple
    "RandomSearch":  "#D62728",   # tab:red
}
MARKERS = {
    "AdvBench":      "o",
    "GCG":           "o",
    "AutoDAN":       "s",
    "PAIR":          "^",
    "DeepInception": "D",
    "RandomSearch":  "v",
}
LINESTYLES = {
    "AdvBench":      ":",
    "GCG":           "-",
    "AutoDAN":       "--",
    "PAIR":          "-",
    "DeepInception": "-.",
    "RandomSearch":  "-",
}

ATTACK_ORDER = ["AdvBench", "GCG", "AutoDAN", "PAIR", "DeepInception", "RandomSearch"]


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def collect(data: dict, model_filter: list[str] | None) -> dict[str, np.ndarray]:
    """Return {attacker: ndarray(P, T) of bool} pooled across selected models."""
    pooled: dict[str, list[np.ndarray]] = {}
    for k, prompts in data.items():
        m, a = k.split("|")
        if model_filter and m not in model_filter: continue
        if not prompts: continue
        arr = np.array([p["series"] for p in prompts], dtype=bool)
        pooled.setdefault(a, []).append(arr)
    return {a: np.concatenate(v, axis=0) for a, v in pooled.items()}


def detection_cdf(pooled):
    """
    Among prompts whose final response is unsafe, what fraction is already
    flagged by prefix length t?
        CDF(t) = mean_{i: y_i^final = unsafe} 1[ exists s <= t : y_i^s unsafe ]

    The left-panel curve is non-decreasing in t and reaches the per-attack
    "detectable jailbreak fraction" at t = full response.
    """
    out = {}
    for a, arr in pooled.items():
        success_mask = arr[:, -1]                              # final unsafe
        if success_mask.sum() == 0:
            continue
        succ = arr[success_mask]                               # [Ps, T]
        # Cumulative OR along the time axis: once flagged, stays flagged
        cum  = np.maximum.accumulate(succ.astype(int), axis=1) # [Ps, T]
        cdf  = cum.mean(axis=0) * 100
        out[a] = (cdf, int(success_mask.sum()))                # also return n
    return out


def conditional(pooled):
    """P[final unsafe | unsafe at t] for each t."""
    out = {}
    for a, arr in pooled.items():
        T = arr.shape[1]
        cond = np.full(T, np.nan)
        for t in range(T):
            mask = arr[:, t]
            if mask.sum() < 3:                  # require >= 3 to keep curve stable
                continue
            cond[t] = arr[mask, -1].mean() * 100
        out[a] = cond
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_single(D, models, out_path, ymax_left, horizon_m, check_c,
                exclude=("AdvBench",), min_jailbreaks=5):
    """Single-panel detection-CDF figure tuned for spotlight presentation.
    Attacks with fewer than ``min_jailbreaks`` successful jailbreaks are
    skipped to avoid plotting noisy single-prompt curves."""
    sns.set_theme(
        context="paper", style="ticks", font="serif", font_scale=1.55,
        rc={
            "axes.titlesize":   17,
            "axes.labelsize":   17,
            "xtick.labelsize":  14,
            "ytick.labelsize":  14,
            "legend.fontsize":  12.5,
            "axes.linewidth":   1.3,
            "axes.edgecolor":   "#1a1a1a",
            "axes.grid":        True,
            "grid.linestyle":   ":",
            "grid.linewidth":   0.7,
            "grid.color":       "#b8b8b8",
            "grid.alpha":       0.65,
            "xtick.direction":  "in",
            "ytick.direction":  "in",
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "lines.linewidth":  2.6,
            "lines.markersize": 9.5,
            "savefig.bbox":     "tight",
            "savefig.dpi":      300,
            "pdf.fonttype":     42,
            "ps.fonttype":      42,
            "mathtext.fontset": "stix",
            "font.family":      "serif",
        },
    )

    prefix = D["prefix_tokens"]
    finite = [p for p in prefix if p is not None]
    last_x = max(finite) + 100
    xs = [p if p is not None else last_x for p in prefix]

    pooled = collect(D["data"], models)
    cdf    = detection_cdf(pooled)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    # Shaded PATROL operating window
    ax.axvspan(check_c, horizon_m, color="#dfe6ee", alpha=0.55, zorder=0)
    ax.axvline(horizon_m, color="#3a3a3a", lw=1.2, ls="--", zorder=1)
    ax.text(
        horizon_m - 4, 6, fr"$m={horizon_m}$",
        fontsize=12.5, color="#3a3a3a",
        ha="right", va="bottom", style="italic",
    )

    for a in ATTACK_ORDER:
        if a in exclude:    continue
        if a not in cdf:    continue
        ys, n_succ = cdf[a]
        if n_succ < min_jailbreaks: continue
        ax.plot(
            xs, ys,
            marker=MARKERS[a], color=PALETTE[a], label=a,
            linestyle=LINESTYLES[a],
            markeredgecolor="white", markeredgewidth=0.7,
            zorder=3,
        )

    ax.set_xlabel(r"Prefix length $t$")
    ax.set_ylabel(r"Detected by $t$ (%)")
    ax.set_ylim(0, ymax_left)
    ax.set_yticks(np.arange(0, ymax_left + 1, 20))
    ax.set_xlim(0, last_x + 20)

    leg = ax.legend(
        loc="lower right",
        frameon=True, framealpha=0.97,
        edgecolor="#aaaaaa", facecolor="white",
        ncols=2, columnspacing=1.1,
        handletextpad=0.5, borderpad=0.4,
        labelspacing=0.35, handlelength=2.4,
    )
    leg.get_frame().set_linewidth(0.7)

    sns.despine()
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"))
    print(f"saved → {out_path}")
    print(f"saved → {out_path.with_suffix('.png')}")


def plot(D, models, out_path, ymax_left, ymax_right, horizon_m, check_c):
    sns.set_theme(
        context="paper", style="ticks", font="serif", font_scale=1.45,
        rc={
            "axes.titlesize":   16,
            "axes.labelsize":   15,
            "xtick.labelsize":  13,
            "ytick.labelsize":  13,
            "legend.fontsize":  11.5,
            "axes.linewidth":   1.1,
            "axes.edgecolor":   "#1a1a1a",
            "axes.grid":        True,
            "grid.linestyle":   ":",
            "grid.linewidth":   0.6,
            "grid.color":       "#b8b8b8",
            "grid.alpha":       0.7,
            "xtick.direction":  "in",
            "ytick.direction":  "in",
            "lines.linewidth":  2.2,
            "lines.markersize": 8,
            "savefig.bbox":     "tight",
            "savefig.dpi":      300,
            "pdf.fonttype":     42,
            "ps.fonttype":      42,
            "mathtext.fontset": "stix",
            "font.family":      "serif",
        },
    )

    prefix = D["prefix_tokens"]
    # Render full-response point as max(prefix) + 50 for visual placement
    finite = [p for p in prefix if p is not None]
    last_x = max(finite) + 100
    xs = [p if p is not None else last_x for p in prefix]

    pooled = collect(D["data"], models)
    cdf    = detection_cdf(pooled)
    cond   = conditional(pooled)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # ── Panel A: detection CDF among successful jailbreaks ───────────────────
    # Shade PATROL's monitoring window [c, m]
    axes[0].axvspan(check_c, horizon_m, color="#dfe6ee", alpha=0.55, zorder=0,
                    label=None)
    axes[0].axvline(horizon_m, color="#444444", lw=1.0, ls="--", zorder=1)
    axes[0].text(horizon_m + 6, 4, fr"$m={horizon_m}$",
                 fontsize=11, color="#444444", ha="left", va="bottom")

    for a in ATTACK_ORDER:
        if a not in cdf: continue
        ys, n_succ = cdf[a]
        axes[0].plot(
            xs, ys,
            marker=MARKERS[a], color=PALETTE[a],
            label=f"{a} (n={n_succ})",
            markeredgecolor="white", markeredgewidth=0.7,
        )
    axes[0].set_xlabel(r"Prefix length $t$ (tokens)")
    axes[0].set_ylabel(r"Detected by $t$ among jailbreaks (\%)")
    axes[0].set_ylim(0, ymax_left)
    axes[0].set_yticks(np.arange(0, ymax_left + 1, 20))
    axes[0].set_title("Detection CDF (successful jailbreaks)", loc="left")
    axes[0].legend(loc="lower right", frameon=True, framealpha=0.95,
                   edgecolor="#cccccc", ncols=2)

    # ── Panel B: conditional commitment ─────────────────────────────────────
    axes[1].axvline(horizon_m, color="#444444", lw=1.0, ls="--", zorder=1)
    for a in ATTACK_ORDER:
        if a not in cond: continue
        ys = cond[a]
        mask = ~np.isnan(ys)
        axes[1].plot(
            np.array(xs)[mask], ys[mask],
            marker=MARKERS[a], color=PALETTE[a],
            label=a, markeredgecolor="white", markeredgewidth=0.7,
        )
    axes[1].set_xlabel(r"Prefix length $t$ (tokens)")
    axes[1].set_ylabel(r"$\Pr[\,\mathrm{final\ unsafe}\mid\mathrm{unsafe\ at}\ t\,]$ (\%)")
    axes[1].set_ylim(50, ymax_right)
    axes[1].set_yticks(np.arange(50, ymax_right + 1, 10))
    axes[1].set_title("Early-commitment probability", loc="left")

    sns.despine()
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"))
    print(f"saved → {out_path}")
    print(f"saved → {out_path.with_suffix('.png')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in",  dest="in_path",  type=Path,
                   default=REPO_ROOT / "figures" / "early_commitment.json")
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "figures" / "early_commitment.pdf")
    p.add_argument("--models", nargs="*", default=None,
                   help="Restrict aggregation to a subset of models.")
    p.add_argument("--ymax_left",  type=float, default=100)
    p.add_argument("--ymax_right", type=float, default=102)
    p.add_argument("--horizon_m",  type=int, default=128,
                   help="PATROL safety horizon; shaded/dashed in the plot.")
    p.add_argument("--check_c",    type=int, default=4,
                   help="PATROL check interval; lower bound of shaded band.")
    args = p.parse_args()

    if not args.in_path.exists():
        raise SystemExit(f"input not found: {args.in_path}\n"
                         "run compute_early_commitment.py first.")

    D = json.loads(args.in_path.read_text())
    plot_single(D, args.models, args.out, args.ymax_left,
                args.horizon_m, args.check_c)


if __name__ == "__main__":
    main()
