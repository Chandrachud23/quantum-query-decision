"""Predicted vs measured fidelity figure for the hardware validation."""

from __future__ import annotations

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

OUT = os.path.join(os.path.dirname(__file__), "output")

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04, "pdf.fonttype": 42, "axes.linewidth": 0.5,
    "xtick.major.width": 0.4, "ytick.major.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.axisbelow": True,
})
BLUE, ORANGE = "#1f77b4", "#ff7f0e"


def main():
    with open(os.path.join(OUT, "validation_hw.json")) as f:
        d = json.load(f)
    rows = d["rows"]
    pred = np.array([r["pred_log10F"] for r in rows])
    meas = np.array([r["meas_log10F"] for r in rows])
    err = np.array([r.get("meas_std", 0.0) for r in rows])
    multi = np.array([r["marks"] > 1 for r in rows])
    r_p = d.get("pearson", pearsonr(pred, meas)[0])
    r_s = d.get("spearman", spearmanr(pred, meas)[0])

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    lo = float(min(pred.min(), meas.min())) - 0.08
    hi = float(max(pred.max(), meas.max())) + 0.08
    ax.plot([lo, hi], [lo, hi], "--", color="0.5", lw=0.8, label="$y=x$", zorder=1)
    ax.errorbar(pred[~multi], meas[~multi], yerr=err[~multi], fmt="o", ms=4,
                c=BLUE, capsize=2, elinewidth=0.6, zorder=3, label="point retrieval")
    ax.errorbar(pred[multi], meas[multi], yerr=err[multi], fmt="s", ms=4,
                c=ORANGE, capsize=2, elinewidth=0.6, zorder=3, label="multi-target ($k$NN)")
    ax.set_xlabel(r"predicted $\log_{10}F$")
    ax.set_ylabel(r"measured $\log_{10}F$ (ibm\_fez)")
    ax.grid(which="major", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.legend(loc="lower right", handlelength=1.4, borderpad=0.3, labelspacing=0.3,
              title=rf"$r={r_p:.2f}$, $\rho={r_s:.2f}$", title_fontsize=7)
    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_hw.{ext}"))
    plt.close(fig)
    print(f"wrote fig_hw.pdf/png  (r={r_p:.3f}, rho={r_s:.3f}, n={len(rows)})")


if __name__ == "__main__":
    main()
