"""Bar chart of the EER per eval attack, with bootstrap 95 % intervals.

Known attacks (A16, A19) and unseen ones get different colours, so the
generalisation gap is visible at a glance.

Usage:
    python scripts/plot_per_attack.py results/lfcc_gmm/metrics_eval.csv results/lfcc_gmm/eer_per_attack.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adg.protocols import EVAL_ATTACKS, KNOWN_EVAL_ATTACKS  # noqa: E402

# Two series: teal + violet. Blue and orange are kept for when nothing else is left.
COLOR_KNOWN = "#6250d6"    # violet
COLOR_UNKNOWN = "#1baf7a"  # teal
COLOR_REFERENCE = "#8a8986"  # grey, for the pooled line


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("metrics_csv", type=Path)
    ap.add_argument("out_png", type=Path)
    ap.add_argument("--title", default="LFCC + GMM baseline, ASVspoof 2019 LA eval")
    args = ap.parse_args()

    df = pd.read_csv(args.metrics_csv).set_index("condition")
    attacks = sorted(a for a in df.index if a in EVAL_ATTACKS)
    if not attacks:
        raise SystemExit(f"no per-attack rows in {args.metrics_csv}")
    sub = df.loc[attacks]
    colors = [COLOR_KNOWN if a in KNOWN_EVAL_ATTACKS else COLOR_UNKNOWN for a in attacks]
    err = [sub["eer_pct"] - sub["eer_low_pct"], sub["eer_high_pct"] - sub["eer_pct"]]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(attacks, sub["eer_pct"], color=colors, yerr=err, capsize=3)
    pooled = df.loc["pooled", "eer_pct"]
    ax.axhline(pooled, color=COLOR_REFERENCE, linestyle="--", linewidth=1)
    ax.set_ylim(0, 1.3 * float(sub["eer_high_pct"].max()))  # head room for the legend
    ax.set_ylabel("EER (%)")
    ax.set_title(args.title)
    ax.legend(handles=[Patch(color=COLOR_KNOWN, label="known (same algorithm as a training attack)"),
                       Patch(color=COLOR_UNKNOWN, label="unseen generator"),
                       Line2D([], [], color=COLOR_REFERENCE, linestyle="--",
                              label=f"pooled over all attacks: {pooled:.2f} %")],
              frameon=False, fontsize=9, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=150)
    print(f"saved {args.out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
