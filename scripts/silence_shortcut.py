"""Milestone 4: how much of the label leaks through silence alone?

A "detector" that never looks at the voice: a logistic regression on the
length of the leading / trailing silence. Trained on train, evaluated on eval.
Two controls: total file duration only, and speech duration only (file minus
the two silences). Everything is repeated for the three silence thresholds.

Needs results/m4/silence_{train,eval}.csv from scripts/silence_stats.py.

Outputs in results/m4/:
    silence_only.csv            EER (+ 95 % CI) per feature set and threshold
    silence_stats_by_class.csv  median / quartiles of the silences per class
    silence_hist.png

Usage:
    python scripts/silence_shortcut.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adg.metrics import bootstrap_eer  # noqa: E402
from adg.protocols import UNKNOWN_EVAL_ATTACKS  # noqa: E402

TEAL, VIOLET = "#1baf7a", "#6250d6"
THRESHOLDS_DB = (30, 40, 50)


def feature_sets(df: pd.DataFrame, t: int) -> dict[str, np.ndarray]:
    lead, trail = df[f"lead_s_{t}db"].to_numpy(), df[f"trail_s_{t}db"].to_numpy()
    dur = df["duration_s"].to_numpy()
    # log1p: silences are skewed (many short, a few long); the log keeps the
    # logistic regression from being driven by a handful of extreme files.
    return {
        "leading + trailing silence": np.c_[np.log1p(lead), np.log1p(trail)],
        "file duration only": np.c_[np.log1p(dur)],
        "speech duration only": np.c_[np.log1p(dur - lead - trail)],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, default=Path("results/m4"))
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    train = pd.read_csv(args.dir / "silence_train.csv")
    ev = pd.read_csv(args.dir / "silence_eval.csv")
    unseen = (ev["label"] == 0) & ev["attack"].isin(UNKNOWN_EVAL_ATTACKS)

    rows = []
    for t in THRESHOLDS_DB:
        ft, fe = feature_sets(train, t), feature_sets(ev, t)
        for name in ft:
            model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))
            model.fit(ft[name], train["label"])
            s = model.decision_function(fe[name])
            bona = s[ev["label"] == 1]
            for subset, spoof in (("pooled", s[ev["label"] == 0]), ("unseen attacks", s[unseen])):
                b = bootstrap_eer(bona, spoof, n_resamples=args.n_boot, seed=0)
                rows.append({"threshold_db": t, "features": name, "eval subset": subset,
                             "eer_pct": 100 * b.estimate, "low": 100 * b.low, "high": 100 * b.high})
    table = pd.DataFrame(rows)
    table.to_csv(args.dir / "silence_only.csv", index=False, float_format="%.3f")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    stats = []
    for part, df in (("train", train), ("eval", ev)):
        for cls, sub in df.groupby("key"):
            for col in ("lead_s_40db", "trail_s_40db", "duration_s"):
                q = sub[col].quantile([0.25, 0.5, 0.75])
                stats.append({"partition": part, "class": cls, "quantity": col,
                              "q25": q.iloc[0], "median": q.iloc[1], "q75": q.iloc[2]})
    stats = pd.DataFrame(stats)
    stats.to_csv(args.dir / "silence_stats_by_class.csv", index=False, float_format="%.3f")
    print(stats.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    bins = np.linspace(0, 1.5, 61)
    for ax, col, title in ((axes[0], "lead_s_40db", "leading silence"), (axes[1], "trail_s_40db", "trailing silence")):
        for key, color, label in (("bonafide", TEAL, "bona fide"), ("spoof", VIOLET, "spoof")):
            v = ev.loc[ev["key"] == key, col].clip(upper=1.5)
            # outlines only: two filled, overlapping histograms mix into blue
            ax.hist(v, bins=bins, density=True, histtype="step", linewidth=1.8, color=color, label=label)
        ax.set_title(f"ASVspoof 2019 LA eval, {title} (40 dB below peak)", fontsize=9)
        ax.set_xlabel("seconds (values above 1.5 s shown at 1.5 s)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("density")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.dir / "silence_hist.png", dpi=150)
    return 0


if __name__ == "__main__":
    sys.exit(main())
