"""Milestone 3: compare the four systems on ASVspoof 2019 LA eval and on In-the-Wild.

Reads score files only (nothing is trained here):
    LFCC + GMM            results/lfcc_gmm_asvspoof2019_b02/scores_{eval,itw}.csv     (1 run)
    LFCC + LCNN           results/lfcc_lcnn/scores_{eval,itw}_seed{0,1,2}.csv        (3 seeds, Kaggle)
    wav2vec 2.0 probe     results/w2v2_probes/scores_{eval,itw}_layer3.csv            (layer picked on dev)
    wav2vec 2.0 MLP       results/w2v2_mlp/scores_{eval,itw}_seed{0,1,2}.csv          (3 seeds)

Outputs in results/m3/:
    per_run.csv      one row per system x run x test set: EER + 95 % bootstrap CI (+ min t-DCF on eval)
    summary.csv/.md  mean +- std over seeds (sample std, n-1), or the single run
    per_attack.csv   eval EER per attack, averaged over seeds
    eer_eval_vs_itw.png, eer_per_attack.png

Usage:
    python scripts/evaluate_m3.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adg.datasets import load_labels  # noqa: E402
from adg.evaluation import evaluate, join_scores, load_asv_rates  # noqa: E402
from adg.metrics import bootstrap_eer  # noqa: E402

TEAL, VIOLET, GREY = "#1baf7a", "#6250d6", "#8a8986"

SYSTEMS = {
    "LFCC + GMM": ("lfcc_gmm_asvspoof2019_b02", ["scores_{part}.csv"]),
    "LFCC + LCNN": ("lfcc_lcnn", [f"scores_{{part}}_seed{s}.csv" for s in range(3)]),
    "wav2vec 2.0 + linear probe (layer 3)": ("w2v2_probes", ["scores_{part}_layer3.csv"]),
    "wav2vec 2.0 + MLP (weighted layers)": ("w2v2_mlp", [f"scores_{{part}}_seed{s}.csv" for s in range(3)]),
}


def mean_std(values: list[float]) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1)) if v.size > 1 else float("nan"))


def fmt(mean: float, std: float) -> str:
    return f"{mean:.2f}" if np.isnan(std) else f"{mean:.2f} ± {std:.2f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--la-root", type=Path, default=Path("data/LA"))
    ap.add_argument("--itw-root", type=Path, default=Path("data/in_the_wild"))
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    out = args.results / "m3"
    out.mkdir(parents=True, exist_ok=True)

    labels = {"eval": load_labels("eval", la_root=args.la_root), "itw": load_labels("itw", itw_root=args.itw_root)}
    asv = load_asv_rates(args.la_root)

    runs, attacks = [], []
    for system, (folder, patterns) in SYSTEMS.items():
        for run, pattern in enumerate(patterns):
            for part in ("eval", "itw"):
                path = args.results / folder / pattern.format(part=part)
                merged = join_scores(pd.read_csv(path), labels[part])
                if part == "eval":
                    table = evaluate(merged, asv, n_boot=args.n_boot, seed=run).set_index("condition")
                    for cond in ("pooled", "unknown (11 attacks)"):
                        row = table.loc[cond]
                        runs.append({"system": system, "run": run, "test": "eval" if cond == "pooled" else "eval unseen",
                                     "eer_pct": row["eer_pct"], "low": row["eer_low_pct"], "high": row["eer_high_pct"],
                                     "min_tdcf": row["min_tdcf"], "file": str(path)})
                    for att, row in table.iterrows():
                        if att.startswith("A"):
                            attacks.append({"system": system, "run": run, "attack": att, "eer_pct": row["eer_pct"]})
                else:
                    y = merged["label"].to_numpy()
                    s = merged["score"].to_numpy()
                    b = bootstrap_eer(s[y == 1], s[y == 0], n_resamples=args.n_boot, seed=run)
                    runs.append({"system": system, "run": run, "test": "In-the-Wild", "eer_pct": 100 * b.estimate,
                                 "low": 100 * b.low, "high": 100 * b.high, "min_tdcf": np.nan, "file": str(path)})
            print(f"{system}, run {run}: done", flush=True)

    per_run = pd.DataFrame(runs)
    per_run.to_csv(out / "per_run.csv", index=False, float_format="%.4f")

    summary = []
    for system in SYSTEMS:
        row = {"system": system, "runs": int(per_run.loc[per_run.system == system, "run"].nunique())}
        for test in ("eval", "eval unseen", "In-the-Wild"):
            sub = per_run[(per_run.system == system) & (per_run.test == test)]
            row[f"{test} EER mean"], row[f"{test} EER std"] = mean_std(sub["eer_pct"].tolist())
            if row["runs"] == 1:  # single run: report its bootstrap interval instead of a std
                row[f"{test} CI"] = f"[{sub['low'].iloc[0]:.2f}, {sub['high'].iloc[0]:.2f}]"
        ev = per_run[(per_run.system == system) & (per_run.test == "eval")]
        row["min t-DCF mean"], row["min t-DCF std"] = mean_std(ev["min_tdcf"].tolist())
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "summary.csv", index=False, float_format="%.4f")

    lines = ["| System | Runs | eval EER (%) | unseen attacks EER (%) | In-the-Wild EER (%) | eval min t-DCF |",
             "|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        cells = []
        for test in ("eval", "eval unseen", "In-the-Wild"):
            c = fmt(r[f"{test} EER mean"], r[f"{test} EER std"])
            if r["runs"] == 1:
                c += f" {r[f'{test} CI']}"
            cells.append(c)
        tdcf = (f"{r['min t-DCF mean']:.3f}" if np.isnan(r["min t-DCF std"])
                else f"{r['min t-DCF mean']:.3f} ± {r['min t-DCF std']:.3f}")
        lines.append(f"| {r['system']} | {r['runs']} | " + " | ".join(cells) + f" | {tdcf} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    att = pd.DataFrame(attacks).groupby(["system", "attack"], sort=False)["eer_pct"].mean().unstack()
    att = att.loc[list(SYSTEMS)]
    att.to_csv(out / "per_attack.csv", float_format="%.3f")

    # Figure 1: eval vs In-the-Wild, error bars = std over seeds (or 95 % CI for single runs).
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(SYSTEMS))
    for i, (test, color) in enumerate((("eval", TEAL), ("In-the-Wild", VIOLET))):
        means, lo, hi = [], [], []
        for system in SYSTEMS:
            sub = per_run[(per_run.system == system) & (per_run.test == test)]
            m, sd = mean_std(sub["eer_pct"].tolist())
            means.append(m)
            if np.isnan(sd):
                lo.append(m - sub["low"].iloc[0])
                hi.append(sub["high"].iloc[0] - m)
            else:
                lo.append(sd)
                hi.append(sd)
        bars = ax.bar(x + (i - 0.5) * 0.38, means, 0.38, yerr=[lo, hi], capsize=3, color=color,
                      label="ASVspoof 2019 LA eval" if test == "eval" else "In-the-Wild (real-world deepfakes)")
        for b, m, h in zip(bars, means, hi):
            ax.text(b.get_x() + b.get_width() / 2, m + h + 0.8, f"{m:.1f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(50, color=GREY, linestyle="--", linewidth=1, label="chance level (EER 50 %)")
    ax.set_xticks(x, [s.replace(" (", "\n(") for s in SYSTEMS], fontsize=8.5)
    ax.set_ylabel("EER (%)")
    ax.set_title("Trained on ASVspoof 2019 LA, tested in and out of domain")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "eer_eval_vs_itw.png", dpi=150)

    # Figure 2: per-attack heat map (white -> violet, no blue/orange).
    cmap = LinearSegmentedColormap.from_list("white_violet", ["#ffffff", VIOLET])
    fig, ax = plt.subplots(figsize=(10, 3.2))
    im = ax.imshow(att.to_numpy(), cmap=cmap, aspect="auto")
    ax.set_xticks(range(att.shape[1]), att.columns, fontsize=8)
    ax.set_yticks(range(att.shape[0]), att.index, fontsize=8)
    vmax = float(np.nanmax(att.to_numpy()))
    for (i, j), v in np.ndenumerate(att.to_numpy()):
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7, color="white" if v > 0.55 * vmax else "black")
    ax.set_title("EER (%) per eval attack, mean over runs (A16 and A19: algorithms seen in training)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.025, label="EER (%)")
    fig.tight_layout()
    fig.savefig(out / "eer_per_attack.png", dpi=150)

    (out / "run_info.json").write_text(json.dumps({"n_boot": args.n_boot, "bootstrap_seed": "run index",
                                                    "std": "sample std over seeds (ddof=1)"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
