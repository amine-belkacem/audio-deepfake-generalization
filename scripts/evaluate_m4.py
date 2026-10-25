"""Milestone 4: silence and frequency-band results for the four systems.

Reads score files only. For each system and test set: EER per run with a 95 %
bootstrap interval (1000 resamples), then mean +- sample std over seeds.

    silence.csv / silence.md   as is | silence removed, same model | silence removed, retrained
                               on ASVspoof eval and on In-the-Wild
    bands.csv / bands.md       EER on the 5,900-file eval subset, untouched and with each 1 kHz band removed
    silence_eer.png, bands_eer.png

Usage:
    python scripts/evaluate_m4.py
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adg.datasets import load_labels  # noqa: E402
from adg.metrics import bootstrap_eer  # noqa: E402

TEAL, VIOLET, MAGENTA, YELLOW, GREY = "#1baf7a", "#6250d6", "#e87ba4", "#eda100", "#8a8986"
BANDS = [f"band{lo}-{lo + 1000}" for lo in range(0, 8000, 1000)]
R = Path("results")
M4 = R / "m4"
SEEDS = (0, 1, 2)

# system -> condition -> test set -> list of score files (one per run)
def seeded(folder, part):
    return [M4 / folder / f"scores_{part}_seed{s}.csv" for s in SEEDS]


SILENCE = {
    "LFCC + GMM": {
        ("eval", "as is"): [R / "lfcc_gmm_asvspoof2019_b02/scores_eval.csv"],
        ("eval", "trimmed, same model"): [M4 / "lfcc_gmm/scores_eval_trimtest_40db.csv"],
        ("eval", "trimmed, retrained"): [M4 / "lfcc_gmm/scores_eval_trimboth_40db.csv"],
        ("itw", "as is"): [R / "lfcc_gmm_asvspoof2019_b02/scores_itw.csv"],
        ("itw", "trimmed, same model"): [M4 / "lfcc_gmm/scores_itw_trim40.csv"],
    },
    "LFCC + LCNN": {
        ("eval", "as is"): seeded("lfcc_lcnn_normal", "eval"),
        ("eval", "trimmed, same model"): seeded("lfcc_lcnn_normal", "eval_trim40"),
        ("eval", "trimmed, retrained"): seeded("lfcc_lcnn_trim40", "eval_trim40"),
        ("itw", "as is"): seeded("lfcc_lcnn_normal", "itw"),
        ("itw", "trimmed, same model"): seeded("lfcc_lcnn_normal", "itw_trim40"),
        ("itw", "trimmed, retrained"): seeded("lfcc_lcnn_trim40", "itw_trim40"),
    },
    "wav2vec 2.0 + probe (layer 3)": {
        ("eval", "as is"): [M4 / "w2v2_probe_normal/scores_eval.csv"],
        ("eval", "trimmed, same model"): [M4 / "w2v2_probe_normal/scores_eval_trim40.csv"],
        ("eval", "trimmed, retrained"): [M4 / "w2v2_probe_trim40/scores_eval_trim40.csv"],
        ("itw", "as is"): [M4 / "w2v2_probe_normal/scores_itw.csv"],
        ("itw", "trimmed, same model"): [M4 / "w2v2_probe_normal/scores_itw_trim40.csv"],
        ("itw", "trimmed, retrained"): [M4 / "w2v2_probe_trim40/scores_itw_trim40.csv"],
    },
    "wav2vec 2.0 + MLP": {
        ("eval", "as is"): seeded("w2v2_mlp_normal", "eval"),
        ("eval", "trimmed, same model"): seeded("w2v2_mlp_normal", "eval_trim40"),
        ("eval", "trimmed, retrained"): seeded("w2v2_mlp_trim40", "eval_trim40"),
        ("itw", "as is"): seeded("w2v2_mlp_normal", "itw"),
        ("itw", "trimmed, same model"): seeded("w2v2_mlp_normal", "itw_trim40"),
        ("itw", "trimmed, retrained"): seeded("w2v2_mlp_trim40", "itw_trim40"),
    },
}

BAND_FILES = {
    "LFCC + GMM": lambda part: [M4 / f"lfcc_gmm/scores_{part}.csv"],
    "LFCC + LCNN": lambda part: seeded("lfcc_lcnn_normal", part),
    "wav2vec 2.0 + probe (layer 3)": lambda part: [M4 / f"w2v2_probe_normal/scores_{part}.csv"],
    "wav2vec 2.0 + MLP": lambda part: seeded("w2v2_mlp_normal", part),
}
COLORS = dict(zip(BAND_FILES, (TEAL, VIOLET, MAGENTA, YELLOW)))
CONDITIONS = ("as is", "trimmed, same model", "trimmed, retrained")


def run_eers(files, labels, n_boot):
    out = []
    for run, f in enumerate(files):
        m = labels.merge(pd.read_csv(f), on="utt_id", validate="one_to_one")
        if m["score"].isna().any():
            raise ValueError(f"{f}: missing scores")
        y, s = m["label"].to_numpy(), m["score"].to_numpy()
        b = bootstrap_eer(s[y == 1], s[y == 0], n_resamples=n_boot, seed=run)
        out.append((100 * b.estimate, 100 * b.low, 100 * b.high))
    return out


def summarise(eers):
    est = np.array([e[0] for e in eers])
    if len(eers) == 1:
        return est[0], np.nan, f"{est[0]:.2f} [{eers[0][1]:.2f}, {eers[0][2]:.2f}]"
    sd = est.std(ddof=1)
    return est.mean(), sd, f"{est.mean():.2f} ± {sd:.2f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    labels = {"eval": load_labels("eval"), "itw": load_labels("itw")}
    subset_ids = set(pd.read_csv(M4 / "band_subset_eval.csv")["utt_id"])
    labels["evalsub"] = labels["eval"][labels["eval"]["utt_id"].isin(subset_ids)].reset_index(drop=True)

    # --- silence
    rows = []
    for system, conds in SILENCE.items():
        for (test, cond), files in conds.items():
            mean, sd, text = summarise(run_eers(files, labels[test], args.n_boot))
            rows.append({"system": system, "test": test, "condition": cond, "runs": len(files),
                         "eer_mean": mean, "eer_std": sd, "text": text})
        print(f"silence: {system} done", flush=True)
    sil = pd.DataFrame(rows)
    sil.to_csv(M4 / "silence.csv", index=False, float_format="%.4f")
    lines = ["| System | eval, as is | eval, silence removed (same model) | eval, silence removed (retrained) "
             "| In-the-Wild, as is | In-the-Wild, silence removed (same model) | In-the-Wild, silence removed (retrained) |",
             "|---|---|---|---|---|---|---|"]
    for system in SILENCE:
        cells = []
        for test in ("eval", "itw"):
            for cond in CONDITIONS:
                hit = sil[(sil.system == system) & (sil.test == test) & (sil.condition == cond)]
                cells.append(hit["text"].iloc[0] if len(hit) else "not run")
        lines.append(f"| {system} | " + " | ".join(cells) + " |")
    (M4 / "silence.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # --- bands
    rows = []
    for system, files_for in BAND_FILES.items():
        for part in ["evalsub"] + [f"evalsub_{b}" for b in BANDS]:
            mean, sd, text = summarise(run_eers(files_for(part), labels["evalsub"], args.n_boot))
            rows.append({"system": system, "band": "none" if part == "evalsub" else part.split("_", 1)[1],
                         "eer_mean": mean, "eer_std": sd, "text": text})
        print(f"bands: {system} done", flush=True)
    bands = pd.DataFrame(rows)
    ref = bands[bands.band == "none"].set_index("system")["eer_mean"]
    bands["delta_vs_untouched"] = bands["eer_mean"] - bands["system"].map(ref)
    bands.to_csv(M4 / "bands.csv", index=False, float_format="%.4f")
    header = "| System | untouched | " + " | ".join(b.replace("band", "").replace("-", "–") + " Hz removed" for b in BANDS) + " |"
    lines = [header, "|" + "---|" * (len(BANDS) + 2)]
    for system in BAND_FILES:
        sub = bands[bands.system == system].set_index("band")
        lines.append(f"| {system} | {sub.loc['none', 'text']} | "
                     + " | ".join(f"{sub.loc[b, 'eer_mean']:.2f} ({sub.loc[b, 'delta_vs_untouched']:+.2f})" for b in BANDS) + " |")
    (M4 / "bands.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # --- figure: silence (eval and In-the-Wild side by side)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=False)
    x = np.arange(len(SILENCE))
    for ax, test, title in ((axes[0], "eval", "ASVspoof 2019 LA eval"), (axes[1], "itw", "In-the-Wild")):
        for i, (cond, color) in enumerate(zip(CONDITIONS, (TEAL, VIOLET, MAGENTA))):
            means, errs = [], []
            for system in SILENCE:
                hit = sil[(sil.system == system) & (sil.test == test) & (sil.condition == cond)]
                means.append(hit["eer_mean"].iloc[0] if len(hit) else np.nan)
                errs.append(hit["eer_std"].iloc[0] if len(hit) and not np.isnan(hit["eer_std"].iloc[0]) else 0)
            bars = ax.bar(x + (i - 1) * 0.27, means, 0.27, yerr=errs, capsize=2, color=color, label=cond)
            for b, m, e in zip(bars, means, errs):
                if not np.isnan(m):
                    ax.text(b.get_x() + b.get_width() / 2, m + e + 0.6, f"{m:.1f}", ha="center", fontsize=7)
        if test == "itw":
            ax.axhline(50, color=GREY, linestyle="--", linewidth=1, label="chance level")
        ax.set_xticks(x, [s.replace(" + ", "\n+ ") for s in SILENCE], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("EER (%)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, title="leading / trailing silence", title_fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("What is left once the silence at the edges is removed (40 dB below peak)", fontsize=11)
    fig.tight_layout()
    fig.savefig(M4 / "silence_eer.png", dpi=150)

    # --- figure: bands
    fig, ax = plt.subplots(figsize=(9, 4))
    centres = np.arange(len(BANDS))
    for system, color in COLORS.items():
        sub = bands[bands.system == system].set_index("band")
        ax.plot(centres, [sub.loc[b, "eer_mean"] for b in BANDS], marker="o", color=color, label=system)
        ax.axhline(sub.loc["none", "eer_mean"], color=color, linestyle=":", linewidth=1)
    ax.set_xticks(centres, [f"{lo // 1000}-{lo // 1000 + 1}" for lo in range(0, 8000, 1000)])
    ax.set_xlabel("band removed (kHz)   dotted lines: same system, nothing removed")
    ax.set_ylabel("EER (%) on the 5,900-file eval subset")
    ax.set_title("Which frequency band does each detector rely on?", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(M4 / "bands_eer.png", dpi=150)
    return 0


if __name__ == "__main__":
    sys.exit(main())
