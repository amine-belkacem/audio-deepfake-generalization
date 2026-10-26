"""Milestone 5: EER through ten simulated channels, with and without channel augmentation.

All systems are the milestone 4 ones trained without leading / trailing silence
(trim40); "aug" versions are retrained on clean + randomly degraded copies of
train and dev. Test set: the 5,900-file eval subset, silence trimmed, then sent
through each channel. Score files only; nothing is trained here.

Outputs in results/m5/: channels.csv, channels.md, channels_eer.png

Usage:
    python scripts/evaluate_m5.py
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from adg.datasets import load_labels  # noqa: E402
from adg.transforms import CHANNELS  # noqa: E402
from evaluate_m4 import run_eers, summarise  # noqa: E402

M5 = Path("results/m5")
SEEDS = (0, 1, 2)
TEAL, VIOLET, MAGENTA, YELLOW, GREY = "#1baf7a", "#6250d6", "#e87ba4", "#eda100", "#8a8986"
LABELS = {"none": "clean", "tel8k": "8 kHz", "g711": "G.711", "gsm": "GSM", "amr12k": "AMR-NB 12.2k",
          "opus6k": "Opus 6k", "opus12k": "Opus 12k", "opus24k": "Opus 24k",
          "loss1": "loss 1 %", "loss5": "loss 5 %", "loss10": "loss 10 %"}


def part(ch: str) -> str:
    return "evalsub_trim40" if ch == "none" else f"evalsub_trim40-{ch}"


# (system, training) -> function giving the score files of a test part
SYSTEMS = {
    ("LFCC + GMM", "no augmentation"): lambda p: [M5 / f"lfcc_gmm_trim40/scores_{p}.csv"],
    ("LFCC + LCNN", "no augmentation"): lambda p: [M5 / f"lfcc_lcnn_trim40/scores_{p}_seed{s}.csv" for s in SEEDS],
    ("LFCC + LCNN", "channel augmentation"): lambda p: [M5 / f"lfcc_lcnn_aug/scores_{p}_seed{s}.csv" for s in SEEDS],
    ("wav2vec 2.0 + probe", "no augmentation"): lambda p: [M5 / f"w2v2_probe_trim40/scores_{p}.csv"],
    ("wav2vec 2.0 + probe", "channel augmentation"): lambda p: [M5 / f"w2v2_probe_aug/scores_{p}.csv"],
    ("wav2vec 2.0 + MLP", "no augmentation"): lambda p: [M5 / f"w2v2_mlp_trim40/scores_{p}_seed{s}.csv" for s in SEEDS],
    ("wav2vec 2.0 + MLP", "channel augmentation"): lambda p: [M5 / f"w2v2_mlp_aug/scores_{p}_seed{s}.csv" for s in SEEDS],
}
COLORS = {"LFCC + GMM": TEAL, "LFCC + LCNN": VIOLET, "wav2vec 2.0 + probe": MAGENTA, "wav2vec 2.0 + MLP": YELLOW}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    keep = set(pd.read_csv("results/m4/band_subset_eval.csv")["utt_id"])
    ev = load_labels("eval")
    labels = ev[ev["utt_id"].isin(keep)].reset_index(drop=True)

    rows = []
    for (system, training), files_for in SYSTEMS.items():
        for ch in ("none", *CHANNELS):
            mean, sd, text = summarise(run_eers(files_for(part(ch)), labels, args.n_boot))
            rows.append({"system": system, "training": training, "channel": ch,
                         "eer_mean": mean, "eer_std": sd, "text": text})
        print(f"{system}, {training}: done", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(M5 / "channels.csv", index=False, float_format="%.4f")

    chans = ("none", *CHANNELS)
    lines = ["| System | Training | " + " | ".join(LABELS[c] for c in chans) + " | mean over the 10 channels |",
             "|" + "---|" * (len(chans) + 3)]
    for (system, training) in SYSTEMS:
        sub = table[(table.system == system) & (table.training == training)].set_index("channel")
        cells = [f"{sub.loc[c, 'eer_mean']:.2f}" for c in chans]
        lines.append(f"| {system} | {training} | " + " | ".join(cells)
                     + f" | {sub.loc[list(CHANNELS), 'eer_mean'].mean():.2f} |")
    (M5 / "channels.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    fig, ax = plt.subplots(figsize=(11, 5.6))
    x = np.arange(len(chans))
    for (system, training) in SYSTEMS:
        sub = table[(table.system == system) & (table.training == training)].set_index("channel")
        y = [sub.loc[c, "eer_mean"] for c in chans]
        dashed = training != "no augmentation"
        ax.plot(x, y, marker="s" if dashed else "o", linestyle="--" if dashed else "-",
                color=COLORS[system], label=f"{system}, {training}")
    ax.axvline(0.5, color=GREY, linewidth=0.8)
    ax.set_xticks(x, [LABELS[c] for c in chans], rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("EER (%) on the 5,900-file eval subset")
    ax.set_title("Silence removed, then each channel; dashed: retrained with channel augmentation", fontsize=10)
    # below the axes: inside, it covered the GMM curve
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(M5 / "channels_eer.png", dpi=150)
    return 0


if __name__ == "__main__":
    sys.exit(main())
