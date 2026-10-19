"""Milestone 2: one linear probe per wav2vec 2.0 layer, EER as a function of depth.

Needs the features produced by scripts/extract_wav2vec2.py (on Kaggle).

Outputs in results/w2v2_probes/:
    metrics_per_layer.csv   pooled / known / unknown EER (+ CI) and min t-DCF per layer
    metrics_best_layer.csv  full per-attack breakdown for the layer with the best dev EER
    scores_eval_layer<k>.csv
    eer_per_layer.png

Usage:
    python scripts/run_probes.py --feat-dir features/w2v2_base
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adg.evaluation import evaluate, join_scores, load_asv_rates  # noqa: E402
from adg.probes import fit_layer_probe, load_split, probe_scores  # noqa: E402
from adg.protocols import load_partition  # noqa: E402


def plot(table: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    # teal, violet, magenta: blue and orange only once every other colour is used.
    styles = {"pooled": ("#1baf7a", "o", "pooled (13 attacks)"),
              "known (A16, A19)": ("#6250d6", "s", "known attacks (A16, A19)"),
              "unknown (11 attacks)": ("#e87ba4", "^", "unseen attacks (11)")}
    for cond, (color, marker, label) in styles.items():
        sub = table[table["condition"] == cond].sort_values("layer")
        ax.plot(sub["layer"], sub["eer_pct"], marker=marker, color=color, label=label)
        ax.fill_between(sub["layer"], sub["eer_low_pct"], sub["eer_high_pct"], color=color, alpha=0.15)
    ax.set_xlabel("wav2vec 2.0 BASE transformer layer")
    ax.set_ylabel("eval EER (%)")
    ax.set_xticks(range(1, 13))
    ax.set_title("Linear probe per layer, ASVspoof 2019 LA eval (bands: 95 % bootstrap CI)")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--la-root", type=Path, default=Path("data/LA"))
    ap.add_argument("--feat-dir", type=Path, default=Path("features/w2v2_base"))
    ap.add_argument("--out", type=Path, default=Path("results/w2v2_probes"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    proto_dir = args.la_root / "ASVspoof2019_LA_cm_protocols"
    protos = {p: load_partition(proto_dir, p) for p in ("train", "dev", "eval")}
    feats = {p: load_split(args.feat_dir, p, protos[p]) for p in protos}
    asv = load_asv_rates(args.la_root)
    n_layers = feats["train"].shape[1]

    rows, chosen, best = [], {}, None
    for k in range(n_layers):
        t0 = time.perf_counter()
        probe = fit_layer_probe(k, feats["train"][:, k], protos["train"]["label"],
                                feats["dev"][:, k], protos["dev"]["label"], seed=args.seed)
        scores = pd.DataFrame({"utt_id": protos["eval"]["utt_id"],
                               "score": probe_scores(probe.model, feats["eval"][:, k])})
        scores.to_csv(args.out / f"scores_eval_layer{k + 1}.csv", index=False)
        table = evaluate(join_scores(scores, protos["eval"]), asv, n_boot=args.n_boot,
                         seed=args.seed, per_attack=False)
        table.insert(0, "layer", k + 1)
        table["dev_eer_pct"] = 100 * probe.dev_eer
        table["C"] = probe.c
        rows.append(table)
        chosen[k + 1] = probe.c
        if best is None or probe.dev_eer < best[1]:
            best = (k + 1, probe.dev_eer)
        pooled = table.loc[table["condition"] == "pooled", "eer_pct"].iloc[0]
        print(f"layer {k + 1:2d}: C={probe.c:g}, dev EER {100 * probe.dev_eer:.2f} %, "
              f"eval EER {pooled:.2f} % ({time.perf_counter() - t0:.0f} s)", flush=True)

    per_layer = pd.concat(rows, ignore_index=True)
    per_layer.to_csv(args.out / "metrics_per_layer.csv", index=False, float_format="%.4f")
    plot(per_layer, args.out / "eer_per_layer.png")

    # Per-attack detail only for the layer picked on dev (not the best on eval).
    best_layer = best[0]
    scores = pd.read_csv(args.out / f"scores_eval_layer{best_layer}.csv")
    detail = evaluate(join_scores(scores, protos["eval"]), asv, n_boot=args.n_boot, seed=args.seed)
    detail.to_csv(args.out / "metrics_best_layer.csv", index=False, float_format="%.4f")
    (args.out / "run_info.json").write_text(json.dumps(
        {"best_layer_on_dev": best_layer, "best_dev_eer": best[1], "C_per_layer": chosen,
         "seed": args.seed, "n_boot": args.n_boot}, indent=2))
    print(f"best layer on dev: {best_layer}")
    print(detail.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
