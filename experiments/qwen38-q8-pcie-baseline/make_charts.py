#!/usr/bin/env python3
"""Regenerate the baseline charts from the committed results.csv.

Declared plotting dependency set (nothing else is imported):
    pandas >= 2.0
    matplotlib >= 3.7

Charts produced (SVG + PNG, identical content):
  * generation-throughput-vs-cards   — warm generation throughput (tok/s)
    per context bucket against card count
  * prompt-throughput-vs-cards       — warm prompt-processing throughput
    (tok/s) per context bucket against card count
  * warm-ttft-vs-cards               — warm TTFT (ms) per context bucket
    against card count

Accessibility choices: Okabe–Ito colorblind-safe palette, one distinct
marker per series, direct unit labels on both axes, min–max band across the
observed warm repetitions, and an explicit "n=k" annotation whenever a cell
has fewer than the contracted 3 warm repetitions (reduced sample counts are
labeled, never hidden). Cold measurements are excluded from all charts and
stated in each figure footer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Okabe–Ito colorblind-safe palette + distinct markers
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
MARKERS = ["o", "s", "^", "D"]

METRICS = {
    "generation": (
        "generation_throughput_tok_s",
        "Generation throughput (tok/s)",
        "generation-throughput-vs-cards",
    ),
    "prompt": (
        "prompt_throughput_tok_s",
        "Prompt-processing throughput (tok/s)",
        "prompt-throughput-vs-cards",
    ),
    "ttft": (
        "ttft_s",
        "Warm TTFT (ms)",
        "warm-ttft-vs-cards",
    ),
}
CONTRACTED_WARM_REPS = 3


def require_success_warm(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        df["phase"].astype(str).eq("warm")
        & df["success"].astype(str).str.lower().eq("true")
    )
    return df[mask]


def plot_metric(
    df: pd.DataFrame,
    column: str,
    ylabel: str,
    stem: str,
    outdir: Path,
    title: str,
    ttft_ms: bool = False,
) -> list[str]:
    notes: list[str] = []
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for i, (bucket, sub) in enumerate(
        sorted(df.groupby("context_bucket"), key=lambda kv: int(kv[0]))
    ):
        color = PALETTE[i % len(PALETTE)]
        marker = MARKERS[i % len(MARKERS)]
        grouped = (
            sub.groupby("card_count")
            .agg(
                mean=(column, "mean"),
                lo=(column, "min"),
                hi=(column, "max"),
                n=(column, "count"),  # successful warm reps actually averaged
            )
            .reset_index()
            .sort_values("card_count")
        )
        x = grouped["card_count"].astype(int).to_numpy()
        y = grouped["mean"]
        if ttft_ms:
            y = y * 1000.0
            lo = grouped["lo"] * 1000.0
            hi = grouped["hi"] * 1000.0
        else:
            lo = grouped["lo"]
            hi = grouped["hi"]
        ax.errorbar(
            x,
            y.to_numpy(),
            yerr=[(y - lo).to_numpy(), (hi - y).to_numpy()],
            color=color,
            marker=marker,
            markersize=6,
            linewidth=1.6,
            capsize=3,
            label=f"{bucket}-token prompt",
        )
        reduced = grouped[grouped["n"].fillna(0) < CONTRACTED_WARM_REPS]
        for _, row in reduced.iterrows():
            xpos = int(row["card_count"])
            ypos = float(row["mean"] if not ttft_ms else row["mean"] * 1000.0)
            ax.annotate(
                f"n={int(row['n'])}",
                (xpos, ypos),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
                color=color,
            )
            notes.append(
                f"{stem}: bucket {bucket}, {xpos} card(s) averages only "
                f"n={int(row['n'])} successful warm repetition(s) "
                f"(contracted: {CONTRACTED_WARM_REPS})"
            )
    ax.set_xlabel("Card count (tensor parallel = card count)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(sorted(df["card_count"].astype(int).unique()))
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.text(
        0.01,
        0.01,
        "Warm repetitions only (cold excluded); error bars = min–max across "
        f"observed reps; n= labels mark cells below {CONTRACTED_WARM_REPS} reps.",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for suffix in ("svg", "png"):
        out = outdir / f"{stem}.{suffix}"
        fig.savefig(out, dpi=160)
    plt.close(fig)
    return notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument(
        "--outdir", type=Path, default=None, help="default: charts/ next to the CSV"
    )
    args = parser.parse_args()
    outdir = args.outdir or args.csv.parent / "charts"
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv, dtype={"context_bucket": str})
    missing = {
        column
        for column in ("card_count", "context_bucket", "phase", "success")
        if column not in df.columns
    }
    if missing:
        raise SystemExit(f"results.csv missing required columns: {sorted(missing)}")

    warm = require_success_warm(df)
    if warm.empty:
        raise SystemExit(
            "no successful warm rows in results.csv — nothing to chart; "
            "failed/partial rounds are disclosed, not charted"
        )

    all_notes: list[str] = []
    for metric, (column, ylabel, stem) in METRICS.items():
        if column not in warm.columns:
            raise SystemExit(f"results.csv missing metric column: {column}")
        usable = warm.dropna(subset=[column])
        if usable.empty:
            print(f"skip {stem}: column {column} present but empty (all failed rows)")
            continue
        title = {
            "generation-throughput-vs-cards": (
                "Generation throughput vs card count (Qwen3.8-27B Q8, 256 output tokens)"
            ),
            "prompt-throughput-vs-cards": (
                "Prompt-processing throughput vs card count (Qwen3.8-27B Q8)"
            ),
            "warm-ttft-vs-cards": (
                "Warm TTFT vs card count (Qwen3.8-27B Q8)"
            ),
        }[stem]
        all_notes += plot_metric(
            usable,
            column,
            ylabel,
            stem,
            outdir,
            title,
            ttft_ms=(metric == "ttft"),
        )
        print(f"wrote {outdir / stem}.svg and .png")

    if all_notes:
        print("\nReduced sample counts disclosed in charts:")
        for note in all_notes:
            print(f"  - {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
