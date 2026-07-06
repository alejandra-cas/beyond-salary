#!/usr/bin/env python3
"""
Plot SME and large-firm precision by posting-count cutoff.

This is a standalone companion to scripts/validate_firm_size_cutoffs.py. It
reads an existing labeled-samples CSV (the per-firm LLM labels that validation
already produced) and recomputes precision per cutoff, so no API calls are
needed. SME is treated as the positive class, matching the validation script.

Example
-------
    uv run python scripts/plot_firm_size_precision.py \
        --labeled results/tables_2026/validation/firm_size_cutoff_labeled_samples.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

DEFAULT_LABELED = (
    Path("results")
    / "tables_2026"
    / "validation"
    / "firm_size_cutoff_labeled_samples.csv"
)
DEFAULT_OUTPUT = (
    Path("results")
    / "figures_2026"
    / "validation"
    / "firm_size_cutoff_precision_plot.png"
)


def safe_divide(num: int, den: int) -> float:
    """Return NaN for undefined rates instead of dividing by zero."""
    return np.nan if den == 0 else num / den


def wilson_ci(num: int, den: int) -> tuple[float, float]:
    """Wilson 95% interval for a proportion, matching the recall plot's CIs."""
    if den == 0:
        return (np.nan, np.nan)
    lower, upper = proportion_confint(num, den, alpha=0.05, method="wilson")
    return (lower, upper)


def precision_by_cutoff(labeled: pd.DataFrame) -> pd.DataFrame:
    """Compute SME and large-firm precision per cutoff from labeled samples."""
    df = labeled.copy()
    # Only sme/large LLM labels define a true class; "unknown" rows are dropped
    # from the denominators, matching score_cutoffs in the validation script.
    df = df[df["llm_label"].isin(["sme", "large"])].copy()
    df["truth_sme"] = df["llm_label"] == "sme"
    df["truth_large"] = df["llm_label"] == "large"
    # proxy_sme may load as the strings "True"/"False" from CSV; coerce to bool.
    df["proxy_sme"] = df["proxy_sme"].astype(str).str.lower().isin(["true", "1"])

    rows = []
    for cutoff, sub in df.groupby("cutoff"):
        # SME = positive class.
        #   TP = predicted SME and LLM says SME
        #   FP = predicted SME but LLM says large
        #   FN = predicted large but LLM says SME
        #   TN = predicted large and LLM says large
        tp = int((sub["proxy_sme"] & sub["truth_sme"]).sum())
        fp = int((sub["proxy_sme"] & sub["truth_large"]).sum())
        fn = int((~sub["proxy_sme"] & sub["truth_sme"]).sum())
        tn = int((~sub["proxy_sme"] & sub["truth_large"]).sum())
        # SME precision denominator is the predicted-SME count (tp+fp); large
        # precision denominator is the predicted-large count (tn+fn). Wilson CIs
        # use those same denominators.
        sme_lo, sme_hi = wilson_ci(tp, tp + fp)
        large_lo, large_hi = wilson_ci(tn, tn + fn)
        rows.append(
            {
                "cutoff": cutoff,
                "n_labeled": tp + fp + fn + tn,
                # SME precision: of firms predicted SME, how many truly are.
                "sme_precision": safe_divide(tp, tp + fp),
                "sme_precision_95ci_lower": sme_lo,
                "sme_precision_95ci_upper": sme_hi,
                # Large-firm precision: of firms predicted large, how many truly are.
                "large_precision": safe_divide(tn, tn + fn),
                "large_precision_95ci_lower": large_lo,
                "large_precision_95ci_upper": large_hi,
                # Balanced accuracy drives the near-optimal range band, the same
                # basis the recall plot uses, so both plots highlight the same
                # cutoffs.
                "balanced_accuracy": np.nanmean(
                    [safe_divide(tp, tp + fn), safe_divide(tn, tn + fp)]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("cutoff")


def plot_precision(metrics: pd.DataFrame, output_path: Path) -> None:
    """Plot SME and large-firm precision against the posting-count cutoff."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.8))
    sme_line = ax.plot(
        metrics["cutoff"],
        metrics["sme_precision"],
        marker="o",
        linewidth=2,
        label="SME precision",
    )[0]
    # Shade the Wilson 95% interval with the line's own color, matching the
    # recall plot's CI bands.
    ax.fill_between(
        metrics["cutoff"],
        metrics["sme_precision_95ci_lower"],
        metrics["sme_precision_95ci_upper"],
        color=sme_line.get_color(),
        alpha=0.2,
        label="SME precision 95% CI",
    )
    large_line = ax.plot(
        metrics["cutoff"],
        metrics["large_precision"],
        marker="o",
        linewidth=2,
        label="Large firm precision",
    )[0]
    ax.fill_between(
        metrics["cutoff"],
        metrics["large_precision_95ci_lower"],
        metrics["large_precision_95ci_upper"],
        color=large_line.get_color(),
        alpha=0.2,
        label="Large firm precision 95% CI",
    )

    optimal = metrics["balanced_accuracy"].max()
    optimal_band = metrics[metrics["balanced_accuracy"] >= optimal - 0.02]
    if not optimal_band.empty:
        # Shade all cutoffs within two percentage points of the best balanced
        # accuracy, matching the recall plot's near-optimal band.
        lo = optimal_band["cutoff"].min()
        hi = optimal_band["cutoff"].max()
        ax.axvspan(lo, hi, color="#9ecae1", alpha=0.25, label="Near-optimal range")

    ax.set_title("Precision by Firm-Size Classification Threshold")
    ax.set_xlabel("Posting-count cutoff")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot firm-size precision from an existing labeled-samples CSV."
    )
    parser.add_argument("--labeled", type=Path, default=DEFAULT_LABELED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.labeled.exists():
        raise FileNotFoundError(f"Labeled samples not found at {args.labeled}.")

    labeled = pd.read_csv(args.labeled)
    metrics = precision_by_cutoff(labeled)
    plot_precision(metrics, args.output)
    print(f"Saved precision plot: {args.output}")
    print("\nPrecision by cutoff:")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
