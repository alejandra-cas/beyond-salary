#!/usr/bin/env python3
"""
Validate posting-count firm-size cutoffs with sampled LLM labels.

This script:
  1. draws one random firm sample;
  2. optionally asks an OpenAI model whether each sampled firm is truly an SME
     or a large firm;
  3. evaluates every posting-count cutoff on that same labeled sample;
  4. reports recall, precision, F1, accuracy, and post-weighted variants where
     each sampled firm is weighted by its number of postings.

Examples
--------
Dry run that exports samples for manual/LLM review:
    uv run python scripts/validate_firm_size_cutoffs.py --cutoffs 5:100:5

Run LLM labels and metrics:
    # Put OPENAI_API_KEY=... in .env first.
    uv run python scripts/validate_firm_size_cutoffs.py --cutoffs 5:100:5 --llm
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAIError
from tqdm.auto import tqdm

# Allow the script to reuse the repo's shared config helpers when run directly
# from the command line, e.g. `uv run python scripts/...`.
sys.path.append(str(Path(__file__).parent.parent / "src"))
from package_files.config_utils import get_processed_dir, get_repo_root


# Load OPENAI_API_KEY and optional OPENAI_MODEL from a local .env file before
# creating the OpenAI SDK client. override=True makes the repo-local .env the
# source of truth even if the shell already has another OPENAI_API_KEY set.
load_dotenv(override=True)

# All generated validation artifacts live together so Matthew can review the
# sampled firms, cached LLM labels, metrics table, and plot side by side.
DEFAULT_OUTPUT_DIR = (
    get_repo_root()
    / "results"
    / "tables_2026"
    / "validation"
    / "firm_size_cutoff_validation"
)

# Optional industry context for the LLM. We use industry names rather than job
# titles or posting counts to avoid nudging the model toward the proxy being
# validated.
NAICS_CONTEXT_COLUMN = "NAICS_2022_6_NAME"


def parse_cutoffs(value: str) -> list[int]:
    """Parse either '5,10,15' or '5:100:5' into a cutoff list."""
    value = value.strip()
    if ":" in value:
        # Python-style range shorthand: start:stop:step, with stop included
        # because analysts usually expect `5:100:5` to include 100.
        parts = [int(x) for x in value.split(":")]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("Range cutoffs must be start:stop:step.")
        start, stop, step = parts
        if step <= 0:
            raise argparse.ArgumentTypeError("Cutoff step must be positive.")
        return list(range(start, stop + 1, step))

    cutoffs = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not cutoffs:
        raise argparse.ArgumentTypeError("Provide at least one cutoff.")
    return sorted(set(cutoffs))


def load_data(input_path: Path | None) -> pd.DataFrame:
    """Load the source posting data."""
    # By default, use the first processed file from scripts/prepare_data.py.
    # Passing --input-path makes it easy to test on a small parquet.
    path = input_path or get_processed_dir() / "data_v1.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Input data not found at {path}. Run scripts/prepare_data.py first "
            "or pass --input-path."
        )
    print(f"Loading postings from {path}")
    return pd.read_parquet(path)


def build_firm_frame(df: pd.DataFrame, max_naics: int) -> pd.DataFrame:
    """Collapse posting rows into one record per classified firm."""
    # The validation operates at the firm level, not the posting level. This
    # function turns many job postings into one row per COMPANY ID.
    required = {"COMPANY"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input data is missing required columns: {sorted(missing)}")

    data = df.copy()
    # COMPANY == 0 is the Lightcast/OII placeholder for unclassified employers.
    # It is not a real firm ID, so including it would create one fake giant firm.
    data = data[data["COMPANY"].notna() & (data["COMPANY"] != 0)].copy()

    # Prefer the pipeline's full-sample posting count when present. If the input is
    # a custom parquet without that column, compute the same proxy from the rows.
    if "FIRM_POSTING_COUNT" not in data.columns:
        counts = data["COMPANY"].value_counts()
        data["FIRM_POSTING_COUNT"] = data["COMPANY"].map(counts).astype("Int64")

    agg_map = {
        "FIRM_POSTING_COUNT": "max",
    }
    # Carry firm names through when available, but do not require them. COMPANY
    # is the stable key in this dataset.
    for col in ["COMPANY_NAME", "COMPANY_RAW"]:
        if col in data.columns:
            agg_map[col] = first_nonempty

    firms = data.groupby("COMPANY", dropna=False).agg(agg_map).reset_index()

    # Add unique industries observed for the firm. This gives the LLM light
    # context without exposing job titles or the posting-count proxy.
    if NAICS_CONTEXT_COLUMN in data.columns:
        naics_context = (
            data.groupby("COMPANY")[NAICS_CONTEXT_COLUMN]
            .apply(lambda x: summarize_unique_values(x, max_naics))
            .rename("UNIQUE_NAICS_2022_6_NAMES")
            .reset_index()
        )
        firms = firms.merge(naics_context, on="COMPANY", how="left")
    else:
        firms["UNIQUE_NAICS_2022_6_NAMES"] = ""
    firms["FIRM_POSTING_COUNT"] = pd.to_numeric(
        firms["FIRM_POSTING_COUNT"], errors="coerce"
    ).astype("Int64")
    firms = firms[firms["FIRM_POSTING_COUNT"].notna()].copy()
    return firms


def exclude_sp500_firms(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude confirmed S&P 500 postings when SP500 labels are available."""
    # The methodology treats S&P 500 firms separately, so they should not
    # influence the SME/large posting-count cutoff search.
    if "SP500" not in df.columns:
        print("SP500 column not found; no S&P 500 exclusion applied.")
        return df

    before = len(df)
    out = df[~df["SP500"].fillna(False).astype(bool)].copy()
    print(f"Excluded {before - len(out):,} S&P 500 postings before sampling.")
    return out


def first_nonempty(series: pd.Series) -> str:
    """Return the first nonblank value from a grouped column."""
    for value in series.dropna():
        text = str(value).strip()
        if text:
            return text
    return ""


def summarize_unique_values(series: pd.Series, n: int) -> str:
    """Return up to n unique nonblank values as a compact pipe-delimited string."""
    if n <= 0:
        return ""

    values = []
    seen = set()
    for value in series.dropna():
        text = str(value).strip()
        if not text or text in seen:
            continue
        values.append(text)
        seen.add(text)
        if len(values) >= n:
            break
    return " | ".join(values)


def split_context_values(value: Any) -> list[str]:
    """Convert the saved pipe-delimited industry context into a JSON list."""
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def draw_validation_sample(
    firms: pd.DataFrame,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    """Draw one reproducible firm sample for validating every cutoff."""
    if sample_size < 1:
        raise ValueError("--sample-size must be at least 1.")

    rng = np.random.default_rng(seed)
    n = min(sample_size, len(firms))
    if n == 0:
        return pd.DataFrame()

    sampled_idx = rng.choice(firms.index.to_numpy(), size=n, replace=False)
    return firms.loc[sampled_idx].copy().reset_index(drop=True)


def load_label_cache(path: Path) -> dict[str, dict[str, Any]]:
    """Load previously completed LLM labels keyed by COMPANY."""
    if not path.exists():
        return {}
    cache_df = pd.read_csv(path, keep_default_na=False)
    cache = {}
    for _, row in cache_df.iterrows():
        item = row.dropna().to_dict()
        item["COMPANY"] = str(item["COMPANY"])
        cache[item["COMPANY"]] = item
    return cache


def save_label_cache(cache: dict[str, dict[str, Any]], path: Path) -> None:
    """Persist the cache after each new label to make interrupted runs resumable."""
    if not cache:
        return
    rows = list(cache.values())
    frame = pd.DataFrame(rows)
    frame["COMPANY"] = frame["COMPANY"].astype(str)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    frame.sort_values("COMPANY").to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def build_label_cache_entry(
    company_key: str,
    row: pd.Series,
    model: str,
    label: str,
    confidence: Any = "",
    error: str = "",
) -> dict[str, Any]:
    """Build one cached LLM-label row, including N/A rows for failures."""
    entry = {
        "COMPANY": company_key,
        "COMPANY_NAME": row.get("COMPANY_NAME", ""),
        "llm_label": label,
        "llm_confidence": confidence,
        "llm_model": model,
        "labeled_at": pd.Timestamp.utcnow().isoformat(),
    }
    if error:
        entry["llm_error"] = error
    return entry


def classify_samples_with_llm(
    samples: pd.DataFrame,
    cache_path: Path,
    model: str,
    base_url: str,
    sleep_seconds: float,
    max_retries: int,
    concurrency: int,
) -> pd.DataFrame:
    """Add LLM firm labels to sampled rows, using a COMPANY-level cache."""
    return asyncio.run(
        classify_samples_with_llm_async(
            samples,
            cache_path,
            model,
            base_url,
            sleep_seconds,
            max_retries,
            concurrency,
        )
    )


async def classify_samples_with_llm_async(
    samples: pd.DataFrame,
    cache_path: Path,
    model: str,
    base_url: str,
    sleep_seconds: float,
    max_retries: int,
    concurrency: int,
) -> pd.DataFrame:
    """Add LLM firm labels to sampled rows concurrently, using a cache."""
    api_key = os.environ.get("OPENAI_API_KEY")
    # print last 4 characters of the key for debugging, but do not print the whole key.
    # print(f"Using OpenAI API key ending with: {api_key[-4:] if api_key else 'None'}")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when --llm is set.")
    if concurrency < 1:
        raise ValueError("--llm-concurrency must be at least 1.")

    cache = load_label_cache(cache_path)
    # Label each unique firm once, then merge labels back to every cutoff sample
    # row that used that firm.
    unique_firms = samples.drop_duplicates("COMPANY").copy()
    pending_rows: list[tuple[str, pd.Series]] = []

    for _, row in unique_firms.iterrows():
        company_key = str(row["COMPANY"])
        # The same firm can be sampled at many cutoffs; cache by COMPANY so
        # reruns and overlapping thresholds do not spend tokens twice.
        if company_key in cache and cache[company_key].get("llm_label"):
            continue
        pending_rows.append((company_key, row))

    workers_count = min(concurrency, len(pending_rows)) if pending_rows else 0
    if pending_rows:
        client_kwargs = {"api_key": api_key}
        if base_url.rstrip("/") != "https://api.openai.com/v1":
            client_kwargs["base_url"] = base_url.rstrip("/")
        client = AsyncOpenAI(**client_kwargs)

        queue: asyncio.Queue[tuple[str, pd.Series] | None] = asyncio.Queue()
        for item in pending_rows:
            queue.put_nowait(item)
        for _ in range(workers_count):
            queue.put_nowait(None)

        cache_lock = asyncio.Lock()
        failed_count = 0
        progress = tqdm(
            total=len(pending_rows),
            desc=f"Classifying firms ({workers_count} workers)",
            unit="firm",
        )

        async def worker(worker_id: int) -> None:
            nonlocal failed_count
            while True:
                item = await queue.get()
                try:
                    if item is None:
                        return
                    company_key, row = item
                    result = await classify_one_firm_async(
                        row, model, client, max_retries
                    )
                    async with cache_lock:
                        cache[company_key] = build_label_cache_entry(
                            company_key,
                            row,
                            model,
                            result["label"],
                            result.get("confidence", ""),
                        )
                        save_label_cache(cache, cache_path)
                    if sleep_seconds > 0:
                        await asyncio.sleep(sleep_seconds)
                except Exception as exc:
                    if item is not None:
                        company_key, row = item
                        failed_count += 1
                        error = f"{type(exc).__name__}: {exc}"
                        async with cache_lock:
                            cache[company_key] = build_label_cache_entry(
                                company_key,
                                row,
                                model,
                                "N/A",
                                error=error[:500],
                            )
                            save_label_cache(cache, cache_path)
                        progress.set_postfix(failed=failed_count)
                finally:
                    if item is not None:
                        progress.update(1)
                    queue.task_done()

        try:
            tasks = [asyncio.create_task(worker(i + 1)) for i in range(workers_count)]
            await queue.join()
            for task in tasks:
                await task
        finally:
            progress.close()
            await client.close()

    labels = pd.DataFrame(list(cache.values()))
    labels["COMPANY"] = labels["COMPANY"].astype(str)
    out = samples.copy()
    # Cast both keys to string so cached labels still merge if pandas inferred
    # COMPANY as int in one file and string in another.
    out["COMPANY"] = out["COMPANY"].astype(str)
    return out.merge(labels, on="COMPANY", how="left", suffixes=("", "_cache"))


async def classify_one_firm_async(
    row: pd.Series,
    model: str,
    client: AsyncOpenAI,
    max_retries: int,
) -> dict[str, Any]:
    """Call the OpenAI Responses API for one firm."""
    company_name = row.get("COMPANY_NAME", "") or row.get("COMPANY_RAW", "")

    prompt = {
        "company_id": str(row["COMPANY"]),
        "company_name": company_name,
        "unique_naics_2022_6_names": split_context_values(
            row.get("UNIQUE_NAICS_2022_6_NAMES", "")
        ),
    }
    instructions = (
        "You classify employers by their true real-world organization size.\n\n"
        "Use the employer name, industry context, and general knowledge of "
        "the employer's real-world scale.\n\n"
        "Return only strict JSON with keys:\n"
        '- label: one of "sme", "large", or "unknown"\n'
        "- confidence: number from 0 to 1\n\n"
        "Definitions:\n"
        'Use "large" for employers that are likely large enterprises, including '
        "national or multinational corporations, public companies, subsidiaries "
        "or brands of large parent companies, large chains, large hospital or "
        "university systems, government entities, national nonprofits, major "
        "staffing firms, or professional-services firms operating across many "
        "locations.\n\n"
        'Use "sme" for employers that are likely small or medium-sized independent '
        "organizations, including local or regional businesses, independent "
        "clinics, restaurants, shops, agencies, contractors, schools, nonprofits, "
        "or niche private firms without evidence of national scale.\n\n"
        'Use "unknown" only when the employer name is missing, generic, ambiguous, '
        "confidential, a job-board/staffing placeholder, or otherwise impossible "
        "to classify from the name and industry context.\n\n"
        'Do not label a firm "unknown" merely because it is unfamiliar. If the '
        "employer appears to be an independent local, regional, or niche "
        "organization and there is no evidence of national or multinational "
        'scale, classify it as "sme".'
    )
    user_input = (
        "Classify this firm. Return JSON with keys: "
        "label ('sme', 'large', or 'unknown') and confidence (0-1).\n\n"
        f"{json.dumps(prompt, ensure_ascii=True)}"
    )
    json_schema = {
        "type": "json_schema",
        "name": "firm_size_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "label": {"type": "string", "enum": ["sme", "large", "unknown"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["label", "confidence"],
        },
    }

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = await client.responses.create(
                model=model,
                input=[
                    {"role": "developer", "content": instructions},
                    {"role": "user", "content": user_input},
                ],
                text={"format": json_schema, "verbosity": "low"},
            )
            result = json.loads(response.output_text)
            label = normalize_label(result.get("label", "unknown"))
            return {
                "label": label,
                "confidence": result.get("confidence"),
            }
        except (OpenAIError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
            # Retry transient API/network/JSON issues with exponential backoff.
            last_error = exc
            if attempt < max_retries:
                await asyncio.sleep(2**attempt)

    raise RuntimeError(f"LLM classification failed for {company_name}: {last_error}")


def normalize_label(label: Any) -> str:
    """Map minor label variants into the three labels used downstream."""
    text = str(label).strip().lower()
    if text in {"sme", "small", "small_medium", "small/medium"}:
        return "sme"
    if text in {"large", "large_firm", "enterprise"}:
        return "large"
    return "unknown"


def score_cutoffs(labeled_samples: pd.DataFrame, cutoffs: list[int]) -> pd.DataFrame:
    """Compute validation metrics for each cutoff on the same labeled sample."""
    df = labeled_samples.copy()
    if "llm_label" not in df.columns:
        return pd.DataFrame()
    if "FIRM_POSTING_COUNT" not in df.columns:
        raise ValueError("Labeled samples must include FIRM_POSTING_COUNT.")

    df["truth_sme"] = df["llm_label"] == "sme"
    df["truth_large"] = df["llm_label"] == "large"
    df["post_weight"] = pd.to_numeric(df["FIRM_POSTING_COUNT"], errors="coerce")
    df = df[df["post_weight"].notna() & (df["post_weight"] > 0)].copy()
    df["FIRM_POSTING_COUNT"] = df["post_weight"]
    # Unknown labels are useful to audit, but they are excluded from metric
    # denominators because they do not define a true positive/negative class.
    df = df[df["llm_label"].isin(["sme", "large"])].copy()
    if df.empty:
        return pd.DataFrame()

    rows = []
    for cutoff in cutoffs:
        sub = df.copy()
        # At a given threshold, firms below the cutoff are predicted SMEs and
        # firms at/above the cutoff are predicted large.
        sub["proxy_sme"] = sub["post_weight"] < cutoff
        unweighted = cutoff_metric_dict(sub)
        post_weighted = cutoff_metric_dict(sub, weight_column="post_weight")

        row = {
            "cutoff": cutoff,
            "n_labeled": int(round(unweighted["n"])),
            "posting_weight_total": post_weighted["n"],
            "true_sme": int(round(unweighted["true_sme"])),
            "true_large": int(round(unweighted["true_large"])),
            "true_sme_post_weighted": post_weighted["true_sme"],
            "true_large_post_weighted": post_weighted["true_large"],
            "false_positive_count": int(round(unweighted["fp"])),
            "false_negative_count": int(round(unweighted["fn"])),
            "false_positive_count_post_weighted": post_weighted["fp"],
            "false_negative_count_post_weighted": post_weighted["fn"],
            "false_positive_rate": unweighted["false_positive_rate"],
            "false_negative_rate": unweighted["false_negative_rate"],
            "false_positive_rate_post_weighted": post_weighted["false_positive_rate"],
            "false_negative_rate_post_weighted": post_weighted["false_negative_rate"],
        }

        for metric in [
            "sme_recall",
            "large_recall",
            "sme_precision",
            "large_precision",
            "sme_f1",
            "large_f1",
            "combined_f1",
            "accuracy",
            "balanced_accuracy",
        ]:
            row[metric] = unweighted[metric]
            row[f"{metric}_post_weighted"] = post_weighted[metric]

        # Keep the old sensitivity column names for downstream scripts that may
        # already read the metrics CSV.
        row["sme_sensitivity"] = row["sme_recall"]
        row["large_sensitivity"] = row["large_recall"]
        row["sme_sensitivity_post_weighted"] = row["sme_recall_post_weighted"]
        row["large_sensitivity_post_weighted"] = row["large_recall_post_weighted"]
        rows.append(row)

    return pd.DataFrame(rows).sort_values("cutoff")


def cutoff_metric_dict(
    sub: pd.DataFrame,
    weight_column: str | None = None,
) -> dict[str, float]:
    """Return confusion-matrix metrics, optionally weighted by posting count."""
    if weight_column is None:
        weights = pd.Series(1.0, index=sub.index)
    else:
        weights = sub[weight_column].astype(float)

    def weighted_count(mask: pd.Series) -> float:
        return float(weights[mask].sum())

    # Treat SME as the positive class:
    #   TP = predicted SME and LLM says SME
    #   FP = predicted SME but LLM says large
    #   FN = predicted large but LLM says SME
    #   TN = predicted large and LLM says large
    tp = weighted_count(sub["proxy_sme"] & sub["truth_sme"])
    fp = weighted_count(sub["proxy_sme"] & sub["truth_large"])
    fn = weighted_count(~sub["proxy_sme"] & sub["truth_sme"])
    tn = weighted_count(~sub["proxy_sme"] & sub["truth_large"])
    n = tp + fp + fn + tn
    sme_recall = safe_divide(tp, tp + fn)
    large_recall = safe_divide(tn, tn + fp)
    sme_precision = safe_divide(tp, tp + fp)
    large_precision = safe_divide(tn, tn + fn)
    sme_f1 = f1_score(sme_precision, sme_recall)
    large_f1 = f1_score(large_precision, large_recall)

    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "true_sme": tp + fn,
        "true_large": tn + fp,
        "false_positive_rate": safe_divide(fp, fp + tn),
        "false_negative_rate": safe_divide(fn, fn + tp),
        "sme_recall": sme_recall,
        "large_recall": large_recall,
        # SME precision falls as the cutoff rises: a higher threshold pulls more
        # truly-large firms into the predicted-SME bucket.
        "sme_precision": sme_precision,
        # Large-firm precision is TN/(TN+FN): of firms predicted large, how many
        # the LLM agrees are large.
        "large_precision": large_precision,
        "sme_f1": sme_f1,
        "large_f1": large_f1,
        "combined_f1": np.nanmean([sme_f1, large_f1]),
        "accuracy": safe_divide(tp + tn, n),
        # Balanced accuracy gives equal weight to SME and large-firm recall.
        "balanced_accuracy": np.nanmean([sme_recall, large_recall]),
    }


def f1_score(precision: float, recall: float) -> float:
    """Return F1 from precision and recall, preserving NaN when undefined."""
    if pd.isna(precision) or pd.isna(recall) or precision + recall == 0:
        return np.nan
    return 2 * precision * recall / (precision + recall)


def safe_divide(num: float, den: float) -> float:
    """Return NaN for undefined rates instead of raising ZeroDivisionError."""
    return np.nan if den == 0 else num / den


def plot_cutoff_metrics(metrics: pd.DataFrame, output_path: Path) -> None:
    """Plot class-specific and overall cutoff performance metrics."""
    if metrics.empty:
        return

    if "MPLCONFIGDIR" not in os.environ:
        mpl_config_dir = Path("/tmp") / f"matplotlib-{os.getuid()}"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.3), sharex=True, sharey=True)
    fig.suptitle("Firm-Size Classification Metrics by Posting-Count Cutoff", y=0.98)
    blue = "#1f77b4"
    orange = "#ff7f0e"
    gray = "#555555"

    def add_line(ax, y_column: str, label: str, color: str, linestyle: str = "-") -> None:
        ax.plot(
            metrics["cutoff"],
            metrics[y_column],
            marker="o",
            linewidth=2,
            linestyle=linestyle,
            color=color,
            label=label,
        )

    optimal = metrics["balanced_accuracy"].max()
    optimal_band = metrics[metrics["balanced_accuracy"] >= optimal - 0.02]
    band_limits = None
    if not optimal_band.empty:
        # Shade all cutoffs within two percentage points of the best balanced
        # accuracy. This avoids over-interpreting one noisy sampled threshold.
        band_limits = (optimal_band["cutoff"].min(), optimal_band["cutoff"].max())

    recall_ax, precision_ax, f1_ax, overall_ax = axes.ravel()

    add_line(recall_ax, "sme_recall", "SME recall", blue)
    add_line(recall_ax, "large_recall", "Large recall", orange)
    add_line(
        recall_ax,
        "sme_recall_post_weighted",
        "SME recall, post-weighted",
        blue,
        "--",
    )
    add_line(
        recall_ax,
        "large_recall_post_weighted",
        "Large recall, post-weighted",
        orange,
        "--",
    )
    recall_ax.set_title("Recall")

    add_line(precision_ax, "sme_precision", "SME precision", blue)
    add_line(precision_ax, "large_precision", "Large precision", orange)
    add_line(
        precision_ax,
        "sme_precision_post_weighted",
        "SME precision, post-weighted",
        blue,
        "--",
    )
    add_line(
        precision_ax,
        "large_precision_post_weighted",
        "Large precision, post-weighted",
        orange,
        "--",
    )
    precision_ax.set_title("Precision")

    add_line(f1_ax, "combined_f1", "Combined F1", gray)
    add_line(f1_ax, "combined_f1_post_weighted", "Combined F1, post-weighted", gray, "--")
    add_line(f1_ax, "sme_f1", "SME F1", blue)
    add_line(f1_ax, "large_f1", "Large F1", orange)
    f1_ax.set_title("F1")

    add_line(overall_ax, "accuracy", "Accuracy", gray)
    add_line(overall_ax, "accuracy_post_weighted", "Accuracy, post-weighted", gray, "--")
    add_line(overall_ax, "balanced_accuracy", "Balanced accuracy", orange)
    overall_ax.set_title("Overall")

    for ax in axes.ravel():
        if band_limits is not None:
            ax.axvspan(
                band_limits[0],
                band_limits[1],
                color="#9ecae1",
                alpha=0.18,
                label="Near-optimal range" if ax is recall_ax else None,
            )
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.28)
        ax.legend(loc="lower right", frameon=False)

    recall_ax.set_ylabel("Score")
    f1_ax.set_ylabel("Score")
    f1_ax.set_xlabel("Posting-count cutoff")
    overall_ax.set_xlabel("Posting-count cutoff")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    # CLI options are kept explicit so Matthew can run cheap dry runs, small LLM
    # tests, and full validation runs from the same script.
    parser = argparse.ArgumentParser(
        description="Sample and validate posting-count firm-size cutoffs."
    )
    parser.add_argument("--input-path", type=Path, help="Posting parquet to validate.")
    parser.add_argument(
        "--cutoffs",
        type=parse_cutoffs,
        default=parse_cutoffs("5:100:5"),
        help="Comma list (5,10,15) or range (5:100:5). Default: 5:100:5.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of firms sampled once for all cutoffs. Default: 1000.",
    )
    parser.add_argument(
        "--sample-per-class",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-naics",
        type=int,
        default=5,
        help="Maximum number of unique NAICS industry names passed to the LLM.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--include-sp500",
        action="store_true",
        help="Include S&P 500 firms. By default they are excluded when SP500 exists.",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Call the LLM for sampled firms. Without this, only samples are exported.",
    )
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.4-mini"))
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--llm-concurrency",
        type=int,
        default=10,
        help="Number of concurrent OpenAI classification workers. Default: 10.",
    )
    args = parser.parse_args()

    # Define all output paths once so the filenames stay consistent across dry
    # runs and LLM-scored runs.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / "firm_size_validation_sample.csv"
    label_cache_path = args.output_dir / "firm_size_llm_labels_cache.csv"
    labeled_path = args.output_dir / "firm_size_labeled_validation_sample.csv"
    metrics_path = args.output_dir / "firm_size_cutoff_metrics.csv"
    plot_path = (
        get_repo_root()
        / "results"
        / "figures_2026"
        / "validation"
        / "firm_size_cutoff_metrics_plot.png"
    )

    if args.sample_per_class is not None:
        print(
            "--sample-per-class is deprecated; using its value as --sample-size "
            "for this one-sample validation design."
        )
        args.sample_size = args.sample_per_class

    df = load_data(args.input_path)
    if not args.include_sp500:
        df = exclude_sp500_firms(df)

    firms = build_firm_frame(df, args.max_naics)
    print(f"Classified firms available for sampling: {len(firms):,}")

    # Always write the sampled firms, even if --llm is not used. This makes it
    # possible to inspect the sample manually before spending API tokens.
    samples = draw_validation_sample(firms, args.sample_size, args.seed)
    if samples.empty:
        raise RuntimeError("No firms were sampled. Check cutoffs and input data.")
    samples.to_csv(sample_path, index=False)
    print(f"Saved validation sample: {sample_path}")

    if not args.llm:
        print("LLM labeling skipped. Re-run with --llm to classify and score samples.")
        return

    # The scoring branch only runs after labels exist; dry runs stop above.
    labeled = classify_samples_with_llm(
        samples,
        label_cache_path,
        args.model,
        args.base_url,
        args.sleep_seconds,
        args.max_retries,
        args.llm_concurrency,
    )
    labeled.to_csv(labeled_path, index=False)
    print(f"Saved labeled samples: {labeled_path}")

    metrics = score_cutoffs(labeled, args.cutoffs)
    metrics.to_csv(metrics_path, index=False)
    print(f"Saved cutoff metrics: {metrics_path}")
    plot_cutoff_metrics(metrics, plot_path)
    print(f"Saved metrics plot: {plot_path}")
    print("\nCutoff metrics:")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
