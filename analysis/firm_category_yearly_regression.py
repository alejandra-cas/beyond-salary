#!/usr/bin/env python3
"""Run firm-category logit regressions by year and period groups."""

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from regression_models import (
    add_llm_firm_arguments,
    prepare_firm_run,
    plot_firm_category_logit_results,
    run_firm_category_logit_models,
    plot_firm_category_time_logit_results,
    run_firm_category_period_logit_models,
    run_firm_category_yearly_logit_models,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run firm-category benefit logit models by year and period group."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--pooled-only', action='store_true',
                       help='Run only pooled firm-category coefficients across benefits.')
    group.add_argument(
        "--yearly-only",
        action="store_true",
        help="Run only yearly firm-category models.",
    )
    group.add_argument(
        "--period-only",
        action="store_true",
        help="Skip yearly models and run only through-2022/post-2022 firm-category models.",
    )
    add_llm_firm_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    data, tables_dir, figures_dir = prepare_firm_run(args)
    title_suffix = ' (LLM-labeled firms)' if args.llm_only else ''
    if args.pooled_only:
        results = run_firm_category_logit_models(data, output_dir=tables_dir)
        plot_firm_category_logit_results(results, output_dir=figures_dir,
                                        title='LLM-labeled firms' if args.llm_only else None)
        return

    if not args.period_only:
        yearly_results = run_firm_category_yearly_logit_models(data, output_dir=tables_dir)
        plot_firm_category_time_logit_results(
            yearly_results,
            "firm_category_yearly_ai_role_logit_coefficients.png",
            "Yearly Firm-Category AI-Role Logit Coefficients" + title_suffix,
            output_dir=figures_dir,
        )

    if not args.yearly_only:
        period_results = run_firm_category_period_logit_models(data, output_dir=tables_dir)
        plot_firm_category_time_logit_results(
            period_results,
            "firm_category_period_ai_role_logit_coefficients.png",
            "Firm-Category AI-Role Logit Coefficients: Through 2022 vs Post-2022" + title_suffix,
            period_order=["Through 2022", "Post-2022"],
            output_dir=figures_dir,
        )


if __name__ == "__main__":
    main()
