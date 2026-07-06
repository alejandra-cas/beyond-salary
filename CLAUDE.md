# Beyond Salary — AI Jobs and Non-Monetary Benefits

Research project analyzing 10M US job postings (2018–2025): whether AI roles offer more non-monetary benefits, and how benefits interact with the AI wage premium across firm sizes.

## Key facts

- Two collaborators run this repo: Alejandra (local subsample, ~100k rows) and Matthew (full 10M dataset). Full-data figures/tables in `results/` come from Matthew's runs — **do not overwrite them by running analysis scripts on subsample data**.
- Main labeled dataset: `data/processed/labeled_v2.parquet`. Benefit flags are keyword-matched from posting BODY text; the full-size BODY file exists only on Matthew's machine.
- `scripts/relabel_sp500.py` refreshes the `SP500` column in existing labeled parquets from `data/sp500_snapshot.csv` without redoing benefit labeling.
- Several plotting scripts can regenerate figures from committed results CSVs without raw data, e.g. `analysis/wage_perk_interaction_analysis.py --plot-only` and `descriptive_analysis.py --use-existing-betas`.
- Benefit label rename: "Workplace Culture" → "Inclusive Workplace" (internal key remains `CULTURE`). Labels live in `src/package_files/benefits_defns.py`.

## Paper figure numbering vs internal names

Internal "Figure 1" (`figure1_ai_demand_wage_beta.png`) is **paper Figure 5**. The full mapping of paper Figures 2–7 to output files and generating scripts is in `README.md` → "Paper Figure Mapping". Consult it before editing any figure — paper numbers do not match file names.
