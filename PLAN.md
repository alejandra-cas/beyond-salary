# Next Steps: Beyond Salary Revision

## Context
Following reviewer feedback and co-author discussion, the paper needs several extensions:
expanded timeframe (through end 2025), stronger controls (firm + regional FE), causal framing
adjustments, and new robustness analyses. The MAY26 subsample is the expanded dataset
covering through end 2025.

---

## 1. Pipeline: Carry new columns through
**Files:** `scripts/prepare_data.py`, `src/package_files/benefits_defns.py`

- Add `COMPANY` (firm ID) and `COMPANY_NAME` to the columns retained in `prepare_data.py`
  so they flow into `data_v1.parquet` and downstream parquets. `STATE_NAME` already flows
  through. Use `COMPANY` for fixed effects; `COMPANY_NAME` for inspection/labeling.
- Add `AI_SKILL_COUNT` column: in `classify_ai_roles()`, count AI/ML skills per posting
  (groupby ID on skills_df) and store the count alongside the binary `AI ROLE` flag.
  This supports the threshold robustness check later.
- Update `benefits_defns.py` to add `firm = "COMPANY"` and `firm_name = "COMPANY_NAME"`
  to the standard variable names.

---

## 2. Rerun all existing analyses
**Files:** all analysis scripts, `scripts/occ_year_analysis_raw.py`

The MAY26 subsample is the expanded dataset. The pipeline paths are already set up.
- Rerun the full pipeline: `prepare_data.py` -> `label_benefits.py` -> `label_benefits_remote.py`
  -> `occ_year_analysis_raw.py` -> `export_samples_2024.py`
- Rerun all analysis scripts (descriptive, regression, salary, scatterplot)
- Verify outputs and update any hardcoded year filters (e.g., `2024Q3` removal)

---

## 3. Add firm and state fixed effects to regression models
**File:** `analysis/regression_models_2024.py`

Current model specs:
- M1: Year + Industry FE
- M2: Year + Industry + Education + Experience
- M3: M2 + Log Salary

Update to add `COMPANY` and `STATE_NAME` as categorical controls. Since firm FE will create
a very large number of dummies, consider:
- Option A: Add firm + state FE as a 4th specification (M4)
- Option B: Replace industry FE with firm FE (firm absorbs industry)

**Implementation:** `run_logit_model()` in `src/package_files/logit_model.py` uses
`pd.get_dummies()` for categorical controls. With potentially thousands of firms, this may
hit memory limits. May need to switch to `statsmodels` formula interface with `C()` or use
`linearmodels.PanelOLS` with absorbed fixed effects.

**Decision needed at implementation time:** test memory/performance with the subsample first.

---

## 4. Industry-year wage and perk coefficient comparison
**New analysis — new file recommended:** `analysis/industry_year_coefficient_analysis.py`

This is the key new analysis connecting monetary and non-monetary compensation:

**Step A — Wage models at industry-year level:**
- For each industry-year cell, run OLS: `log_wage ~ AI_ROLE + controls`
- Extract the beta coefficient on `AI ROLE` for each industry-year
- This gives a panel of "AI wage premiums" by industry-year

**Step B — Perk models at industry-year level:**
- For each industry-year cell and each perk, run: `perk ~ AI_ROLE + controls`
- Extract the beta coefficient on `AI ROLE` for each industry-year-perk
- This gives a panel of "AI perk premiums" by industry-year

**Step C — Compare and scatterplot:**
- For each perk, scatterplot the AI wage beta (x) against the AI perk beta (y) across
  industry-years
- A positive relationship suggests perks complement wages in attracting AI talent

**Data source:** Use `ind_year_analysis_raw.parquet` from `occ_year_analysis_raw.py` for
the aggregated data, or run the regressions directly on the job-level data grouped by
industry-year.

---

## 5. High-AI-prevalence firm analysis
**New analysis — can be added to descriptive or as a new file**

- Identify firms with high share of AI postings (e.g., top quartile by % AI ROLE)
- For these firms, examine: is perk prevalence uniformly high across all their postings
  (AI and non-AI), or is there still meaningful within-firm variation?
- This addresses the concern that large tech firms drive the results simply by offering
  perks to everyone

**Implementation:** Group by `COMPANY`, compute `% AI ROLE` and `% perk` for each firm,
then compare perk prevalence between AI and non-AI postings *within* high-AI firms.

---

## 6. AI classification threshold robustness
**Files:** `scripts/prepare_data.py`, `analysis/regression_models_2024.py`

- `AI_SKILL_COUNT` is added in step 1
- Create alternative AI ROLE flags: `AI_ROLE_2PLUS`, `AI_ROLE_3PLUS` (requiring 2+ or 3+
  AI/ML skills)
- Rerun the main regression models with each threshold
- Report how beta coefficients and sample sizes change as classification becomes more
  conservative
- Can be a loop in the regression script or a separate robustness script

---

## 7. LLM-based perk classification validation
**New script/notebook**

- Take a subsample of postings (e.g., 1000-2000)
- Run a small LLM (e.g., Claude Haiku) to classify perks from the BODY text
- Compare LLM classifications against the keyword-based labels from `label_benefits.py`
- Report precision/recall/F1 for each perk category
- This validates the keyword approach or identifies where it falls short

**Note:** Requires API access. Can use the Anthropic SDK (already in the project context).

---

## 8. Perk positioning analysis
**File:** `scripts/label_benefits.py` (modify `check_benefits()`)

Currently `check_benefits()` returns True/False. Extend to also capture position:

- When a keyword match is found, record `match.start() / len(text) * 100`
- This gives a 0-100 value: where in the posting the perk first appears
- Store as a new column per benefit (e.g., `EDU_ASSISTANCE_POSITION`)
- A value of 95 means the perk appears in the first 5% of text (prominent)
- A value of 5 means it appears in the last 5% (buried)

**Downstream analysis:** Regress perk position on AI ROLE + controls. Test whether AI
postings place perks more prominently.

**Alternative:** Simpler decile approach — classify position into top/middle/bottom third.

---

## 9. Additional benefit categories from structured fields
**Files:** `scripts/label_benefits.py`, `analysis/regression_models.py`, analysis scripts

The structured `BENEFIT_SUBCATEGORY_NAME` field contains categories not currently
captured by keyword labels. Add two new benefit variables:

- **Flexible Work Schedules**: Use the structured "Flexible Work Schedules" subcategory
  (9,816 rows in subsample). This complements the existing `REMOTE_KW` keyword label
  which captures remote/hybrid language in body text — the structured field captures
  a formal benefit designation.
- **Professional Development / Leadership Development / Mentorships**: Use the structured
  "Professional Development" (7,344 rows), "Leadership Development" (765 rows), and
  "Mentorships" (504 rows) subcategories. This is broader than `EDU_ASSISTANCE` which
  focuses on tuition/education reimbursement keywords.

**Implementation:** Either derive from the structured fields directly, or add new keyword
lists to `label_benefits.py` to capture these from body text (for consistency with existing
approach). Decide after reviewing the `keyword_vs_structured_benefits.ipynb` comparison.

**Downstream:** Add to `benefits4` list, rerun regression models and descriptive analyses
with the expanded benefit set.

---

## Suggested priority order

| Priority | Task | Complexity | Dependency |
|----------|------|-----------|------------|
| 1 | Pipeline: carry COMPANY + AI_SKILL_COUNT (step 1) | Low | None |
| 2 | Add firm + state FE to regressions (step 3) | Medium | Step 1 |
| 3 | Industry-year wage vs perk coefficient analysis (step 4) | Medium | Step 1 |
| 4 | High-AI firm analysis (step 5) | Low | Step 1 |
| 5 | AI threshold robustness (step 6) | Low | Step 1 |
| 6 | Perk positioning analysis (step 8) | Low | None |
| 7 | LLM perk validation (step 7) | Medium | None |
| 8 | Rerun full pipeline + all analyses (step 2) | Low | Step 1 |
| 9 | Flexible Work + Prof Dev/Mentorship benefit categories (step 9) | Low | Keyword vs structured comparison |

Steps 1-6 can proceed now with the MAY26 subsample (the expanded dataset).
Step 7 (LLM validation) is independent and can happen anytime.

---

## Verification
- After step 1: check that `data_v1.parquet` contains `COMPANY` and `AI_SKILL_COUNT` columns
- After step 3: compare model results with and without firm/state FE; check memory usage
- After step 4: produce scatterplots of wage betas vs perk betas
- After step 6: produce a table of beta coefficients across thresholds (1+, 2+, 3+ AI skills)
- After step 8: compare keyword vs LLM perk labels on the subsample
