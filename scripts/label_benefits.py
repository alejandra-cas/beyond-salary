import pandas as pd
import re
import time

from pathlib import Path

_base = Path(__file__).parent.parent / "data" / "processed"
input_path = _base / "data_v1.parquet"
output_path = _base / "labeled_v1.parquet"

# --- Keyword definitions ---

edu_assistance = (
    "education assistance",
    "tuition reimbursement",
    "tuition assistance",
    "education reimbursement",
)
leave = (
    "paid time off",
    "PTO",
    "extra vacation",
    "flexible vacation",
    "mental health day",
    "generous time-off",
    "generous time off",
    "paid vacation",
    "paid holidays",
    "holiday pay",
    "vacation days",
    "company holiday",
    "sick leave",
    "vacation time",
    "paid days off",
    "paid flexible holidays",
)
wellbeing = (
    "wellness stipend",
    "mental health support",
    "gym membership",
    "wellness program",
    "well-being stipend",
    "wellness programs",
    "mental health benefits",
    "employee well-being",
    "mental wellness support",
    "wellness and global well-being",
    "wellness stipend",
)
health_wellbeing = (
    "wellness stipend",
    "health benefits",
    "mental health support",
    "gym membership",
    "wellness program",
    "well-being stipend",
    "wellness programs",
    "mental health benefits",
    "employee well-being",
    "health care benefits",
)
parental_leave = (
    "parental leave",
    "family support",
    "flexible maternity leave",
    "paternity leave",
    "maternity leave",
    "paid family leave",
    "paid caregiver/parental",
    "paid parental",
    "paid new parent leave",
    "paid bonding leave",
    "parental bonding leave",
)
culture = (
    "diversity and inclusion",
    "team culture",
    "creative freedom",
    "values-driven",
    "inclusive culture",
    "diverse team",
    "inclusive environment",
    "value diversity",
    "diversity, equity",
    "culture of diversity",
    "diversity, inclusion",
    "inclusion and diversity",
    "commitment to diversity",
    "diversity, inclusion",
    "diversity, equity and inclusion",
    "diversity is respected",
    "inclusive diversity",
    "workforce diversity",
    "embracing diversity",
    "value diversity",
    "values diversity",
    "committed to diversity",
    "equity and diversity",
    "promoting diversity",
    "celebrate diversity",
    "encourage diversity",
    "support diversity",
    "diversity in the workplace",
    "equity, inclusion",
)

# --- Exclusion lists ---

career_dev_to_exclude = [
    "youth leadership development",
    "provide mentorship",
    "professional development expertise",
    "planning professional development",
    "forecasting growth opportunities",
    "teen leadership development",
    "develop training programs",
    "professional development experience",
    "training programs as required",
    "providing mentorship",
    "providing and encouraging mentorship",
    "implementing training programs",
    "provide leadership and mentorship",
    "provide mentorship",
    "identify growth opportunities",
    "manage the Symbotic Tuition Reimbursement",
]
tuition_to_exclude = ["manage the Symbotic Tuition Reimbursement"]
wellbeing_to_exclude = [
    "health benefits companies",
    "wellness program coordinator",
    "familiarity with mental health support",
    "health benefits administration",
]
recognition_to_exclude = ["managing employee recognition"]
family_to_exclude = [
    "accredited childcare program",
    "maui family support services",
    "teaching assistant - childcare",
    "experience in childcare",
    "childcare state licensing",
    "experience with children",
]
culture_to_exclude = ["maintaining a collaborative environment"]


def check_benefits(body_series, keywords, exclusions=None):
    """Vectorized benefit labeling using compiled regex.

    Uses pandas str.contains() for the bulk of rows, then falls back to
    row-level overlap checking only for the small subset that matched
    both inclusion and exclusion patterns.

    Returns a boolean numpy array.
    """
    include_re = re.compile(
        r"\b(?:" + "|".join(map(re.escape, keywords)) + r")\b", re.IGNORECASE
    )

    # Vectorized inclusion check
    body_filled = body_series.fillna("")
    included = body_filled.str.contains(include_re.pattern, regex=True, case=False, na=False)

    if exclusions is None:
        return included.to_numpy(dtype=bool)

    # Only check exclusions on the subset that matched inclusion
    exclude_re = re.compile(
        r"\b(?:" + "|".join(map(re.escape, exclusions)) + r")\b", re.IGNORECASE
    )
    excluded = body_filled.str.contains(exclude_re.pattern, regex=True, case=False, na=False)

    # Rows with inclusion but no exclusion are True
    # Rows with both need overlap check
    needs_overlap_check = included & excluded

    if not needs_overlap_check.any():
        return included.to_numpy(dtype=bool)

    # Row-level overlap check only for the small subset with both matches
    result = included.to_numpy(dtype=bool, copy=True)
    check_idx = needs_overlap_check[needs_overlap_check].index

    for idx in check_idx:
        text = body_filled.iloc[idx]
        include_matches = list(include_re.finditer(text))
        exclude_matches = list(exclude_re.finditer(text))

        # Check if any exclusion overlaps with an inclusion match
        has_overlap = False
        for exc in exclude_matches:
            for inc in include_matches:
                if inc.start() <= exc.start() < inc.end() or inc.start() <= exc.end() <= inc.end():
                    has_overlap = True
                    break
            if has_overlap:
                break

        if has_overlap:
            result[idx] = False

    return result


def main():
    load_time = time.time()
    print("Loading the input parquet file...")
    data = pd.read_parquet(input_path)
    print(f"Loaded {len(data):,} rows in {time.time() - load_time:.1f}s")

    empty_body_count = data["BODY"].isna().sum() + data["BODY"].str.strip().eq("").sum()
    print(f"Empty body text: {empty_body_count:,}")

    # Extract BODY series once for all benefit checks
    body = data["BODY"]

    print("Labeling benefits...")
    start = time.time()

    # check_benefits(body, career_dev, exclusions=career_dev_to_exclude)
    data["EDU_ASSISTANCE"] = check_benefits(body, edu_assistance, exclusions=tuition_to_exclude)
    print(f"  EDU_ASSISTANCE: {time.time() - start:.1f}s")

    data["PAID LEAVE"] = check_benefits(body, leave)
    print(f"  PAID LEAVE: {time.time() - start:.1f}s")

    data["WELLBEING"] = check_benefits(body, wellbeing, wellbeing_to_exclude)
    print(f"  WELLBEING: {time.time() - start:.1f}s")

    data["HEALTH_WELLBEING"] = check_benefits(body, health_wellbeing, wellbeing_to_exclude)
    print(f"  HEALTH_WELLBEING: {time.time() - start:.1f}s")

    data["PARENTAL_LEAVE"] = check_benefits(body, parental_leave)
    print(f"  PARENTAL_LEAVE: {time.time() - start:.1f}s")

    data["CULTURE"] = check_benefits(body, culture)
    print(f"  CULTURE: {time.time() - start:.1f}s")

    print(f"Total labeling time: {time.time() - start:.1f}s")

    save_time = time.time()
    print("Saving to parquet...")
    data.to_parquet(output_path, compression='gzip')
    print(f"Saved in {time.time() - save_time:.1f}s")


if __name__ == "__main__":
    main()
