import pandas as pd
import re
import time
import yaml
from pathlib import Path


def load_processed_paths():
    """Load processed parquet paths from config.yaml.

    Falls back to repo-local defaults if config.yaml is not found.
    """
    repo_root = Path(__file__).parent.parent
    config_path = repo_root / "config.yaml"

    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["data"]
        processed_dir = Path(cfg["processed_dir"])
        if not processed_dir.is_absolute():
            processed_dir = repo_root / processed_dir
    else:
        print(
            "Warning: config.yaml not found, using default processed paths. "
            "Copy config.example.yaml to config.yaml to configure."
        )
        processed_dir = repo_root / "data" / "processed"

    return {
        "input_path": processed_dir / "labeled_v1.parquet",
        "output_path": processed_dir / "labeled_v2.parquet",
    }


paths = load_processed_paths()
input_path = paths["input_path"]
output_path = paths["output_path"]

remote_keywords = [
    "fully remote", "100% remote", "work from home", "remote role", "work remotely",
    "remote eligible", "open to remote", "remote work environment", "remote work policy",
    "remote and onsite", "virtual role", "remote first", "home office", "telecommute",
    "distributed team", "remote-first", "remote-friendly", "remote options", "virtual work", "work from home",
    "remote position", "wfh", "Remote - us", "telework", "home office", "(remote)", "remote work flexibility", "remote work: hybrid",
    "remote: yes", "location: remote", "remote usa", "remote flexibility", "(remote", "- remote", "remote-flexibility",
    "remote,", "\nremote\n", "\n remote \n", "remotely", "us-remote", "remote work eligible", "remote - united states",
    "remote - work at home", "can be remote", "remote within the us", "remote in north america", "remote available" ", remote",
    "remote full-time", "us remote", "#li-remote", "open to remote",
    "hybrid work", "split time between office and remote", "office and remote options",
    "partially remote", "hybrid position", "3 days remote", "hybrid workplace",
    "some remote days", "in-office and remote", "hybrid onsite", "remote or in-office", "work-from-home days", "mix of working in the office and from home", "#li-hybrid", "hybrid remote"
]

remote_to_exclude = [
    "not considering remote", "in-office only",
    "remote monitoring", "remote sensing", "remote access systems", "remote control",
    "remote diagnostics", "remote delivery", "#li-onsite", "onsite job", "work from home not available", "telework:no",
    "remote: no", "100% on-site", "work at home option: No",
    "remotely: no", "remote: n", "remotely: n", "telework: no", "remote: * no", "remotely: * no",
    "remotely piloted", "data remotely", "remote type on-site", "interviewed remotely", "work remotely: * no",
    "remote desktop", "must be able to work on-site", "not applicable for 100% remote", "remote testing", "remotely upgrading", "remote usability",
    "remote site", "remote machine", "remote iot", "no remote", "work remotely no", "remotely:no", "remotely? n", "remotely no", "remote areas",
    "remote access", "remote position? no", "remotely tucked away", "supporting remote", "remotely sensed"
]


def check_benefit_position(body_series, keywords, exclusions=None):
    """Return the prominence score (0–100) of the first keyword match in each posting.

    Score = (1 - match.start() / len(text)) * 100
      - 100: match at the very start of the text (most prominent)
      - 0:   match at the very end of the text (least prominent)
      - NaN: no match (or empty body)
    """
    include_re = re.compile(
        r"\b(?:" + "|".join(map(re.escape, keywords)) + r")\b", re.IGNORECASE
    )
    exclude_re = (
        re.compile(
            r"\b(?:" + "|".join(map(re.escape, exclusions)) + r")\b", re.IGNORECASE
        )
        if exclusions
        else None
    )

    body_filled = body_series.fillna("")
    positions = pd.array([float("nan")] * len(body_series), dtype="Float64")

    for i, text in enumerate(body_filled):
        if not text:
            continue
        text_len = len(text)
        for m in include_re.finditer(text):
            if exclude_re:
                skip = False
                for exc in exclude_re.finditer(text):
                    if exc.start() <= m.start() < exc.end() or exc.start() < m.end() <= exc.end():
                        skip = True
                        break
                if skip:
                    continue
            positions[i] = (1 - m.start() / text_len) * 100
            break

    return positions


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
    # Only read ID and BODY columns for keyword labeling
    body_df = pd.read_parquet(input_path, columns=['ID', 'BODY'])
    print(f"Loaded {len(body_df):,} rows in {time.time() - load_time:.1f}s")

    empty_body_count = body_df['BODY'].isna().sum() + body_df['BODY'].str.strip().eq("").sum()
    print(f"Empty body text: {empty_body_count:,}")

    # Clean body text
    print("Cleaning body text...")
    clean_time = time.time()
    body_df['BODY'] = (
        body_df['BODY']
        .fillna("")
        .str.strip()
        .str.replace(r'\s+', ' ', regex=True)
        .str.lower()
    )
    print(f"Cleaned in {time.time() - clean_time:.1f}s")

    print("Labeling remote work...")
    start = time.time()
    body_df['REMOTE_KW'] = check_benefits(body_df['BODY'], remote_keywords, remote_to_exclude)
    print(f"Labeled in {time.time() - start:.1f}s")

    print("Computing remote work prominence position...")
    pos_start = time.time()
    body_df['REMOTE_KW_POSITION'] = check_benefit_position(body_df['BODY'], remote_keywords, remote_to_exclude)
    valid = body_df['REMOTE_KW_POSITION'].dropna()
    print(f"  REMOTE_KW_POSITION: {len(valid):,} matches, mean prominence {valid.mean():.1f} ({time.time() - pos_start:.1f}s)")

    # Merge REMOTE_KW and REMOTE_KW_POSITION into the full labeled dataset and overwrite labeled_v1
    print("Merging REMOTE_KW and REMOTE_KW_POSITION into labeled_v1.parquet...")
    data = pd.read_parquet(input_path)
    for col in ['REMOTE_KW', 'REMOTE_KW_POSITION']:
        if col in data.columns:
            data = data.drop(columns=[col])
    data = data.merge(body_df[['ID', 'REMOTE_KW', 'REMOTE_KW_POSITION']], on='ID', how='left')

    save_time = time.time()
    data.to_parquet(output_path, compression='gzip')
    print(f"Saved labeled_v1.parquet with REMOTE_KW in {time.time() - save_time:.1f}s")


if __name__ == "__main__":
    main()
