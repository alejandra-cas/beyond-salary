#!/usr/bin/env python3
"""Build a posting-system S&P 500 snapshot from a scraped constituent table."""

import argparse
from difflib import get_close_matches
from pathlib import Path
import re

import pandas as pd


COMPANY_SUFFIXES = {
    "and",
    "co",
    "com",
    "company",
    "cos",
    "corp",
    "corporation",
    "group",
    "holdings",
    "inc",
    "incorporated",
    "ltd",
    "limited",
    "plc",
    "the",
}
TRAILING_QUALIFIERS = {"de", "md", "ny", "oh"}
REVIEWED_ALIASES = {
    "alphabet": "google",
    "bank of new york mellon": "bny mellon",
    "bunge global sa": "bunge",
    "capital one financial": "capital one",
    "carrier global": "carrier",
    "casey s general stores": "casey s",
    "charles river laboratories international": "charles river laboratories",
    "chipotle mexican grill": "chipotle",
    "cisco systems": "cisco",
    "costco wholesale": "costco",
    "deckers outdoor": "deckers",
    "digital realty trust": "digital realty",
    "dupont de nemours": "dupont",
    "expeditors international of washington": "expeditors",
    "fifth third bancorp": "fifth third",
    "ford motor": "ford",
    "garmin": "garmin international",
    "ge healthcare technologies": "ge healthcare",
    "generac": "generac power systems",
    "hartford insurance": "hartford",
    "hilton worldwide": "hilton",
    "hormel foods": "hormel",
    "idexx laboratories": "idexx",
    "international business machines": "ibm",
    "intuitive surgical": "intuitive",
    "jb hunt transport services": "jb hunt",
    "johnson controls international": "johnson controls",
    "keysight technologies": "keysight",
    "kimco realty": "kimco",
    "live nation entertainment": "live nation",
    "lululemon athletica": "lululemon",
    "lyondellbasell industries nv": "lyondellbasell",
    "martin marietta materials": "martin marietta",
    "meta platforms": "meta",
    "mettler toledo international": "mettler toledo",
    "moderna": "moderna therapeutics",
    "molson coors beverage": "molson coors",
    "oneok": "oneok services",
    "pnc financial services": "pnc",
    "ppg industries": "ppg",
    "ralph lauren": "ralph lauren retail",
    "royal caribbean cruises": "royal caribbean",
    "synchrony financial": "synchrony",
    "uber technologies": "uber",
    "union pacific": "union pacific railroad",
    "westinghouse air brake technologies": "wabtec",
}


def normalize_company_name(value):
    """Normalize a company name for deterministic matching."""
    tokens = re.sub(r"[^a-z0-9]+", " ", str(value).lower()).split()
    while tokens and tokens[-1] in TRAILING_QUALIFIERS:
        tokens.pop()
    return " ".join(token for token in tokens if token not in COMPANY_SUFFIXES)


def parse_constituents(input_path):
    """Parse rank, company, and symbol from a scraped tab-separated constituent table."""
    rows = []
    for line in Path(input_path).read_text().splitlines():
        match = re.match(r"^(\d+)\t([^\t]+)\t([^\t]+)\t", line)
        if match:
            rank, company_name, symbol = match.groups()
            rows.append({
                "RANK": int(rank),
                "SP500_COMPANY_NAME": company_name.strip(),
                "SYMBOL": symbol.strip(),
            })
    if not rows:
        raise ValueError("No S&P 500 constituent rows were parsed from the input file.")
    return pd.DataFrame(rows)


def build_company_lookup(posts_csv):
    """Build a unique normalized-name lookup from the posting-system company IDs."""
    companies = pd.read_csv(posts_csv, usecols=["COMPANY", "COMPANY_NAME"]).drop_duplicates()
    companies = companies[companies["COMPANY"].notna() & (companies["COMPANY"] != 0)].copy()
    companies["NORMALIZED_NAME"] = companies["COMPANY_NAME"].map(normalize_company_name)
    return companies


def build_snapshot(constituents, companies, snapshot_date):
    """Match normalized S&P names to posting-system company IDs and return snapshot plus audit."""
    company_groups = companies.groupby("NORMALIZED_NAME")
    companies["COMPACT_NAME"] = companies["NORMALIZED_NAME"].str.replace(" ", "", regex=False)
    compact_groups = companies.groupby("COMPACT_NAME")
    available_names = set(companies["NORMALIZED_NAME"])
    rows = []

    for company_name, group in constituents.groupby("SP500_COMPANY_NAME", sort=False):
        normalized_name = normalize_company_name(company_name)
        normalized_name = REVIEWED_ALIASES.get(normalized_name, normalized_name)
        matches = company_groups.get_group(normalized_name) if normalized_name in company_groups.groups else companies.iloc[0:0]
        if matches.empty:
            compact_name = normalized_name.replace(" ", "")
            matches = compact_groups.get_group(compact_name) if compact_name in compact_groups.groups else companies.iloc[0:0]
        unique_matches = matches[["COMPANY", "COMPANY_NAME"]].drop_duplicates()
        suggestions = get_close_matches(normalized_name, available_names, n=3, cutoff=0.75)

        if len(unique_matches) == 1:
            company_id = unique_matches.iloc[0]["COMPANY"]
            posting_name = unique_matches.iloc[0]["COMPANY_NAME"]
            status = "matched"
        elif len(unique_matches) > 1:
            company_id = pd.NA
            posting_name = ""
            status = "ambiguous"
        else:
            company_id = pd.NA
            posting_name = ""
            status = "unmatched"

        rows.append({
            "COMPANY": company_id,
            "COMPANY_NAME": posting_name,
            "SP500_COMPANY_NAME": company_name,
            "SYMBOLS": ",".join(group["SYMBOL"]),
            "SNAPSHOT_DATE": snapshot_date,
            "MATCH_STATUS": status,
            "NORMALIZED_NAME": normalized_name,
            "SUGGESTED_NORMALIZED_NAMES": " | ".join(suggestions),
        })

    audit = pd.DataFrame(rows)
    snapshot = audit[audit["MATCH_STATUS"] == "matched"][
        ["COMPANY", "COMPANY_NAME", "SP500_COMPANY_NAME", "SYMBOLS", "SNAPSHOT_DATE"]
    ].copy()
    snapshot["COMPANY"] = snapshot["COMPANY"].astype(int)
    return snapshot, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Scraped S&P 500 constituent text file")
    parser.add_argument("--posts-csv", required=True, help="Posting CSV containing COMPANY and COMPANY_NAME")
    parser.add_argument("--output", required=True, help="Output company-ID snapshot CSV")
    parser.add_argument("--audit-output", required=True, help="Output matching audit CSV")
    parser.add_argument("--snapshot-date", required=True, help="Snapshot date in YYYY-MM-DD format")
    args = parser.parse_args()

    constituents = parse_constituents(args.input)
    companies = build_company_lookup(args.posts_csv)
    snapshot, audit = build_snapshot(constituents, companies, args.snapshot_date)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.audit_output).parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(args.output, index=False)
    audit.to_csv(args.audit_output, index=False)

    print(f"Parsed constituent rows: {len(constituents):,}")
    print(f"Unique constituent companies: {len(audit):,}")
    print(audit["MATCH_STATUS"].value_counts().to_string())
    print(f"Snapshot saved: {args.output}")
    print(f"Audit saved: {args.audit_output}")


if __name__ == "__main__":
    main()
