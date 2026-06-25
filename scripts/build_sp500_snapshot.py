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


# Manual ticker -> posting-system COMPANY id overrides for constituents the
# conservative name matcher leaves unmatched or ambiguous. Each id was confirmed
# by inspecting the dominant posting-system company for that constituent in the
# full posts CSV (e.g. UPS posts as "UPS", Charter as "Spectrum", GE as
# "GE Aerospace", US Bancorp as "US Bank"). Tickers not listed here fall back to
# normal name matching.
SYMBOL_COMPANY_OVERRIDES = {
    # Unmatched: real firm posts under a different name than the constituent name
    "LLY": 40425754,   # Eli Lilly -> "Lilly"
    "AMD": 9175501,    # Advanced Micro Devices -> "AMD"
    "GE": 76540,       # General Electric -> "GE Aerospace"
    "DIS": 89628099,   # Walt Disney -> "Disney"
    "DE": 3580599,     # Deere -> "John Deere"
    "USB": 37723136,   # US Bancorp -> "US Bank"
    "UPS": 12330310,   # United Parcel Service -> "UPS"
    "ADP": 7897782,    # Automatic Data Processing -> "ADP"
    "ORLY": 42049638,  # O'Reilly Automotive -> "O'Reilly Auto Parts"
    "AIG": 105502207,  # American International Group -> "AIG"
    "CCL": 99461686,   # Carnival -> "Carnival Cruise Lines"
    "EL": 89628126,    # Estee Lauder -> "The Estée Lauder Companies"
    "KEY": 36031738,   # KeyCorp -> "KeyBank"
    "SMCI": 37448805,  # Super Micro Computer -> "Supermicro"
    "CHTR": 41110818,  # Charter Communications -> "Spectrum"
    "HOOD": 36254093,  # Robinhood Markets -> "Robinhood"
    # Ambiguous: multiple posting-system names normalized alike; pick the
    # dominant real firm by posting volume
    "LIN": 8821559,    # Linde
    "SCHW": 89628114,  # Charles Schwab
    "BX": 5227848,     # Blackstone -> "The Blackstone Group"
    "DHR": 36926534,   # Danaher
    "PGR": 38522287,   # Progressive
    "COF": 38360945,   # Capital One
    "LOW": 37662275,   # Lowe's
    "SYK": 4855312,    # Stryker
    "SO": 36798433,    # Southern -> "Southern Company"
    "WMB": 7749595,    # Williams
    "AON": 4241295,    # Aon -> "AON"
    "LITE": 39930333,  # Lumentum -> "Lumentum Holdings"
    "FLEX": 61624047,  # Flex
    "XYZ": 7968440,    # Block
    "VTR": 99459648,   # Ventas
    "WAT": 3125543,    # Waters
    "SYF": 10106637,   # Synchrony
    "DOW": 62009627,   # Dow -> "Dow Chemical"
    "ALB": 404315,     # Albemarle
    "CDW": 2463641,    # CDW
    "EG": 40621853,    # Everest -> "The Everest Group"
    "CSGP": 41418228,  # CoStar -> "CoStar Group"
    "SJM": 40363361,   # J M Smucker -> "The J M Smucker Company"
    "AES": 41501248,   # AES -> "The AES Corporation"
    "FOX": 2859385,    # Fox
    "FOXA": 2859385,   # Fox (class A) -> same posting-system company
    "MOS": 9469562,    # Mosaic
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

    companies_by_id = companies.drop_duplicates("COMPANY").set_index("COMPANY")
    for company_name, group in constituents.groupby("SP500_COMPANY_NAME", sort=False):
        normalized_name = normalize_company_name(company_name)
        normalized_name = REVIEWED_ALIASES.get(normalized_name, normalized_name)

        # Manual ticker override takes precedence over name matching so confirmed
        # firms (e.g. UPS, Charter/Spectrum, GE/GE Aerospace) resolve directly to
        # their posting-system COMPANY id.
        override_id = next(
            (SYMBOL_COMPANY_OVERRIDES[s] for s in group["SYMBOL"]
             if s in SYMBOL_COMPANY_OVERRIDES),
            None,
        )
        if override_id is not None and override_id in companies_by_id.index:
            rows.append({
                "COMPANY": override_id,
                "COMPANY_NAME": companies_by_id.loc[override_id, "COMPANY_NAME"],
                "SP500_COMPANY_NAME": company_name,
                "SYMBOLS": ",".join(group["SYMBOL"]),
                "SNAPSHOT_DATE": snapshot_date,
                "MATCH_STATUS": "matched",
                "NORMALIZED_NAME": normalized_name,
                "SUGGESTED_NORMALIZED_NAMES": "",
            })
            continue

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
