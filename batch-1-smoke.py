

#!/usr/bin/env python3
"""
batch-one-smoke.py

Purpose:
  Fast, deterministic smoke tests for a single batch (Batch 1) to ensure the
  student-sidecar pipeline is wired correctly and producing plausible outputs.

Design principles:
  - No re-extraction or mutation of artifacts.
  - Uses existing artifacts under ./artifacts/.
  - Fails loudly and early on structural regressions.
  - Anchored on two known cases:
      * Group 9: trivial, expected ~100% success.
      * Group 5: non-trivial, expected ~50% success.

This is NOT a correctness test of NLP quality.
It is a pipeline integrity and invariants test.
"""

from pathlib import Path
import sys
import pandas as pd

ARTIFACTS = Path("artifacts")
PARQUET = ARTIFACTS / "parquet"
VERIFICATION = ARTIFACTS / "verification"

REQUIRED_PARQUETS = {
    "pairs_raw": PARQUET / "pairs_raw.parquet",
    "sources": PARQUET / "sources.parquet",
    "tables_report": PARQUET / "tables_report.parquet",
}


EXPECTED_GROUPS = {
    "Group 9 supplemental information": {
        "min_pairs": 1,
        "min_success_ratio": 0.95,
    },
    "Group 5 supplemental information": {
        "min_pairs": 1,
        "min_success_ratio": 0.40,
        "max_success_ratio": 0.60,
    },
}

# --- Normalization helper for group id comparison ---
import re
def norm_group(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def fail(msg: str):
    print(f"[SMOKE FAIL] {msg}", file=sys.stderr)
    sys.exit(1)


def ok(msg: str):
    print(f"[OK] {msg}")


def check_parquets_exist():
    for name, path in REQUIRED_PARQUETS.items():
        if not path.exists():
            fail(f"Missing required parquet: {path}")
        ok(f"Found {name}: {path.name}")


def load_pairs() -> pd.DataFrame:
    try:
        df = pd.read_parquet(REQUIRED_PARQUETS["pairs_raw"])
    except Exception as e:
        fail(f"Could not read pairs_raw.parquet: {e}")
    required_cols = {
        "group_id",
        "quote_from_report",
        "quote_from_source",
        "source_sha256",
        "extract_ok",
    }
    missing = required_cols - set(df.columns)
    if missing:
        fail(f"pairs_raw.parquet missing columns: {missing}")
    ok("pairs_raw.parquet schema OK")
    return df


def load_verification_summary() -> pd.DataFrame:
    summary_csv = VERIFICATION / "verification_summary.csv"
    if not summary_csv.exists():
        fail("Missing artifacts/verification/verification_summary.csv")
    try:
        df = pd.read_csv(summary_csv)
    except Exception as e:
        fail(f"Could not read verification_summary.csv: {e}")

    # Accept either legacy `group` or newer `group_id` column names
    if "group_id" not in df.columns and "group" in df.columns:
        df = df.rename(columns={"group": "group_id"})
    # Always treat group_id as string (to avoid int/float issues)
    df["group_id"] = df["group_id"].astype(str)

    required_cols = {"group_id", "success_ratio"}
    missing = required_cols - set(df.columns)
    if missing:
        fail(f"verification_summary.csv missing columns: {missing}")

    ok("verification_summary.csv schema OK")
    return df


def check_group_expectations(pairs: pd.DataFrame, summary: pd.DataFrame):
    for group, spec in EXPECTED_GROUPS.items():
        target = norm_group(group)
        g_pairs = pairs[pairs["group_id"].astype(str).map(norm_group) == target]
        if len(g_pairs) < spec["min_pairs"]:
            observed = sorted(set(pairs["group_id"].astype(str).map(norm_group)))
            sample = ", ".join(observed[:15])
            fail(
                f"{group}: expected ≥{spec['min_pairs']} rows in pairs_raw, found {len(g_pairs)}. "
                f"Observed group_id values (normalized, first 15): {sample}"
            )

        g_sum = summary[summary["group_id"].astype(str).map(norm_group) == target]
        if g_sum.empty:
            fail(f"{group}: missing from verification_summary.csv")

        ratio = float(g_sum.iloc[0]["success_ratio"])

        if ratio < spec["min_success_ratio"]:
            fail(f"{group}: success_ratio {ratio:.2f} < expected minimum {spec['min_success_ratio']:.2f}")

        if "max_success_ratio" in spec and ratio > spec["max_success_ratio"]:
            fail(f"{group}: success_ratio {ratio:.2f} > expected maximum {spec['max_success_ratio']:.2f}")

        ok(f"{group}: success_ratio={ratio:.2f} within expected bounds")


def main():
    print("=== Batch 1 smoke test ===")

    check_parquets_exist()
    pairs = load_pairs()
    summary = load_verification_summary()
    check_group_expectations(pairs, summary)

    print("=== Smoke test PASSED ===")


if __name__ == "__main__":
    main()