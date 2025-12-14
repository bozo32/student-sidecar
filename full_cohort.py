#!/usr/bin/env python3
import argparse
import subprocess
import sys
import shutil
import os
from pathlib import Path

def run(cmd: list[str]):
    # Print a shell-like command line for readability; keep argv safe for spaces.
    printable = " ".join(cmd)
    print("\n>>>", printable)
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"[ERROR] Command failed with exit code {res.returncode}: {printable}", file=sys.stderr)
        sys.exit(res.returncode)

def main():
    ap = argparse.ArgumentParser(
        description="Full authoritative rebuild for one cohort. The --cohort-id argument is mandatory and is used for cross-run provenance."
    )
    ap.add_argument("root", help="Root directory of cohort submissions")
    ap.add_argument("--wipe", action="store_true", help="Delete existing artifacts before running")
    ap.add_argument("--grobid-url", default="http://localhost:8070")
    ap.add_argument("--cheap-only", action="store_true", help="Skip NLI in cherry-picking sniff")
    ap.add_argument("--cohort-id", required=True, help="Cohort identifier for cross-run provenance (required)")
    ap.add_argument("--cohort-note", default=None, help="Optional note describing the academic context of this cohort")
    args = ap.parse_args()

    # Ensure stdout is unbuffered (better progress visibility in long runs)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    artifacts = Path("artifacts")

    if args.wipe:
        print(">>> Wiping artifacts/")
        shutil.rmtree(artifacts, ignore_errors=True)

    (artifacts / "text").mkdir(parents=True, exist_ok=True)
    (artifacts / "parquet").mkdir(parents=True, exist_ok=True)
    (artifacts / "verification").mkdir(parents=True, exist_ok=True)
    (artifacts / "cherrypicking").mkdir(parents=True, exist_ok=True)

    extract_cmd = [
        sys.executable, "extract_text.py", args.root,
        "--texts-dir", "artifacts/text",
        "--parquet-dir", "artifacts/parquet",
        "--extensions", ".pdf,.docx,.html,.htm,.txt,.md,.rst",
        "--grobid-url", args.grobid_url,
        "--cohort-id", args.cohort_id,
    ]
    if args.cohort_note is not None:
        extract_cmd.extend(["--cohort-note", args.cohort_note])
    run(extract_cmd)

    run([
        sys.executable, "build_pairs.py", args.root,
        "--texts-dir", "artifacts/text",
        "--parquet-dir", "artifacts/parquet",
        "--prefer-excel", "--fallback-csv", "--csv-encoding-sweep",
        "--consolidate-urls", "--urls-subdir", "urls"
    ])

    run([
        sys.executable, "verify_quotes.py",
        "--parquet-dir", "artifacts/parquet",
        "--texts-dir", "artifacts/text",
        "--out-dir", "artifacts/verification",
        "--encoder", "all-MiniLM-L6-v2",
        "--bm25-topk", "20",
        "--cos-thresh", "0.82",
        "--fuzzy-thresh", "85",
        "--summary-csv"
    ])

    sniff_cmd = [
        sys.executable, "sniff_cherrypicking.py",
        "--parquet-dir", "artifacts/parquet",
        "--texts-dir", "artifacts/text",
        "--out-dir", "artifacts/cherrypicking",
        "--query-field", "quote_from_report"
    ]
    if args.cheap_only:
        sniff_cmd.append("--cheap-only")

    run(sniff_cmd)

    print("\n=== FULL COHORT REBUILD COMPLETE ===")

if __name__ == "__main__":
    main()