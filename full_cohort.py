#!/usr/bin/env python3

import argparse
import subprocess
import sys
import shutil
import os
from pathlib import Path

PROCESS_SUBMISSIONS_SPEC = """
process_submissions.py (teacher-proof spec)

Purpose
  - Find URL-only citations embedded in messy spreadsheet cells (e.g., a long sentence containing https://... somewhere in the middle).
  - Download those URL targets into each group folder so the pipeline can treat them like local source files.
  - Emit a URL manifest so build_pairs.py can resolve a row that cites a URL into:
      (a) a concrete downloaded file path, and then
      (b) a stable sha256 (via extract_text.py), enabling downstream quote verification.

Input expectations (cohort root folder)
  cohort_root/
    <group folder>/
      *.xlsx or *.csv           (a sources table)
      cited_file_1.pdf/docx/... (optional local sources)
      cited_file_2.pdf/docx/... (optional local sources)

  - The sources table may have URLs in any of these fields:
      * file_name_raw  (often contains URLs or filenames)
      * source_uri      (preferred if present)
      * source_ref      (may contain DOI/URL-like content)
    and URLs may be embedded inside other text.

What it writes (per group)
  cohort_root/<group folder>/<urls-subdir>/
    <downloaded files...>
    urls_manifest.csv or urls_manifest.json (implementation-defined)

How the rest of the pipeline uses this
  - extract_text.py will later extract text for these downloaded files and assign sha256.
  - build_pairs.py (with --consolidate-urls --urls-subdir <urls-subdir>) will read the URL manifest(s)
    and attempt to map each cited URL to a downloaded file path, then to sha256.

Notes
  - process_submissions.py does NOT accept cohort provenance flags. Provenance (cohort_id/cohort_note)
    is recorded by extract_text.py into sources.parquet.
  - If fetch fails (network, paywall), the row may remain URL-only and will not receive a sha256.
"""

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
        description=(
            "Full authoritative rebuild for one cohort. The --cohort-id argument is mandatory and is used for cross-run provenance. "
            "Optionally fetch URL-only sources via process_submissions.py (note: process_submissions.py does not accept cohort flags; "
            "cohort provenance is recorded by extract_text.py)."
        )
    )
    ap.add_argument("root", help="Root directory of cohort submissions")
    ap.add_argument("--wipe", action="store_true", help="Delete existing artifacts before running")
    ap.add_argument(
        "--artifacts-root",
        default=None,
        help="Root directory for all pipeline outputs. Defaults to artifacts/<basename of root submission folder>."
    )
    ap.add_argument("--grobid-url", default="http://localhost:8070")
    ap.add_argument("--cheap-only", action="store_true", help="Skip NLI in cherry-picking sniff")
    ap.add_argument("--fetch-urls", "--no-fetch-urls", action=argparse.BooleanOptionalAction, default=True,
                    help="Download URL-only sources into per-group urls/ folders and write URL manifests for build_pairs.py --consolidate-urls")
    ap.add_argument("--urls-subdir", default="urls", help="Where downloaded URL files/manifests are placed under each group folder")
    ap.add_argument("--cohort-id", required=True, help="Cohort identifier for cross-run provenance (required)")
    ap.add_argument("--cohort-note", default=None, help="Optional note describing the academic context of this cohort")
    args = ap.parse_args()
    if args.artifacts_root is None:
        root_name = Path(args.root).resolve().name
        args.artifacts_root = str(Path("artifacts") / root_name)

    # Ensure stdout is unbuffered (better progress visibility in long runs)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    artifacts = Path(args.artifacts_root)

    if args.wipe:
        print(f">>> Wiping {artifacts}/")
        shutil.rmtree(artifacts, ignore_errors=True)

    (artifacts / "text").mkdir(parents=True, exist_ok=True)
    (artifacts / "parquet").mkdir(parents=True, exist_ok=True)
    (artifacts / "verification").mkdir(parents=True, exist_ok=True)
    (artifacts / "cherrypicking").mkdir(parents=True, exist_ok=True)

    # Optional teacher-proof step: pre-fetch URL-only citations.
    # See PROCESS_SUBMISSIONS_SPEC above for the contract and expected artifacts.
    if args.fetch_urls:
        run([
            sys.executable, "process_submissions.py", args.root,
            "--fetch-urls",
            "--urls-subdir", args.urls_subdir,
        ])

    extract_cmd = [
        sys.executable, "extract_text.py", args.root,
        "--texts-dir", str(artifacts / "text"),
        "--parquet-dir", str(artifacts / "parquet"),
        "--extensions", ".pdf,.docx,.html,.htm,.txt,.md,.rst",
        "--grobid-url", args.grobid_url,
        "--cohort-id", args.cohort_id,
    ]
    if args.cohort_note is not None:
        extract_cmd.extend(["--cohort-note", args.cohort_note])
    run(extract_cmd)

    run([
        sys.executable, "build_pairs.py", args.root,
        "--texts-dir", str(artifacts / "text"),
        "--parquet-dir", str(artifacts / "parquet"),
        "--prefer-excel", "--fallback-csv", "--csv-encoding-sweep",
        "--consolidate-urls", "--urls-subdir", args.urls_subdir
    ])

    run([
        sys.executable, "verify_quotes.py",
        "--parquet-dir", str(artifacts / "parquet"),
        "--texts-dir", str(artifacts / "text"),
        "--out-dir", str(artifacts / "verification"),
        "--encoder", "all-MiniLM-L6-v2",
        "--bm25-topk", "20",
        "--cos-thresh", "0.82",
        "--fuzzy-thresh", "85",
        "--summary-csv"
    ])

    sniff_cmd = [
        sys.executable, "sniff_cherrypicking.py",
        "--parquet-dir", str(artifacts / "parquet"),
        "--texts-dir", str(artifacts / "text"),
        "--out-dir", str(artifacts / "cherrypicking"),
        "--query-field", "quote_from_report"
    ]
    if args.cheap_only:
        sniff_cmd.append("--cheap-only")

    run(sniff_cmd)

    print("\n=== FULL COHORT REBUILD COMPLETE ===")

if __name__ == "__main__":
    main()