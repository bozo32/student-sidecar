#!/usr/bin/env python3
"""
Batch 2 smoke tests

Purpose:
- Exercise *new or higher-risk behaviors* introduced after the initial pipeline stabilized.
- Validate identity semantics, cohort-aware provenance, and diagnostic richness.
- Intended for debugging, regression checking, and bug hunting by developers or advanced users.

When to run:
- AFTER batch-1-smoke.py has passed.
- AFTER a full or partial rebuild using extract_text.py, build_pairs.py, and verify_quotes.py.
- Especially useful when modifying extraction logic, source identity handling, or cohort/group semantics.

What this checks (non-exhaustive):
1. Required artifacts exist (pairs_raw, sources, tables_report; urls_manifest optional).
2. tables_report.parquet contains meaningful diagnostics (not empty counters).
3. Source identity invariants:
   - source_sha256 is the sole join key between pairs_raw and sources.
   - source_sha256 values are unique in sources.parquet.
   - sha-backed rows have consistent source_ref_type and source_ref values.
4. Resolved-path semantics:
   - Any resolved source path implies a valid source_sha256.
5. Artifact presence:
   - Spot-checks that extracted text/TEI artifacts exist for sha-backed sources.
6. Group normalization:
   - Ensures group_id normalization is stable and non-expansive.
7. Verification alignment:
   - verification_summary.csv exists, has expected schema,
     and covers all groups present in tables_report.
8. Sanity bounds:
   - success_ratio values lie within [0,1].

What this does NOT do:
- It does not validate semantic correctness of matches.
- It does not enforce pedagogical thresholds.
- It does not require perfect extraction coverage.

Failure philosophy:
- FAIL on broken invariants or missing required structure.
- WARN on partial coverage, optional artifacts, or broader-than-expected runs.

If this script fails, the pipeline outputs should be treated as suspect
until the reported issue is understood and resolved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent
PARQUET_DIR = ROOT / "artifacts" / "parquet"
VERIFY_DIR = ROOT / "artifacts" / "verification"


def fail(msg):
    print(f"[SMOKE FAIL] {msg}")
    sys.exit(1)


def ok(msg):
    print(f"[OK] {msg}")


def warn(msg):
    print(f"[WARN] {msg}")


def main():
    print("=== Batch 2 smoke test ===")

    # --- 1. Artifacts exist ---
    pairs_path = PARQUET_DIR / "pairs_raw.parquet"
    sources_path = PARQUET_DIR / "sources.parquet"
    tables_path = PARQUET_DIR / "tables_report.parquet"

    for p in (pairs_path, sources_path, tables_path):
        if not p.exists():
            fail(f"Missing artifact: {p.name}")
        ok(f"Found {p.name}")

    # Load pairs_raw for identity semantics checks
    try:
        pairs = pd.read_parquet(pairs_path)
    except Exception as e:
        fail(f"Could not read pairs_raw.parquet: {e}")

    # Load sources for identity semantics checks
    try:
        sources = pd.read_parquet(sources_path)
    except Exception as e:
        fail(f"Could not read sources.parquet: {e}")

    # urls_manifest is optional (only present if consolidate-urls was used)
    urls_manifest_path = PARQUET_DIR / "urls_manifest.parquet"
    if urls_manifest_path.exists():
        ok("Found urls_manifest.parquet")
    else:
        warn("urls_manifest.parquet not found (OK if consolidate-urls not used)")

    # --- 2. Load tables_report and check diagnostics richness ---
    tables = pd.read_parquet(tables_path)

    required_cols = {
        "group_id",
        "nonempty_rows",
        "resolved_count",
        "extract_ok_count",
        "extract_success_ratio",
        "graph_like_count",
    }
    missing = required_cols - set(tables.columns)
    if missing:
        fail(f"tables_report.parquet missing columns: {missing}")
    ok("tables_report.parquet schema OK")

    # Ensure we actually recorded something diagnostic
    if tables["nonempty_rows"].sum() == 0:
        fail("tables_report has zero nonempty_rows across all tables")

    if tables["extract_ok_count"].sum() == 0:
        fail("tables_report shows zero successful extractions")

    ok("tables_report contains non-trivial diagnostics")

    # --- 2b. Source identity semantics (Batch 2) ---
    # Contract: source_sha256 is the only identity join key; paths are metadata.
    # We enforce this by requiring explicit source_ref fields.
    identity_required = {"source_sha256", "source_ref_type", "source_ref"}
    missing = identity_required - set(pairs.columns)
    if missing:
        fail(f"pairs_raw.parquet missing identity columns: {missing}")

    # sources.parquet must have source_sha256 as the primary identity key
    if "source_sha256" not in sources.columns:
        fail("sources.parquet missing required column: source_sha256")

    def norm_sha(v):
        """Normalize sha-ish values; return None for null/sentinels."""
        if v is None:
            return None
        # Keep pandas NA/NaN as null
        try:
            if pd.isna(v):
                return None
        except Exception:
            pass
        s = str(v).strip().lower()
        if s in {"", "none", "nan", "null"}:
            return None
        return s

    # Enforce uniqueness of source_sha256 in sources.parquet (after normalizing & dropping null/sentinels)
    ssha = sources["source_sha256"].map(norm_sha).dropna()
    if ssha.duplicated().any():
        dup = ssha[ssha.duplicated()].iloc[0]
        fail(f"sources.parquet has duplicate source_sha256 value: {dup!r}")
    ok("sources.parquet source_sha256 unique")

    # If pairs has a resolved_source_path (or resolved_source_*), it must also have a sha
    resolved_col = None
    for c in ("resolved_source_path", "resolved_path", "resolved_source_file"):
        if c in pairs.columns:
            resolved_col = c
            break

    if resolved_col is not None:
        resolved_mask = pairs[resolved_col].notna() & (pairs[resolved_col].astype(str).str.len() > 0)
        if resolved_mask.any():
            missing_sha = pairs.loc[resolved_mask, "source_sha256"].isna() | (
                pairs.loc[resolved_mask, "source_sha256"].astype(str).str.len() == 0
            )
            if missing_sha.any():
                idx = pairs.loc[resolved_mask].index[missing_sha][0]
                fail(
                    f"Row {idx} has {resolved_col} set but missing source_sha256"
                )
        ok(f"sha present when {resolved_col} is present")
    else:
        warn("No resolved_source_path-like column found in pairs_raw; skipping sha-vs-resolved checks")

    # For sha-backed rows, ensure they exist in sources.parquet
    pairs_sha = pairs["source_sha256"].map(norm_sha).dropna()
    if len(pairs_sha) > 0:
        missing_in_sources = set(pairs_sha.unique()) - set(ssha.unique())
        if missing_in_sources:
            sample = sorted(list(missing_in_sources))[:10]
            fail(f"pairs_raw contains source_sha256 not found in sources.parquet: {sample}")
        ok("pairs_raw sha values are covered by sources.parquet")
    else:
        warn("pairs_raw has no non-empty source_sha256 values")

    allowed_ref_types = {"sha256", "url", "unresolved"}
    bad_types = set(pairs["source_ref_type"].dropna().astype(str).str.lower()) - allowed_ref_types
    if bad_types:
        fail(f"pairs_raw.parquet has unexpected source_ref_type values: {bad_types}")

    # For sha-backed rows, source_ref must be sha256:<hex> and ref_type must be sha256
    sha_mask = pairs["source_sha256"].map(norm_sha).notna()
    if sha_mask.any():
        sha_ref_type_bad = pairs.loc[sha_mask, "source_ref_type"].astype(str).str.lower().ne("sha256")
        if sha_ref_type_bad.any():
            n = int(sha_ref_type_bad.sum())
            fail(f"{n} rows have source_sha256 but source_ref_type != 'sha256'")

        expected_refs = "sha256:" + pairs.loc[sha_mask, "source_sha256"].map(norm_sha)
        actual_refs = pairs.loc[sha_mask, "source_ref"].astype(str)
        ref_mismatch = actual_refs.ne(expected_refs)
        if ref_mismatch.any():
            # pick the first failing row by position within the sha-backed subset
            bad_pos = ref_mismatch.to_numpy().nonzero()[0][0]
            bad_idx = actual_refs.index[bad_pos]
            fail(
                "source_ref mismatch for sha-backed row at index "
                f"{bad_idx}: expected={expected_refs.loc[bad_idx]!r} got={actual_refs.loc[bad_idx]!r}"
            )

        ok("pairs_raw source identity fields consistent for sha-backed rows")
    else:
        warn("No sha-backed rows found in pairs_raw (unexpected for normal runs)")

    # Artifact existence check by sha (sample a few rows)
    texts_dir = ROOT / "artifacts" / "text"
    if not texts_dir.exists():
        warn("artifacts/text not found; skipping sha artifact existence checks")
    else:
        sample_shas = pairs.loc[sha_mask, "source_sha256"].astype(str).dropna().unique()[:5]
        if len(sample_shas) == 0:
            warn("No sha values available for artifact checks")
        else:
            missing_artifacts = []
            for sha in sample_shas:
                txt = texts_dir / f"{sha}.txt"
                tei = texts_dir / f"{sha}.tei.xml"
                if not (txt.exists() or tei.exists()):
                    missing_artifacts.append(sha)
            if missing_artifacts:
                warn(f"Missing text/tei artifacts for sha(s): {missing_artifacts} (allowed in Batch 2; indicates extract_text not yet rerun)")
            else:
                ok("sha artifacts present for sampled sources")

    # --- 3. Group-id normalization sanity ---
    normalized = tables["group_id"].str.lower().str.strip()
    if normalized.nunique() > tables["group_id"].nunique():
        fail("group_id normalization appears inconsistent")
    ok("group_id normalization stable")

    # --- 4. Verification summary presence & schema ---
    summary_path = VERIFY_DIR / "verification_summary.csv"
    if not summary_path.exists():
        fail("verification_summary.csv missing")

    summary = pd.read_csv(summary_path)

    summary_required = {"group", "success_ratio", "total", "hits", "misses"}
    missing = summary_required - set(summary.columns)
    if missing:
        fail(f"verification_summary.csv missing columns: {missing}")
    ok("verification_summary.csv schema OK")

    # --- 5. Cross-check: group overlap between verification and tables_report ---
    # The verification_summary.csv may come from a broader run than the current parquet set.
    # We require that every group in tables_report is present in verification_summary,
    # but we only WARN if verification_summary contains additional groups.
    tables_groups = set(tables["group_id"].astype(str).str.lower().str.strip())
    summary_groups = set(summary["group"].astype(str).str.lower().str.strip())

    missing_in_summary = tables_groups - summary_groups
    if missing_in_summary:
        fail(f"verification_summary missing groups present in tables_report: {missing_in_summary}")

    extra_in_summary = summary_groups - tables_groups
    if extra_in_summary:
        # Don't spam: show count and a small sample
        sample = sorted(list(extra_in_summary))[:10]
        warn(
            "verification_summary contains groups not present in tables_report "
            f"(likely from a broader run): count={len(extra_in_summary)} sample={sample}"
        )

    ok("verification_summary covers tables_report groups")

    # --- 6. Sanity bounds on success ratios ---
    if not ((summary["success_ratio"] >= 0.0) & (summary["success_ratio"] <= 1.0)).all():
        fail("success_ratio outside [0,1] bounds")

    ok("success_ratio values within bounds")

    print("=== Batch 2 smoke test PASSED ===")


if __name__ == "__main__":
    main()
