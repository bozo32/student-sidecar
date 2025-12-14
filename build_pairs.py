from __future__ import annotations
import re

# --- SHA256 normalization helper ---
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
def normalize_sha256(v: object) -> Optional[str]:
    """Return a validated lowercase sha256 hex string, or None.

    Defensive: some upstream extractors may accidentally emit the literal string "None".
    """
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    low = s.lower()
    if low in {"none", "nan", "null"}:
        return None
    if _SHA256_RE.match(low):
        return low
    return None


#!/usr/bin/env python3
"""
build_pairs.py — normalize student tables into training/eval pairs and extract source texts.

Inputs (expected layout under ROOT):
  ROOT/
    Group X .../
      <one or more CSV/XLS/XLSX with three logical columns>
      <submitted source files: pdf/docx/html/txt/...>
      urls/_manifest.csv    # optional, created by process_submissions.py

Outputs (under ./artifacts by default):
  artifacts/parquet/pairs_raw.parquet   # one row per citation row
  artifacts/parquet/sources.parquet     # unique sources by sha256 + paths + text/tei paths
  artifacts/parquet/urls_manifest.parquet  # consolidated URL fetch logs (if any)
  artifacts/text/<sha>.txt              # extracted plain text for each unique source (if available)
  artifacts/text/<sha>.tei.xml          # TEI from GROBID when available

This script is robust to messy CSVs and odd Excel workbooks.
Heuristics:
  * If both CSV and XLSX exist, prefer the Excel.
  * Normalize headers to: quote_from_report, file_name, quote_from_source
  * If >3 columns, attempt to pick “last non-empty texty” column as quote_from_source.
  * Strip Excel hyperlinks; keep their visible text only.
  * Col 2 may contain a URL; pass-through to URL consolidator, but we still try to match
    to a local file in the group folder via fuzzy filename.

Flags:
  --urls-subdir: Specify the subdirectory for URL manifests (default: "urls")
  --prefer-excel: Prefer Excel files when present in a group (see code for details)
  --fallback-csv: When preferring Excel, also use CSVs if no Excel files found
  --csv-encoding-sweep: Try multiple encodings for CSV reading if needed
"""

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


# ----------------- Graph/figure reference tagging -----------------
GRAPH_TOKENS = [
    "figure", "fig.", "fig ", "graph", "diagram", "chart",
    "table", "image", "photo", "schematic", "map", "plot"
]

def _is_graph_like(quote_from_source: str, quote_from_report: str) -> tuple[bool, str]:
    """
    Heuristic to tag rows likely referring to non-textual elements (figures/graphs/tables)
    rather than quoting running text.

    Rules (any TRUE -> tag as graph_like):
      A) The source quote is very short (< 40 chars) AND the report quote is at least 3x longer.
      B) The source quote mentions indicative tokens (figure/fig./graph/chart/table/etc.).
      C) The source quote is empty/near-empty but the report quote is long (> 80 chars).

    Returns (is_graph_like, reason_string)
    """
    qs = (quote_from_source or "").strip()
    qr = (quote_from_report or "").strip()
    qs_len = len(qs)
    qr_len = len(qr)

    # A) length ratio
    if qs_len < 40 and qr_len > 0 and (qr_len / max(1, qs_len + 1)) >= 3:
        return True, "short_source_vs_long_report"

    # B) token hits (in the source quote)
    low = qs.lower()
    for tok in GRAPH_TOKENS:
        if tok in low:
            return True, f"token:{tok}"

    # C) empty-ish source vs long report
    if qs_len == 0 and qr_len >= 80:
        return True, "empty_source_long_report"

    return False, ""


# ----------------- Excel HYPERLINK stripper -----------------

HYPERLINK_RE = re.compile(r'^\s*=\s*HYPERLINK\s*\(\s*"([^"]*)"\s*,\s*"([^"]*)"\s*\)\s*$', re.I)

def _strip_excel_hyperlinks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace Excel =HYPERLINK("url","text") cells with just the visible text.
    Operates column-wise on object dtypes.
    """
    df = df.copy()
    for c in df.columns:
        s = df[c]
        if not pd.api.types.is_object_dtype(s):
            continue
        try:
            mask = s.astype(str).str.startswith(("=HYPERLINK(", "=Hyperlink(", "=hyperlink("), na=False)
            if mask.any():
                def _unhyper(x: str) -> str:
                    m = HYPERLINK_RE.match(str(x))
                    if m:
                        return m.group(2)
                    return str(x)
                df.loc[mask, c] = s[mask].astype(str).map(_unhyper)
        except Exception:
            continue
    return df

# Local module providing robust, layered extraction, OCR, and GROBID
import extract_text
from extract_text import extract_any

# Optional: only present if extract_text defines it
try:
    from extract_text import RateLimiter  # type: ignore
except Exception:  # pragma: no cover
    RateLimiter = None  # type: ignore

# ----------------- Defaults / Paths -----------------

ARTIFACTS_DIR = Path("artifacts")
PARQUET_DIR = ARTIFACTS_DIR / "parquet"
TEXT_DIR = ARTIFACTS_DIR / "text"

PARQUET_DIR.mkdir(parents=True, exist_ok=True)
TEXT_DIR.mkdir(parents=True, exist_ok=True)

CANON = ["quote_from_report", "file_name", "quote_from_source"]
PDF_LIKE_EXTS = {".pdf"}
DOC_LIKE_EXTS = {".docx", ".doc", ".rtf"}
HTML_LIKE_EXTS = {".html", ".htm"}
TEXT_LIKE_EXTS = {".txt", ".md"}
ACCEPTED_EXTS = PDF_LIKE_EXTS | DOC_LIKE_EXTS | HTML_LIKE_EXTS | TEXT_LIKE_EXTS

# Regex for URLs in cells (also used to detect URL-ish "file names")
URL_RE = re.compile(r"""(?i)\bhttps?://[^\s<>'"]+""")

# Group-14 style extra column / “last column is the actual target” fallback
TRAILING_TARGET_OK = True

# ----------------- GROBID controls helper -----------------
from typing import Optional
def _init_grobid_controls(grobid_rps: Optional[float], grobid_timeout: int, grobid_max_retries: int) -> None:
    """Initialize throttling/retry globals inside extract_text (used by extract_any).

    This keeps GROBID behavior consistent between extract_text.py and build_pairs.py.
    """
    # Not all versions of extract_text expose these symbols; set them if present.
    try:
        if hasattr(extract_text, "GROBID_MAX_RETRIES"):
            extract_text.GROBID_MAX_RETRIES = int(grobid_max_retries)
        if hasattr(extract_text, "GROBID_TIMEOUT"):
            extract_text.GROBID_TIMEOUT = int(grobid_timeout)
        if hasattr(extract_text, "GROBID_LIMITER"):
            if RateLimiter is not None and grobid_rps and grobid_rps > 0:
                extract_text.GROBID_LIMITER = RateLimiter(grobid_rps)
            else:
                extract_text.GROBID_LIMITER = None
    except Exception:
        # Never fail the pipeline just because throttling globals aren't available
        return

# ----------------- Utilities -----------------

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def normalize_header_names(cols: List[str]) -> List[str]:
    out = []
    for c in cols:
        if c is None:
            out.append(c)
            continue
        k = str(c).strip()
        k_l = k.lower().replace("’", "'").replace("“", '"').replace("”", '"')
        k_l = re.sub(r"\s+", " ", k_l)
        if k_l in {
            "quote from report",
            "text in report",
            "quote in our report",
            "assignment 1 citations",
            "section in report",
        }:
            out.append("quote_from_report" if k_l != "section in report" else k)
        elif k_l in {"file name", "file name ", "filename", "source", "source file", "file", "file_name"}:
            out.append("file_name")
        elif k_l in {"quote from source", "used text from source", "text from source", "quote from resource", "qoute from source", "in-tekst ref", "used text from source "}:
            out.append("quote_from_source")
        else:
            out.append(k)
    return out

def drop_empty_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop fully-empty columns first
    df = df.loc[:, ~(df.isna().all(axis=0))]
    # Drop “Unnamed: xxx” columns that are entirely NA or blanks
    keep_cols = []
    for c in df.columns:
        if isinstance(c, str) and c.lower().startswith("unnamed"):
            s = df[c]
            s_str = s.astype(str)
            if s_str.str.strip().replace("nan", "").replace("None", "").eq("").all():
                continue
        keep_cols.append(c)
    return df[keep_cols]

def collapse_to_three(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to return a DF with exactly three columns mapped to CANON.
    Heuristics for extra columns:
      - If all three canon names present → keep those order.
      - Else, try to infer; if >3, prefer last non-empty for quote_from_source.
    """
    df = drop_empty_unnamed(df)
    cols = list(df.columns)
    cols_norm = normalize_header_names(cols)
    df2 = df.copy()
    df2.columns = cols_norm
    # Drop duplicated columns defensively, keep first occurrence
    df2 = df2.loc[:, ~pd.Index(df2.columns).duplicated(keep="first")]

    # If too many columns and a leading column looks like a section label (short, low punctuation), drop it.
    if len(df2.columns) > 3:
        def looks_section(series: pd.Series) -> bool:
            s = series.astype(str).str.strip()
            avg_len = s.map(len).replace(0, pd.NA).dropna().astype(float).mean()
            punct_rate = s.str.contains(r"[\.!?]", regex=True).mean()
            # short average length and almost no sentence punctuation → section-ish
            return (avg_len is not None and avg_len < 25) and (punct_rate < 0.1)
        # If the very first column looks like a section label, drop it from consideration
        first_col = df2.columns[0]
        try:
            if looks_section(df2[first_col]):
                df2 = df2.drop(columns=[first_col])
        except Exception:
            pass

    if all(c in df2.columns for c in CANON):
        return df2[CANON]

    # Build a candidate list with simple scoring
    # Start with any exact matches we do have
    picks: Dict[str, str] = {}

    # Exact matches
    for t in CANON:
        if t in df2.columns:
            picks[t] = t

    # For missing targets, pick best fallback columns
    remaining = [c for c in df2.columns if c not in picks.values()]

    def looks_fileish(series: pd.Series) -> bool:
        # Many “file name” cells look like "paper.pdf" or have .pdf/.docx links
        s = series.astype(str).str.strip().fillna("")
        pdfish = s.str.contains(r"\.pdf(?:[?#].*)?$", case=False, regex=True)
        urly = s.str.contains(URL_RE)
        return bool((pdfish | urly).mean() > 0.2)

    def looks_quotey(series: pd.Series) -> bool:
        s = series.astype(str).str.strip().fillna("")
        longish = s.str.len().fillna(0) > 50
        punct = s.str.contains(r"[\.!?]")
        return bool(((longish & punct).mean()) > 0.2)

    # Infer file_name if needed
    if "file_name" not in picks:
        candidates = []
        for c in remaining:
            try:
                if c not in picks.values() and looks_fileish(df2[c]):
                    candidates.append(c)
            except Exception:
                pass
        if candidates:
            picks["file_name"] = candidates[0]

    # Infer quote_from_report if needed: pick a texty column
    if "quote_from_report" not in picks:
        candidates = []
        for c in remaining:
            if c in picks.values():
                continue
            try:
                if looks_quotey(df2[c]):
                    candidates.append(c)
            except Exception:
                pass
        if candidates:
            picks["quote_from_report"] = candidates[0]

    # Infer quote_from_source: if there are >3 columns, try last non-empty textual
    if "quote_from_source" not in picks:
        candidates = [c for c in remaining if c not in picks.values()]
        if TRAILING_TARGET_OK and len(candidates) >= 1:
            # Walk from the end, pick the last that looks “quotey”
            for c in reversed(candidates):
                try:
                    if c not in picks.values() and looks_quotey(df2[c]):
                        picks["quote_from_source"] = c
                        break
                except Exception:
                    pass
        # Fallback: first leftover
        if "quote_from_source" not in picks and candidates:
            picks["quote_from_source"] = candidates[-1] if TRAILING_TARGET_OK else candidates[0]

    # Final fallback: just take the first three visible columns
    final_cols = [picks.get(t) for t in CANON]
    if any(v is None for v in final_cols):
        # fallback to first three in df2
        cols3 = list(df2.columns)[:3]
        df3 = df2[cols3].copy()
        df3.columns = CANON[: len(cols3)] + [f"extra_{i}" for i in range(max(0, 3 - len(cols3)))]
        # if fewer than 3, append blanks
        while len(df3.columns) < 3:
            df3[f"extra_{len(df3.columns)}"] = ""
        df3.columns = CANON
        return df3
    else:
        ret = df2[[picks["quote_from_report"], picks["file_name"], picks["quote_from_source"]]].rename(
            columns={picks["quote_from_report"]: "quote_from_report",
                     picks["file_name"]: "file_name",
                     picks["quote_from_source"]: "quote_from_source"}
        )
        # Drop any duplicated column labels defensively
        ret = ret.loc[:, ~pd.Index(ret.columns).duplicated(keep="first")]
        return ret

def read_any_table(path: Path, csv_encoding_sweep: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Return {sheet_name: DataFrame with canon columns}. For CSV, use a single sheet "CSV".
    Robust to delimiter and bad lines.
    If csv_encoding_sweep is True, try multiple encodings for CSVs if needed.
    Also: sniff misnamed Excel workbooks saved with a .csv extension and parse as Excel.
    """
    import sys
    # --- quick binary sniff ---
    def _sniff_excel(p: Path) -> Optional[str]:
        try:
            with p.open("rb") as f:
                sig = f.read(8)
            # XLSX/ZIP: 50 4B 03 04  or 50 4B 05 06 (empty zip) or 50 4B 07 08
            if sig[:2] == b"PK":
                return "xlsx-like"
            # old XLS OLE2 magic: D0 CF 11 E0 A1 B1 1A E1
            if sig == b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1":
                return "xls-like"
        except Exception:
            pass
        return None

    def _parse_excel_any(p: Path) -> Dict[str, pd.DataFrame]:
        xls = pd.ExcelFile(p)  # let pandas pick an engine
        out = {}
        for sheet in xls.sheet_names:
            try:
                df_raw = pd.read_excel(p, sheet_name=sheet, dtype=str, header=0)
                df_raw = _strip_excel_hyperlinks(df_raw)
                df = collapse_to_three(df_raw)
                out[sheet] = df
            except Exception as e:
                print(f"[WARN] {p} sheet={sheet}: {e}", file=sys.stderr)
        if not out:
            raise RuntimeError(f"[READ-ERROR] {p}: no readable sheets")
        return out

    # If extension is xls/xlsx, handle as Excel straight away
    if path.suffix.lower() in {".xls", ".xlsx"}:
        return _parse_excel_any(path)

    # If it's a CSV extension, first sniff for misnamed Excel
    if path.suffix.lower() == ".csv":
        sniff = _sniff_excel(path)
        if sniff is not None:
            # Misnamed Excel saved with .csv; parse via Excel path
            try:
                return _parse_excel_any(path)
            except Exception as e:
                raise RuntimeError(f"[READ-ERROR] {path}: detected {sniff} but failed to parse as Excel: {e}")

        # Try standard CSV reads
        tried = []
        try:
            df = pd.read_csv(path, dtype=str, on_bad_lines="skip", engine="c")
            df = collapse_to_three(df)
            return {"CSV": df}
        except Exception as e:
            tried.append(("utf-8", str(e)))

        # broaden encoding list; include UTF-16 variants
        encodings = ["utf-8-sig", "cp1252", "latin1", "mac_roman", "utf-16", "utf-16le", "utf-16be"]
        if csv_encoding_sweep:
            for enc in encodings:
                try:
                    df = pd.read_csv(path, dtype=str, on_bad_lines="skip", engine="python", encoding=enc, sep=None)
                    df = collapse_to_three(df)
                    return {"CSV": df}
                except Exception as e:
                    tried.append((enc, str(e)))
        else:
            try:
                df = pd.read_csv(path, dtype=str, on_bad_lines="skip", engine="python", sep=None)
                df = collapse_to_three(df)
                return {"CSV": df}
            except Exception as e:
                tried.append(("utf-8-python", str(e)))

        # As a last resort: although extension is .csv, try Excel parsing anyway
        try:
            return _parse_excel_any(path)
        except Exception:
            pass

        raise RuntimeError(f"[READ-ERROR] {path}: could not read as CSV. Attempts: {tried}")

    # Any other extension: try Excel (covers odd cases users renamed)
    try:
        return _parse_excel_any(path)
    except Exception as e:
        raise RuntimeError(f"[READ-ERROR] {path}: {e}")

def iter_group_tables(
    group_dir: Path,
    prefer_excel: bool = False,
    fallback_csv: bool = False,
    csv_encoding_sweep: bool = False,
) -> Iterable[Tuple[Path, str, pd.DataFrame]]:
    """
    Yield (table_path, sheet_name, normalized_df) for each table in a group.
    If prefer_excel is True:
        - If Excel files exist, yield Excel files first.
        - If fallback_csv is False, do NOT yield CSVs at all when any Excel file exists.
        - If fallback_csv is True, yield CSVs only when there were no Excel files.
    If prefer_excel is False: yield all CSVs and Excels as before.
    Pass csv_encoding_sweep into read_any_table for CSVs.
    """
    csvs = sorted(group_dir.glob("*.csv"))
    xls = sorted(group_dir.glob("*.xls")) + sorted(group_dir.glob("*.xlsx"))
    if prefer_excel:
        if xls:
            # Yield Excel files first; skip CSVs if fallback_csv is False
            for p in xls:
                data = read_any_table(p)
                for sheet, df in data.items():
                    yield p, sheet, df
            if fallback_csv:
                if not xls:  # fallback_csv only if there were no Excel files
                    for p in csvs:
                        data = read_any_table(p, csv_encoding_sweep=csv_encoding_sweep)
                        for sheet, df in data.items():
                            yield p, sheet, df
            # If fallback_csv is False, do not yield CSVs at all when Excel files exist
        else:
            # No Excel files: yield CSVs if fallback_csv is True
            if fallback_csv or not xls:
                for p in csvs:
                    data = read_any_table(p, csv_encoding_sweep=csv_encoding_sweep)
                    for sheet, df in data.items():
                        yield p, sheet, df
    else:
        files = csvs + xls
        for p in files:
            if p.suffix.lower() == ".csv":
                data = read_any_table(p, csv_encoding_sweep=csv_encoding_sweep)
            else:
                data = read_any_table(p)
            for sheet, df in data.items():
                yield p, sheet, df

def fuzzy_pick_path(group_dir: Path, fn_raw: str) -> Optional[Path]:
    """
    Try to resolve a messy filename string to an existing file path under the group directory.
    Strategy:
      * If looks like a URL, return None (URLs are handled separately).
      * Case-insensitive exact name match against any file (including subdirs)
      * Loose token match ignoring spaces/underscores and punctuation
    """
    if not fn_raw:
        return None
    fn = str(fn_raw).strip()
    if URL_RE.search(fn):
        return None

    # Gather candidate files under group_dir
    candidates = list(group_dir.rglob("*"))
    file_candidates = [p for p in candidates if p.is_file() and p.suffix.lower() in ACCEPTED_EXTS]

    # direct exact, case-insensitive name match (last path component only)
    fn_tail = Path(fn).name
    for p in file_candidates:
        if p.name.lower() == fn_tail.lower():
            return p

    # normalize tokens (remove spaces, underscores, punctuation)
    def norm(s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^a-z0-9]+", "", s)
        return s

    norm_target = norm(fn_tail)
    # Try removing extension from both sides
    norm_target_noext = norm(Path(fn_tail).stem)
    # Enhanced token-overlap scoring (more robust to punctuation/order/noise)
    def token_set(s: str) -> set[str]:
        s = re.sub(r"[^a-z0-9]+", " ", s.lower())
        toks = [t for t in s.split() if t]
        return set(toks)

    norm_target_tokens = token_set(Path(fn_tail).stem)
    best = None
    best_score = 0.0
    for p in file_candidates:
        t_tokens = token_set(p.stem)
        if not t_tokens:
            continue
        overlap = len(norm_target_tokens & t_tokens)
        score = overlap / max(1, len(norm_target_tokens))
        # boost when the normalized target appears as a substring
        if norm_target and norm_target in re.sub(r"[^a-z0-9]+", "", p.name.lower()):
            score += 0.5
        if norm_target_noext and norm_target_noext in re.sub(r"[^a-z0-9]+", "", p.stem.lower()):
            score += 0.25
        if score > best_score:
            best = p
            best_score = score
    return best

# ----------------- URL manifest consolidation -----------------

def build_url_map(root: Path, urls_subdir: str = "urls", include_groups: Optional[list[str]] = None) -> Dict[Tuple[str, str], Path]:
    """
    Return a mapping {(group_id, url) -> local_saved_file_path} from per-group urls/_manifest.csv files.
    """
    mapping: Dict[Tuple[str, str], Path] = {}
    for group in sorted(p for p in root.iterdir() if p.is_dir()):
        gid = group.name
        if include_groups and not _group_selected(gid, include_groups):
            continue
        m = group / urls_subdir / "_manifest.csv"
        if not m.exists():
            continue
        try:
            df = pd.read_csv(m, dtype=str)
        except Exception:
            continue
        for _, r in df.iterrows():
            url = str(r.get("url", "") or "").strip()
            saved = str(r.get("saved_as", "") or "").strip()
            if not url or not saved:
                continue
            local = (group / saved).resolve()
            if local.exists():
                mapping[(gid, url)] = local
    return mapping

def consolidate_url_manifests(root: Path, urls_subdir: str = "urls") -> pd.DataFrame:
    rows = []
    for group in sorted(p for p in root.iterdir() if p.is_dir()):
        m = group / urls_subdir / "_manifest.csv"
        if m.exists():
            try:
                df = pd.read_csv(m, dtype=str)
                df["group_id"] = group.name
                rows.append(df)
            except Exception as e:
                print(f"[WARN] could not read URL manifest {m}: {e}", file=sys.stderr)
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(columns=["timestamp", "url", "saved_as", "group_id"])

# ----------------- Core processing -----------------

# Treat a cell/string as non-empty when it has at least one non‑whitespace character
_def_nonempty = lambda s: bool(str(s or "").strip())

@dataclass
class PairRow:
    cohort_id: str
    group_id: str
    row_id: str
    quote_from_report: str
    file_name_raw: str
    quote_from_source: str

    table_path: str
    sheet_name: str
    row_index: int

    source_uri: Optional[str] = None
    # Identity semantics
    source_ref_type: str = "unresolved"  # one of: sha256, url, unresolved
    source_ref: str = ""                 # normalized identity: sha256:<hex> | url:<url> | unresolved:<raw>

    resolved_source_path: Optional[str] = None
    source_sha256: Optional[str] = None
    source_text_path: Optional[str] = None
    tei_path: Optional[str] = None
    extract_tool: Optional[str] = None
    ocr_used: Optional[bool] = None
    extract_ok: bool = False
    extract_error: Optional[str] = None
    graph_like: Optional[bool] = None
    graph_reason: Optional[str] = None

# Helper function: robust cell fetch for possibly duplicated column labels
def _cell_str(row: pd.Series, key: str) -> str:
    """
    Robustly fetch a single cell as string even if the Series has duplicate column labels.
    If duplicate labels exist, take the first occurrence. Treat NaN/None as empty string.
    """
    try:
        if key not in row.index:
            return ""
        v = row[key]
        # If there are duplicate column labels, pandas may return a Series here.
        if isinstance(v, pd.Series):
            v = v.iloc[0] if not v.empty else ""
    except Exception:
        v = ""
    if v is None:
        return ""
    try:
        import math
        if isinstance(v, float) and math.isnan(v):
            return ""
    except Exception:
        pass
    return str(v)

def process_one_row(group_dir: Path,
                    texts_dir: Path,
                    cohort_id: str,
                    group_id: str,
                    table_path: Path,
                    sheet_name: str,
                    ridx: int,
                    row: pd.Series,
                    force_academic_pdf: bool,
                    grobid_url: Optional[str],
                    grobid_rps: Optional[float],
                    grobid_timeout: int,
                    grobid_max_retries: int,
                    url_map: Optional[Dict[Tuple[str, str], Path]] = None) -> PairRow:
    # Pull values safely
    q_report = _cell_str(row, "quote_from_report")
    file_raw  = _cell_str(row, "file_name")
    q_source  = _cell_str(row, "quote_from_source")

    # Identify likely figure/graph/table references
    glike, greason = _is_graph_like(q_source, q_report)

    # Row id stable key
    row_id = f"{group_id}:{table_path.name}:{sheet_name}:{ridx}"

    # Is file_name a URL?
    source_uri = None
    if URL_RE.search(file_raw):
        source_uri = file_raw.strip()

    source_ref_type = "unresolved"
    source_ref = f"unresolved:{file_raw.strip()}"

    # Resolve local path (if any), ignoring URLs
    resolved: Optional[Path] = None
    if not source_uri:
        resolved = fuzzy_pick_path(group_dir, file_raw)
    # If the file_name cell was a URL and we have a downloaded copy in the group's manifest, use it
    if source_uri and url_map is not None:
        mapped = url_map.get((group_id, source_uri))
        if mapped and mapped.exists():
            resolved = mapped
            # keep original URI for lineage; identity becomes sha256 once resolved
            
    # If we still have no source and no local file, scan all columns for URLs
    if not source_uri and not resolved:
        for k, v in row.items():
            try:
                s = str(v or "").strip()
            except Exception:
                continue
            if URL_RE.search(s):
                source_uri = s
                # If URL map provides a local copy, use it
                if url_map is not None:
                    mapped = url_map.get((group_id, source_uri))
                    if mapped and mapped.exists():
                        resolved = mapped
                break        
    # If we have a URL but no resolved local file yet, treat the URL as identity for now.
    if source_uri and not resolved:
        source_ref_type = "url"
        source_ref = f"url:{source_uri}"

    # Extract text/tei if we found a file
    sha = None
    text_path = None
    tei_path = None
    tool = None
    ocr_used = None
    extract_ok = False
    extract_err = None

    if resolved and resolved.exists():
        try:
            # Compute sha from source bytes up front to address caching
            byts = resolved.read_bytes()
            sha = sha256_bytes(byts)
            text_path = texts_dir / f"{sha}.txt"
            tei_path = texts_dir / f"{sha}.tei.xml"
            source_ref_type = "sha256"
            source_ref = f"sha256:{sha}"

            # If either artifact already exists, skip expensive extraction
            if text_path.exists() or tei_path.exists():
                tool = "cached"
                ocr_used = None
                extract_ok = True
            else:
                # Ensure GROBID retry/timeout/RPS are consistent with extract_text.py CLI
                _init_grobid_controls(grobid_rps=grobid_rps, grobid_timeout=grobid_timeout, grobid_max_retries=grobid_max_retries)

                # Call extract_any with only the kwargs it supports (backward compatible)
                import inspect
                sig = inspect.signature(extract_any)
                kwargs = {
                    "prefer_ocr": False,
                    "force_academic_pdf": force_academic_pdf,
                    "grobid_url": grobid_url,
                }
                # Some versions may accept extra parameters; add only if present
                if "grobid_timeout" in sig.parameters:
                    kwargs["grobid_timeout"] = grobid_timeout
                if "grobid_max_retries" in sig.parameters:
                    kwargs["grobid_max_retries"] = grobid_max_retries

                res = extract_any(resolved, **kwargs)
                # prefer sha from extractor if provided, but keep computed as fallback
                sha = res.get("sha256", sha)
                if sha:
                    source_ref_type = "sha256"
                    source_ref = f"sha256:{sha}"

                if res.get("text"):
                    if not text_path.exists():
                        text_path.write_text(res["text"], encoding="utf-8")
                if res.get("tei_xml"):
                    if not tei_path.exists():
                        tei_path.write_text(res["tei_xml"], encoding="utf-8")

                tool = res.get("extract_tool")
                ocr_used = bool(res.get("ocr_used", False))
                extract_ok = True

        except Exception as e:
            extract_err = str(e)[:500]

    # Final sanitation: ensure sha is normalized
    sha = normalize_sha256(sha)

    return PairRow(
        cohort_id=cohort_id,
        group_id=group_id,
        row_id=row_id,
        quote_from_report=q_report.strip(),
        file_name_raw=file_raw.strip(),
        quote_from_source=q_source.strip(),
        source_uri=source_uri,
        source_ref_type=source_ref_type,
        source_ref=source_ref,
        table_path=str(table_path),
        sheet_name=sheet_name,
        row_index=int(ridx),
        resolved_source_path=str(resolved) if resolved else None,
        source_sha256=sha,
        source_text_path=str(text_path) if text_path else None,
        tei_path=str(tei_path) if tei_path else None,
        extract_tool=tool,
        ocr_used=ocr_used,
        extract_ok=extract_ok,
        extract_error=extract_err,
        graph_like=bool(glike),
        graph_reason=greason,
    )

def _parse_include_groups(spec: str) -> list[str]:
    if not spec:
        return []
    parts = [p.strip() for p in str(spec).split(",")]
    return [p for p in parts if p]

def _group_selected(group_name: str, include: list[str]) -> bool:
    """Return True if group_name should be processed under include list.

    Matching rules:
      - Exact folder-name match (case-sensitive or case-insensitive): token equals group_name.
      - Numeric token like '5' matches case-insensitive substring 'group 5' in folder name.
      - Non-numeric token matches case-insensitive substring in folder name.
    """
    if not include:
        return True
    g_low = str(group_name).lower()
    for tok in include:
        t = str(tok).strip()
        if not t:
            continue
        if t == group_name or t.lower() == g_low:
            return True
        if t.isdigit():
            if f"group {int(t)}" in g_low:
                return True
        else:
            if t.lower() in g_low:
                return True
    return False

def build_all_pairs(
    root: Path,
    texts_dir: Path,
    force_academic_pdf: bool,
    grobid_url: Optional[str],
    grobid_rps: Optional[float],
    grobid_timeout: int,
    grobid_max_retries: int,
    prefer_excel: bool = False,
    fallback_csv: bool = False,
    csv_encoding_sweep: bool = False,
    url_map: Optional[Dict[Tuple[str, str], Path]] = None,
    workers: int = 6,
    include_groups: Optional[list[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return (pairs_df, sources_df, stats_df)
    stats_df provides one row per (group_id, table_path, sheet_name) with counts to help QA.
    """
    pairs: List[Dict] = []
    stats_rows: List[Dict] = []

    # Iterate groups
    for group in sorted(p for p in root.iterdir() if p.is_dir()):
        group_id = group.name
        cohort_id = "cohort-1"  # placeholder; extend if you have multiple cohorts
        if include_groups and not _group_selected(group_id, include_groups):
            continue

        # Tables
        for table_path, sheet, df in iter_group_tables(
            group,
            prefer_excel=prefer_excel,
            fallback_csv=fallback_csv,
            csv_encoding_sweep=csv_encoding_sweep,
        ):
            # Per-table counters
            total_rows = int(len(df))
            nonempty_any = 0
            nonempty_qsrc = 0
            nonempty_fname = 0
            url_count = 0
            resolved_count = 0
            extract_ok_count = 0
            extract_err_count = 0
            graphlike_count = 0

            # Iterate rows (threaded: I/O-bound for file reads + HTTP to GROBID)
            from concurrent.futures import ThreadPoolExecutor, as_completed

            futures = []
            with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
                for ridx, row in df.iterrows():
                    futures.append(
                        ex.submit(
                            process_one_row,
                            group,
                            texts_dir,
                            cohort_id,
                            group_id,
                            table_path,
                            sheet,
                            ridx,
                            row,
                            force_academic_pdf,
                            grobid_url,
                            grobid_rps,
                            grobid_timeout,
                            grobid_max_retries,
                            url_map,
                        )
                    )

                for fut in as_completed(futures):
                    pr = fut.result()
                    pairs.append(pr.__dict__)

                    # Update stats
                    any_nonempty = (
                        _def_nonempty(pr.quote_from_report)
                        or _def_nonempty(pr.file_name_raw)
                        or _def_nonempty(pr.quote_from_source)
                    )
                    if any_nonempty:
                        nonempty_any += 1
                    if _def_nonempty(pr.quote_from_source):
                        nonempty_qsrc += 1
                    if _def_nonempty(pr.file_name_raw):
                        nonempty_fname += 1
                    if _def_nonempty(pr.source_uri):
                        url_count += 1
                    if _def_nonempty(pr.resolved_source_path):
                        resolved_count += 1
                    if pr.extract_ok:
                        extract_ok_count += 1
                    if _def_nonempty(pr.extract_error):
                        extract_err_count += 1
                    if getattr(pr, "graph_like", False):
                        graphlike_count += 1

            stats_rows.append({
                "cohort_id": cohort_id,
                "group_id": group_id,
                "table_path": str(table_path),
                "sheet_name": sheet,
                "total_rows": total_rows,
                "nonempty_rows": nonempty_any,
                "nonempty_qsource": nonempty_qsrc,
                "nonempty_filename": nonempty_fname,
                "url_count": url_count,
                "resolved_count": resolved_count,
                "extract_ok_count": extract_ok_count,
                "extract_error_count": extract_err_count,
                # Simple quality ratios
                "resolved_ratio": (resolved_count / nonempty_fname) if nonempty_fname else 0.0,
                "extract_success_ratio": (extract_ok_count / max(1, nonempty_any)),
                "graph_like_count": graphlike_count,
                "graph_like_ratio": (graphlike_count / max(1, total_rows)),
            })

    pairs_df = pd.DataFrame(pairs)

    # Cleanup: ensure parquet never contains sentinel strings
    if not pairs_df.empty and "source_sha256" in pairs_df.columns:
        pairs_df["source_sha256"] = pairs_df["source_sha256"].map(normalize_sha256)

    # Build sources_df: unique by sha256 (non-null)
    src_rows: List[Dict] = []
    if not pairs_df.empty and "source_sha256" in pairs_df.columns:
        # Ensure we only build sources for valid sha256 identities
        uniq = pairs_df.dropna(subset=["source_sha256"]).drop_duplicates(subset=["source_sha256"])
        for _, r in uniq.iterrows():
            src_rows.append({
                "source_sha256": r["source_sha256"],
                "canonical_ext": Path(r["resolved_source_path"]).suffix.lower().lstrip(".") if r["resolved_source_path"] else None,
                "bytes": Path(r["resolved_source_path"]).stat().st_size if r["resolved_source_path"] and Path(r["resolved_source_path"]).exists() else None,
                "mime": None,  # could populate at extract time if desired
                "first_seen_path": r["resolved_source_path"],
                "all_paths": [r["resolved_source_path"]] if r["resolved_source_path"] else [],
                "text_path": r["source_text_path"],
                "tei_path": r["tei_path"],
                "extract_tool": r["extract_tool"],
                "ocr_used": r["ocr_used"],
                "extract_ok": r["extract_ok"],
                "extract_error": r["extract_error"],
            })
    sources_df = pd.DataFrame(src_rows)

    stats_df = pd.DataFrame(stats_rows)
    return pairs_df, sources_df, stats_df

# ----------------- CLI -----------------

def main():
    ap = argparse.ArgumentParser(description="Normalize student tables and extract source texts")
    ap.add_argument("root", help="Root directory containing group subfolders")
    ap.add_argument("--texts-dir", default=str(TEXT_DIR), help="Directory to write extracted text/tei files")
    ap.add_argument("--parquet-dir", default=str(PARQUET_DIR), help="Directory to write parquet tables")
    ap.add_argument("--force-academic-pdf", action="store_true", help="Prefer GROBID for PDFs and save TEI/XML")
    ap.add_argument("--grobid-url", default="http://localhost:8070", help="GROBID service URL (default http://localhost:8070)")
    ap.add_argument("--workers", type=int, default=6, help="Max concurrent workers for table row processing (default: 6)")
    ap.add_argument("--grobid-rps", type=float, default=0.0, help="Throttle GROBID requests to RPS (0 disables throttling)")
    ap.add_argument("--grobid-timeout", type=int, default=45, help="GROBID request timeout in seconds (default: 45)")
    ap.add_argument("--grobid-max-retries", type=int, default=8, help="Max retries for transient GROBID failures (default: 8)")
    ap.add_argument("--consolidate-urls", action="store_true", help="Also write urls_manifest.parquet from per-group manifests")
    ap.add_argument("--urls-subdir", default="urls", help="Subdirectory for URL manifests (default: urls)")
    ap.add_argument("--prefer-excel", action="store_true", help="Prefer Excel files when present in a group")
    ap.add_argument("--fallback-csv", action="store_true", help="When preferring Excel, also use CSVs if no Excel files found")
    ap.add_argument("--csv-encoding-sweep", action="store_true", help="Try multiple encodings for CSV reading if needed")
    ap.add_argument(
        "--include-groups",
        default="",
        help=(
            "Comma-separated group filters. Each token may be a full folder name (exact match) "
            "or a group number like '5' (matches folders containing 'Group 5' case-insensitively). "
            "If empty, all groups are processed."
        ),
    )

    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    include_groups = _parse_include_groups(args.include_groups)
    text_dir = Path(args.texts_dir).expanduser().resolve()
    parquet_dir = Path(args.parquet_dir).expanduser().resolve()
    text_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    # Build URL map for resolving downloaded URLs
    url_map = build_url_map(root, urls_subdir=args.urls_subdir, include_groups=include_groups)   
    pairs_df, sources_df, stats_df = build_all_pairs(
        root=root,
        texts_dir=text_dir,
        force_academic_pdf=args.force_academic_pdf,
        grobid_url=args.grobid_url,
        grobid_rps=args.grobid_rps,
        grobid_timeout=args.grobid_timeout,
        grobid_max_retries=args.grobid_max_retries,
        prefer_excel=args.prefer_excel,
        fallback_csv=args.fallback_csv,
        csv_encoding_sweep=args.csv_encoding_sweep,
        url_map=url_map,
        workers=args.workers,
        include_groups=include_groups,
    )

    # Persist
    if not pairs_df.empty:
        if "source_sha256" in pairs_df.columns:
            pairs_df["source_sha256"] = pairs_df["source_sha256"].map(normalize_sha256)
        pairs_out = parquet_dir / "pairs_raw.parquet"
        pairs_df.to_parquet(pairs_out, index=False)
        print(f"[OK] wrote {pairs_out}")
    else:
        print("[WARN] no pairs produced")

    sources_out = parquet_dir / "sources.parquet"

    # If prior sources exist, merge them so build_pairs doesn't regress/lose state.
    prev_sources = None
    if sources_out.exists():
        try:
            prev_sources = pd.read_parquet(sources_out)
            if "source_sha256" in prev_sources.columns:
                prev_sources["source_sha256"] = prev_sources["source_sha256"].map(normalize_sha256)
                prev_sources = prev_sources.dropna(subset=["source_sha256"])
        except Exception as e:
            print(f"[WARN] could not read existing {sources_out}: {e}", file=sys.stderr)

    merged_sources = sources_df
    if prev_sources is not None and not prev_sources.empty:
        if merged_sources is None or merged_sources.empty:
            merged_sources = prev_sources
        else:
            merged_sources = pd.concat([prev_sources, merged_sources], ignore_index=True)

        # Deduplicate by sha; merge all_paths lists conservatively
        if "source_sha256" in merged_sources.columns:
            def _merge_paths(series: pd.Series) -> list:
                out: list = []
                for v in series.dropna().tolist():
                    if isinstance(v, list):
                        out.extend([str(x) for x in v if x])
                    else:
                        out.append(str(v))
                # preserve order, unique
                seen = set()
                uniq = []
                for p in out:
                    if p not in seen:
                        seen.add(p)
                        uniq.append(p)
                return uniq

            # groupby merge
            gb = merged_sources.groupby("source_sha256", dropna=True, as_index=False)
            merged_sources = gb.agg({
                **{c: "first" for c in merged_sources.columns if c not in {"all_paths", "source_sha256"}},
                "all_paths": _merge_paths,
            })

    if merged_sources is not None and not merged_sources.empty:
        merged_sources["source_sha256"] = merged_sources["source_sha256"].map(normalize_sha256)
        merged_sources = merged_sources.dropna(subset=["source_sha256"])
        merged_sources.to_parquet(sources_out, index=False)
        print(f"[OK] wrote {sources_out}")
    else:
        # Actionable diagnostics
        if pairs_df is None or pairs_df.empty:
            print("[WARN] no sources produced (pairs_raw is empty)")
        else:
            resolved = int(pairs_df["resolved_source_path"].notna().sum()) if "resolved_source_path" in pairs_df.columns else 0
            with_sha = int(pairs_df["source_sha256"].notna().sum()) if "source_sha256" in pairs_df.columns else 0
            extract_ok = int(pairs_df["extract_ok"].fillna(False).sum()) if "extract_ok" in pairs_df.columns else 0
            print(
                "[WARN] no sources produced. Diagnostics: "
                f"pairs={len(pairs_df)} resolved_paths={resolved} with_sha={with_sha} extract_ok={extract_ok}. "
                "This usually means filenames could not be resolved to local files, or extraction never produced sha/text artifacts."
            )

    # --- SMOKE CHECK: pairs_raw source_sha256 all present in sources.parquet ---
    # Use normalize_sha256 to normalize, treat null/None/Nan as missing
    try:
        pairs_shas = set(
            pairs_df.get("source_sha256")
            .map(normalize_sha256)
            .dropna()
            .tolist()
        ) if "source_sha256" in pairs_df.columns else set()

        if sources_out.exists():
            sources_df_chk = pd.read_parquet(sources_out)
        else:
            sources_df_chk = pd.DataFrame()
        sources_shas = set(
            sources_df_chk.get("source_sha256")
            .map(normalize_sha256)
            .dropna()
            .tolist()
        ) if "source_sha256" in sources_df_chk.columns else set()

        missing = sorted(pairs_shas - sources_shas)
        if missing:
            print(f"[SMOKE FAIL] pairs_raw contains source_sha256 not found in sources.parquet: {missing[:25]}")
            sys.exit(1)
        else:
            print("[OK] pairs_raw source_sha256 values all present in sources.parquet")
    except Exception as e:
        print(f"[WARN] smoke check failed with error: {e}")

    # Per-table QA report
    if not stats_df.empty:
        report_parquet = parquet_dir / "tables_report.parquet"
        report_csv = parquet_dir / "tables_report.csv"
        stats_df.to_parquet(report_parquet, index=False)
        stats_df.to_csv(report_csv, index=False)
        print(f"[OK] wrote {report_parquet}")
        print(f"[OK] wrote {report_csv}")

        # Print top offenders where extract_success_ratio << 1 for non-empty tables
        offenders = stats_df[stats_df["nonempty_rows"] > 0].copy()
        offenders = offenders.sort_values(["extract_success_ratio", "resolved_ratio"])[:10]
        if not offenders.empty:
            print("\n[SUMMARY] Lowest extract success ratios (top 10):")
            for _, r in offenders.iterrows():
                print(
                    f"  - {r['group_id']} | {Path(r['table_path']).name} | {r['sheet_name']}  "
                    f"nonempty={int(r['nonempty_rows'])}  extract_ok={int(r['extract_ok_count'])}  "
                    f"success={r['extract_success_ratio']:.2%}  resolved={r['resolved_ratio']:.2%}"
                )

    # Consolidated URL manifest (optional)
    if args.consolidate_urls:
        urls_df = consolidate_url_manifests(root, urls_subdir=args.urls_subdir)
        if include_groups:
            # Filter to included groups only
            urls_df = urls_df[urls_df["group_id"].apply(lambda g: _group_selected(g, include_groups))]
        if not urls_df.empty:
            urls_out = parquet_dir / "urls_manifest.parquet"
            urls_df.to_parquet(urls_out, index=False)
            print(f"[OK] wrote {urls_out}")
        else:
            print("[INFO] no URL manifests to consolidate")
    else:
        if stats_df.empty:
            print("[WARN] no stats generated")

if __name__ == "__main__":
    main()