#!/usr/bin/env python3
"""
process_submissions.py — walk subdirectories and capture a 4-row head from each CSV/XLS(X).
Optionally discover URLs embedded in tables, fetch them, and save into each group's folder.

Usage:
  python process_submissions.py /path/to/root [--fetch-urls] [--urls-subdir urls] [--url-timeout 20]

Outputs:
  - Console summary (file path, type, sheet, inferred delimiter, columns)
  - Per-file/per-sheet previews saved under ./previews/ as CSV
  - (optional) For each table, fetched URLs saved under <group>/<urls-subdir>/ with a per-group _manifest.csv
"""

import sys
import os
import io
import csv
from pathlib import Path
from typing import Optional

import pandas as pd

import argparse
import re
import hashlib
import requests
from urllib.parse import urlparse
from typing import List, Dict

# ---------- helpers ----------

def sniff_delimiter(p: Path, max_bytes: int = 8192) -> Optional[str]:
    """Try to infer CSV delimiter using csv.Sniffer on the first `max_bytes`."""
    try:
        with p.open("rb") as f:
            sample = f.read(max_bytes)
        # try common encodings quickly
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                text = sample.decode(enc, errors="strict")
                dialect = csv.Sniffer().sniff(text, delimiters=[",", ";", "\t", "|", ":"])
                return dialect.delimiter
            except Exception:
                continue
    except Exception:
        pass
    return None

def read_csv_head(p: Path, nrows: int = 4) -> pd.DataFrame:
    """Read first `nrows` of a CSV, robust to unknown delimiter/encoding."""
    # Let pandas try to infer delimiter; fallbacks on encoding
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(
                p,
                nrows=nrows,
                sep=None,              # infer
                engine="python",       # needed for sep=None
                encoding=enc,
                on_bad_lines="skip",
            )
        except Exception:
            continue
    # Last resort: assume comma + permissive encoding
    return pd.read_csv(
        p,
        nrows=nrows,
        sep=",",
        engine="python",
        encoding="latin-1",
        on_bad_lines="skip",
    )

def sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)

def _drop_empty_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that (a) are named like 'Unnamed: N' and (b) are entirely empty/NA/whitespace.
    This handles messy Excel sheets with thousands of blank columns created by merged cells.
    """
    keep_cols = []
    for c in df.columns:
        s = df[c]
        # treat non-string gracefully
        s_str = s.astype(str).str.strip()
        all_empty = s.isna().all() or (s_str.eq("").all())
        if isinstance(c, str) and c.startswith("Unnamed:") and all_empty:
            continue
        keep_cols.append(c)
    return df.loc[:, keep_cols]

def _print_columns_capped(df: pd.DataFrame, prefix: str = "columns=") -> str:
    cols = list(df.columns)
    cap = 20
    if len(cols) <= cap:
        return f"{prefix}{cols}"
    head = cols[:cap]
    return f"{prefix}{head} ... (+{len(cols)-cap} more)"

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize messy student headers into a canonical trio where possible:
    quote_from_report, file_name, quote_from_source.
    Non-matching headers are left as-is.
    """
    def norm(s: str) -> str:
        s = (s or "").replace("\ufeff", "")  # strip BOM
        s = re.sub(r"\s+", " ", s).strip().lower()
        return s

    mapping = {
        "quote from report": "quote_from_report",
        "quote in our report": "quote_from_report",
        "text in report": "quote_from_report",
        "section in report": "quote_from_report",
        "text from report": "quote_from_report",
        "assignment 1 citations": "quote_from_report",

        "quote from source": "quote_from_source",
        "used text from source": "quote_from_source",
        "in-tekst ref": "quote_from_source",
        "qoute from source": "quote_from_source",
        "quote from resource": "quote_from_source",
        "text in source": "quote_from_source",
        "used text from source ": "quote_from_source",
        "quote from source ": "quote_from_source",

        "file name": "file_name",
        "file name ": "file_name",
        "file": "file_name",
        "source": "file_name",
        "file name  ": "file_name",
        "file name\u00a0": "file_name",  # handles non-breaking space
    }

    new_cols = []
    for c in df.columns:
        c_norm = norm(str(c))
        new_cols.append(mapping.get(c_norm, c))
    df.columns = new_cols
    return df

# ---------- URL helpers ----------

URL_RE = re.compile(r"https?://\S+", re.I)

def extract_urls_from_table(p: Path, max_rows: int = 2000) -> List[Dict]:
    """Scan a CSV/XLS/XLSM for URLs; return list of {'url','cell','sheet'}."""
    out: List[Dict] = []
    try:
        if p.suffix.lower() == ".csv":
            df = pd.read_csv(p, dtype=str, keep_default_na=False, encoding_errors="ignore", nrows=max_rows)
            sheet = None
        elif p.suffix.lower() == ".tsv":
            df = pd.read_csv(p, dtype=str, keep_default_na=False, encoding_errors="ignore", sep="\t", nrows=max_rows)
            sheet = None
        else:
            xl = pd.ExcelFile(p)
            if not xl.sheet_names:
                return out
            sheet = xl.sheet_names[0]
            df = pd.read_excel(
                p,
                sheet_name=sheet,
                dtype=str,
                nrows=max_rows,
                engine="openpyxl" if p.suffix.lower() == ".xlsx" else None
            )
        for r_idx, row in df.iterrows():
            for c_name, val in row.items():
                s = str(val).strip()
                if not s:
                    continue
                m = URL_RE.search(s)
                if m:
                    out.append({
                        "url": m.group(0),
                        "cell": f"row={r_idx}, col={c_name}",
                        "sheet": sheet
                    })
    except Exception:
        pass
    return out

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".", "+") else "_" for ch in s)

def guess_ext_from_ct(ct: str | None) -> str:
    if not ct:
        return ""
    ct = ct.split(";")[0].strip().lower()
    return {
        "text/html": ".html",
        "application/pdf": ".pdf",
        "text/plain": ".txt",
        "application/msword": ".doc",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    }.get(ct, "")

def fetch_url(url: str, timeout: int = 20) -> tuple[bool, Dict]:
    """Fetch URL; return (ok, meta) where meta includes status, content_type, bytes, sha256, filename, content (bytes) or error."""
    meta: Dict = {"status": None, "content_type": None, "error": None, "bytes": 0, "sha256": None, "filename": None}
    try:
        headers = {"User-Agent": "student-sidecar/1.0 (+https://example.local)"}
        r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        meta["status"] = r.status_code
        if r.status_code != 200 or not r.content:
            meta["error"] = f"HTTP {r.status_code}"
            return False, meta
        meta["content_type"] = r.headers.get("Content-Type", "")
        meta["bytes"] = len(r.content)
        meta["sha256"] = sha256_bytes(r.content)
        parsed = urlparse(url)
        base = safe_name(parsed.path.split("/")[-1]) or "download"
        ext = Path(base).suffix
        if not ext:
            ext = guess_ext_from_ct(meta["content_type"]) or ""
        meta["filename"] = (base if ext else base) + ext
        meta["content"] = r.content
        return True, meta
    except Exception as e:
        meta["error"] = str(e)
        return False, meta

def save_url_payload(group_dir: Path, urls_subdir: str, url: str, meta: Dict) -> Dict:
    """Save fetched bytes under group/<urls_subdir>/<sha256>.<ext>; also derive HTML text companion if possible."""
    out_dir = group_dir / urls_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(meta.get("filename") or "").suffix
    sha = meta["sha256"]
    bin_path = out_dir / f"{sha}{ext}"
    if not bin_path.exists():
        bin_path.write_bytes(meta["content"])
    text_path = None
    ct = (meta.get("content_type") or "").split(";")[0].lower()
    if ct == "text/html":
        try:
            import trafilatura
            txt = trafilatura.extract(
                meta["content"].decode("utf-8", errors="ignore"),
                include_comments=False,
                include_tables=False
            ) or ""
            if txt.strip():
                text_path = out_dir / f"{sha}.txt"
                text_path.write_text(txt, encoding="utf-8")
        except Exception:
            pass
    return {"saved_path": str(bin_path), "text_path": str(text_path) if text_path else None}

import csv as _csv
def append_manifest_row(manifest_path: Path, row: Dict):
    header = ["url","status","content_type","bytes","sha256","saved_path","text_path","source_table","source_cell","source_sheet","error"]
    exists = manifest_path.exists()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k) for k in header})

def main(root: Path, fetch_urls: bool = False, urls_subdir: str = "urls", url_timeout: int = 20):
    root = root.expanduser().resolve()
    if not root.exists():
        print(f"[ERROR] root path does not exist: {root}")
        return
    if not root.is_dir():
        print(f"[ERROR] root path is not a directory: {root}")
        return

    previews_root = Path("previews")
    previews_root.mkdir(parents=True, exist_ok=True)

    exts_csv = {".csv", ".tsv"}
    exts_xls = {".xlsx", ".xls", ".xlsm"}

    files_seen = 0

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            p = Path(dirpath) / fname
            ext = p.suffix.lower()

            if ext in exts_csv:
                delim = sniff_delimiter(p) if ext == ".csv" else "\\t"
                try:
                    df = read_csv_head(p, nrows=4)
                    df = normalize_headers(df)
                    df = _drop_empty_unnamed(df)
                except Exception as e:
                    print(f"[CSV-ERROR] {p}: {e}")
                    continue

                print(
                    f"[CSV] {p}\n"
                    f"      delimiter={repr(delim)}  {_print_columns_capped(df)}  rows_shown={len(df)}"
                )

                try:
                    rel = p.relative_to(root)
                except ValueError:
                    rel = p
                out_name = f"{sanitize(str(rel))}__head.csv"
                out_path = previews_root / out_name
                try:
                    df.to_csv(out_path, index=False)
                except Exception as e:
                    print(f"   ↳ [write-failed] {out_path}: {e}")

            elif ext in exts_xls:
                try:
                    xls = pd.ExcelFile(p)
                    sheets = xls.sheet_names
                except Exception as e:
                    print(f"[XLS-ERROR] {p}: {e}")
                    continue

                print(f"[XLS/XLSX] {p}  sheets={sheets}")
                if not sheets:
                    print("   ↳ [sheet-error] no sheets found")
                    continue

                sheet = sheets[0]
                try:
                    df = pd.read_excel(
                        p,
                        sheet_name=sheet,
                        nrows=4,
                        engine="openpyxl" if ext == ".xlsx" else None,
                    )
                    df = normalize_headers(df)
                    df = _drop_empty_unnamed(df)
                except Exception as e:
                    print(f"   ↳ [sheet-error] '{sheet}': {e}")
                    continue

                print(f"   ↳ sheet='{sheet}'  {_print_columns_capped(df)}  rows_shown={len(df)}")

                try:
                    rel = p.relative_to(root)
                except ValueError:
                    rel = p
                out_name = f"{sanitize(str(rel))}__{sanitize(sheet)}__head.csv"
                out_path = previews_root / out_name
                try:
                    df.to_csv(out_path, index=False)
                except Exception as e:
                    print(f"      [write-failed] {out_path}: {e}")

            # --- URL discovery & optional fetch ---
            if fetch_urls and (ext in exts_csv or ext in exts_xls):
                try:
                    hits = extract_urls_from_table(p, max_rows=2000)
                except Exception:
                    hits = []
                if hits:
                    # Save under the *group directory* = the parent folder of the table file.
                    # (This is robust even if there is no cohort subfolder or if filenames are in the first level.)
                    group_dir = p.parent
                    manifest = group_dir / urls_subdir / "_manifest.csv"
                    for hit in hits:
                        ok, meta = fetch_url(hit["url"], timeout=url_timeout)
                        saved = {"saved_path": None, "text_path": None}
                        if ok:
                            saved = save_url_payload(group_dir, urls_subdir, hit["url"], meta)
                        append_manifest_row(manifest, {
                            "url": hit["url"],
                            "status": meta.get("status"),
                            "content_type": meta.get("content_type"),
                            "bytes": meta.get("bytes"),
                            "sha256": meta.get("sha256"),
                            "saved_path": saved.get("saved_path"),
                            "text_path": saved.get("text_path"),
                            "source_table": str(p),
                            "source_cell": hit.get("cell"),
                            "source_sheet": hit.get("sheet"),
                            "error": meta.get("error"),
                        })

            # ignore everything else
            else:
                continue

            files_seen += 1

    if files_seen == 0:
        print(f"[INFO] No CSV/XLS workbooks found under {root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preview tables and optionally fetch URLs embedded in CSV/XLS files.")
    parser.add_argument("root", help="Root directory with cohorts/groups")
    parser.add_argument("--fetch-urls", action="store_true", help="Fetch and save URLs found in tables")
    parser.add_argument("--urls-subdir", default="urls", help="Relative folder under each group to save fetched URLs")
    parser.add_argument("--url-timeout", type=int, default=20, help="HTTP timeout per URL in seconds")
    args = parser.parse_args()
    main(Path(args.root).resolve(), fetch_urls=args.fetch_urls, urls_subdir=args.urls_subdir, url_timeout=args.url_timeout)
