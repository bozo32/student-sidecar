

#!/usr/bin/env python3
"""
extract_text.py — layered document extraction used by build_pairs.py

Goal:
  Provide a single entry point `extract_any(path, prefer_ocr=False, force_academic_pdf=False, grobid_url="http://localhost:8070")`
  that:
    • computes sha256 of the file bytes
    • extracts plain text from PDFs/Word/HTML/TXT
    • optionally calls a local GROBID service to produce TEI for academic PDFs
    • gracefully falls back across multiple extractors
    • optionally OCRs scanned PDFs (via `ocrmypdf --sidecar`)
Returns:
  {
    "sha256": str,
    "text": str or None,
    "tei_xml": str or None,
    "extract_tool": str,   # which extractor produced the text
    "ocr_used": bool,
  }

Notes:
  • We avoid rewriting PDFs with Ghostscript: OCR uses `--sidecar` to capture text only.
  • If `force_academic_pdf` is True and file is .pdf, we try GROBID first; else we try fast text extractors first and GROBID later.
"""

from __future__ import annotations

import io
import os
import re
import sys
import json
import time
import shutil
import hashlib
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Dict, Tuple

import requests

# Optional import dependencies. Each is fully optional and failure is handled.
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except Exception:
    pdfminer_extract_text = None

try:
    from docx import Document as DocxDocument  # python-docx
except Exception:
    DocxDocument = None

try:
    from bs4 import BeautifulSoup  # bs4
except Exception:
    BeautifulSoup = None


log = logging.getLogger(__name__)


# ---------------------- utilities ----------------------

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def read_bytes(path: Path) -> bytes:
    with path.open("rb") as f:
        return f.read()


def looks_like_pdf(path: Path) -> bool:
    return path.suffix.lower() == ".pdf"


def looks_like_docx(path: Path) -> bool:
    # python-docx supports .docx; legacy .doc is not supported here
    return path.suffix.lower() == ".docx"


def looks_like_html(path: Path) -> bool:
    return path.suffix.lower() in {".html", ".htm"}


def looks_like_text(path: Path) -> bool:
    return path.suffix.lower() in {".txt", ".md", ".rst"}


def _normalize_ws(s: str) -> str:
    # Collapse whitespace but keep paragraph structure roughly intact
    return re.sub(r"[ \t]+", " ", s).replace("\r", "").strip()


# ---------------------- PDF helpers ----------------------

def _pdf_text_with_pymupdf(path: Path) -> Optional[str]:
    if fitz is None:
        return None
    try:
        with fitz.open(str(path)) as doc:
            parts = []
            for page in doc:
                parts.append(page.get_text("text"))
        text = "\n".join(parts)
        text = _normalize_ws(text)
        return text if text else None
    except Exception as e:
        log.debug(f"PyMuPDF failed on {path}: {e}")
        return None


def _pdf_text_with_pdfminer(path: Path, timeout: int = 60) -> Optional[str]:
    if pdfminer_extract_text is None:
        return None
    try:
        # pdfminer can be slow; run in-process but catch errors
        text = pdfminer_extract_text(str(path))
        text = _normalize_ws(text or "")
        return text if text else None
    except Exception as e:
        log.debug(f"pdfminer failed on {path}: {e}")
        return None


def _should_ocr(text: Optional[str], path: Path) -> bool:
    """
    Very simple heuristic: if PyMuPDF/pdfminer return too little text and file has multiple pages,
    assume scanned PDF and consider OCR.
    """
    try:
        n_pages = 0
        if fitz is not None:
            with fitz.open(str(path)) as d:
                n_pages = d.page_count
        else:
            # fall back to guessing by file size if PyMuPDF is not available
            n_pages = 2 if path.stat().st_size > 512 * 1024 else 1
    except Exception:
        n_pages = 0

    length = len(text or "")
    return (n_pages >= 2 and length < 400) or (length == 0)


def _available(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _ocr_text_sidecar(path: Path, timeout: int = 300) -> Optional[str]:
    """
    Use ocrmypdf to produce a sidecar text file (no PDF rewrite).
      ocrmypdf --skip-text --sidecar out.txt in.pdf out.pdf
    We still must produce an output PDF argument; use a temporary file and ignore it.
    """
    if not _available("ocrmypdf"):
        return None
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        out_pdf = td / "out.pdf"
        sidecar = td / "out.txt"
        cmd = [
            "ocrmypdf",
            "--skip-text",              # don't overwrite existing text objects
            "--sidecar", str(sidecar),  # write text here
            "--quiet",
            str(path),
            str(out_pdf),
        ]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
            if proc.returncode != 0:
                log.debug(f"ocrmypdf failed: rc={proc.returncode} stderr={proc.stderr.decode(errors='ignore')[:500]}")
                return None
            if sidecar.exists():
                txt = sidecar.read_text(encoding="utf-8", errors="ignore")
                txt = _normalize_ws(txt)
                return txt if txt else None
        except subprocess.TimeoutExpired:
            log.debug("ocrmypdf timed out")
        except Exception as e:
            log.debug(f"ocrmypdf error: {e}")
    return None


# ---------------------- GROBID ----------------------

def _grobid_fulltext_tei(path: Path, grobid_url: str = "http://localhost:8070", timeout: int = 120) -> Optional[str]:
    """
    Call a local GROBID service to get TEI XML for the PDF.
    Endpoint: POST {grobid_url}/api/processFulltextDocument
    """
    url = f"{grobid_url.rstrip('/')}/api/processFulltextDocument"
    data = {
        # You can tweak consolidation params if you want
        "consolidateHeader": "1",
        "consolidateCitations": "1",
    }
    try:
        with path.open("rb") as fh:
            files = {"input": (path.name, fh, "application/pdf")}
            r = requests.post(url, files=files, data=data, timeout=timeout)
        r.raise_for_status()
        tei = r.text
        # simple sanity check: TEI header
        if "<TEI" in tei or "<tei" in tei:
            return tei
    except Exception as e:
        log.debug(f"GROBID failed on {path}: {e}")
    return None


# ---------------------- Other file types ----------------------

def _docx_text(path: Path) -> Optional[str]:
    if DocxDocument is None:
        return None
    try:
        doc = DocxDocument(str(path))
        parts = []
        for p in doc.paragraphs:
            parts.append(p.text)
        # tables
        for table in doc.tables:
            for row in table.rows:
                parts.append("  ".join(cell.text for cell in row.cells))
        text = "\n".join(parts)
        text = _normalize_ws(text)
        return text if text else None
    except Exception as e:
        log.debug(f"python-docx failed: {e}")
        return None


def _html_text(path: Path) -> Optional[str]:
    if BeautifulSoup is None:
        return None
    try:
        html = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        # Drop nav/script/style
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.extract()
        txt = soup.get_text("\n")
        txt = _normalize_ws(txt)
        return txt if txt else None
    except Exception as e:
        log.debug(f"BeautifulSoup failed: {e}")
        return None


def _txt_text(path: Path) -> Optional[str]:
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
        txt = _normalize_ws(txt)
        return txt if txt else None
    except Exception as e:
        log.debug(f"read txt failed: {e}")
        return None


# ---------------------- main entry ----------------------

def extract_any(path: Path,
                prefer_ocr: bool = False,
                force_academic_pdf: bool = False,
                grobid_url: str = "http://localhost:8070") -> Dict[str, object]:
    """
    Unified extractor.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    byts = read_bytes(path)
    sha = sha256_bytes(byts)

    tei_xml: Optional[str] = None
    text: Optional[str] = None
    tool: Optional[str] = None
    ocr_used = False

    # PDF path
    if looks_like_pdf(path):
        # Strategy ordering
        tried = []

        def try_grobid() -> Optional[str]:
            nonlocal tei_xml
            tei = _grobid_fulltext_tei(path, grobid_url=grobid_url)
            if tei:
                tei_xml = tei
                return tei
            return None

        def try_pymupdf() -> Optional[str]:
            return _pdf_text_with_pymupdf(path)

        def try_pdfminer() -> Optional[str]:
            return _pdf_text_with_pdfminer(path)

        def try_ocr() -> Optional[str]:
            return _ocr_text_sidecar(path)

        # Decide order
        if force_academic_pdf:
            tei = try_grobid()
            # Regardless of TEI success, attempt fast text extraction
            text = try_pymupdf() or try_pdfminer()
            if tei and text:
                tool = "grobid+pymupdf"
            elif tei:
                tool = "grobid"
            elif text:
                tool = "pymupdf/pdfminer"
            if text and _should_ocr(text, path) and prefer_ocr:
                ocr_txt = try_ocr()
                if ocr_txt and len(ocr_txt) > len(text):
                    text = ocr_txt
                    tool = (tool + "+ocr") if tool else "ocrmypdf"
                    ocr_used = True
        else:
            # Try fast text first
            text = try_pymupdf()
            tool = "pymupdf" if text else None
            if not text:
                text = try_pdfminer()
                tool = "pdfminer" if text else None

            if text and _should_ocr(text, path) and prefer_ocr:
                ocr_txt = try_ocr()
                if ocr_txt and len(ocr_txt) > len(text):
                    text = ocr_txt
                    tool = "ocrmypdf"
                    ocr_used = True

            # Optionally also get TEI if it looks academic or if forced
            # lightweight heuristic: if file name contains conf/journal-ish words
            if ("doi" in path.name.lower() or "conf" in path.name.lower() or "journal" in path.name.lower()) or force_academic_pdf:
                tei = _grobid_fulltext_tei(path, grobid_url=grobid_url)
                if tei:
                    tei_xml = tei
                    tool = (tool + "+grobid") if tool else "grobid"

        return {
            "sha256": sha,
            "text": text,
            "tei_xml": tei_xml,
            "extract_tool": tool or "unknown",
            "ocr_used": bool(ocr_used),
        }

    # DOCX
    if looks_like_docx(path):
        text = _docx_text(path)
        return {
            "sha256": sha,
            "text": text,
            "tei_xml": None,
            "extract_tool": "python-docx" if text else "unknown",
            "ocr_used": False,
        }

    # HTML
    if looks_like_html(path):
        text = _html_text(path)
        return {
            "sha256": sha,
            "text": text,
            "tei_xml": None,
            "extract_tool": "beautifulsoup" if text else "unknown",
            "ocr_used": False,
        }

    # TXT
    if looks_like_text(path):
        text = _txt_text(path)
        return {
            "sha256": sha,
            "text": text,
            "tei_xml": None,
            "extract_tool": "plaintext" if text else "unknown",
            "ocr_used": False,
        }

    # Fallback: unknown binary → no text

    return {
        "sha256": sha,
        "text": None,
        "tei_xml": None,
        "extract_tool": "unsupported",
        "ocr_used": False,
    }


# ---------------------- CLI runner ----------------------
if __name__ == "__main__":
    import argparse
    from collections import defaultdict
    import pandas as pd

    parser = argparse.ArgumentParser(description="Extract text/TEI from a tree of documents and write a sources.parquet manifest.")
    parser.add_argument("root", type=str, help="Root directory to walk (group folders under here).")
    parser.add_argument("--texts-dir", type=str, default="artifacts/text", help="Directory to write .txt and .tei.xml artifacts.")
    parser.add_argument("--parquet-dir", type=str, default="artifacts/parquet", help="Directory to write sources.parquet.")
    parser.add_argument("--grobid-url", type=str, default="http://localhost:8070", help="Base URL of the GROBID service.")
    parser.add_argument("--force-academic-pdf", action="store_true", help="Try GROBID first for PDFs (assume academic PDFs).")
    parser.add_argument("--prefer-ocr", action="store_true", help="If PDF text looks sparse, allow OCR sidecar and prefer it when longer.")
    parser.add_argument("--reextract", action="store_true", help="Overwrite existing text/tei artifacts; otherwise skip files already extracted.")
    parser.add_argument("--extensions", type=str, default=".pdf,.docx,.html,.htm,.txt,.md,.rst", help="Comma-separated list of file extensions to process.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    texts_dir = Path(args.texts_dir).expanduser().resolve()
    parquet_dir = Path(args.parquet_dir).expanduser().resolve()
    parquet_dir.mkdir(parents=True, exist_ok=True)
    texts_dir.mkdir(parents=True, exist_ok=True)

    wanted_exts = {e.strip().lower() if e.strip().startswith(".") else f".{e.strip().lower()}"
                   for e in args.extensions.split(",") if e.strip()}

    records = []
    seen_sha = set()

    def write_artifacts(src_path: Path, out: Dict[str, object]):
        sha = out.get("sha256")
        if not sha:
            return
        txt_path = texts_dir / f"{sha}.txt"
        tei_path = texts_dir / f"{sha}.tei.xml"

        # Respect --reextract: only skip if both targets exist and user didn't request overwrite
        if not args.reextract and txt_path.exists() and tei_path.exists():
            return

        # Write text if present (empty string is treated as absent)
        text = out.get("text")
        if text:
            txt_path.write_text(text, encoding="utf-8", errors="ignore")
        # Write TEI if present
        tei_xml = out.get("tei_xml")
        if tei_xml:
            tei_path.write_text(tei_xml, encoding="utf-8", errors="ignore")

    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden directories like .git, .DS_Store, etc.
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fname in filenames:
            if fname.startswith("."):
                continue
            fpath = Path(dirpath) / fname
            if fpath.suffix.lower() not in wanted_exts:
                continue

            try:
                out = extract_any(
                    fpath,
                    prefer_ocr=bool(args.prefer_ocr),
                    force_academic_pdf=bool(args.force_academic_pdf),
                    grobid_url=args.grobid_url,
                )
            except Exception as e:
                # Produce a minimal failure record
                records.append({
                    "source_sha256": None,
                    "canonical_ext": fpath.suffix.lower().lstrip("."),
                    "bytes": fpath.stat().st_size if fpath.exists() else None,
                    "mime": None,
                    "first_seen_path": str(fpath),
                    "all_paths": [str(fpath)],
                    "text_path": None,
                    "tei_path": None,
                    "extract_tool": f"error:{type(e).__name__}",
                    "ocr_used": None,
                    "extract_ok": False,
                    "extract_error": str(e),
                })
                print(f"[EXTRACT-ERROR] {fpath}: {e}")
                continue

            sha = out.get("sha256")
            if not sha:
                print(f"[WARN] No sha produced for {fpath}")
                continue

            write_artifacts(fpath, out)

            # Build a manifest record; consolidate duplicate SHA entries
            rec = {
                "source_sha256": sha,
                "canonical_ext": fpath.suffix.lower().lstrip("."),
                "bytes": fpath.stat().st_size,
                "mime": None,  # optional; could be added with python-magic
                "first_seen_path": str(fpath),
                "all_paths": [str(fpath)],
                "text_path": str((texts_dir / f"{sha}.txt")) if out.get("text") else None,
                "tei_path": str((texts_dir / f"{sha}.tei.xml")) if out.get("tei_xml") else None,
                "extract_tool": out.get("extract_tool"),
                "ocr_used": bool(out.get("ocr_used")),
                "extract_ok": bool(out.get("text") or out.get("tei_xml")),
                "extract_error": None,
            }

            if sha in seen_sha:
                # Merge paths into existing record
                for r in records:
                    if r.get("source_sha256") == sha:
                        paths = set(r.get("all_paths", []))
                        paths.add(str(fpath))
                        r["all_paths"] = sorted(paths)
                        break
            else:
                records.append(rec)
                seen_sha.add(sha)

            print(f"[OK] extracted {fpath}  → sha={sha[:8]} tool={rec['extract_tool']} ocr={rec['ocr_used']}")

    # Write manifest parquet
    if records:
        df = pd.DataFrame.from_records(records)
        out_path = parquet_dir / "sources.parquet"
        df.to_parquet(out_path, index=False)
        print(f"[OK] wrote manifest {out_path}  rows={len(df)}")
    else:
        print("[INFO] No supported files found; nothing written.")