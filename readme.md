MIT License © 2025 Bozo32
# 🧩 student-sidecar

Tools for auditing and normalizing messy “supplemental information” submissions, extracting text from cited sources, and verifying that quoted text actually appears in those sources.

This repo is tuned for real-world student data: weird CSV encodings, Excel sheets with extra columns, filenames that are URLs or half-typed references, PDFs that may be scanned, and academic papers processed via a local GROBID service.

---

## What’s in the box

**CLI scripts (run them in this order):**

1) **`process_submissions.py`** — Walk group folders, preview tables, and (optionally) download any URLs found in CSV/XLSX.
2) **`extract_text.py`** — Layered text extraction for PDFs/Word/HTML/TXT with optional GROBID TEI; writes sidecar files by SHA.
3) **`build_pairs.py`** — Normalize tables to a standard 3-column schema, resolve each row’s source file/URL, and produce:
   - a rows dataset (`pairs_raw.parquet`)
   - a unique sources manifest (`sources.parquet`)
   - a consolidated URL manifest (`urls_manifest.parquet`)
   - a per-table diagnostic report (`tables_report.csv`/`.parquet`)
4) **`verify_quotes.py`** — For each row, try to find the “quote from source” in the extracted source text using BM25 + embeddings + fuzzy matching; writes one JSON per group (+ optional summary CSV).

**Utilities**

- `check_missing_sidecars.py` — quick check for extracted sidecars that are referenced by `pairs_raw.parquet` but missing in `artifacts/text/`.

---

## Folder layout (expected)

```
/path/to/root/
├── Group 1 supplemental information/
│   ├── group_1.xlsx
│   ├── Some Paper.pdf
│   └── ...
├── Group 2 supplemental information/
│   ├── group_2.csv
│   ├── A_Web_Article.html
│   └── ...
└── ...
```

Artifacts are written under `./artifacts/` (created on first run):

```
artifacts/
  parquet/
    pairs_raw.parquet
    sources.parquet
    urls_manifest.parquet
    tables_report.parquet
    tables_report.csv
  text/
    <sha256>.txt
    <sha256>.tei.xml
  verification/
    Group_<group>.json
    verification_summary.csv   # when enabled
previews/
  <group>_<table>[_<sheet>]_head.csv
```

---

## Environment

Python 3.10+ recommended. Minimal setup:

```bash
conda create -n student_sidecar python=3.10
conda activate student_sidecar

# Install project deps
pip install -r requirements.txt
```

Optional tools:
- **GROBID** service (Java) for TEI extraction from academic PDFs (default `http://localhost:8070`).
- **ocrmypdf** if you want OCR sidecars for scanned PDFs (not required by default).

> Apple Silicon note: prefer running GROBID locally with `./gradlew run` inside the GROBID repo (no Docker needed).

---

## 1) Preview & fetch URLs

Scan groups, print normalized headers + 4-row head per table, and (optionally) download URLs to a per‑group `urls/` folder.

```bash
python process_submissions.py "/path/to/root"
# or include URL fetching:
python process_submissions.py "/path/to/root" --fetch-urls --url-timeout 20
```

Behavior:
- Handles `.csv/.tsv/.xls/.xlsx/.xlsm`
- Normalizes to canonical headers: `quote_from_report`, `file_name`, `quote_from_source`
- Writes 4-row previews under `./previews/`
- When `--fetch-urls` is set:
  - Downloads only `http(s)` URLs (local `file://` / drive-path links are ignored)
  - Saves to `<group>/urls/` and logs to `_manifest.csv`
  - Excel cell hyperlinks are captured via OpenPyXL as well as plain text detections

---

## 2) Extract text / TEI sidecars

Walk the **source files** under the same root (PDF/DOCX/HTML/TXT) and create sidecars under `artifacts/text/` keyed by SHA‑256.

```bash
python extract_text.py "/path/to/root" \
  --texts-dir artifacts/text \
  --parquet-dir artifacts/parquet \
  --grobid-url http://localhost:8070 \
  --force-academic-pdf     # try GROBID first for PDFs
  # --prefer-ocr           # optionally allow OCR via ocrmypdf
  # --reextract            # overwrite existing sidecars
```

Heuristics:
- PDF order: fast text (PyMuPDF/pdfminer), optional OCR (sidecar only), optional GROBID TEI (or first when `--force-academic-pdf`)
- DOCX via `python-docx`; HTML via `BeautifulSoup`; TXT read as UTF‑8 (errors ignored)
- Output rows -> `artifacts/parquet/sources.parquet`

---

## 3) Build normalized pairs

Read student tables again (with stronger cleanup), resolve each row’s source (local file, or URL → downloaded copy), attach SHA and sidecar paths, and write datasets.

```bash
python build_pairs.py "/path/to/root" \
  --texts-dir artifacts/text \
  --parquet-dir artifacts/parquet \
  --consolidate-urls \
  --urls-subdir urls \
  --grobid-url http://localhost:8070 \
  --prefer-excel \
  --fallback-csv \
  --csv-encoding-sweep
```

Key behavior & heuristics:
- **Excel preferred** when present; CSVs used if no Excel (or if you omit `--prefer-excel`)
- Robust CSV reading with encoding sweep (`utf-8[-sig]`, `cp1252`, `latin1`, `mac_roman`, and UTF‑16 variants)
- Excel `=HYPERLINK("url","text")` cells: we keep **visible text**, not the link
- “Group 14 pattern”: if more than 3 columns, we try to infer the last texty column as the true `quote_from_source`
- Column normalization to **exactly 3 canonical columns**
- **URL logic**:
  - If `file_name` is a URL and a downloaded copy exists in `<group>/urls/_manifest.csv`, we use the saved file
  - Otherwise the original URL is kept in `source_uri`
- **File resolution**:
  - Fuzzy matching of messy filenames to actual files under the group (token overlap + substring boosts)
  - Resolved files are hashed (SHA‑256) once and linked to existing sidecars (no re-extraction here)
- Writes:
  - `artifacts/parquet/pairs_raw.parquet`
  - `artifacts/parquet/sources.parquet` (de‑duplicated unique sources, merged with paths)
  - `artifacts/parquet/urls_manifest.parquet` (when `--consolidate-urls`)
  - `artifacts/parquet/tables_report.parquet` & `tables_report.csv` (per‑sheet diagnostics:
    non‑empty rows, rows resolved to files, extraction success %)
- Console summary prints the **lowest success ratios** to help you spot problematic tables quickly.

---

## 4) Verify quotes against sources

For each row in `pairs_raw.parquet`, verify whether the *claimed* `quote_from_source` appears in the corresponding sidecar (TEI sentences when available, else regex-split sentences from TXT).

```bash
python verify_quotes.py \
  --parquet-dir artifacts/parquet \
  --texts-dir artifacts/text \
  --out-dir artifacts/verification \
  --encoder all-MiniLM-L6-v2 \
  --bm25-topk 20 --cos-thresh 0.82 --fuzzy-thresh 85 \
  --summary-csv
```

How it works:
- **BM25** retrieves top‑K candidate sentences (and small windows around them)
- **Embeddings** (Sentence‑Transformers) compute cosine similarity
- **Fuzzy match** (rapidfuzz partial ratio) + **5‑gram Jaccard** as backstops
- A hit if any of the above passes adaptive thresholds (short quotes require stricter fuzzy; long quotes allow lower cosine)
- Trivial claims like `table`, `figure`, `n.a.` → “non-verifiable” miss with note

Outputs:
- One JSON per group: success ratio, counts, and **misses with diagnostics** (`candidate`, cosine, fuzzy, jaccard, window span)
- Optional `verification_summary.csv` (group, totals, success ratio)

---

## Data model (selected columns)

**`pairs_raw.parquet`**
- `group_id` — group folder name
- `quote_from_report` — student’s reported text
- `file_name_raw` — raw “source filename” cell (may be URL)
- `quote_from_source` — claimed quote to verify
- `source_uri` — URL if present in row/columns
- `resolved_source_path` — local path found via fuzzy match or URL download
- `source_sha256` — SHA for de‑duplication and sidecar lookup
- `source_text_path` / `tei_path` — sidecar locations (when available)
- `extract_ok` / `extract_error` — result of extraction for that row’s source (if extraction was attempted here)

**`sources.parquet`**
- One row per unique `source_sha256`; includes `all_paths`, `text_path`, `tei_path`, `extract_tool`, `ocr_used`, `extract_ok`.

**`urls_manifest.parquet`**
- Consolidation of all `<group>/urls/_manifest.csv` files (if present).

**`tables_report.[parquet|csv]`**
- Per (group, table, sheet): non‑empty row count, extraction success %, and resolution %.

---

## GROBID notes

- Default endpoint: `http://localhost:8070`
- You can run it inside the GROBID repo:

```bash
cd /path/to/grobid
./gradlew run
# service starts on :8070; stop with Ctrl+C
```

---

## Performance tips

- **Skip re‑extraction:** `build_pairs.py` never re‑extracts if a sidecar already exists for a SHA. To re‑extract everything, do it explicitly via `extract_text.py --reextract`.
- **CPU‑bound steps:** GROBID and OCR are the slowest paths. Use them selectively (e.g., `--force-academic-pdf` during `extract_text.py` only).
- **Embeddings:** `verify_quotes.py` loads a Sentence‑Transformers model; reuse the same model across runs for consistency.

---

## Troubleshooting

- **CSV encoding errors:** Use `--csv-encoding-sweep` in `build_pairs.py`. It tries common Windows/mac encodings and UTF‑16.
- **Excel with extra/empty columns:** The normalizer drops empty `Unnamed:` columns and tries to infer the last “texty” column as `quote_from_source`.
- **Links to local files in Excel:** We capture and ignore local non‑HTTP hyperlinks; only `http(s)` URLs are fetched.
- **Ghostscript/ocrmypdf warnings:** We never rewrite PDFs; OCR uses `--sidecar` text only.
- **Missing sidecars:** Run `check_missing_sidecars.py` to list SHAs referenced by `pairs_raw.parquet` that have no `.txt`/`.tei.xml` in `artifacts/text/`.

---

## Example end‑to‑end

```bash
# 1) Preview + fetch URLs
python process_submissions.py "/path/to/root" --fetch-urls --url-timeout 20

# 2) Extract text/tei (optionally favor GROBID for academic PDFs)
python extract_text.py "/path/to/root" \
  --texts-dir artifacts/text \
  --parquet-dir artifacts/parquet \
  --grobid-url http://localhost:8070 \
  --force-academic-pdf

# 3) Normalize & pair
python build_pairs.py "/path/to/root" \
  --texts-dir artifacts/text \
  --parquet-dir artifacts/parquet \
  --consolidate-urls \
  --urls-subdir urls \
  --grobid-url http://localhost:8070 \
  --prefer-excel --fallback-csv --csv-encoding-sweep

# 4) Verify quotes
python verify_quotes.py \
  --parquet-dir artifacts/parquet \
  --texts-dir artifacts/text \
  --out-dir artifacts/verification \
  --encoder all-MiniLM-L6-v2 \
  --bm25-topk 20 --cos-thresh 0.82 --fuzzy-thresh 85 \
  --summary-csv
```

---

## License

MIT © 2025 Bozo32