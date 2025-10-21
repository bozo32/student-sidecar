

# 🧩 student-sidecar

A utility for auditing and normalizing student "supplemental information" submissions.  
It walks through nested group folders, previews tabular data, standardizes column names, and optionally downloads URLs embedded in student CSV/XLS files.

---

## ✨ Features

- Recursively scans subdirectories for `.csv`, `.tsv`, `.xls`, `.xlsx`, or `.xlsm` files  
- Prints a clean console summary of:
  - file path
  - detected delimiter
  - normalized column names
  - first 4 rows of data
- Saves a normalized 4-row preview of each table under `./previews/`
- Optionally extracts and downloads any URLs found in tables:
  - Saves to `<group_folder>/urls/`
  - Generates a `_manifest.csv` with metadata (status, content type, SHA256, saved path, etc.)
  - Optionally extracts readable text from HTML via `trafilatura`

---

## 🧠 Typical Folder Layout

Each group submission should look something like:

```
source/
├── group_1/
│   ├── group_1.csv
│   ├── source_file_1.pdf
│   └── ...
├── group_2/
│   ├── group_2.xlsx
│   ├── source_file_1.pdf
│   └── ...
└── ...
```

---

## 🛠️ Installation

```bash
conda create -n student_sidecar python=3.10
conda activate student_sidecar

# install dependencies
pip install pandas openpyxl requests trafilatura
```

---

## 🚀 Usage

Preview all student tables (CSV/XLS) recursively:
```bash
python process_submissions.py /path/to/root
```

Preview and fetch URLs found in those tables:
```bash
python process_submissions.py /path/to/root --fetch-urls --url-timeout 20
```

This will:
- print console summaries for every detected file,
- write normalized previews to `./previews/`,
- and download all reachable URLs to each group’s `urls/` folder.

---

## 🧩 Output Structure

```
previews/
    Group_7_supplemental_information_Literature_Group_7_csv__head.csv
group_7_supplemental_information/
    urls/
        _manifest.csv
        9a3e1f….pdf
        5bfe2c….html
```

---

## ⚙️ Manifest Fields

| Column | Description |
|:--|:--|
| `url` | Original URL found in table |
| `status` | HTTP status code |
| `content_type` | MIME type returned |
| `bytes` | Payload size |
| `sha256` | Hash of downloaded bytes |
| `saved_path` | Where the file was saved |
| `text_path` | Extracted plaintext (if available) |
| `source_table` | Which CSV/XLS file it came from |
| `source_cell` | Row and column in table |
| `source_sheet` | Excel sheet (if applicable) |
| `error` | Error message if failed |

---

## 📦 Planned Extensions

- Detect and extract quoted sentences from PDFs
- Validate that “quote from source” text actually appears in the cited PDF
- Build a clean, databricks-friendly dataset linking report text → source file → verified evidence

---

## 🧾 License

MIT License © 2025 Bozo32
# 🧩 student-sidecar

Tools to audit and normalize student “supplemental information” submissions and build a dataset for **citation checking**. The pipeline:

1) **process_submissions.py** — scan folders, preview tables, (optionally) capture and download URLs found in CSV/XLS.
2) **extract_text.py** — content-address files (SHA256) and extract text/TEI from PDFs/HTML/DOCX/… (GROBID for academic PDFs).
3) **build_pairs.py** — normalize messy tables into a clean row-level dataset linking *quote_from_report ↔ source file ↔ quote_from_source*; consolidates URL downloads into the dataset; writes summary diagnostics.
4) **verify_quotes.py** — attempt to verify that *quote_from_source* actually appears in the referenced source using BM25 + embedding similarity; per‑group JSON reports.

> Designed to be resilient to messy student data (odd encodings, inconsistent filenames, extra columns, Excel hyperlinks, etc.).

---

## 🚀 Quick Start

```bash
# 1) Create a clean environment
conda create -n student_sidecar python=3.10 -y
conda activate student_sidecar

# 2) Install requirements
pip install -r requirements.txt

# 3) (Recommended) Start GROBID for academic PDFs
# Option A: Docker (Intel/AMD and Apple Silicon both supported via multi-arch)
docker run --rm -it -p 8070:8070 -p 8071:8071 lfoppiano/grobid:0.8.0
# Option B: local build
#  (from grobid repo) ./gradlew :grobid-service:run
# The scripts expect the service at http://localhost:8070 by default.
```

> **Data layout expected** (you can point the tools at the root folder):
>
> ```
> /path/to/root/
> ├── Group 1 supplemental information/
> │   ├── group_1.xlsx (or .csv)
> │   ├── source_1.pdf
> │   └── …
> ├── Group 2 supplemental information/
> │   ├── group_2.csv
> │   ├── urls/ (optional; created by process_submissions)
> │   └── …
> └── …
> ```

---

## 📦 What gets created

All scripts write into `artifacts/` (relative to your current working directory):

```
artifacts/
  text/                   # SHA256-named sidecars: <sha>.txt and <sha>.tei.xml
  parquet/
    sources.parquet       # one row per unique source (sha, paths, mime, extract status)
    pairs_raw.parquet     # one row per table row (normalized), with resolved source path/sha
    urls_manifest.parquet # normalized records for every URL captured/downloaded
    tables_report.{parquet,csv}  # per-table diagnostics (nonempty rows vs extracted/resolved)
  verification/
    Group_<…>.json        # per‑group verification reports (see verify_quotes)
```

Sidecar file names are **content addressed** by SHA256 of the original bytes, so the same file from two groups only extracts once: `artifacts/text/<sha>.txt` and (if academic PDF) `artifacts/text/<sha>.tei.xml`.

---

## 1) `process_submissions.py`

Walk subdirectories, preview the first 4 rows of each CSV/XLS, and (optionally) download URLs embedded in cells or Excel hyperlinks.

```bash
python process_submissions.py /path/to/root \
  [--fetch-urls] [--urls-subdir urls] [--url-timeout 20]
```

**Behavior**
- Detects `.csv`, `.tsv`, `.xls`, `.xlsx`, `.xlsm`.
- Prints per‑file summary (path, inferred delimiter, normalized columns, 4‑row head).
- Writes normalized 4‑row previews to `./previews/`.
- If `--fetch-urls` is set:
  - Ignores *local* file links (e.g., `file:///`, `C:\\…`, macOS file paths).
  - Downloads **HTTP/HTTPS** links into `<group>/urls/` and records `_manifest.csv`.
  - Extracts text from HTML via `trafilatura` when possible.

---

## 2) `extract_text.py`

Content-address every discovered binary and produce text/TEI sidecars. This is safe to run repeatedly; it skips sources that already have sidecars.

```bash
python extract_text.py /path/to/root \
  --texts-dir artifacts/text \
  --parquet-dir artifacts/parquet \
  [--grobid-url http://localhost:8070] \
  [--force-academic-pdf]              # force PDF → GROBID even if text is extractable
  [--overwrite]                       # re-extract even if sidecars already exist
```

**Notes**
- Hashing is on **original file bytes**. Output files are `<sha>.txt` and optionally `<sha>.tei.xml`.
- Academic PDFs are routed to GROBID (header+fulltext TEI). Non‑academic PDFs and other formats fall back to `pdfminer.six`, `pypdf`, `mammoth` (DOCX→HTML→text), `trafilatura`, etc.
- Writes/updates `artifacts/parquet/sources.parquet`.

**Sanity check**
```bash
python check_missing_sidecars.py --parquet artifacts/parquet/sources.parquet \
  --texts-dir artifacts/text
```

---

## 3) `build_pairs.py`

Normalize messy student tables into a single dataset and line them up with extracted sources. Also consolidates any URLs captured by `process_submissions.py`.

```bash
python build_pairs.py "/path/to/root" \
  --texts-dir artifacts/text \
  --parquet-dir artifacts/parquet \
  --consolidate-urls \
  --urls-subdir urls \
  --grobid-url http://localhost:8070 \
  [--prefer-excel] [--fallback-csv] [--csv-encoding-sweep]
```

**What it does**
- Reads each group’s CSV/XLS(X) and normalizes column names to:
  - `quote_from_report`, `file_name`, `quote_from_source` (handles common variants and extra columns; for sheets that include an extra leading label column like *Section*, it drops/renames appropriately).
- Handles Excel hyperlinks: uses the **displayed text** as filename when the link points to a local file; preserves **HTTP/HTTPS** as URL candidates.
- Fuzzy‑matches `file_name` to actual files in the group folder; if multiple candidates tie, it prefers PDFs.
- Emits **diagnostic table** `tables_report.{parquet,csv}` with, per table/sheet: nonempty rows, extracted rows, resolved paths, success ratios. Use it to find messy submissions fast.
- Updates/creates:
  - `artifacts/parquet/pairs_raw.parquet`
  - `artifacts/parquet/sources.parquet` (merged paths and hashes)
  - `artifacts/parquet/urls_manifest.parquet` (if `--consolidate-urls`)

**Tip**: If you have already run `extract_text.py`, `build_pairs.py` will **not** re‑extract content; it will just align table rows to existing sidecars via SHA/paths.

---

## 4) `verify_quotes.py`

Try to verify that each `quote_from_source` actually appears in the referenced source. It uses a robust cascade:

1. **BM25** to pull top‑K candidate sentences from the source sidecar.
2. **Sentence‑BERT** cosine for semantic matching.
3. **Fuzzy** (token sort ratio) and **5‑gram Jaccard** to catch near‑verbatim snippets.
4. **Multi‑line** handling: splits quoted text into sentences and searches sliding windows.
5. Skips trivial entries (e.g., “table”, “figure”, “n.a.”).

```bash
python verify_quotes.py \
  --parquet-dir artifacts/parquet \
  --texts-dir artifacts/text \
  --out-dir artifacts/verification \
  --encoder all-MiniLM-L6-v2 \
  --bm25-topk 20 --cos-thresh 0.82 --fuzzy-thresh 85 \
  [--summary parquet]   # also write a flat parquet with per-row results
```

**Outputs**
- One JSON per group under `artifacts/verification/` summarizing:
  - success ratio, 
  - per‑source filename → found/missed quotes,
  - notes for misses (e.g., trivial generic text, encoding/sanitization issues).

---

## 🔍 Suggested end‑to‑end

```bash
# 0) Ensure GROBID is running (optional but recommended for academic PDFs)
# docker run --rm -it -p 8070:8070 lfoppiano/grobid:0.8.0

# 1) Quick scan + (optional) URL capture
python process_submissions.py "/path/to/root" --fetch-urls --url-timeout 20

# 2) Extract text once for all sources
python extract_text.py "/path/to/root" \
  --texts-dir artifacts/text --parquet-dir artifacts/parquet --force-academic-pdf
python check_missing_sidecars.py --parquet artifacts/parquet/sources.parquet --texts-dir artifacts/text

# 3) Normalize tables and build pairs + diagnostics
python build_pairs.py "/path/to/root" \
  --texts-dir artifacts/text --parquet-dir artifacts/parquet \
  --consolidate-urls --urls-subdir urls --grobid-url http://localhost:8070 \
  --prefer-excel --fallback-csv --csv-encoding-sweep

# 4) Verify quotes
python verify_quotes.py --parquet-dir artifacts/parquet --texts-dir artifacts/text \
  --out-dir artifacts/verification --encoder all-MiniLM-L6-v2 \
  --bm25-topk 20 --cos-thresh 0.82 --fuzzy-thresh 85 --summary parquet
```

---

## 🧪 Troubleshooting & Tips

- **CSV encoding errors**: use `--csv-encoding-sweep` (build_pairs) or convert manually to UTF‑8.
- **Ghostscript warnings** during OCR/repair: they’re harmless here; we avoid destructive rewrite by default.
- **Performance**: extraction is content‑addressed; repeated runs are fast. Keep `artifacts/text/` around for reuse.
- **Apple Silicon**: the GROBID Docker image is multi‑arch; if you built locally, ensure Java 17+ and enough heap (`JAVA_OPTS=-Xms1g -Xmx4g`).

---

## 📄 License

MIT © 2025 Bozo32