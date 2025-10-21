
### Why this matters

Unchecked citations are a weak link in both education and research. This sidecar workflow turns citation checking into an educational, auditable, data‑rich process: it strengthens the integrity of student work by requiring sound citation practice, reveals how sources were actually used (facilitating assessment and naturalising transparent reporting), and generates aligned text‑evidence pairs suitable for training and benchmarking discipline specific language models. 

This repository serves **two distinct audiences**:

1. **Teachers** who want to make sound citation practice and transparent reporting a natural habit. The sidecar system enables instructors better to understand student engagement with the literature buy guiding and making visible proof of that engagement through steps that are trivial for good students and which block the most common forms of gen‑AI shortcutting.

2. **Developers and researchers** building or testing **natural language inference (NLI)**, citation verification, or retrieval pipelines. The dataset produced links student claims to verified source evidence with full provenance, and includes all source documents plus content‑addressed sidecars so you can evaluate both *extraction/parsing* and *reasoning* robustness.

---


# 🧩 student-sidecar

## 👩‍🏫 For Teachers: Auditing Student Submissions

The approach in this repository enables teachers to review student citation practice by operating on metadata (“sidecars”) produced by those students in the course of their normal engagement with the literature.

The script `process_submissions.py` scans each submission folder, creates previews of the first few rows of each CSV/XLS(X) table, and can **optionally fetch URLs** embedded in tables or Excel hyperlinks. This helps instructors quickly to spot missing sources.

With `build_pairs.py`, the toolchain **identifies misaligned or problematic citations**. It includes a “graph_like” heuristic that flags cases where the quoted material seems to refer to a figure, table, or other non-running-text element, rather than actual text from the source. This is especially useful for highlighting when students cite images or data visualizations instead of quoting or paraphrasing written content and relevant for testing students, and models, extraction and interpretation of non-textual information from documents.

Finally, the verification reports produced by `verify_quotes.py` provide teachers with a detailed summary of citation integrity. These reports indicate which quotes could be matched to the submitted sources, which were missed (and why). These diagnostics can be used as a basis for discussion with students about proper citation practices, integrity, and common pitfalls in academic writing.

## 🧠 For Developers: Data for NLI Training

This toolchain builds **three-layer paired data** connecting (1) a student’s quoted text from their report (`quote_from_report`), (2) the text they claim it came from in the source (`quote_from_source`), and (3) the full original source document from which that passage was drawn. The source is checked to determine whether the quoted portion actually appears there. The resulting dataset can be used to train or evaluate systems for citation verification, fact-checking, and natural language inference (NLI).

The main outputs are:
- `pairs_raw.parquet`: one row per citation pair.
- JSON verification reports (from `verify_quotes.py`).
- Plain-text and TEI sidecars for every unique source file.

Extraction is handled in layers. GROBID is used for structured academic PDFs, OCR is applied only when text can’t be read directly, and lighter extractors handle HTML, DOCX, and similar files. The goal is full coverage with minimal noise.

Because the dataset includes the **original source documents**, it supports not only NLI testing but also evaluation of extraction and parsing from messy, real-world documents. Every step records hashes and provenance, so results are reproducible.

The pipeline:

1) **process_submissions.py**
   - **Input:** root folder with submission subfolders.
   - **Does:** reads CSV/XLS(X), prints 4‑row previews, optionally fetches HTTP/HTTPS links discovered in cells or Excel hyperlinks.
   - **Creates:** `previews/` CSV heads; `<group>/urls/` with downloaded files and a `_manifest.csv`.

2) **extract_text.py**
   - **Input:** root (or a single group/folder).
   - **Does:** computes SHA256 of each source; extracts plain text (PyMuPDF/HTML/DOCX/TXT), calls **GROBID** for academic PDFs (TEI), OCR only when needed.
   - **Creates:** `artifacts/text/<sha>.txt` (+ `<sha>.tei.xml` when TEI exists) and `artifacts/parquet/sources.parquet`.
   - **Notes:** idempotent (skips already‑extracted SHA sidecars); content‑addressed and reproducible.

3) **build_pairs.py**
   - **Input:** normalized tables + extracted sidecars (`artifacts/text`) and `sources.parquet`.
   - **Does:** normalizes columns to `quote_from_report | file_name | quote_from_source`, resolves `file_name` to actual files (fuzzy match, Excel hyperlink handling), consolidates any `<group>/urls/_manifest.csv`, tags **graph_like** candidates.
   - **Creates:** `artifacts/parquet/pairs_raw.parquet`, `artifacts/parquet/tables_report.{parquet,csv}`, and `artifacts/parquet/urls_manifest.parquet` (if URLs were consolidated).

4) **verify_quotes.py**
   - **Input:** `pairs_raw.parquet` + sidecars in `artifacts/text/`.
   - **Does:** verifies that `quote_from_source` appears in the referenced source via **BM25 → SBERT cosine → fuzzy/Jaccard** cascade; handles multi‑sentence windows; skips trivial figure/table strings.
   - **Creates:** per‑group JSON in `artifacts/verification/` (+ optional flat summary parquet/csv when `--summary` is set).

> Designed to be resilient to messy student data (odd encodings, inconsistent filenames, extra columns, Excel hyperlinks, etc.).

At this point student assessment asks whether the 'quote from source' provided by the student can, indeed, be found in that source. For easily parsable source documents where the relevant information is in body text, this expectation performs very well. The same can not be said for messy documents or supporting information found in non-textual features. At this point I am running the assumption that students will not willfully submit false information in these tables so failures to detect are more likely attributable to shortcomings in these scripts. 

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