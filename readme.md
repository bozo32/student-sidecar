
# student-sidecar

A workflow for turning messy student submissions of their supplemental active citation log into auditable evidence for teaching and provenance-rich datasets for retrieval, verification, and NLI research.

The core idea is simple: keep student submissions intact, extract sidecar metadata, and make citation practice inspectable without automated accusation.

---

## What this repository is for

### For teachers
This toolchain helps instructors:
- see whether cited sources resolve to real documents,
- check whether quoted passages actually appear in those sources,
- flag rows that likely refer to figures, tables, or graphs rather than text,
- optionally surface places where surrounding context may qualify or contradict a student’s claim.

Outputs are designed for human review and discussion, not automated misconduct detection.

### For developers / researchers
The pipeline produces reproducible, mergeable datasets linking:
1. `quote_from_report` — the student’s claim or paraphrase,
2. `quote_from_source` — what the student says they copied,
3. the **full original source document**.

All sources are content-addressed by **SHA256**, enabling cross-cohort merging and downstream work.

---

## Core design principle: content-addressed sources

- `source_sha256` is the identity.
- File paths are metadata, not identifiers.
- Text sidecars live at `artifacts/text/<sha>.txt` (and `<sha>.tei.xml` when available).

This makes results reproducible, mergeable across classes, and robust to renamed files.

---

## Quick start

```bash
conda create -n student_sidecar python=3.10 -y
conda activate student_sidecar
pip install -r requirements.txt
```

### Run GROBID (recommended for academic PDFs)

```bash
docker run --rm -it -p 8070:8070 -p 8071:8071 lfoppiano/grobid:0.8.0
```

---

## Data requirements (what students submit)

This pipeline assumes a very specific submission structure. Most failures and warnings are caused by deviations from this structure.

### Required folder structure
When processing, reference the <class_or_assignment_folder> as the target path. Starting from there, the structure must be:

```
<class_or_assignment_folder>/
  <group_identifier_1>/
    references.xlsx   (or .csv)
    cited_file_1.pdf
    cited_file_2.pdf
    ...
  <group_identifier_2>/
    references.xlsx
    cited_file_1.pdf
    ...
```

- Each **group** must have its **own folder**.
- The group folder name is treated as the authoritative `group_id`.
- There must be **exactly one citation table per group folder**.
- All locally cited files must live **in the same folder as the table**.

Nested folders inside a group folder are not supported.

---
### What students submit

Students should submit **one ZIP file**, which when unzipped contains **a single top-level folder**.

Each group must submit **one table** (`.xlsx`, `.xls`, or `.csv`) listing their sources.

The table must contain at least the following columns (column names are matched case-insensitively and with normalization):

| Column name | Meaning |
|------------|--------|
| `quote_from_report` | The sentence or claim from the student’s report that relies on the source |
| `quote_from_source` | The passage the student claims comes from the source (may be paraphrased) |
| `filename` | Name of the cited file **exactly as it appears in the group folder** |
| `url` | Optional. Direct URL to the source if no local file is provided |

Additional columns are allowed and preserved, but ignored by the analysis.

---

### Local files vs URLs (important)

Students must include **all cited sources as files**, **except** when the source is:

- directly (i.e. no click-through) and publicly accessible via a stable URL (e.g. journal article, report page),
- not behind authentication,
- not dynamically generated,

Rules:

- If a row contains a `filename`, that file **must exist** in the group folder.
- If a row contains a `url` **and no filename**, the pipeline will attempt to fetch it.
- If both are present, the local file takes precedence.

Broken filenames and inaccessible URLs are reported but do not stop the pipeline.

---

### What is *not* supported

- Multiple citation tables per group
- Citation tables split across folders
- References to files outside the group folder
- Handwritten PDFs or images without OCR
- Nested group folders

---

### Why these constraints exist

These constraints are intentional:

- they keep student submissions auditable,
- they prevent accidental cross-group contamination,
- they allow sources to be content-addressed and merged across cohorts.

When teaching, it is strongly recommended to **provide students with a template ZIP** that already follows this structure.

---


## Recommended workflow

### One-command processing almost all of the way 

This script:
- runs text extraction,
- rebuilds citation pairs,
- verifies quoted passages,
- runs cherry-picking triage,
- and prints a clear completion banner.

It is the recommended entry point for routine checking of files submitted by student analysis.

Example:

```bash
python full_cohort.py "/path/to/cohort/root"
```

This will **replace** all derived artifacts in `artifacts/` for that cohort.  
Use this only when you are ready to commit to a full rebuild.

full_cohort.py automatically runs the scripts below in sequence. These scripts can also be run individually for more control.

1) **Preview submissions and (optionally) capture URLs**
```bash
python process_submissions.py "/path/to/root" --fetch-urls
```

2) **Extract text and TEI sidecars**
```bash
python extract_text.py "/path/to/root" \
  --texts-dir artifacts/text \
  --parquet-dir artifacts/parquet \
  --extensions ".pdf,.docx,.html,.htm,.txt,.md,.rst" \
  --grobid-url http://localhost:8070
```

3) **Normalize tables and build citation pairs**
```bash
python build_pairs.py "/path/to/root" \
  --texts-dir artifacts/text \
  --parquet-dir artifacts/parquet \
  --prefer-excel --fallback-csv --csv-encoding-sweep \
  --consolidate-urls --urls-subdir urls
```

4) **Verify quoted source text**
```bash
python verify_quotes.py \
  --parquet-dir artifacts/parquet \
  --texts-dir artifacts/text \
  --out-dir artifacts/verification \
  --encoder all-MiniLM-L6-v2 \
  --bm25-topk 20 --cos-thresh 0.82 --fuzzy-thresh 85 \
  --summary-csv
```

5) **Optional: sniff for contextual qualification or contradiction**
```bash
python sniff_cherrypicking.py \
  --parquet-dir artifacts/parquet \
  --texts-dir artifacts/text \
  --out-dir artifacts/cherrypicking \
  --query-field quote_from_report \
  --cheap-only
```

---

## Outputs

All scripts write to `artifacts/`:

```
artifacts/
  text/                   # <sha>.txt and <sha>.tei.xml
  parquet/
    sources.parquet
    pairs_raw.parquet
    tables_report.{parquet,csv}
    urls_manifest.parquet
  verification/
    Group_<...>.json
    verification_summary.csv
  cherrypicking/
    cherrypicking_report.json
    cherrypicking_candidates.parquet
    cherrypicking_summary.csv
```

## Understanding the outputs (for teachers)

This workflow produces several kinds of outputs, each designed for a **different teaching purpose**. None of them are meant to be read raw by students; they are tools for **instructor review, diagnosis, and discussion**.

The outputs do **not** label misconduct. They surface *evidence and context* so that you can decide what, if anything, to do pedagogically.

---

### 1. `tables_report.csv` — How well did groups cite, structurally?

**What it is**  
A per-table diagnostic summary of each group’s submitted citation table(s).

**What it tells you**
- How many rows in each table contained something that *looked like* a citation.
- How many rows could be resolved to an actual source file or URL.
- How many rows were extractable as text (vs figures, tables, graphs, or broken references).

**How a teacher uses it**
- Identify groups who struggled *procedurally* (file naming, broken links, malformed spreadsheets).
- Separate **formatting / workflow problems** from **conceptual citation problems**.
- Decide which groups need technical remediation vs citation instruction.

---

### 2. `verification_summary.csv` — Do the quoted passages appear in the source?

**What it is**  
A per-group summary of quote-matching results.

**What it tells you**
- For each group:
  - how many cited quotes were checked,
  - how many had a close textual match in the cited source,
  - how many did not.

**How a teacher uses it**
- Spot groups where **many quoted passages are not actually present** in the cited documents.
- Decide where to open a discussion about paraphrasing vs quotation.
- Prioritize which groups to inspect more closely.

**Important caveat**  
A “no match” does **not** mean misconduct. The student may have paraphrased, the content may be in a figure or table, or the source may be scanned or poorly extracted.

---

### 3. `verification/Group_<…>.json` — Evidence for discussion

**What it is**  
A detailed, per-group evidence file.

**What it contains**
- the student’s quoted or paraphrased text,
- the best-matching passages found in the source,
- similarity scores (lexical and semantic),
- the exact location in the source text.

**How a teacher uses it**
- Prepare examples for class discussion.
- Walk through *how citation works in practice*.
- Show students what “good” vs “weak” evidence alignment looks like.

---

### 4. `cherrypicking_teacher.csv` — Fast triage for instructors

**What it is**  
A flattened, human-readable table meant specifically for teachers.

**What it shows**
- the student’s claim (from the report),
- the cited source,
- a nearby passage from the source,
- similarity scores,
- lexical signals that the passage may **qualify, limit, or contradict** the claim.

**How a teacher uses it**
- Quickly scan potentially problematic claims.
- Identify cases of selective quoting or over-generalization.
- Choose concrete examples for feedback or seminars.

---

### 5. `cherrypicking_report.json` — Structured context for deeper review

**What it is**  
A structured (JSON) version of the cherry-picking analysis.

**What it contains**
- All candidate passages considered for each claim.
- Grouping by student claim and source.
- Flags indicating whether deeper analysis (e.g. NLI) may be useful.

This is primarily intended for future dashboards or interactive tools.

---

### 6. Optional: `nli_results.csv` / `nli_results.parquet` — Does the source contradict the claim?

**What it is**  
An optional Natural Language Inference (NLI) pass run **only on flagged cases**.

**What it tells you**
- whether a source passage *entails*, *contradicts*, or is *neutral* with respect to the student’s claim,
- with model confidence scores.

**How a teacher uses it**
- As a prioritization aid, not a verdict.
- To distinguish paraphrasing issues from possible misrepresentation.

**Strong warning**  
NLI results are **suggestive, not authoritative**, and should never be used as automated judgments.

---

### What you are not getting

This pipeline deliberately avoids producing:
- automatic accusations,
- plagiarism scores,
- misconduct labels.

Instead, it produces **auditable evidence and context** suitable for teaching, feedback, and reflection.

---

## Multi-class / multi-cohort use

Label each dataset root by cohort (e.g. `2025_p1_yrm20306`).

- `extract_text.py` records `cohort_id` and `group_id`.
- Outputs can be kept per-cohort or merged later by `source_sha256`.
- A shared `artifacts/text/` directory safely deduplicates sources across cohorts.

---

## Details

### process_submissions.py
Scans group folders, previews CSV/XLS(X) tables, and optionally downloads HTTP/HTTPS URLs found in cells or Excel hyperlinks.

### extract_text.py
Hashes original bytes, extracts text sidecars, routes academic PDFs through GROBID, and records timing and provenance in `sources.parquet`. Extraction is idempotent and cohort-aware.

### build_pairs.py
Normalizes messy tables into a single row-level dataset, resolves filenames to actual sources (including Excel hyperlinks), flags graph-like references, and produces table-level diagnostics.

### verify_quotes.py
Checks whether `quote_from_source` appears in the cited source using a BM25 → embedding → fuzzy cascade, producing per-group JSON and summary CSVs.

### sniff_cherrypicking.py (experimental)
Performs a teacher-facing triage pass to surface nearby source passages that may qualify, limit, or contradict a student’s claim. Runs cheaply by default and can optionally apply NLI when explicitly enabled.

This script is intended to **support human judgment**, not automate accusations.

---

## License

MIT © 2025