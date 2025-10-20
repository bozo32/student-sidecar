

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