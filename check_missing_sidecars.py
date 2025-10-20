#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

pairs_path = Path("/Users/peter/ai-stuff/student sidecar/artifacts/parquet/pairs_raw.parquet")
texts_dir = Path("/Users/peter/ai-stuff/student sidecar/artifacts/text")

pairs = pd.read_parquet(pairs_path)

missing = []
for _, row in pairs.iterrows():
    sha = row.get("source_sha256")
    if not isinstance(sha, str) or not sha.strip():
        continue
    txt = texts_dir / f"{sha}.txt"
    tei = texts_dir / f"{sha}.tei.xml"
    if not txt.exists() and not tei.exists():
        missing.append({
            "group_id": row.get("group_id"),
            "file_name_raw": row.get("file_name_raw"),
            "source_sha256": sha,
        })

df = pd.DataFrame(missing).drop_duplicates(subset=["source_sha256"]).sort_values(["group_id", "file_name_raw"])
out_path = pairs_path.parent / "missing_text_sources.csv"
df.to_csv(out_path, index=False)
print(f"[OK] wrote {out_path} with {len(df)} missing sidecar files")