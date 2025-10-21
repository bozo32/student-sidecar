#!/usr/bin/env python3
"""
verify_quotes.py — check whether students' "quote from source" appears in the submitted source
(using BM25 + semantic similarity + fuzzy match), and summarize results per group.

Inputs (default locations):
  artifacts/parquet/pairs_raw.parquet   # from build_pairs.py
  artifacts/parquet/sources.parquet     # from build_pairs.py
  artifacts/text/<sha>.txt              # plain text sidecar (optional but preferred)
  artifacts/text/<sha>.tei.xml          # TEI XML (if GROBID used)

Outputs:
  artifacts/verification/Group_<group>.json   # one JSON per group with success ratio & misses

Heuristics:
  - BM25 retrieves top-K candidate sentences (from TEI sentence tags if present, else regex split).
  - Semantic similarity via Sentence-Transformers (default: all-MiniLM-L6-v2).
  - Fuzzy string match via rapidfuzz.partial_ratio as a fallback.
  - If either cosine >= cos_thresh OR fuzzy >= fuzzy_thresh, we count it as a hit.
  - Trivial claims like "table", "figure", "n.a." are marked non-verifiable (counted under misses with a note).

CLI:
  python verify_quotes.py \
      --parquet-dir artifacts/parquet \
      --texts-dir artifacts/text \
      --out-dir artifacts/verification \
      --encoder all-MiniLM-L6-v2 \
      --bm25-topk 20 --cos-thresh 0.82 --fuzzy-thresh 85
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# light-weight tokenization for BM25
import spacy
from rank_bm25 import BM25Okapi
from rapidfuzz.fuzz import partial_ratio
from sentence_transformers import SentenceTransformer, util

# ------------------- Config -------------------

TRIVIAL_PAT = re.compile(r"^(?:table|figure|fig\.?\s*\d+|graph|content|n\.?a\.?)$", re.I)

DEFAULT_PARQUET_DIR = Path("artifacts/parquet")
DEFAULT_TEXTS_DIR = Path("artifacts/text")
DEFAULT_OUT_DIR = Path("artifacts/verification")

# ------------------- Helpers -------------------

def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\u00A0", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_for_match(text: Optional[str]) -> str:
    if text is None:
        return ""
    s = str(text)
    # replace NBSP and tabs with space
    s = s.replace("\u00A0", " ").replace("\t", " ")
    # normalize quotes and dashes
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    s = s.replace("–", "-").replace("—", "-").replace("\u2013", "-").replace("\u2014", "-")
    # remove soft hyphens
    s = s.replace("\u00AD", "")
    # fuse hyphenations across line breaks
    s = re.sub(r"-\s*\n\s*", "", s)
    # collapse newlines and multiple spaces
    s = re.sub(r"\s*\n\s*", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # remove simple citations like (Author, 2020) or (Author et al., 2020)
    s = re.sub(r"\(\s*[A-Za-z]+(?: et al\.)?,?\s*\d{4}[a-z]?\s*\)", "", s)
    s = s.strip()
    return s

def split_sentences_simple(text: str) -> List[str]:
    parts = re.split(r"(?<=[\.\?\!])\s+(?=[A-Z(])", text)
    return [p.strip() for p in parts if p.strip()]

def jaccard_char_ngrams(a: str, b: str, n: int = 5) -> float:
    def ngrams(s: str, n: int) -> set:
        s = s.lower()
        return set(s[i:i+n] for i in range(max(len(s) - n + 1, 1)))
    a_ngrams = ngrams(a, n)
    b_ngrams = ngrams(b, n)
    if not a_ngrams and not b_ngrams:
        return 1.0
    intersection = len(a_ngrams.intersection(b_ngrams))
    union = len(a_ngrams.union(b_ngrams))
    return intersection / union if union > 0 else 0.0

def sent_split_regex(text: str) -> List[str]:
    # simple, robust sentence splitter when TEI sentences aren't available
    # split on ., ?, ! followed by whitespace+capital OR end of string
    # also keep long lines as sentences
    parts = re.split(r"(?<=[\.\?\!])\s+(?=[A-Z(])", text)
    out = []
    for p in parts:
        p = normalize_text(p)
        if p:
            out.append(p)
    return out

def read_tei_sentences(tei_path: Path) -> Optional[List[str]]:
    try:
        import xml.etree.ElementTree as ET
        root = ET.parse(tei_path).getroot()
        ns = {"tei": "http://www.tei-c.org/ns/1.0"}
        # prefer explicit sentence tags if present
        sents = [normalize_text("".join(s.itertext())) for s in root.findall(".//tei:s", ns)]
        sents = [s for s in sents if s]
        if sents:
            return sents
        # otherwise, paragraphs -> split further
        paras = [normalize_text("".join(p.itertext())) for p in root.findall(".//tei:p", ns)]
        paras = [p for p in paras if p]
        if paras:
            out = []
            for p in paras:
                out.extend(sent_split_regex(p))
            return out if out else None
    except Exception:
        return None
    return None

@dataclass
class SourceIndex:
    sentences: List[str]
    bm25: BM25Okapi
    tokenized: List[List[str]]

@lru_cache(maxsize=1024)
def _load_spacy():
    # use blank English for tokenization only (fast, no model download)
    return spacy.blank("en")

def build_bm25(sentences: List[str]) -> SourceIndex:
    nlp = _load_spacy()
    tokenized = [[tok.text for tok in nlp(s)] for s in sentences]
    # avoid empty docs
    tokenized = [t if t else [""] for t in tokenized]
    bm25 = BM25Okapi(tokenized)
    return SourceIndex(sentences=sentences, bm25=bm25, tokenized=tokenized)

@lru_cache(maxsize=8)
def _load_encoder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)

def cosine_sim(q: str, cands: List[str], model_name: str) -> np.ndarray:
    enc = _load_encoder(model_name)
    qe = enc.encode([q], convert_to_tensor=True, show_progress_bar=False)
    ce = enc.encode(cands, convert_to_tensor=True, show_progress_bar=False)
    sims = util.pytorch_cos_sim(qe, ce).cpu().numpy()[0]
    return sims

def best_literal_bonus(query: str, candidate: str) -> float:
    """
    Give a small bonus if a long literal snippet of the query appears in candidate.
    """
    q = normalize_text(query)
    c = normalize_text(candidate)
    # take the longest token window of length >= 12 chars that appears in both
    # very cheap proxy
    bonus = 0.0
    if len(q) >= 12:
        # take mid slice if very long
        for L in [40, 30, 20, 15, 12]:
            if len(q) >= L:
                frag = q[:L]
                if frag in c:
                    bonus = 0.02
                    break
    return bonus

def verify_one_quote(
    quote_from_source: str,
    sidx: SourceIndex,
    encoder_name: str,
    bm25_topk: int,
    cos_thresh: float,
    fuzzy_thresh: int,
) -> Tuple[bool, Dict]:
    q = normalize_for_match(quote_from_source)
    if not q:
        return False, {"note": "empty quote_from_source"}
    if TRIVIAL_PAT.match(q.strip()):
        return False, {"note": "non-verifiable (generic: figure/table/N.A.)"}

    # BM25 retrieve
    nlp = _load_spacy()
    q_tokens = [t.text for t in nlp(q)]
    scores = sidx.bm25.get_scores(q_tokens)
    if not isinstance(scores, np.ndarray):
        scores = np.array(scores, dtype=float)

    # top-k indices safely
    k = min(bm25_topk, len(scores))
    if k == 0:
        return False, {"note": "no sentences in source"}
    top_idx = np.argpartition(-scores, range(k))[:k]
    top_idx = [int(i) for i in top_idx if 0 <= int(i) < len(sidx.sentences)]
    if not top_idx:
        return False, {"note": "no valid candidates from BM25"}

    # build windowed candidates around top_idx
    q_sents = split_sentences_simple(q)
    win_len = min(max(len(q_sents), 1), 3)
    candidates = []
    spans = []
    seen_texts = set()
    n_sents = len(sidx.sentences)

    for idx in top_idx:
        # windows from size 1 to win_len, forward and backward
        for size in range(1, win_len + 1):
            # forward window
            start_fwd = idx
            end_fwd = min(idx + size, n_sents)
            cand_fwd = " ".join(sidx.sentences[start_fwd:end_fwd])
            cand_fwd_norm = normalize_for_match(cand_fwd)
            if cand_fwd_norm and cand_fwd_norm not in seen_texts:
                candidates.append(cand_fwd_norm)
                spans.append((start_fwd, end_fwd - 1))
                seen_texts.add(cand_fwd_norm)
            # backward window
            start_bwd = max(0, idx - size + 1)
            end_bwd = idx + 1
            if start_bwd != start_fwd or end_bwd != end_fwd:
                cand_bwd = " ".join(sidx.sentences[start_bwd:end_bwd])
                cand_bwd_norm = normalize_for_match(cand_bwd)
                if cand_bwd_norm and cand_bwd_norm not in seen_texts:
                    candidates.append(cand_bwd_norm)
                    spans.append((start_bwd, end_bwd - 1))
                    seen_texts.add(cand_bwd_norm)

    if not candidates:
        return False, {"note": "no valid candidates after windowing"}

    cos = cosine_sim(q, candidates, encoder_name)
    fuzz = [partial_ratio(q, c) for c in candidates]
    jaccard5 = [jaccard_char_ngrams(q, c, 5) for c in candidates]

    best_i = int(np.argmax(cos))
    best_cand = candidates[best_i]
    best_cos = float(cos[best_i])
    best_fz = int(fuzz[best_i])
    best_jac = float(jaccard5[best_i])
    best_span = spans[best_i]

    bonus = best_literal_bonus(q, best_cand)

    # adaptive thresholds
    cos_thr = cos_thresh
    fz_thr = fuzzy_thresh
    if len(q) < 60:
        fz_thr = max(fz_thr, 92)
        cos_thr = max(cos_thr, 0.84)
    elif len(q) > 200:
        cos_thr = min(cos_thr, 0.78)

    ok_cos = (best_cos + bonus) >= cos_thr
    ok_fz = best_fz >= fz_thr
    ok_jac = best_jac >= 0.65

    if ok_cos or ok_fz or ok_jac:
        n_true = sum([ok_cos, ok_fz, ok_jac])
        parts = []
        if ok_cos:
            parts.append("cos")
        if ok_fz:
            parts.append("fuzzy")
        if ok_jac:
            parts.append("jaccard5")
        note = "+".join(parts) + f" (bonus={bonus:.02f})"
        return True, {
            "candidate": best_cand,
            "span": best_span,
            "cosine": best_cos,
            "bm25": float(scores[top_idx[0]]) if top_idx else None,
            "fuzzy": best_fz,
            "jaccard5": best_jac,
            "note": note,
        }
    else:
        # also surface 2nd-best if helpful
        sorted_idx = np.argsort(-cos)
        second_i = int(sorted_idx[1]) if len(candidates) > 1 else best_i
        return False, {
            "candidate": best_cand,
            "span": best_span,
            "cosine": best_cos,
            "bm25": float(scores[top_idx[0]]) if top_idx else None,
            "fuzzy": best_fz,
            "jaccard5": best_jac,
            "alt_idx": second_i if len(candidates) > 1 else None,
            "alt_cosine": float(cos[second_i]) if len(candidates) > 1 else None,
            "note": "below thresholds",
        }

# ------------------- Loading source indices -------------------

def load_sentences_for_row(row: pd.Series, texts_dir: Path) -> Optional[List[str]]:
    # Prefer TEI sentences
    tei_path = row.get("tei_path")
    if isinstance(tei_path, str) and tei_path:
        tp = Path(tei_path)
        if not tp.is_absolute():
            tp = texts_dir / tp.name
        if tp.exists():
            sents = read_tei_sentences(tp)
            if sents:
                return sents
    # Else use plain text
    text_path = row.get("source_text_path")
    if isinstance(text_path, str) and text_path:
        tp = Path(text_path)
        if not tp.is_absolute():
            tp = texts_dir / tp.name
        if tp.exists():
            txt = tp.read_text(encoding="utf-8", errors="ignore")
            return sent_split_regex(txt)
    return None

@lru_cache(maxsize=2048)
def index_for_sha(sha: str, texts_dir: Path) -> Optional[SourceIndex]:
    # This function will be provided with sha and will try to use TEI or TXT by sha file names
    if not sha:
        return None
    tei = texts_dir / f"{sha}.tei.xml"
    txt = texts_dir / f"{sha}.txt"
    sentences: Optional[List[str]] = None
    if tei.exists():
        sentences = read_tei_sentences(tei)
    if not sentences and txt.exists():
        sentences = sent_split_regex(txt.read_text(encoding="utf-8", errors="ignore"))
    if sentences:
        return build_bm25(sentences)
    return None

def source_filename_from_row(row: pd.Series) -> str:
    p = row.get("resolved_source_path")
    if isinstance(p, str) and p:
        return Path(p).name
    # fallback to the raw provided filename
    fn = row.get("file_name_raw")
    if isinstance(fn, str) and fn:
        return fn.strip()
    return "UNRESOLVED"

# ------------------- Main -------------------

def main():
    ap = argparse.ArgumentParser(description="Verify whether quotes appear in sources (BM25+semantic+fuzzy).")
    ap.add_argument("--parquet-dir", default=str(DEFAULT_PARQUET_DIR))
    ap.add_argument("--texts-dir", default=str(DEFAULT_TEXTS_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--encoder", default="all-MiniLM-L6-v2")
    ap.add_argument("--bm25-topk", type=int, default=20)
    ap.add_argument("--cos-thresh", type=float, default=0.82)
    ap.add_argument("--fuzzy-thresh", type=int, default=85)
    ap.add_argument("--summary-csv", action="store_true", help="Write verification_summary.csv with group summary")
    args = ap.parse_args()

    parquet_dir = Path(args.parquet_dir).resolve()
    texts_dir = Path(args.texts_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = parquet_dir / "pairs_raw.parquet"
    sources_path = parquet_dir / "sources.parquet"
    if not pairs_path.exists():
        print(f"[ERROR] {pairs_path} not found", file=sys.stderr)
        sys.exit(2)
    pairs = pd.read_parquet(pairs_path)
    # sources currently unused directly; sentence loading is by sha files
    if sources_path.exists():
        _ = pd.read_parquet(sources_path)

    # Bucket by group
    groups = sorted(pairs["group_id"].dropna().unique())
    encoder_name = args.encoder

    summary_rows = []
    for g in groups:
        gdf = pairs[pairs["group_id"] == g].copy()
        by_source: Dict[str, Dict] = {}

        for ridx, row in gdf.iterrows():
            qsrc_raw = normalize_text(row.get("quote_from_source", ""))
            if not qsrc_raw:
                continue
            if TRIVIAL_PAT.match(qsrc_raw.strip()):
                sfn = source_filename_from_row(row)
                bucket = by_source.setdefault(sfn, {"source_filename": sfn, "hits": 0, "misses": 0, "misses_detail": []})
                bucket["misses"] += 1
                bucket["misses_detail"].append({
                    "row_index": int(row.get("row_index", -1)),
                    "claimed_quote": qsrc_raw,
                    "notes": "non-verifiable (generic: figure/table/N.A.)"
                })
                continue

            sha = row.get("source_sha256")
            sidx = None
            if isinstance(sha, str) and sha:
                sidx = index_for_sha(sha, texts_dir)
            # Fallback: try to load from row-specific paths (rare, but helps when sha missing)
            if sidx is None:
                sents = load_sentences_for_row(row, texts_dir)
                if sents:
                    sidx = build_bm25(sents)

            sfn = source_filename_from_row(row)
            bucket = by_source.setdefault(sfn, {"source_filename": sfn, "hits": 0, "misses": 0, "misses_detail": []})

            if sidx is None or not sidx.sentences:
                bucket["misses"] += 1
                bucket["misses_detail"].append({
                    "row_index": int(row.get("row_index", -1)),
                    "claimed_quote": qsrc_raw,
                    "notes": "no text index available for source"
                })
                continue

            ok, info = verify_one_quote(qsrc_raw, sidx, encoder_name, args.bm25_topk, args.cos_thresh, args.fuzzy_thresh)
            if ok:
                bucket["hits"] += 1
            else:
                bucket["misses"] += 1
                bucket["misses_detail"].append({
                    "row_index": int(row.get("row_index", -1)),
                    "claimed_quote": qsrc_raw,
                    **info
                })

        total_hits = sum(b["hits"] for b in by_source.values()) if by_source else 0
        total_miss = sum(b["misses"] for b in by_source.values()) if by_source else 0
        denom = max(1, total_hits + total_miss)
        summary = {
            "group": g,
            "success_ratio": round(total_hits / denom, 4),
            "total": denom,
            "hits": total_hits,
            "misses": total_miss,
            "by_source": sorted(by_source.values(), key=lambda x: x["source_filename"].lower() if x["source_filename"] else ""),
            "params": {
                "encoder": encoder_name,
                "bm25_topk": args.bm25_topk,
                "cos_thresh": args.cos_thresh,
                "fuzzy_thresh": args.fuzzy_thresh,
            },
        }

        out_path = out_dir / f"Group_{g.replace('/', '_').replace(' ', '_')}.json"
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] wrote {out_path}")
        # Collect summary row for CSV
        summary_rows.append({
            "group": g,
            "success_ratio": round(total_hits / denom, 4),
            "total": denom,
            "hits": total_hits,
            "misses": total_miss,
            "filename": out_path.name,
        })

    # After all groups, optionally write summary CSV
    if getattr(args, "summary_csv", False):
        import csv
        csv_path = out_dir / "verification_summary.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["group", "success_ratio", "total", "hits", "misses", "filename"])
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
        print(f"[OK] wrote {csv_path}")

if __name__ == "__main__":
    main()