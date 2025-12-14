#!/usr/bin/env python3
"""
sniff_cherrypicking.py

Purpose
-------
Given a cleaned row-level citation dataset (`pairs_raw.parquet`) and per-document extracted text sidecars
(`<sha>.txt` produced by `extract_text.py`), this script performs a *cheap-by-default* “cherry-picking sniff”:

- For each row, treat either the student claim (`quote_from_report`) or the student's cited excerpt (`quote_from_source`)
  as the **query/hypothesis**.
- Retrieve nearby/relevant passages from the full source text.
- Flag passages that *look like* qualification, limitation, or contradiction (lexical cue terms).
- Flag candidates as “worth checking” for NLI, but does not run NLI itself.

What you get
------------
This produces *teacher-usable* artifacts (text included) plus a more complete machine-usable candidate table.
Candidate windows are flagged with `recommend_nli` if they look worth further review.

Inputs
------
- `{parquet_dir}/pairs_raw.parquet`
  Must contain: `source_sha256`, `quote_from_report`, `quote_from_source`.
  Recommended (if present): `cohort_id`, `group_id`, `resolved_source_path`.

- `{parquet_dir}/sources.parquet`
  Used to map `source_sha256 -> text_path` when available.

- `{texts_dir}/<source_sha256>.txt`
  Fallback when `sources.parquet` does not provide a `text_path`.

Outputs
-------
Written to `--out-dir` (default: `artifacts/cherrypicking/`):

- `cherrypicking_candidates.parquet`
  One row per candidate window (all candidates kept after BM25 + embedding rerank).

- `cherrypicking_summary.csv`
  Group-level diagnostics (counts + whether any likely-contradiction/limitation signals appear).

- `cherrypicking_teacher.csv`
  Top 3 candidate windows per (group_id, row_id), with short text fields for quick review.

- `cherrypicking_report.json`
  Nested JSON for UI use: group -> row -> claim/cited text + candidate windows.

Typical usage
-------------

  python sniff_cherrypicking.py \
      --parquet-dir artifacts/parquet \
      --texts-dir artifacts/text \
      --out-dir artifacts/cherrypicking \
      --query-field quote_from_report

Key parameters
--------------
- `--query-field {quote_from_report,quote_from_source}`
  Chooses what to treat as the query/hypothesis:
  - `quote_from_report`: the student's *claim/restatement*.
  - `quote_from_source`: the student's *claimed evidence excerpt*.

- `--config PATH.json`
  Override defaults (retrieval sizes, thresholds, cue term lists, models).
  See `DEFAULT_CONFIG` inside this script for all configurable keys.

Notes and limitations
---------------------
- Sentence splitting is heuristic and intentionally dependency-light; windowing is for retrieval, not for gold segmentation.
- Lexical cue hits are *signals*, not proof; use NLI or manual review for high-stakes decisions.
- This script is designed for triage: it surfaces “worth a look” passages near what the student cited/claimed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from rank_bm25 import BM25Okapi
from rapidfuzz.fuzz import partial_ratio

# sentence-transformers is already in your stack (verify_quotes.py)
from sentence_transformers import SentenceTransformer
import numpy as np




def make_candidate_id(
    cohort_id: str,
    group_id: str,
    row_id: int,
    source_sha256: str,
    window_index: int,
    embed_rank: int,
) -> str:
    """Stable id for a candidate window.

    Purpose: allow later stages (e.g., NLI-only reruns) to join back to candidates
    without depending on brittle multi-column merges.
    """
    key = f"{cohort_id}|{group_id}|{row_id}|{source_sha256}|{window_index}|{embed_rank}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


DEFAULT_CONFIG = {
    # Retrieval
    "bm25_topk": 30,
    "embed_rerank_topk": 10,
    "min_claim_chars": 20,

    # Which field to use as query: "quote_from_report" (default) or "quote_from_source"
    "query_field": "quote_from_report",  # or "quote_from_source"

    # Windowing
    "window_sentences": 2,
    "window_stride": 1,
    "max_source_chars": 2_000_000,  # guardrail against giant dumps

    # Lexical conditioning
    "contrast_markers": [
        "however", "but", "although", "though", "nevertheless", "nonetheless",
        "yet", "whereas", "on the other hand", "in contrast", "despite", "while"
    ],
    "limitation_markers": [
        "may", "might", "could", "suggest", "suggests", "suggested", "possibly",
        "likely", "unlikely", "uncertain", "limited", "limitation", "caution",
        "inconclusive", "preliminary", "assumption", "assume"
    ],
    "negation_markers": [
        "no ", "not ", "never", "none", "without", "fail to", "fails to", "cannot", "can't"
    ],

    # Heuristics: when to recommend NLI rerun
    "recommend_nli_if_any_contrast": True,
    "recommend_nli_if_limitation_hits": 2,
    "recommend_nli_if_negation_hits": 1,

    # Similarity thresholds (for ranking / pruning)
    "min_embed_cosine": 0.25,   # low-ish; cheap mode is exploratory
    "min_fuzzy_partial": 60,    # for optional “does candidate resemble cited quote?” checks

    # Models
    "embed_model": "all-MiniLM-L6-v2",
    "nli_model": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
}


def load_config(path: Optional[str]) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"--config not found: {p}")
        user_cfg = json.loads(p.read_text(encoding="utf-8"))
        cfg.update(user_cfg)
    return cfg


def normalize_text(s: str) -> str:
    s = s or ""
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def shorten(s: str, n: int = 280) -> str:
    s = normalize_text(s)
    if len(s) <= n:
        return s
    return s[: max(0, n - 1)] + "…"




def simple_sentence_split(text: str) -> List[str]:
    """
    Pure-Python sentence splitter (no native deps).

    Heuristic: split on [.?!] followed by whitespace, while trying not to split on
    common abbreviations and initials. This is not perfect, but it's fast and
    portable, and good enough for retrieval windowing.
    """
    text = normalize_text(text)
    if not text:
        return []

    # Normalize newlines to spaces
    t = re.sub(r"\s*\n+\s*", " ", text).strip()

    # Protect a small set of common abbreviations (extend via config later if needed)
    # Replace the dot with a placeholder so we don't split there.
    placeholder = "∯"
    abbr = [
        "e.g.", "i.e.", "et al.", "Fig.", "Figs.", "Dr.", "Prof.", "Mr.", "Ms.", "Mrs.",
        "vs.", "No.", "St.", "Inc.", "Ltd.", "Jr.", "Sr."
    ]
    for a in abbr:
        t = t.replace(a, a.replace(".", placeholder))

    # Protect single-letter initials like "A. B. Smith"
    t = re.sub(r"\b([A-Z])\.\s", rf"\1{placeholder} ", t)

    # Split on sentence end punctuation.
    parts = re.split(r"(?<=[.!?])\s+", t)
    sents = []
    for p in parts:
        p = p.replace(placeholder, ".")
        p = normalize_text(p)
        if p:
            sents.append(p)
    return sents


def make_windows(sentences: List[str], window_size: int, stride: int) -> List[str]:
    if window_size <= 0:
        return []
    out = []
    i = 0
    n = len(sentences)
    while i < n:
        w = sentences[i:i + window_size]
        if not w:
            break
        out.append(" ".join(w))
        i += max(1, stride)
        if i + window_size > n and i < n:
            # allow last partial window? no: keep consistent windowing
            break
    return out


def tokenize_for_bm25(s: str) -> List[str]:
    s = normalize_text(s).lower()
    # simple tokenization: split on non-word
    toks = re.split(r"[^a-z0-9]+", s)
    return [t for t in toks if t]


def count_markers(text: str, markers: List[str]) -> int:
    t = normalize_text(text).lower()
    hits = 0
    for m in markers:
        m2 = m.lower()
        if m2.endswith(" "):  # already spaced cue
            hits += t.count(m2)
        else:
            # word boundary-ish search
            hits += len(re.findall(rf"\b{re.escape(m2)}\b", t))
    return hits


@dataclass
class CandidateHit:
    candidate_id: str
    cohort_id: str
    group_id: str
    row_id: int
    source_sha256: str
    source_path: str
    report_quote: str
    source_quote: str
    claim: str
    cited_quote: str
    candidate_window: str
    bm25_rank: int
    bm25_score: float
    embed_rank: int
    embed_cosine: float
    contrast_hits: int
    limitation_hits: int
    negation_hits: int
    fuzzy_to_cited_quote: int
    recommend_nli: bool


def build_doc_index(source_text: str, cfg: dict) -> Tuple[List[str], BM25Okapi, List[List[str]]]:
    sents = simple_sentence_split(source_text)
    windows = make_windows(
        sents,
        window_size=int(cfg["window_sentences"]),
        stride=int(cfg["window_stride"]),
    )
    tokenized = [tokenize_for_bm25(w) for w in windows]
    bm25 = BM25Okapi(tokenized) if windows else BM25Okapi([["empty"]])
    return windows, bm25, tokenized


def embed_rerank(
    model: SentenceTransformer,
    claim: str,
    windows: List[str],
    bm25_scores: np.ndarray,
    cfg: dict
) -> List[Tuple[int, float]]:
    # Take top bm25 candidates then rerank by cosine
    topk = int(cfg["bm25_topk"])
    rerank_k = int(cfg["embed_rerank_topk"])

    if not windows:
        return []

    idxs = np.argsort(-bm25_scores)[:topk]
    cands = [windows[i] for i in idxs]

    claim_emb = model.encode([claim], normalize_embeddings=True)
    cand_emb = model.encode(cands, normalize_embeddings=True)
    sims = (cand_emb @ claim_emb[0]).astype(float)

    order = np.argsort(-sims)[:rerank_k]
    out = []
    for r, j in enumerate(order):
        out.append((int(idxs[j]), float(sims[j])))
    return out




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", required=True, help="Directory containing pairs_raw.parquet and sources.parquet")
    ap.add_argument("--texts-dir", required=True, help="Directory containing <sha>.txt sidecars")
    ap.add_argument("--out-dir", default="artifacts/cherrypicking", help="Output directory")
    ap.add_argument("--config", default=None, help="JSON config overriding defaults (cue terms, thresholds, models)")
    ap.add_argument("--limit-rows", type=int, default=0, help="For testing: process only first N rows (0=all)")
    ap.add_argument(
        "--query-field",
        default=None,
        choices=["quote_from_report", "quote_from_source"],
        help="Which pairs_raw column to use as the query/hypothesis for retrieval.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.query_field:
        cfg["query_field"] = args.query_field

    parquet_dir = Path(args.parquet_dir)
    texts_dir = Path(args.texts_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = parquet_dir / "pairs_raw.parquet"
    sources_path = parquet_dir / "sources.parquet"

    if not pairs_path.exists():
        print(f"[FAIL] missing {pairs_path}", file=sys.stderr)
        sys.exit(2)
    if not sources_path.exists():
        print(f"[FAIL] missing {sources_path}", file=sys.stderr)
        sys.exit(2)

    pairs = pd.read_parquet(pairs_path)
    sources = pd.read_parquet(sources_path)

    # Required columns
    query_field = str(cfg.get("query_field") or "quote_from_report")
    required_pairs = {"source_sha256", "quote_from_source", "quote_from_report", query_field}
    missing = required_pairs - set(pairs.columns)
    if missing:
        print(f"[FAIL] pairs_raw missing columns: {missing}", file=sys.stderr)
        sys.exit(2)

    # Map sha -> text_path (or fallback to texts_dir/<sha>.txt)
    sha_to_text = {}
    if "text_path" in sources.columns:
        for _, r in sources.iterrows():
            sha = str(r.get("source_sha256") or "")
            tp = r.get("text_path")
            if sha and isinstance(tp, str) and tp:
                sha_to_text[sha] = tp

    # Initialize embedding model once
    embed_model_name = cfg["embed_model"]
    embedder = SentenceTransformer(embed_model_name)


    # Cache per-doc indices (windows + bm25) to avoid recompute
    doc_cache: Dict[str, Tuple[List[str], BM25Okapi]] = {}

    hits: List[CandidateHit] = []
    n_rows = len(pairs)
    limit = int(args.limit_rows or 0)
    if limit > 0:
        pairs = pairs.head(limit)

    processed = 0
    skipped_no_sha = 0
    skipped_short_claim = 0
    skipped_missing_text = 0

    for idx, row in pairs.iterrows():
        sha = row.get("source_sha256")
        if sha is None or (isinstance(sha, float) and np.isnan(sha)):
            skipped_no_sha += 1
            continue
        sha = str(sha)

        report_quote = normalize_text(str(row.get("quote_from_report") or ""))
        source_quote = normalize_text(str(row.get("quote_from_source") or ""))
        query_text = report_quote if query_field == "quote_from_report" else source_quote
        cited = source_quote  # keep naming: 'cited' remains the claimed evidence string

        if len(query_text) < int(cfg["min_claim_chars"]):
            skipped_short_claim += 1
            continue

        cohort_id = str(row.get("cohort_id") or "")
        group_id = str(row.get("group_id") or row.get("group") or "")

        # Load source text
        tp = sha_to_text.get(sha)
        if not tp:
            # default location
            cand = texts_dir / f"{sha}.txt"
            tp = str(cand) if cand.exists() else ""
        if not tp or not Path(tp).exists():
            skipped_missing_text += 1
            continue

        src_path = str(row.get("resolved_source_path") or row.get("source_path") or "")
        source_text = Path(tp).read_text(encoding="utf-8", errors="ignore")
        if len(source_text) > int(cfg["max_source_chars"]):
            source_text = source_text[: int(cfg["max_source_chars"])]

        # Build/reuse doc index
        if sha not in doc_cache:
            windows, bm25, _tok = build_doc_index(source_text, cfg)
            doc_cache[sha] = (windows, bm25)

        windows, bm25 = doc_cache[sha]
        if not windows:
            continue

        scores = np.array(bm25.get_scores(tokenize_for_bm25(query_text)), dtype=float)
        reranked = embed_rerank(embedder, query_text, windows, scores, cfg)

        # Emit CandidateHit for top reranked windows
        for erank, (w_idx, cos) in enumerate(reranked):
            wtxt = windows[w_idx]
            if cos < float(cfg["min_embed_cosine"]):
                continue

            contrast_hits = count_markers(wtxt, cfg["contrast_markers"])
            limitation_hits = count_markers(wtxt, cfg["limitation_markers"])
            negation_hits = count_markers(wtxt, cfg["negation_markers"])
            fuzzy = int(partial_ratio(cited.lower(), wtxt.lower())) if cited else 0

            recommend = False
            if cfg["recommend_nli_if_any_contrast"] and contrast_hits > 0:
                recommend = True
            if limitation_hits >= int(cfg["recommend_nli_if_limitation_hits"]):
                recommend = True
            if negation_hits >= int(cfg["recommend_nli_if_negation_hits"]):
                recommend = True

            # bm25 rank among all windows (approx): use w_idx ordering by score
            bm25_rank = int(np.where(np.argsort(-scores) == w_idx)[0][0]) if len(scores) else -1

            cand_id = make_candidate_id(
                cohort_id=cohort_id,
                group_id=group_id,
                row_id=int(idx),
                source_sha256=sha,
                window_index=int(w_idx),
                embed_rank=int(erank),
            )

            hits.append(CandidateHit(
                candidate_id=cand_id,
                cohort_id=cohort_id,
                group_id=group_id,
                row_id=int(idx),
                source_sha256=sha,
                source_path=src_path,
                report_quote=report_quote,
                source_quote=source_quote,
                claim=query_text,
                cited_quote=cited,
                candidate_window=wtxt,
                bm25_rank=bm25_rank,
                bm25_score=float(scores[w_idx]),
                embed_rank=erank,
                embed_cosine=float(cos),
                contrast_hits=int(contrast_hits),
                limitation_hits=int(limitation_hits),
                negation_hits=int(negation_hits),
                fuzzy_to_cited_quote=int(fuzzy),
                recommend_nli=bool(recommend),
            ))

        processed += 1


    # Write outputs
    out_rows = []
    for h in hits:
        out_rows.append({
            "candidate_id": h.candidate_id,
            "cohort_id": h.cohort_id,
            "group_id": h.group_id,
            "row_id": h.row_id,
            "source_sha256": h.source_sha256,
            "source_path": h.source_path,
            "query_field": query_field,
            "query_text": h.claim,
            "quote_from_report": h.report_quote,
            "quote_from_source": h.source_quote,
            "candidate_window": h.candidate_window,
            "bm25_rank": h.bm25_rank,
            "bm25_score": h.bm25_score,
            "embed_rank": h.embed_rank,
            "embed_cosine": h.embed_cosine,
            "contrast_hits": h.contrast_hits,
            "limitation_hits": h.limitation_hits,
            "negation_hits": h.negation_hits,
            "fuzzy_to_cited_quote": h.fuzzy_to_cited_quote,
            "recommend_nli": h.recommend_nli,
        })

    df_out = pd.DataFrame(out_rows)
    cand_path = out_dir / "cherrypicking_candidates.parquet"
    df_out.to_parquet(cand_path, index=False)

    # Summary (diagnostics) + teacher-friendly exports
    summ_path = out_dir / "cherrypicking_summary.csv"
    teacher_path = out_dir / "cherrypicking_teacher.csv"
    report_path = out_dir / "cherrypicking_report.json"

    if len(df_out) == 0:
        summary = pd.DataFrame([{
            "groups": 0,
            "rows_processed": int(processed),
            "candidates": 0,
            "recommend_nli": 0,
            "skipped_no_sha": skipped_no_sha,
            "skipped_short_claim": skipped_short_claim,
            "skipped_missing_text": skipped_missing_text,
        }])
        summary.to_csv(summ_path, index=False)
        # still write empty teacher/report files for UI pipelines
        pd.DataFrame([]).to_csv(teacher_path, index=False)
        report_path.write_text(json.dumps({"groups": {}}, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        # Diagnostic summary: one row per group
        grp = df_out.groupby("group_id", dropna=False).agg(
            rows_processed=("row_id", "nunique"),
            candidates=("candidate_window", "count"),
            recommend_nli=("recommend_nli", "sum"),
            avg_cosine=("embed_cosine", "mean"),
            any_contrast=("contrast_hits", lambda s: int((s > 0).any())),
            any_negation=("negation_hits", lambda s: int((s > 0).any())),
            any_limitations=("limitation_hits", lambda s: int((s > 0).any())),
        ).reset_index()
        summary = grp.sort_values(["recommend_nli", "avg_cosine"], ascending=[False, False])
        summary.to_csv(summ_path, index=False)

        # Teacher view: keep only the most useful candidates per row
        # Priority: recommend_nli, then high cosine.
        df_teacher = df_out.copy()

        # Make shorter text fields for quick scanning
        df_teacher["claim_short"] = df_teacher["query_text"].apply(lambda s: shorten(str(s), 240))
        df_teacher["cited_short"] = df_teacher["quote_from_source"].apply(lambda s: shorten(str(s), 240))
        df_teacher["candidate_short"] = df_teacher["candidate_window"].apply(lambda s: shorten(str(s), 320))

        df_teacher = df_teacher.sort_values(
            ["group_id", "row_id", "recommend_nli", "embed_cosine", "bm25_score"],
            ascending=[True, True, False, False, False],
        )
        # top 3 candidates per row is usually enough for a teacher pass
        df_teacher = df_teacher.groupby(["group_id", "row_id"], dropna=False).head(3)

        teacher_cols = [
            "cohort_id",
            "group_id",
            "row_id",
            "candidate_id",
            "source_sha256",
            "source_path",
            "query_field",
            "claim_short",
            "cited_short",
            "candidate_short",
            "embed_cosine",
            "contrast_hits",
            "limitation_hits",
            "negation_hits",
            "recommend_nli",
        ]
        for c in teacher_cols:
            if c not in df_teacher.columns:
                df_teacher[c] = None
        import csv
        df_teacher[teacher_cols].to_csv(
            teacher_path,
            index=False,
            quoting=csv.QUOTE_ALL,
            escapechar="\\",
        )

        # JSON report: nested for UI (group -> row -> list of candidates)
        report: Dict[str, dict] = {"groups": {}}
        # Pull one row-level record per (group_id,row_id)
        base_cols = [
            "cohort_id",
            "group_id",
            "row_id",
            "source_sha256",
            "source_path",
            "query_field",
            "query_text",
            "quote_from_report",
            "quote_from_source",
        ]
        # Use teacher-filtered candidates for the report (keeps size sane)
        for (g, rid), sub in df_teacher.groupby(["group_id", "row_id"], dropna=False):
            gk = str(g)
            rk = str(int(rid))
            if gk not in report["groups"]:
                report["groups"][gk] = {"rows": {}}

            first = sub.iloc[0]
            row_obj = {
                "cohort_id": str(first.get("cohort_id") or ""),
                "row_id": int(rid),
                "source_sha256": str(first.get("source_sha256") or ""),
                "source_path": str(first.get("source_path") or ""),
                "query_field": str(first.get("query_field") or ""),
                "quote_from_report": str(first.get("quote_from_report") or ""),
                "quote_from_source": str(first.get("quote_from_source") or ""),
                "query_text": str(first.get("query_text") or ""),
                "candidates": [],
            }

            for _, r in sub.iterrows():
                row_obj["candidates"].append({
                    "candidate_id": str(r.get("candidate_id") or ""),
                    "candidate_window": str(r.get("candidate_window") or ""),
                    "embed_cosine": float(r.get("embed_cosine") or 0.0),
                    "bm25_score": float(r.get("bm25_score") or 0.0),
                    "contrast_hits": int(r.get("contrast_hits") or 0),
                    "limitation_hits": int(r.get("limitation_hits") or 0),
                    "negation_hits": int(r.get("negation_hits") or 0),
                    "recommend_nli": bool(r.get("recommend_nli") or False),
                })

            report["groups"][gk]["rows"][rk] = row_obj

        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Console guidance
    total_reco = int(df_out["recommend_nli"].sum()) if len(df_out) else 0
    print(f"=== Cherry-picking sniff (cheap) ===")
    print(f"[OK] processed_rows={processed} candidates={len(df_out)} recommend_nli={total_reco}")
    print(f"[SKIP] no_sha={skipped_no_sha} short_claim={skipped_short_claim} missing_text={skipped_missing_text}")
    print(f"[OUT] {cand_path}")
    print(f"[OUT] {summ_path}")
    print(f"[OUT] {teacher_path}")
    print(f"[OUT] {report_path}")

    if total_reco > 0:
        print("\nNLI looks potentially useful on flagged candidates.")
        print("Proposed rerun:")

        qf = f" --query-field {query_field}"
        lim = f" --limit-rows {args.limit_rows}" if args.limit_rows else ""

    print(
        "python nli_cherrypicking.py "
        f"--candidates-parquet {out_dir}/cherrypicking_candidates.parquet "
        f"--out-dir {out_dir} "
        f"--flagged-only "
        f"--nli-model \"{cfg['nli_model']}\""
    )


if __name__ == "__main__":
    main()