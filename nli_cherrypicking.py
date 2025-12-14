#!/usr/bin/env python3
"""
nli_cherrypicking.py — Stage-2 NLI pass for cherry-picking candidates

Purpose
- Run an expensive Natural Language Inference (NLI) model only on precomputed candidates
  produced by sniff_cherrypicking.py, without recomputing BM25/embeddings/candidate selection.

Inputs
- artifacts/cherrypicking/cherrypicking_candidates.parquet
  Must contain at least:
    - recommend_nli (bool-like)
    - query_text (str)
    - candidate_window (str)
  Strongly recommended:
    - candidate_id (stable join key)
    - cohort_id, group_id, source_sha256, pair_id (or row id), cosine, bm25_rank

Outputs (written to --out-dir)
- nli_results.parquet            (NLI-scored rows only, keyed by candidate_id)
- nli_results.csv                (same as parquet, for quick inspection)
- nli_summary.csv                (group-level counts + contradiction rates)
- nli_results.json               (teacher-friendly structured JSON)

Usage
  # NLI only on flagged candidates (recommended)
  python nli_cherrypicking.py --candidates-parquet artifacts/cherrypicking/cherrypicking_candidates.parquet \
      --out-dir artifacts/cherrypicking \
      --flagged-only \
      --nli-model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

  # NLI on all candidates
  python nli_cherrypicking.py --candidates-parquet artifacts/cherrypicking/cherrypicking_candidates.parquet \
      --out-dir artifacts/cherrypicking \
      --all \
      --nli-model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd

# transformers/torch are already in your stack
from transformers import pipeline


def _boolish(x) -> bool:
    if x is None:
        return False
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    return s in ("1", "true", "t", "yes", "y")


def load_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Normalize expected columns from sniff_cherrypicking.py
    if "query_text" not in df.columns:
        raise SystemExit(f"[ERROR] Missing required column in {path}: query_text")

    # Candidate text can be stored as candidate_window (preferred) or candidate_text (legacy)
    if "candidate_window" in df.columns and "candidate_text" not in df.columns:
        df["candidate_text"] = df["candidate_window"]
    if "candidate_text" not in df.columns:
        raise SystemExit(f"[ERROR] Missing required column in {path}: candidate_window (or candidate_text)")

    # Flag column can be recommend_nli (preferred) or recommended_nli (legacy)
    if "recommend_nli" in df.columns and "recommended_nli" not in df.columns:
        df["recommended_nli"] = df["recommend_nli"]
    if "recommended_nli" not in df.columns:
        # If candidates lack it, default False (user can still run --all)
        df["recommended_nli"] = False

    # Strongly preferred join key
    if "candidate_id" not in df.columns:
        df["candidate_id"] = None

    # Ensure string dtype-ish
    df["query_text"] = df["query_text"].astype(str)
    df["candidate_text"] = df["candidate_text"].astype(str)
    return df


def run_nli(
    df: pd.DataFrame,
    model_name: str,
    device: str | None,
    batch_size: int,
    max_length: int,
    hypothesis_template: str,
) -> pd.DataFrame:
    """
    Uses a zero-shot-classification style NLI wrapper OR text-classification pipeline
    depending on model head. For mnli-style models, text-classification with premise/hypothesis
    pairs is typically: entailment / neutral / contradiction.

    We encode as: premise=candidate_text (from source), hypothesis=query_text (student claim).
    If you want the reverse, swap them.
    """
    # Pipeline will auto-pick device if you pass device=0/-1. On Apple silicon, MPS works if available.
    # transformers pipeline 'device' expects int: 0 for GPU/MPS, -1 for CPU.
    # We'll map:
    #   device="cpu" -> -1
    #   device="mps" -> 0
    dev = -1
    if device:
        d = device.strip().lower()
        if d in ("mps", "gpu", "cuda"):
            dev = 0
        elif d in ("cpu",):
            dev = -1

    clf = pipeline(
        "text-classification",
        model=model_name,
        device=dev,
        truncation=True,
        max_length=max_length,
        top_k=None,
    )

    labels = {"entailment", "neutral", "contradiction"}

    # Prepare batched inputs as list of dicts with text pairs.
    # Many MNLI heads accept (premise, hypothesis) via {"text":..., "text_pair":...}
    inputs = [{"text": prem, "text_pair": hyp} for prem, hyp in zip(df["candidate_text"], df["query_text"])]

    out_label: List[str] = []
    out_score: List[float] = []
    out_ent: List[float] = []
    out_neu: List[float] = []
    out_con: List[float] = []

    # Batch manually for predictable memory use
    n = len(inputs)
    for i in range(0, n, batch_size):
        batch = inputs[i : i + batch_size]
        preds = clf(batch)

        # preds: list of list[{"label":..., "score":...}] or list[{"label":..., "score":...}]
        for p in preds:
            if isinstance(p, dict):
                # Already top-1
                lab = p.get("label", "")
                sc = float(p.get("score", 0.0))
                dist = {lab: sc}
            else:
                # Distribution
                dist = {d["label"]: float(d["score"]) for d in p}

                # Some models label like "LABEL_0" etc; try to map common MNLI label sets
                # If the model outputs plain entailment/neutral/contradiction, good.
                # If it outputs FEVER labels, they may already be text.
                # Otherwise we fall back to choosing max label as-is.
            # Normalize label names
            # Common variants: "ENTAILMENT", "NEUTRAL", "CONTRADICTION"
            norm = {}
            for k, v in dist.items():
                kk = str(k).strip().lower()
                if "entail" in kk:
                    norm["entailment"] = v
                elif "contrad" in kk:
                    norm["contradiction"] = v
                elif "neutral" in kk:
                    norm["neutral"] = v
                else:
                    # keep unknown labels
                    norm[k] = v

            # If we have a full MNLI distribution, compute top label among the 3
            if labels.issubset(set(norm.keys())):
                ent = float(norm.get("entailment", 0.0))
                neu = float(norm.get("neutral", 0.0))
                con = float(norm.get("contradiction", 0.0))
                best = max([("entailment", ent), ("neutral", neu), ("contradiction", con)], key=lambda x: x[1])
                out_label.append(best[0])
                out_score.append(best[1])
                out_ent.append(ent)
                out_neu.append(neu)
                out_con.append(con)
            else:
                # Unknown label space: pick max
                best = max(norm.items(), key=lambda x: x[1])
                out_label.append(str(best[0]))
                out_score.append(float(best[1]))
                out_ent.append(float("nan"))
                out_neu.append(float("nan"))
                out_con.append(float("nan"))

    df = df.copy()
    df["nli_label"] = out_label
    df["nli_score"] = out_score
    df["nli_entailment"] = out_ent
    df["nli_neutral"] = out_neu
    df["nli_contradiction"] = out_con
    return df


def make_teacher_report(df: pd.DataFrame, topk_per_pair: int) -> Dict[str, Any]:
    """
    Teacher-facing JSON: grouped by group_id, then by pair_id (or fallback),
    with top candidate contradictions highlighted.
    """
    # Determine identity field
    pair_key = "pair_id" if "pair_id" in df.columns else ("row_index" if "row_index" in df.columns else None)

    report: Dict[str, Any] = {"groups": []}

    # Group by group_id
    gcol = "group_id" if "group_id" in df.columns else ("group" if "group" in df.columns else None)
    if gcol is None:
        gcol = "_group"
        df = df.copy()
        df[gcol] = "unknown"

    for gname, gdf in df.groupby(gcol, dropna=False):
        grp: Dict[str, Any] = {
            "group_id": str(gname),
            "rows": [],
        }
        if pair_key:
            for pid, pdf in gdf.groupby(pair_key, dropna=False):
                # Rank candidates: contradiction first, then high contradiction prob/score, then cosine if present
                tmp = pdf.copy()
                if "nli_contradiction" in tmp.columns and tmp["nli_contradiction"].notna().any():
                    tmp["_rank"] = tmp["nli_contradiction"].fillna(-1.0)
                else:
                    tmp["_rank"] = tmp["nli_score"].fillna(-1.0)
                # Prefer contradictions
                tmp["_is_con"] = (tmp["nli_label"] == "contradiction").astype(int)
                # Optional cosine tie-break
                if "cosine" in tmp.columns:
                    tmp["_cos"] = pd.to_numeric(tmp["cosine"], errors="coerce").fillna(-1.0)
                else:
                    tmp["_cos"] = -1.0

                tmp = tmp.sort_values(by=["_is_con", "_rank", "_cos"], ascending=False).head(topk_per_pair)

                row_entry = {
                    "pair_id": str(pid),
                    "query_text": str(tmp["query_text"].iloc[0]) if len(tmp) else "",
                    "source_sha256": str(tmp["source_sha256"].iloc[0]) if "source_sha256" in tmp.columns and len(tmp) else None,
                    "candidates": [],
                }
                for _, r in tmp.iterrows():
                    row_entry["candidates"].append(
                        {
                            "candidate_id": r.get("candidate_id") if "candidate_id" in tmp.columns else None,
                            "nli_label": r.get("nli_label"),
                            "nli_score": float(r.get("nli_score")) if r.get("nli_score") is not None and not pd.isna(r.get("nli_score")) else None,
                            "nli_contradiction": float(r.get("nli_contradiction")) if "nli_contradiction" in tmp.columns and not pd.isna(r.get("nli_contradiction")) else None,
                            "cosine": float(r.get("cosine")) if "cosine" in tmp.columns and not pd.isna(r.get("cosine")) else None,
                            "bm25_rank": int(r.get("bm25_rank")) if "bm25_rank" in tmp.columns and not pd.isna(r.get("bm25_rank")) else None,
                            "candidate_text": r.get("candidate_text"),
                        }
                    )
                grp["rows"].append(row_entry)
        else:
            # No stable per-row key; dump top contradictions overall
            tmp = gdf[gdf["nli_label"] == "contradiction"].copy()
            tmp = tmp.sort_values(by=["nli_score"], ascending=False).head(50)
            grp["rows"].append(
                {
                    "pair_id": None,
                    "query_text": None,
                    "note": "No pair_id/row_index column found; showing top contradictions overall.",
                    "candidates": tmp[["query_text", "candidate_text", "nli_label", "nli_score"]].to_dict(orient="records"),
                }
            )

        report["groups"].append(grp)

    return report


def main():
    ap = argparse.ArgumentParser(description="Stage-2 NLI over cherry-picking candidates (reads sniff outputs)")
    grp_in = ap.add_mutually_exclusive_group(required=True)
    grp_in.add_argument("--candidates-parquet", help="Candidates parquet from sniff_cherrypicking.py")
    grp_in.add_argument("--in-parquet", help="(alias) Candidates parquet from sniff_cherrypicking.py")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--flagged-only", action="store_true", help="Run NLI only on recommend_nli/recommended_nli==True (default)")
    ap.add_argument("--all", action="store_true", help="Run NLI on all candidates in the parquet")
    ap.add_argument("--nli-model", default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    ap.add_argument("--device", default=None, help="cpu or mps (default: auto)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=384, help="Token truncation length for NLI model")
    ap.add_argument("--topk-per-pair", type=int, default=3, help="How many top contradictions/alerts per row in JSON report")
    args = ap.parse_args()

    in_path = Path((args.candidates_parquet or args.in_parquet).strip())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_candidates(in_path)

    # Default: flagged-only unless --all is explicitly set
    flagged_only = True
    if args.all:
        flagged_only = False
    if args.flagged_only:
        flagged_only = True

    if flagged_only:
        df_run = df[df["recommended_nli"].apply(_boolish)].copy()
    else:
        df_run = df.copy()

    if len(df_run) == 0:
        print("[OK] No rows selected for NLI (0 candidates). Nothing to do.")
        return

    print(f"[INFO] Running NLI on candidates={len(df_run)} model={args.nli_model}")

    df_run = run_nli(
        df_run,
        model_name=args.nli_model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        hypothesis_template="{}",
    )

    # NLI outputs: (1) scored rows only (preferred), (2) merged convenience parquet
    nli_cols = ["nli_label", "nli_score", "nli_entailment", "nli_neutral", "nli_contradiction"]

    # (1) scored-only results keyed by candidate_id when available
    df_scored = df_run.copy()
    out_scored_parquet = out_dir / "nli_results.parquet"
    df_scored.to_parquet(out_scored_parquet, index=False)
    print(f"[OUT] {out_scored_parquet}")

    out_scored_csv = out_dir / "nli_results.csv"
    df_scored.to_csv(out_scored_csv, index=False)
    print(f"[OUT] {out_scored_csv}")

    # (2) merged convenience file (preserve non-run rows with NaNs)
    if "candidate_id" in df.columns and df["candidate_id"].notna().any() and "candidate_id" in df_run.columns:
        df_out = df.copy()
        # Ensure base columns exist
        for c in nli_cols:
            if c not in df_out.columns:
                df_out[c] = pd.NA

        # Merge NLI columns from the scored subset. Use a suffix, then coalesce.
        df_out = df_out.merge(
            df_run[["candidate_id"] + nli_cols],
            on="candidate_id",
            how="left",
            suffixes=("", "_nli"),
        )
        for c in nli_cols:
            cn = f"{c}_nli"
            if cn in df_out.columns:
                # Prefer newly computed values where present
                df_out[c] = df_out[cn].combine_first(df_out[c])
                df_out.drop(columns=[cn], inplace=True)
    else:
        # Fallback: index alignment (works if df_run is a filtered view preserving original index)
        df_out = df.copy()
        for c in nli_cols:
            if c not in df_out.columns:
                df_out[c] = pd.NA
        for c in nli_cols:
            df_out.loc[df_run.index, c] = df_run[c]

    out_merged_parquet = out_dir / "cherrypicking_candidates_nli.parquet"
    df_out.to_parquet(out_merged_parquet, index=False)
    print(f"[OUT] {out_merged_parquet}")

    # Summary CSV: contradiction rates by group
    gcol = "group_id" if "group_id" in df_out.columns else ("group" if "group" in df_out.columns else None)
    if gcol is None:
        gcol = "_group"
        df_out[gcol] = "unknown"

    def _is_con(x): return str(x).strip().lower() == "contradiction"
    summ = (
        df_out.assign(_is_con=df_out["nli_label"].apply(_is_con))
             .groupby(gcol, dropna=False)
             .agg(
                 candidates=("candidate_text", "count"),
                 nli_scored=("nli_label", lambda s: s.notna().sum()),
                 contradictions=("nli_label", lambda s: (s.astype(str).str.lower() == "contradiction").sum()),
             )
             .reset_index()
    )
    summ["contradiction_rate"] = summ.apply(
        lambda r: (r["contradictions"] / r["nli_scored"]) if r["nli_scored"] else 0.0, axis=1
    )

    out_csv = out_dir / "nli_summary.csv"
    summ.to_csv(out_csv, index=False)
    print(f"[OUT] {out_csv}")

    # Teacher JSON
    report = make_teacher_report(df_out[df_out["nli_label"].notna()].copy(), topk_per_pair=args.topk_per_pair)
    out_json = out_dir / "nli_results.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OUT] {out_json}")


if __name__ == "__main__":
    main()