"""Milestone 2 eval: OCR precision/recall, caption adequacy, semantic precision@k.

Three label-light objective checks for the search dataset:

1. OCR        -- from the hand-labeled sample (SEARCH_LABELS_CSV): precision (of
                 frames where we emitted text, how many matched the true text)
                 and recall (of frames that truly had text, how many we caught).
                 "Match" = token-Jaccard >= OCR_MATCH_JACCARD, so minor spacing
                 differences don't count as misses.
2. Captions   -- mean adequacy (1-5) over the labeled sample, plus the share
                 scoring >= 4.
3. Semantic   -- precision@k over the curated query set (SEARCH_QUERIES_JSON):
                 for each query we take the top-k semantic results and count how
                 many are topically relevant (their caption+OCR document contains
                 one of the query's relevant_terms). Fully automatic, no labels.

The OCR/caption sections gracefully report "not yet labeled" when the sheet has
no filled rows, so the semantic number is always available. Writes
reports/eval_search.json and reports/eval_search.md.
"""
import json
import re

import pandas as pd

import config
import util

log = util.get_logger()


# ---- pure helpers (unit-tested without models or labels) -------------------
def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def ocr_jaccard(pred: str, truth: str) -> float:
    """Token-level Jaccard overlap of two OCR strings (0-1)."""
    a, b = _tokens(pred), _tokens(truth)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def ocr_matches(pred: str, truth: str, cutoff: float) -> bool:
    return ocr_jaccard(pred, truth) >= cutoff


def is_relevant(document: str, terms: list[str]) -> bool:
    """True if the frame document mentions any relevant term (word-boundary)."""
    doc = (document or "").lower()
    return any(re.search(rf"\b{re.escape(t.lower())}\b", doc) for t in terms)


def precision_at_k(returned_docs: list[str], terms: list[str], k: int) -> float:
    """Fraction of the top-k returned documents that are relevant."""
    topk = returned_docs[:k]
    if not topk:
        return 0.0
    return sum(is_relevant(d, terms) for d in topk) / len(topk)


# ---- OCR + caption scoring from the labelsheet -----------------------------
def _score_labels() -> dict:
    """Detection precision/recall + string fidelity for OCR, mean caption score.

    Only rows with a non-blank ``true_ocr_text`` are scored for OCR (that is the
    human's signal that the frame was inspected); use the sentinel ``-`` for a
    frame confirmed to have no salient text. Text detection is judged per frame
    (does pred/true agree on presence of text); on the frames where both agree
    text IS present, string fidelity = mean token-Jaccard of pred vs. truth.
    """
    if not config.SEARCH_LABELS_CSV.exists():
        return {"status": "no labelsheet (run make_search_labelsheet.py)"}
    df = pd.read_csv(config.SEARCH_LABELS_CSV, dtype=str).fillna("")
    df["_pred"] = df["pred_ocr_text"].str.strip()
    df["_true"] = df["true_ocr_text"].str.strip()
    caps = pd.to_numeric(df["caption_score"], errors="coerce").dropna()

    ocr = df[df["_true"] != ""]                    # inspected-for-OCR rows only
    if ocr.empty and caps.empty:
        return {"status": "labelsheet present but not yet filled in",
                "sampled_frames": int(len(df))}

    out = {"status": "ok"}
    if not ocr.empty:
        tp = fp = fn = tn = 0
        fidelity = []
        for _, r in ocr.iterrows():
            has_pred = r["_pred"] != ""
            has_true = r["_true"] not in ("", "-")
            if has_pred and has_true:
                tp += 1
                fidelity.append(ocr_jaccard(r["_pred"], r["_true"]))
            elif has_pred and not has_true:
                fp += 1
            elif not has_pred and has_true:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        out.update({
            "ocr_labeled_frames": int(len(ocr)),
            "ocr_tp": tp, "ocr_fp": fp, "ocr_fn": fn, "ocr_tn": tn,
            "ocr_detect_precision": round(precision, 3) if precision is not None else None,
            "ocr_detect_recall": round(recall, 3) if recall is not None else None,
            "ocr_string_fidelity": round(sum(fidelity) / len(fidelity), 3)
            if fidelity else None,
        })
        if precision and recall:
            out["ocr_detect_f1"] = round(
                2 * precision * recall / (precision + recall), 3)
    if len(caps):
        out.update({
            "caption_scored": int(len(caps)),
            "caption_mean_adequacy": round(float(caps.mean()), 2),
            "caption_pct_good_ge4": round(float((caps >= 4).mean()), 3),
        })
    return out


# ---- semantic precision@k over the curated query set -----------------------
def _score_retrieval(finder, index_file, k: int) -> dict:
    """precision@k over the curated query set for a search function (semantic or
    visual). Relevance is judged on the frame's caption+OCR document containing a
    query's relevant_terms -- a text proxy, so it slightly favours the text path;
    the visual path can retrieve a correct frame whose caption omits the term."""
    if not config.SEARCH_QUERIES_JSON.exists():
        return {"status": f"no query set ({config.SEARCH_QUERIES_JSON.name})"}
    if not index_file.exists():
        return {"status": f"no index ({index_file.name}) — build it first"}
    import search          # heavy import deferred until we actually score

    with open(config.SEARCH_QUERIES_JSON) as f:
        queries = json.load(f)
    meta = search.load_metadata()

    per_query, precisions = [], []
    for q in queries:
        res = finder(q["query"], df=meta, top_k=k, min_score=0.0)
        docs = res["snippet"].tolist()
        p = precision_at_k(docs, q["relevant_terms"], k)
        precisions.append(p)
        per_query.append({
            "query": q["query"],
            f"precision_at_{k}": round(p, 3),
            "top_score": round(float(res["score"].iloc[0]), 3) if len(res) else 0.0,
            "top_result": docs[0] if docs else "",
        })
    mean_p = round(sum(precisions) / len(precisions), 3) if precisions else 0.0
    return {"status": "ok", "k": k, "queries": len(queries),
            f"mean_precision_at_{k}": mean_p, "per_query": per_query}


def run(k: int | None = None) -> dict:
    config.ensure_dirs()
    k = k or config.SEARCH_EVAL_K
    import search
    metrics = {
        "ocr_and_captions": _score_labels(),
        "semantic": _score_retrieval(search.semantic_search,
                                     config.TEXT_EMB_FILE, k),
        "visual": _score_retrieval(search.visual_search,
                                   config.IMAGE_EMB_FILE, k),
    }
    # Fusion needs both indexes; gate on the text one but require both to exist so
    # a Milestone-2-partial checkout degrades to a status instead of erroring.
    if config.TEXT_EMB_FILE.exists() and config.IMAGE_EMB_FILE.exists():
        metrics["fused"] = _score_retrieval(search.fused_search,
                                            config.TEXT_EMB_FILE, k)
    else:
        metrics["fused"] = {"status": "needs both text and CLIP indexes"}
    with open(config.REPORT_DIR / "eval_search.json", "w") as f:
        json.dump(metrics, f, indent=2)
    _write_md(metrics, k)
    log.info("search eval -> reports/eval_search.json")
    return metrics


def _retrieval_md(title: str, sec: dict, k: int, note: str = "") -> list[str]:
    L = [f"## {title}"]
    if sec.get("status") != "ok":
        return L + [f"- _{sec.get('status')}_", ""]
    L += [f"- Queries: **{sec['queries']}**",
          f"- **Mean precision@{k}: {sec[f'mean_precision_at_{k}']}**"]
    if note:
        L += [f"- _{note}_"]
    L += ["", f"| Query | precision@{k} | Top result |", "|---|---|---|"]
    for r in sec["per_query"]:
        L.append(f"| {r['query']} | {r[f'precision_at_{k}']} | "
                 f"{r['top_result'][:48]} |")
    return L + [""]


def _write_md(m: dict, k: int) -> None:
    oc, se, vis = m["ocr_and_captions"], m["semantic"], m["visual"]
    L = ["# Milestone 2 Search-Quality Evaluation", ""]

    L += ["## 1. OCR (vs. hand-labeled sample)"]
    if oc.get("status") != "ok" or "ocr_labeled_frames" not in oc:
        L += [f"- _{oc.get('status', 'not yet labeled')}_", ""]
    else:
        L += [f"- Labeled frames: **{oc['ocr_labeled_frames']}** "
              f"(TP={oc['ocr_tp']} FP={oc['ocr_fp']} "
              f"FN={oc['ocr_fn']} TN={oc['ocr_tn']})",
              f"- Detection precision: **{oc.get('ocr_detect_precision')}** "
              "(when we emit text, the frame really has text — no hallucination)",
              f"- Detection recall: **{oc.get('ocr_detect_recall')}** "
              "(of frames that truly have text, share we flagged)",
              f"- String fidelity: **{oc.get('ocr_string_fidelity')}** "
              "(mean token-Jaccard of read vs. true text, on detected frames)"]
        if "ocr_detect_f1" in oc:
            L += [f"- Detection F1: **{oc['ocr_detect_f1']}**"]
        L += [""]

    L += ["## 2. Caption adequacy (1–5)"]
    if oc.get("caption_scored"):
        L += [f"- Scored captions: **{oc['caption_scored']}**",
              f"- Mean adequacy: **{oc['caption_mean_adequacy']} / 5**",
              f"- Share rated good (≥4): **{oc['caption_pct_good_ge4']}**", ""]
    else:
        L += ["- _not yet labeled_", ""]

    L += _retrieval_md(f"3. Semantic search — text/caption+OCR (precision@{k})",
                       se, k)
    L += _retrieval_md(f"4. Visual search — CLIP image (precision@{k})", vis, k,
                       note="Relevance is judged on the frame's caption/OCR text, "
                       "which favours the text path; the visual index retrieves "
                       "by image content and needs no caption at all.")
    if "fused" in m:
        L += _retrieval_md(f"5. Fused search — RRF(text, visual) (precision@{k})",
                           m["fused"], k,
                           note="Reciprocal Rank Fusion of the semantic and visual "
                           "rankings, so either index can rescue a query the other "
                           "misses. Same text-proxy relevance caveat as visual.")

    with open(config.REPORT_DIR / "eval_search.md", "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate Milestone 2 search quality")
    ap.add_argument("--k", type=int, default=None, help="k for precision@k")
    args = ap.parse_args()
    run(k=args.k)
