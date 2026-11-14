"""Evaluation: the baseline system plus the comparators the proposal requires."""
from __future__ import annotations

import csv
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

from ..query.bm25_search import Hit
from ..query.engine import SearchEngine
from ..query.router import DOCUMENT, SCENE
from .metrics import evaluate_run


def load_queries(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            qrels[row["query_id"]][row["item_id"]] = int(row["grade"])
    return dict(qrels)


def filename_search(conn: sqlite3.Connection, query: str, top_k: int) -> list[Hit]:
    """The trivial floor: substring match on the file path only.

    This is what a phone's file manager already gives you, and is the bar any
    real system has to clear.

    It must NOT touch `gold_subtype` -- that is the ground-truth label, and
    matching against it turns the floor into an oracle. An earlier version did,
    which inflated this row and made routing look less valuable than it is.
    """
    terms = [t for t in query.lower().split() if len(t) > 2]
    if not terms:
        return []
    clause = " OR ".join(["LOWER(path) LIKE ?"] * len(terms))
    params: list[str] = [f"%{t}%" for t in terms]
    rows = conn.execute(
        f"SELECT item_id FROM items WHERE {clause} LIMIT ?", (*params, top_k)
    ).fetchall()
    return [Hit(item_id=r["item_id"], score=1.0, branch="filename") for r in rows]


def build_systems(engine: SearchEngine, conn: sqlite3.Connection, depth: int) -> dict[str, Callable]:
    """The baseline plus its three required comparators.

    Each returns (ranked_ids, elapsed_ms, branch) so query latency is captured
    from the same pass the metrics come from -- no second timing run.
    """
    def _routed(q):
        r = engine.search(q, depth)
        return r.ranked_ids, r.elapsed_ms, r.route.branch

    def _forced(branch):
        def inner(q):
            r = engine.search(q, depth, force_branch=branch)
            return r.ranked_ids, r.elapsed_ms, branch
        return inner

    def _filename(q):
        start = time.perf_counter()
        hits = filename_search(conn, q, depth)
        return [h.item_id for h in hits], (time.perf_counter() - start) * 1000.0, "filename"

    return {
        # The baseline: router picks exactly one branch.
        "routed": _routed,
        # Single-model comparators -- do these make routing look worthwhile?
        "clip_only": _forced(SCENE),
        "bm25_only": _forced(DOCUMENT),
        # The trivial floor.
        "filename": _filename,
    }


def run_eval(engine: SearchEngine, conn: sqlite3.Connection, queries, qrels, k_values,
             top_k, depth: int | None = None):
    """Per-system metrics, split by query kind and by bucket, plus latencies.

    `depth` is the retrieval depth used for scoring and defaults to at least
    the largest relevant set in the qrels. R-precision needs the ranking to run
    R deep, and with 20 items per subtype a top_k of 10 would silently cap it
    at 0.5 -- reintroducing the very ceiling artefact the metric exists to
    avoid. Prefix metrics (p@1, recall@10, ...) are unaffected by the extra
    depth, so one deeper pass serves every metric.
    """
    depth = depth or retrieval_depth(qrels, top_k, k_values)
    systems = build_systems(engine, conn, depth)
    runs: dict[str, dict[str, list[str]]] = {}
    latencies: dict[str, list[tuple[str, float]]] = {}

    for name, fn in systems.items():
        run: dict[str, list[str]] = {}
        lat: list[tuple[str, float]] = []
        for q in queries:
            ranked, ms, branch = fn(q["text"])
            run[q["query_id"]] = ranked
            lat.append((branch, ms))
        runs[name] = run
        latencies[name] = lat

    return score_runs(runs, queries, qrels, k_values), runs, latencies, depth


def retrieval_depth(qrels, top_k: int, k_values) -> int:
    """Depth every system is scored at: deep enough for R-precision (see run_eval)."""
    max_r = max((len(v) for v in qrels.values()), default=0)
    return max(top_k, max_r, max(k_values))


def score_runs(runs: dict[str, dict[str, list[str]]], queries, qrels, k_values) -> dict[str, dict]:
    """Metrics per system: overall, by query kind, by bucket, and bucket macro-average.

    Split out of run_eval so rankings produced elsewhere (the on-device harness,
    scripts/evaluate_device.py) are scored by exactly the same code.
    """
    # Bucket comes from each query's gold_route: the document/scene split the
    # report asks for. Kind (doc_lookup/scene_search/ambiguous) is kept too.
    groups: dict[str, dict[str, list[str]]] = {"kind": defaultdict(list),
                                               "bucket": defaultdict(list)}
    for q in queries:
        groups["kind"][q["kind"]].append(q["query_id"])
        groups["bucket"][q["gold_route"]].append(q["query_id"])

    results: dict[str, dict] = {}
    for name, run in runs.items():
        results[name] = {"overall": evaluate_run(run, qrels, k_values)}

        for grouping in groups.values():
            for label, qids in grouping.items():
                sub_qrels = {qid: qrels[qid] for qid in qids if qid in qrels}
                if sub_qrels:
                    subset = {qid: run[qid] for qid in qids}
                    results[name][label] = evaluate_run(subset, sub_qrels, k_values)

        # Macro-average across buckets, stated explicitly: each bucket counts
        # once regardless of how many queries it holds. Micro-averaging would
        # silently weight whichever bucket has more queries.
        bucket_blocks = [results[name][b] for b in groups["bucket"] if b in results[name]]
        if bucket_blocks:
            keys = {k for b in bucket_blocks for k in b if k != "n_queries"}
            results[name]["bucket_macro"] = {
                k: sum(b.get(k, 0.0) for b in bucket_blocks) / len(bucket_blocks) for k in keys
            }
            results[name]["bucket_macro"]["n_queries"] = sum(
                b.get("n_queries", 0) for b in bucket_blocks)

    return results


def latency_report(latencies: dict[str, list[tuple[str, float]]]) -> dict[str, dict]:
    """Mean/p50/p95 per system, and per branch within the routed system.

    The two branches are bimodal -- BM25 is a SQL query, CLIP needs a text
    encoder forward pass -- so a single mean hides the distribution that the
    on-device latency claim actually rests on.
    """
    def stats(xs: list[float]) -> dict[str, float]:
        if not xs:
            return {}
        s = sorted(xs)
        pick = lambda p: s[min(len(s) - 1, int(round(p * (len(s) - 1))))]  # noqa: E731
        return {"n": float(len(s)), "mean_ms": sum(s) / len(s), "p50_ms": pick(0.50),
                "p95_ms": pick(0.95), "max_ms": s[-1]}

    out: dict[str, dict] = {}
    for name, rows in latencies.items():
        out[name] = {"all": stats([ms for _, ms in rows])}
        by_branch: dict[str, list[float]] = defaultdict(list)
        for branch, ms in rows:
            by_branch[branch].append(ms)
        for branch, xs in by_branch.items():
            out[name][branch] = stats(xs)
    return out


def doctype_benchmark(conn: sqlite3.Connection) -> dict:
    """Accuracy of `doctype.py` -- gold_subtype vs the tagged doc_type.

    This module drives the FTS column weighted 5x against body text, which the
    design notes credit for making document lookup reliable, and it has never
    had a benchmark. Per-class P/R/F1 plus the rule that fired, so a
    misclassification is attributable to a specific rule rather than to "OCR".
    """
    rows = conn.execute(
        "SELECT i.item_id, i.gold_subtype, i.origin, f.doc_type, f.doc_type_rule "
        "FROM items i JOIN doc_fields f ON f.item_id = i.item_id "
        "WHERE i.gold_bucket = 'document'"
    ).fetchall()
    if not rows:
        return {}

    pairs = [(r["gold_subtype"], r["doc_type"] or "unknown") for r in rows]
    labels = sorted({g for g, _ in pairs} | {p for _, p in pairs})
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    for g, p in pairs:
        confusion[(g, p)] += 1

    per_class: dict[str, dict[str, float]] = {}
    for lab in labels:
        tp = confusion[(lab, lab)]
        fp = sum(c for (g, p), c in confusion.items() if p == lab and g != lab)
        fn = sum(c for (g, p), c in confusion.items() if g == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        support = sum(c for (g, _), c in confusion.items() if g == lab)
        if support or tp or fp:
            per_class[lab] = {"precision": prec, "recall": rec, "f1": f1,
                              "support": float(support)}

    correct = sum(c for (g, p), c in confusion.items() if g == p)
    rule_counts: dict[str, int] = defaultdict(int)
    for r in rows:
        rule_counts[r["doc_type_rule"] or "(none)"] += 1

    # Synthetic documents are rendered text, so they are the easy half; the
    # fetched ones are photographs and scans. Splitting keeps the headline
    # number honest.
    by_origin: dict[str, dict[str, float]] = {}
    for origin in {r["origin"] for r in rows}:
        sub = [r for r in rows if r["origin"] == origin]
        hit = sum(1 for r in sub if r["gold_subtype"] == (r["doc_type"] or "unknown"))
        by_origin[origin] = {"n": float(len(sub)), "accuracy": hit / len(sub)}

    return {"n": len(rows), "accuracy": correct / len(rows), "per_class": per_class,
            "confusion": dict(confusion), "rules": dict(rule_counts), "by_origin": by_origin}


def routing_accuracy(engine: SearchEngine, queries) -> tuple[float, list[tuple[str, str, str]]]:
    """Router accuracy against each query's gold_route, with the misroutes."""
    wrong: list[tuple[str, str, str]] = []
    correct = 0
    for q in queries:
        got = engine.router.route(q["text"]).branch
        if got == q["gold_route"]:
            correct += 1
        else:
            wrong.append((q["query_id"], q["gold_route"], got))
    return (correct / len(queries) if queries else 0.0), wrong


def classifier_confusion(conn: sqlite3.Connection) -> dict[tuple[str, str], int]:
    rows = conn.execute(
        "SELECT gold_bucket, bucket_pred, COUNT(*) c FROM items "
        "WHERE bucket_pred IS NOT NULL GROUP BY 1,2"
    ).fetchall()
    return {(r["gold_bucket"], r["bucket_pred"]): r["c"] for r in rows}
