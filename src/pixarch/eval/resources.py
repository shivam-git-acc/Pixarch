"""Storage, memory and ingestion-throughput accounting.

These are the on-device feasibility numbers: what the index costs per item,
what the models cost in RAM, and how fast ingestion runs. The extrapolation to
a realistic gallery size is the figure the whole "it fits on a phone" claim
rests on, and nothing else in the codebase computes it.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

# A plausible phone gallery. Used only to project measured per-item costs; the
# projection is linear and stated as such.
GALLERY_SIZE = 10_000


def index_breakdown(conn: sqlite3.Connection, db_path: Path) -> dict:
    """Per-table byte usage via dbstat, plus per-item cost split by bucket.

    dbstat is a virtual table compiled into SQLite that reports the real page
    usage per b-tree, so this is measured rather than estimated. FTS5 spreads
    across several shadow tables (doc_fts_data, doc_fts_idx, ...), which are
    folded together so the inverted index shows as one line.
    """
    total_bytes = db_path.stat().st_size

    per_table: dict[str, int] = {}
    try:
        for name, size in conn.execute(
            "SELECT name, SUM(pgsize) FROM dbstat GROUP BY name"
        ).fetchall():
            key = "doc_fts (inverted index)" if str(name).startswith("doc_fts") else str(name)
            per_table[key] = per_table.get(key, 0) + int(size or 0)
    except sqlite3.OperationalError:
        per_table = {}

    counts = dict(conn.execute(
        "SELECT gold_bucket, COUNT(*) FROM items GROUP BY gold_bucket").fetchall())
    n_items = sum(counts.values())
    n_doc = int(counts.get("document", 0))
    n_scene = int(counts.get("scene", 0))

    # Every item carries a CLIP vector and a row in `items`; only documents
    # additionally carry OCR blocks and FTS postings. Attributing the shared
    # cost per item and the document-only cost per document keeps the
    # projection honest for any document:scene mix.
    doc_only = sum(v for k, v in per_table.items()
                   if k.startswith(("doc_fts", "ocr_blocks", "doc_fields")))
    shared = total_bytes - doc_only

    shared_per_item = shared / n_items if n_items else 0.0
    doc_extra_per_doc = doc_only / n_doc if n_doc else 0.0

    return {
        "total_bytes": total_bytes,
        "n_items": n_items, "n_doc": n_doc, "n_scene": n_scene,
        "per_table": dict(sorted(per_table.items(), key=lambda kv: -kv[1])),
        "bytes_per_item_overall": total_bytes / n_items if n_items else 0.0,
        "shared_bytes_per_item": shared_per_item,
        "document_extra_bytes_per_doc": doc_extra_per_doc,
        "doc_ratio": n_doc / n_items if n_items else 0.0,
        "projection": {
            "gallery_size": GALLERY_SIZE,
            # Projected at the corpus's own document:scene ratio.
            "at_corpus_ratio_mb": (
                GALLERY_SIZE * (shared_per_item + doc_extra_per_doc * (n_doc / n_items))
                / 1e6 if n_items else 0.0),
            # Bracketing cases, since the mix dominates the answer.
            "all_scene_mb": GALLERY_SIZE * shared_per_item / 1e6,
            "all_document_mb": GALLERY_SIZE * (shared_per_item + doc_extra_per_doc) / 1e6,
        },
    }


def model_footprints(encoder=None) -> dict:
    """Parameter counts and RAM cost of the models the baseline loads."""
    out: dict[str, dict] = {}
    if encoder is not None and getattr(encoder, "model", None) is not None:
        n = sum(p.numel() for p in encoder.model.parameters())
        out[f"CLIP {getattr(encoder, 'id', '')}"] = {
            "params_m": n / 1e6,
            "fp32_mb": n * 4 / 1e6,
            "fp16_mb": n * 2 / 1e6,
            "int8_mb": n / 1e6,
        }
    try:
        import torch
        if torch.cuda.is_available():
            out["_cuda_peak_mb"] = {"value": torch.cuda.max_memory_allocated() / 1e6}
    except Exception:  # noqa: BLE001
        pass
    return out


def ingestion_report(conn: sqlite3.Connection, run_id: str | None = None) -> dict:
    """Per-stage ms/image and images/sec for the most recent indexing run.

    Stages run over different denominators -- `ocr` touches document items
    only, while classify/clip_encode/ingest touch everything -- so each stage
    is reported against its own n_items. Averaging a single "ms/image" across
    stages would be meaningless.
    """
    if run_id is None:
        row = conn.execute(
            "SELECT run_id FROM timings ORDER BY created_at DESC, rowid DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return {}
        run_id = row[0]

    stages = []
    for r in conn.execute(
        "SELECT stage, n_items, total_ms FROM timings WHERE run_id=? ORDER BY rowid", (run_id,)
    ):
        n, ms = int(r["n_items"] or 0), float(r["total_ms"] or 0.0)
        stages.append({
            "stage": r["stage"], "n_items": n, "total_ms": ms,
            "ms_per_image": ms / n if n else 0.0,
            "images_per_sec": n / (ms / 1000.0) if ms else 0.0,
        })

    wall_ms = sum(s["total_ms"] for s in stages)
    n_all = max((s["n_items"] for s in stages), default=0)

    # OCR is the only stage with per-item timings stored (doc_fields.ocr_ms),
    # and it is the one whose tail matters: a 4080x3072 photographed passport
    # and a 509x314 card differ by orders of magnitude, which a mean hides.
    ocr_dist = {}
    vals = [float(r[0]) for r in conn.execute(
        "SELECT ocr_ms FROM doc_fields WHERE ocr_ms IS NOT NULL") if r[0] is not None]
    if vals:
        s = sorted(vals)
        pick = lambda p: s[min(len(s) - 1, int(round(p * (len(s) - 1))))]  # noqa: E731
        ocr_dist = {"n": len(s), "mean_ms": sum(s) / len(s), "p50_ms": pick(0.50),
                    "p95_ms": pick(0.95), "max_ms": s[-1], "min_ms": s[0]}

    # Per-bucket ingestion cost. Documents dominate (OCR is ~35x classify), so
    # mixed-gallery throughput depends entirely on the document:scene ratio.
    counts = dict(conn.execute(
        "SELECT gold_bucket, COUNT(*) FROM items GROUP BY gold_bucket").fetchall())
    n_doc, n_scene = int(counts.get("document", 0)), int(counts.get("scene", 0))
    ocr_total = sum(s["total_ms"] for s in stages if s["stage"] == "ocr")
    shared_total = wall_ms - ocr_total
    shared_per_item = shared_total / n_all if n_all else 0.0
    per_bucket = {
        "scene": {"ms_per_image": shared_per_item,
                  "images_per_sec": 1000.0 / shared_per_item if shared_per_item else 0.0},
        "document": {"ms_per_image": shared_per_item + (ocr_total / n_doc if n_doc else 0.0),
                     "images_per_sec": 0.0},
    }
    d = per_bucket["document"]["ms_per_image"]
    per_bucket["document"]["images_per_sec"] = 1000.0 / d if d else 0.0

    return {
        "run_id": run_id, "stages": stages,
        "wall_ms": wall_ms, "n_items": n_all,
        "overall_ms_per_image": wall_ms / n_all if n_all else 0.0,
        "overall_images_per_sec": n_all / (wall_ms / 1000.0) if wall_ms else 0.0,
        "ocr_per_item": ocr_dist,
        "per_bucket": per_bucket,
        "n_doc": n_doc, "n_scene": n_scene,
        "projection_hours_10k": (GALLERY_SIZE * (wall_ms / n_all) / 1000.0 / 3600.0)
                                if n_all else 0.0,
    }
