#!/usr/bin/env python3
"""Evaluate the baseline and its comparators; write a markdown results table.

    python scripts/evaluate.py
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from pixarch.clip_model import ClipEncoder  # noqa: E402
from pixarch.eval import ocr_quality, resources, runner  # noqa: E402
from pixarch.query.engine import SearchEngine  # noqa: E402
from pixarch.store import sqlite_store as store  # noqa: E402

# Precision leads: recall saturates at k/|relevant|, so recall@1 on a scene
# query with 10 relevant items tops out at 0.10 and misreads as a failure.
METRIC_ORDER = ["p@1", "p@5", "mrr@10", "ndcg@10", "r_precision", "recall@5", "recall@10"]
# Recall@k cannot exceed k/|relevant|; shown next to it so the figure is readable.
RECALL_ORDER = ["recall@1", "ceiling@1", "recall@1_pct_of_max",
                "recall@5", "ceiling@5", "recall@5_pct_of_max",
                "recall@10", "ceiling@10", "recall@10_pct_of_max"]


def fmt_table(results: dict, section: str) -> list[str]:
    lines = ["| system | " + " | ".join(METRIC_ORDER) + " | n |",
             "|---|" + "---|" * (len(METRIC_ORDER) + 1)]
    for name, sections in results.items():
        block = sections.get(section)
        if not block:
            continue
        cells = [f"{block.get(m, float('nan')):.3f}" for m in METRIC_ORDER]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {int(block.get('n_queries', 0))} |")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/baseline.yaml"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    conn = store.connect(ROOT / cfg["paths"]["index_db"], create=False)
    queries = runner.load_queries(ROOT / cfg["paths"]["queries"])
    qrels = runner.load_qrels(ROOT / cfg["paths"]["qrels"])

    encoder = ClipEncoder(
        model_name=cfg["clip"]["model"], pretrained=cfg["clip"]["pretrained"],
        device=cfg["clip"]["device"], batch_size=cfg["clip"]["batch_size"],
    )
    engine = SearchEngine(conn, encoder, cfg)

    k_values = cfg["eval"]["k_values"]
    top_k = cfg["search"]["top_k"]
    results, _, latencies, depth = runner.run_eval(engine, conn, queries, qrels, k_values, top_k)
    acc, misroutes = runner.routing_accuracy(engine, queries)
    confusion = runner.classifier_confusion(conn)

    n_items = conn.execute("SELECT COUNT(*) c FROM items").fetchone()["c"]
    db_mb = (ROOT / cfg["paths"]["index_db"]).stat().st_size / 1e6

    out: list[str] = [
        f"# Baseline results -- {date.today().isoformat()}",
        "",
        f"Corpus: **{n_items} items** | index **{db_mb:.1f} MB** | "
        f"CLIP `{encoder.id}` on `{encoder.device}` | OCR `{cfg['ocr']['engine']}` | "
        f"{len(queries)} queries | retrieval depth {depth}",
        "",
        "> Indicative, not publishable: the corpus is small and auto-fetched, and",
        "> relevance is single-judge. Enough to catch regressions and prove the",
        "> plumbing; not enough to support a claim.",
        "",
        "## Retrieval -- overall",
        "",
        *fmt_table(results, "overall"),
        "",
        "**Reading the `filename` row:** it is inflated and should be treated as a",
        "loose upper bound, not the real floor. The fetcher names files after their",
        "class (`scenes/beach/scn_beach_000.jpg`), so substring matching on the path",
        "succeeds far more often than it would on a phone, where names look like",
        "`IMG_20231205_123456.jpg`. Preserving original filenames in the curated",
        "milestone will make this row meaningful.",
        "",
    ]
    for kind in ("doc_lookup", "scene_search", "ambiguous"):
        rows = fmt_table(results, kind)
        if len(rows) > 2:
            out += [f"## Retrieval -- {kind}", "", *rows, ""]

    out += ["## Retrieval -- by bucket", "",
            "Macro-averaged **within** each bucket over its queries, then macro-averaged",
            "**across** buckets so each bucket counts once regardless of query count.", ""]
    for bucket in ("document", "scene"):
        rows = fmt_table(results, bucket)
        if len(rows) > 2:
            out += [f"### {bucket}", "", *rows, ""]
    rows = fmt_table(results, "bucket_macro")
    if len(rows) > 2:
        out += ["### macro-average across buckets", "", *rows, ""]

    out += ["## Recall with ceilings", "",
            "`ceiling@k = min(k, |relevant|)/|relevant|` is the largest recall@k this",
            "corpus permits: with ~20 items per subtype, recall@10 cannot exceed 0.50.",
            "`pct_of_max` is the fraction of the attainable value actually reached.", "",
            "| system | " + " | ".join(RECALL_ORDER) + " |",
            "|---|" + "---|" * len(RECALL_ORDER)]
    for name, sections in results.items():
        blk = sections.get("overall")
        if blk:
            out.append(f"| {name} | " + " | ".join(
                f"{blk.get(m, float('nan')):.3f}" for m in RECALL_ORDER) + " |")
    out += [""]

    lat = runner.latency_report(latencies)
    out += ["## Query latency", "",
            "Captured during the same evaluation pass. The document and scene branches",
            "are bimodal -- BM25 is a SQL query, CLIP needs a text-encoder forward pass",
            "-- so the per-branch split matters more than the pooled mean.", "",
            "| system | scope | n | mean ms | p50 ms | p95 ms | max ms |",
            "|---|---|---|---|---|---|---|"]
    for name, scopes in lat.items():
        for scope, st in scopes.items():
            if st:
                out.append(f"| {name} | {scope} | {int(st['n'])} | {st['mean_ms']:.1f} | "
                           f"{st['p50_ms']:.1f} | {st['p95_ms']:.1f} | {st['max_ms']:.1f} |")
    out += [""]

    out += [
        "## Routing", "",
        f"Router accuracy vs `gold_route`: **{acc:.1%}** ({len(misroutes)} misrouted)", "",
    ]
    if misroutes:
        out += ["| query | gold | predicted |", "|---|---|---|"]
        out += [f"| {q} | {g} | {p} |" for q, g, p in misroutes] + [""]

    out += ["## Bucket classification (gold -> predicted)", "",
            "| gold | predicted | n |", "|---|---|---|"]
    out += [f"| {g} | {p} | {c} |" for (g, p), c in sorted(confusion.items())] + [""]

    dt = runner.doctype_benchmark(conn)
    if dt:
        out += ["## Document subtype tagging (doctype.py)", "",
                f"Overall accuracy **{dt['accuracy']:.1%}** over {dt['n']} document items. "
                "This module fills the FTS column weighted 5x against body text.", "",
                "| origin | n | accuracy |", "|---|---|---|"]
        for origin, st in sorted(dt["by_origin"].items()):
            out.append(f"| {origin} | {int(st['n'])} | {st['accuracy']:.1%} |")
        out += ["", "| gold subtype | precision | recall | f1 | support |", "|---|---|---|---|---|"]
        for lab, m in sorted(dt["per_class"].items()):
            out.append(f"| {lab} | {m['precision']:.3f} | {m['recall']:.3f} | "
                       f"{m['f1']:.3f} | {int(m['support'])} |")
        wrong = {k: v for k, v in dt["confusion"].items() if k[0] != k[1]}
        if wrong:
            out += ["", "Misclassifications (gold -> predicted):", "",
                    "| gold | predicted | n |", "|---|---|---|"]
            out += [f"| {g} | {p} | {c} |" for (g, p), c in
                    sorted(wrong.items(), key=lambda kv: -kv[1])]
        out += [""]

    ocr_stats = ocr_quality.evaluate_ocr(conn, ROOT / "data/ocr_groundtruth.jsonl")
    if ocr_stats:
        out += [
            "## OCR accuracy", "",
            f"Engine `{ocr_stats['engine']}` over {int(ocr_stats['n_items'])} items with "
            "ground truth: SROIE receipts, FUNSD forms and IAM handwriting carry upstream "
            "transcriptions; the synthetic specimens carry the exact rendered text.", "",
            "> **`bill` is not comparable.** The invoice ground truth upstream is a set of "
            "extracted *field values*, not a full-page transcription, so OCR that correctly "
            "reads the whole page is scored against a much shorter reference and its CER "
            "exceeds 1.0. Treat the `bill` row as a ground-truth mismatch, not an OCR "
            "failure.", "",
            "| metric | value |", "|---|---|",
            f"| CER | {ocr_stats['cer']:.3f} |",
            f"| WER | {ocr_stats['wer']:.3f} |", "",
        ]

    grouped = ocr_quality.evaluate_ocr_grouped(conn, ROOT / "data/ocr_groundtruth.jsonl")
    if grouped.get("by_origin"):
        out += ["### OCR by origin", "",
                "Synthetic specimens are rendered text; fetched items are photographs and",
                "scans. The gap between the two is the domain gap the synthetic corpus", 
                "carries.", "",
                "| origin | n | CER | WER |", "|---|---|---|---|"]
        for origin, st in sorted(grouped["by_origin"].items()):
            out.append(f"| {origin} | {int(st['n'])} | {st['cer']:.3f} | {st['wer']:.3f} |")
        out += ["", "### OCR by subtype (worst CER first)", "",
                "| subtype | n | CER | WER |", "|---|---|---|---|"]
        for st_name, st in list(grouped["by_subtype"].items()):
            out.append(f"| {st_name} | {int(st['n'])} | {st['cer']:.3f} | {st['wer']:.3f} |")
        out += [""]

    ing = resources.ingestion_report(conn)
    if ing:
        out += ["## Ingestion throughput", "",
                f"Run `{ing['run_id']}` over {ing['n_items']} items "
                f"({ing['n_doc']} document, {ing['n_scene']} scene).", "",
                "Each stage is reported against **its own** `n_items`: `ocr` touches",
                "document items only, the rest touch everything, so a single averaged",
                "ms/image across stages would be meaningless.", "",
                "| stage | items | total ms | ms/image | images/sec |",
                "|---|---|---|---|---|"]
        for st in ing["stages"]:
            out.append(f"| {st['stage']} | {st['n_items']} | {st['total_ms']:.1f} | "
                       f"{st['ms_per_image']:.1f} | {st['images_per_sec']:.2f} |")
        out += [f"| **end-to-end** | {ing['n_items']} | {ing['wall_ms']:.1f} | "
                f"{ing['overall_ms_per_image']:.1f} | {ing['overall_images_per_sec']:.2f} |", ""]
        pb = ing["per_bucket"]
        out += ["Per bucket -- documents dominate ingestion cost, so mixed-gallery",
                "throughput depends on the document:scene ratio:", "",
                "| bucket | ms/image | images/sec |", "|---|---|---|"]
        for b in ("scene", "document"):
            out.append(f"| {b} | {pb[b]['ms_per_image']:.1f} | {pb[b]['images_per_sec']:.2f} |")
        out += [""]
        d = ing.get("ocr_per_item") or {}
        if d:
            out += ["OCR per-item distribution (from `doc_fields.ocr_ms`); long-tailed, so",
                    "the mean alone understates the worst case:", "",
                    "| n | min ms | mean ms | p50 ms | p95 ms | max ms |",
                    "|---|---|---|---|---|---|",
                    f"| {d['n']} | {d['min_ms']:.0f} | {d['mean_ms']:.0f} | {d['p50_ms']:.0f} | "
                    f"{d['p95_ms']:.0f} | {d['max_ms']:.0f} |", ""]
        out += [f"Projected full-index time for a {resources.GALLERY_SIZE:,}-item gallery "
                f"at this corpus's document ratio: **{ing['projection_hours_10k']:.2f} hours** "
                "(linear projection).", ""]

    br = resources.index_breakdown(conn, ROOT / cfg["paths"]["index_db"])
    if br:
        out += ["## Storage and memory", "",
                f"`index.db` is **{br['total_bytes']/1e6:.1f} MB** for {br['n_items']} items "
                f"({br['n_doc']} document, {br['n_scene']} scene).", ""]
        if br["per_table"]:
            out += ["| table | bytes | MB | share |", "|---|---|---|---|"]
            for name, b in br["per_table"].items():
                out.append(f"| {name} | {b:,} | {b/1e6:.2f} | "
                           f"{b/br['total_bytes']:.1%} |")
            out += [""]
        out += ["| measure | value |", "|---|---|",
                f"| bytes/item (overall) | {br['bytes_per_item_overall']:,.0f} |",
                f"| bytes/item (shared: items + CLIP vector) | {br['shared_bytes_per_item']:,.0f} |",
                f"| extra bytes/document (OCR + FTS) | {br['document_extra_bytes_per_doc']:,.0f} |",
                ""]
        pr = br["projection"]
        out += [f"Projected index size for {pr['gallery_size']:,} items:", "",
                "| mix | MB |", "|---|---|",
                f"| at this corpus's ratio ({br['doc_ratio']:.0%} documents) | "
                f"{pr['at_corpus_ratio_mb']:.0f} |",
                f"| all scene (lower bound) | {pr['all_scene_mb']:.0f} |",
                f"| all document (upper bound) | {pr['all_document_mb']:.0f} |", ""]

    mf = resources.model_footprints(encoder)
    if mf:
        out += ["### Model memory", "", "| model | params (M) | fp32 MB | fp16 MB | int8 MB |",
                "|---|---|---|---|---|"]
        for name, m in mf.items():
            if name.startswith("_"):
                continue
            out.append(f"| {name} | {m['params_m']:.1f} | {m['fp32_mb']:.0f} | "
                       f"{m['fp16_mb']:.0f} | {m['int8_mb']:.0f} |")
        out += [""]

    timings = conn.execute(
        "SELECT stage, SUM(n_items) n, ROUND(SUM(total_ms),1) ms FROM timings "
        "GROUP BY stage ORDER BY ms DESC"
    ).fetchall()
    if timings:
        out += ["## Indexing cost", "", "| stage | items | total ms | ms/item |", "|---|---|---|---|"]
        for r in timings:
            per = f"{r['ms']/r['n']:.1f}" if r["n"] else "-"
            out += [f"| {r['stage']} | {r['n']} | {r['ms']:.1f} | {per} |"]
        out += [""]

    text = "\n".join(out)
    print(text)

    out_path = Path(args.out) if args.out else (
        ROOT / cfg["paths"]["results_dir"] / f"baseline_{date.today().isoformat()}.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"\nwritten -> {out_path}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
