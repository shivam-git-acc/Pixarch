#!/usr/bin/env python3
"""Score an on-device evaluation run and compare it with the PC baseline.

    python scripts/evaluate_device.py --run data/device_runs/<run>   [--no-pc]

Input: the folder `scripts/run_device_eval.py` pulled from the phone
(`device_run.json` + `device_index.db`). Every metric is computed by the same
code as scripts/evaluate.py (src/pixarch/eval): retrieval via runner.score_runs,
latency via runner.latency_report, bucket confusion, doc-type benchmark, OCR
CER/WER and ingestion throughput. The PC side is re-scored here from index.db on
the same queries, so both columns use identical queries, qrels and depth.

Output: docs/results/device_<model>_<date>.md and a JSON summary next to the run.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from pixarch.eval import ocr_quality, resources, runner  # noqa: E402

METRICS = ["p@1", "p@5", "mrr@10", "ndcg@10", "r_precision", "recall@10"]
STAGES = ["ingest", "clip_encode", "classify", "embed_store", "ocr"]
GT = ROOT / "data" / "ocr_groundtruth.jsonl"


# --------------------------------------------------------------------------- phone
def prepare_device_db(run_dir: Path, manifest: dict[str, dict], run: dict) -> sqlite3.Connection:
    """Make the phone's index look like the desktop schema the eval code expects.

    The app's `items` table has no gold labels (a phone never knows them) and stores
    a file:// URI as path; add gold_bucket / gold_subtype / origin from the manifest,
    set path to the manifest-relative path (so the filename floor behaves exactly as
    on PC), and rebuild a `timings` table from the harness's stage totals.
    """
    scored = run_dir / "device_index_scored.db"
    shutil.copy(run_dir / "device_index.db", scored)
    conn = sqlite3.connect(scored)
    conn.row_factory = sqlite3.Row
    cols = {r[1] for r in conn.execute("PRAGMA table_info(items)")}
    for col in ("gold_bucket", "gold_subtype", "origin"):
        if col not in cols:
            conn.execute(f"ALTER TABLE items ADD COLUMN {col} TEXT")
    conn.executemany(
        "UPDATE items SET path=?, gold_bucket=?, gold_subtype=?, origin=? WHERE item_id=?",
        [(m["path"], m["bucket"], m["subtype"], m.get("origin") or "fetched", i) for i, m in manifest.items()],
    )
    conn.execute("DROP TABLE IF EXISTS timings")
    conn.execute("CREATE TABLE timings (run_id TEXT, stage TEXT, n_items INTEGER, "
                 "total_ms REAL, created_at TEXT)")
    stages = run["index"]["stages"]
    conn.executemany(
        "INSERT INTO timings VALUES ('device', ?, ?, ?, datetime('now'))",
        [(s, stages[s]["n"], stages[s]["total_ms"]) for s in STAGES if s in stages],
    )
    conn.commit()
    return conn


def device_runs(run: dict, conn: sqlite3.Connection, queries, depth: int):
    runs, lat = {}, {}
    for name, rows in run["queries"]["runs"].items():
        runs[name] = {r["query_id"]: list(r["ranked"]) for r in rows}
        lat[name] = [(r["branch"], float(r["ms"])) for r in rows]
    # The filename floor is device-independent (substring match on the manifest
    # path), so it is computed here with the desktop function on the phone's items.
    runs["filename"] = {q["query_id"]: [h.item_id for h in runner.filename_search(conn, q["text"], depth)]
                        for q in queries}
    return runs, lat


def device_routing(run: dict, queries) -> tuple[float, list[tuple[str, str, str]]]:
    by_id = {r["query_id"]: r["branch"] for r in run["queries"]["runs"]["routed"]}
    wrong = [(q["query_id"], q["gold_route"], by_id.get(q["query_id"], "?"))
             for q in queries if by_id.get(q["query_id"]) != q["gold_route"]]
    return 1 - len(wrong) / len(queries), wrong


def device_storage(run: dict, conn: sqlite3.Connection) -> dict:
    """Index size per item and 10k projection; per-table split when dbstat was available."""
    total = int(run["sizes"]["db_bytes"])
    counts = dict(conn.execute("SELECT gold_bucket, COUNT(*) FROM items GROUP BY gold_bucket").fetchall())
    n = sum(counts.values())
    n_doc = int(counts.get("document", 0))
    per_table: dict[str, int] = {}
    for name, size in (run["sizes"].get("db_per_table") or {}).items():
        key = "doc_fts (inverted index)" if name.startswith("doc_fts") else name
        per_table[key] = per_table.get(key, 0) + int(size)
    out = {"total_bytes": total, "n_items": n, "n_doc": n_doc,
           "bytes_per_item": total / n if n else 0.0,
           "per_table": dict(sorted(per_table.items(), key=lambda kv: -kv[1])),
           "projection_10k_mb": resources.GALLERY_SIZE * total / n / 1e6 if n else 0.0}
    if per_table:
        doc_only = sum(v for k, v in per_table.items() if k.startswith(("doc_fts", "ocr_blocks", "doc_fields")))
        shared = total - doc_only
        out["all_scene_mb"] = resources.GALLERY_SIZE * shared / n / 1e6
        out["all_document_mb"] = resources.GALLERY_SIZE * (shared / n + (doc_only / n_doc if n_doc else 0)) / 1e6
    return out


# ------------------------------------------------------------------------------ PC
def pc_side(cfg: dict, queries, qrels, k_values, top_k):
    from pixarch.clip_model import ClipEncoder
    from pixarch.query.engine import SearchEngine
    from pixarch.store import sqlite_store as store

    db_path = ROOT / cfg["paths"]["index_db"]
    conn = store.connect(db_path, create=False)
    encoder = ClipEncoder(model_name=cfg["clip"]["model"], pretrained=cfg["clip"]["pretrained"],
                          device=cfg["clip"]["device"], batch_size=cfg["clip"]["batch_size"])
    engine = SearchEngine(conn, encoder, cfg)
    results, _, latencies, depth = runner.run_eval(engine, conn, queries, qrels, k_values, top_k)
    pc = {
        "label": f"PC · CLIP {encoder.id} on {encoder.device} · OCR {cfg['ocr']['engine']}",
        "depth": depth,
        "results": results,
        "latency": runner.latency_report(latencies),
        "routing": runner.routing_accuracy(engine, queries),
        "confusion": runner.classifier_confusion(conn),
        "doctype": runner.doctype_benchmark(conn),
        "ocr": ocr_quality.evaluate_ocr(conn, GT),
        "ocr_grouped": ocr_quality.evaluate_ocr_grouped(conn, GT),
        "ingestion": resources.ingestion_report(conn),
        "storage": {"total_bytes": db_path.stat().st_size,
                    "n_items": conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]},
        "params_m": sum(p.numel() for p in encoder.model.parameters()) / 1e6,
    }
    pc["storage"]["bytes_per_item"] = pc["storage"]["total_bytes"] / pc["storage"]["n_items"]
    pc["storage"]["projection_10k_mb"] = resources.GALLERY_SIZE * pc["storage"]["bytes_per_item"] / 1e6
    return pc


# -------------------------------------------------------------------------- report
def f3(x) -> str:
    return "—" if x is None else f"{x:.3f}"


def retrieval_rows(label: str, results: dict, section: str) -> list[str]:
    rows = []
    for name, sections in results.items():
        b = sections.get(section)
        if b:
            rows.append(f"| {name} | {label} | " + " | ".join(f3(b.get(m)) for m in METRICS)
                        + f" | {int(b.get('n_queries', 0))} |")
    return rows


def confusion_acc(conf: dict) -> tuple[float, int, int]:
    total = sum(conf.values())
    ok = sum(c for (g, p), c in conf.items() if g == p)
    return (ok / total if total else 0.0), ok, total


def build_report(run: dict, dev: dict, pc: dict | None, queries) -> str:
    d = run["device"]
    name = f"{d['manufacturer']} {d['model']}"
    out = [f"# On-device evaluation — {name} — {date.today().isoformat()}", "",
           f"**Phone:** {name} · Android {d['release']} (SDK {d['sdk']}) · {d['soc']} · "
           f"{d['cores']} cores · {d['total_ram_mb']/1000:.1f} GB RAM",
           f"**Phone pipeline:** {run['config']['clip_model']} · ML Kit OCR ({run['config']['ocr_mode']}) · "
           f"bucket τ = {run['config']['bucket_threshold']:.4f} · retrieval depth {run['config']['depth']}",
           f"**PC pipeline:** {pc['label'] if pc else '(not run: --no-pc)'}",
           f"**Benchmark:** {run['index']['indexed']} items indexed on the phone "
           f"({run['index']['missing_files']} missing, {run['index']['skipped']} duplicate/unreadable) · "
           f"{len(queries)} queries", "",
           "> Indicative, not publishable: relevance is single-judge and part of the corpus is",
           "> generated. Both columns are scored by the same code (src/pixarch/eval) on the same",
           "> queries, qrels and depth. Phone latencies are warm, measured per search on the device.", ""]

    # Retrieval
    hdr = ["| system | platform | " + " | ".join(METRICS) + " | n |", "|---|---|" + "---|" * (len(METRICS) + 1)]
    for section, title in (("overall", "overall"), ("document", "document queries"),
                           ("scene", "scene queries"), ("bucket_macro", "macro-average across buckets")):
        rows = []
        for sysname in ("routed", "clip_only", "bm25_only", "filename"):
            if pc and sysname in pc["results"]:
                rows += retrieval_rows("PC", {sysname: pc["results"][sysname]}, section)
            if sysname in dev["results"]:
                rows += retrieval_rows("phone", {sysname: dev["results"][sysname]}, section)
        if rows:
            out += [f"## Retrieval — {title}", "", *hdr, *rows, ""]
    out += ["The `filename` row matches query words against manifest paths, which are named after",
            "their class, so it is an inflated upper bound on the trivial floor and identical on both.", ""]

    # Routing + buckets
    acc, wrong = dev["routing"]
    out += ["## Query routing", "", "| platform | accuracy | misrouted |", "|---|---|---|"]
    if pc:
        out.append(f"| PC | {pc['routing'][0]:.1%} | {len(pc['routing'][1])} |")
    out += [f"| phone | {acc:.1%} | {len(wrong)} |", ""]
    if wrong:
        texts = {q["query_id"]: q["text"] for q in queries}
        out += ["Phone misroutes:", "", "| query | text | gold | predicted |", "|---|---|---|---|"]
        out += [f"| {q} | {texts[q]} | {g} | {p} |" for q, g, p in wrong] + [""]

    out += ["## Bucket classification (document vs scene, at indexing)", "",
            "| platform | accuracy | gold → predicted |", "|---|---|---|"]
    for label, conf in (("PC", pc["confusion"] if pc else None), ("phone", dev["confusion"])):
        if conf:
            a, ok, total = confusion_acc(conf)
            detail = ", ".join(f"{g}→{p}: {c}" for (g, p), c in sorted(conf.items()) if g != p) or "none"
            out.append(f"| {label} | {a:.1%} ({ok}/{total}) | errors: {detail} |")
    hedged = sum(v for k, v in run["index"]["bucket_counts"].items() if k.endswith("both=1"))
    out += ["", f"Phone hedged {hedged} items into both branches (margin < τ).", ""]

    # Doc type + OCR
    out += ["## Document subtype tagging", "", "| platform | overall | synthetic | fetched |", "|---|---|---|---|"]
    for label, dt in (("PC", pc["doctype"] if pc else None), ("phone", dev["doctype"])):
        if dt:
            bo = dt["by_origin"]
            cell = lambda o: f"{bo[o]['accuracy']:.1%} (n={int(bo[o]['n'])})" if o in bo else "—"  # noqa: E731
            out.append(f"| {label} | {dt['accuracy']:.1%} (n={dt['n']}) | {cell('synthetic')} | {cell('fetched')} |")
    out += ["", "## OCR accuracy (normalised text vs ground truth)", "",
            "| platform | engine | n | CER | WER | synthetic CER | fetched CER |", "|---|---|---|---|---|---|---|"]
    for label, o, g in (("PC", pc["ocr"] if pc else None, pc["ocr_grouped"] if pc else None),
                        ("phone", dev["ocr"], dev["ocr_grouped"])):
        if o:
            bo = (g or {}).get("by_origin", {})
            out.append(f"| {label} | {o['engine']} | {int(o['n_items'])} | {o['cer']:.3f} | {o['wer']:.3f} | "
                       f"{f3(bo.get('synthetic', {}).get('cer'))} | {f3(bo.get('fetched', {}).get('cer'))} |")
    out += ["", "`bill` ground truth is extracted field values, not a page transcription, so its CER can",
            "exceed 1.0 on either platform; compare the synthetic column for recognition quality.", ""]

    # Latency
    out += ["## Query latency (warm)", "", "| platform | system | scope | n | mean ms | p50 ms | p95 ms | max ms |",
            "|---|---|---|---|---|---|---|---|"]
    for label, lat in (("PC", pc["latency"] if pc else None), ("phone", dev["latency"])):
        for sysname, scopes in (lat or {}).items():
            for scope, st in scopes.items():
                if st:
                    out.append(f"| {label} | {sysname} | {scope} | {int(st['n'])} | {st['mean_ms']:.1f} | "
                               f"{st['p50_ms']:.1f} | {st['p95_ms']:.1f} | {st['max_ms']:.1f} |")
    out += ["", f"Phone cold start: the first scene query (loads the text encoder) took "
            f"**{run['queries']['cold_first_scene_query_ms']:.0f} ms**; the image encoder loaded in "
            f"{run['index']['visual_model_load_ms']:.0f} ms.", ""]

    # Ingestion
    out += ["## Indexing throughput", "",
            "Each stage is reported against its own item count (OCR only runs on document/hedged items).", "",
            "| stage | PC items | PC ms/image | phone items | phone ms/image |", "|---|---|---|---|---|"]
    pcs = {s["stage"]: s for s in (pc["ingestion"]["stages"] if pc and pc["ingestion"] else [])}
    dvs = {s["stage"]: s for s in dev["ingestion"]["stages"]}
    for s in STAGES:
        a, b = pcs.get(s), dvs.get(s)
        pa = f"{a['n_items']} | {a['ms_per_image']:.1f}" if a else "— | —"
        pb = f"{b['n_items']} | {b['ms_per_image']:.1f}" if b else "— | —"
        out.append(f"| {s} | {pa} | {pb} |")
    ing = dev["ingestion"]
    pi = run["index"]["per_item"]
    out += ["", "| measure | PC | phone |", "|---|---|---|"]
    if pc and pc["ingestion"]:
        p = pc["ingestion"]
        out += [f"| end-to-end ms/image | {p['overall_ms_per_image']:.1f} | {ing['overall_ms_per_image']:.1f} |",
                f"| images/sec | {p['overall_images_per_sec']:.2f} | {ing['overall_images_per_sec']:.2f} |",
                f"| scene ms/image | {p['per_bucket']['scene']['ms_per_image']:.1f} | "
                f"{ing['per_bucket']['scene']['ms_per_image']:.1f} |",
                f"| document ms/image | {p['per_bucket']['document']['ms_per_image']:.1f} | "
                f"{ing['per_bucket']['document']['ms_per_image']:.1f} |",
                f"| projected 10,000-item gallery | {p['projection_hours_10k']:.2f} h | "
                f"{ing['projection_hours_10k']:.2f} h |"]
    else:
        out += [f"| end-to-end ms/image | — | {ing['overall_ms_per_image']:.1f} |",
                f"| projected 10,000-item gallery | — | {ing['projection_hours_10k']:.2f} h |"]
    out += ["", f"Phone wall-clock for the whole corpus: **{run['index']['wall_ms']/1000:.0f} s**; "
            f"per item mean {pi['mean_ms']:.0f} ms, p50 {pi['p50_ms']:.0f}, p95 {pi['p95_ms']:.0f}, "
            f"max {pi['max_ms']:.0f} ms.", ""]
    od = ing.get("ocr_per_item") or {}
    if od:
        out += [f"Phone OCR per document (ML Kit `ocr_ms`): mean {od['mean_ms']:.0f} ms, p50 {od['p50_ms']:.0f}, "
                f"p95 {od['p95_ms']:.0f}, max {od['max_ms']:.0f} ms over {od['n']} items.", ""]

    # Storage + memory
    st = dev["storage"]
    out += ["## Storage, memory and app size", "", "| measure | PC | phone |", "|---|---|---|"]
    if pc:
        ps = pc["storage"]
        out += [f"| index size | {ps['total_bytes']/1e6:.1f} MB ({ps['n_items']} items) | "
                f"{st['total_bytes']/1e6:.1f} MB ({st['n_items']} items) |",
                f"| bytes per item | {ps['bytes_per_item']:,.0f} | {st['bytes_per_item']:,.0f} |",
                f"| projected index, 10,000 items | {ps['projection_10k_mb']:.0f} MB | {st['projection_10k_mb']:.0f} MB |",
                f"| vision-language model | {pc['params_m']:.1f} M params, {pc['params_m']*4:.0f} MB fp32 | "
                f"image {run['sizes']['visual_model_bytes']/1e6:.1f} MB + text "
                f"{run['sizes']['text_model_bytes']/1e6:.1f} MB (INT8) |"]
    else:
        out += [f"| index size | — | {st['total_bytes']/1e6:.1f} MB |"]
    m = run["memory"]
    out += [f"| app process memory (PSS) | — | baseline {m['baseline_pss_mb']:.0f} MB · peak indexing "
            f"{m['peak_pss_indexing_mb']:.0f} MB · peak querying {m['peak_pss_querying_mb']:.0f} MB |",
            f"| APK size (debug) | — | {run['sizes']['apk_bytes']/1e6:.0f} MB |", ""]
    if st["per_table"]:
        out += ["Phone index by table (dbstat):", "", "| table | MB | share |", "|---|---|---|"]
        out += [f"| {k} | {v/1e6:.2f} | {v/st['total_bytes']:.1%} |" for k, v in st["per_table"].items()]
        out += ["", f"10,000-item projection brackets: all scene {st['all_scene_mb']:.0f} MB, "
                f"all document {st['all_document_mb']:.0f} MB.", ""]
    out += ["Phone memory is sampled every 250 ms from the app process (instrumentation runs in it),",
            "so it includes both ONNX sessions, ML Kit and the test harness itself.", ""]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="folder holding device_run.json + device_index.db")
    ap.add_argument("--no-pc", action="store_true", help="skip re-scoring the PC index")
    ap.add_argument("--config", default=str(ROOT / "configs/baseline.yaml"))
    args = ap.parse_args()

    run_dir = Path(args.run)
    run = json.loads((run_dir / "device_run.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    queries = runner.load_queries(ROOT / cfg["paths"]["queries"])
    qrels = runner.load_qrels(ROOT / cfg["paths"]["qrels"])
    k_values, top_k = cfg["eval"]["k_values"], cfg["search"]["top_k"]
    with (ROOT / cfg["paths"]["manifest"]).open(newline="", encoding="utf-8") as fh:
        manifest = {r["item_id"]: r for r in csv.DictReader(fh)}

    depth = runner.retrieval_depth(qrels, top_k, k_values)
    if run["config"]["depth"] != depth:
        print(f"! phone ran at depth {run['config']['depth']}, PC scoring uses {depth}")
    if run["queries"]["n"] != len(queries):
        print(f"! phone ran {run['queries']['n']} queries, data/queries.jsonl has {len(queries)}")

    conn = prepare_device_db(run_dir, manifest, run)
    runs, lat = device_runs(run, conn, queries, depth)
    dev = {
        "results": runner.score_runs(runs, queries, qrels, k_values),
        "latency": runner.latency_report(lat),
        "routing": device_routing(run, queries),
        "confusion": runner.classifier_confusion(conn),
        "doctype": runner.doctype_benchmark(conn),
        "ocr": ocr_quality.evaluate_ocr(conn, GT),
        "ocr_grouped": ocr_quality.evaluate_ocr_grouped(conn, GT),
        "ingestion": resources.ingestion_report(conn, "device"),
        "storage": device_storage(run, conn),
    }
    pc = None if args.no_pc else pc_side(cfg, queries, qrels, k_values, top_k)

    text = build_report(run, dev, pc, queries)
    model = run["device"]["model"].replace(" ", "_")
    out_md = ROOT / cfg["paths"]["results_dir"] / f"device_{model}_{date.today().isoformat()}.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(text, encoding="utf-8")

    def jsonable(o):
        if isinstance(o, dict):
            return {(" -> ".join(k) if isinstance(k, tuple) else str(k)): jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [jsonable(v) for v in o]
        return o
    (run_dir / "summary.json").write_text(json.dumps(jsonable({"phone": dev, "pc": pc}), indent=1), encoding="utf-8")
    sys.stdout.reconfigure(errors="replace")  # Windows consoles can't print "τ" etc.
    print(text)
    print(f"\nwritten -> {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
