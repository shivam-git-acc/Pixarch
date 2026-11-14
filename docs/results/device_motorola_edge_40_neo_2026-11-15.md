# On-device evaluation — motorola motorola edge 40 neo — 2026-11-15

**Phone:** motorola motorola edge 40 neo · Android 15 (SDK 35) · Mediatek MT6879V_T/ZA · 8 cores · 7.8 GB RAM
**Phone pipeline:** MobileCLIP2-S0 (dfndr2b) INT8 ONNX · ML Kit OCR (DEVANAGARI) · bucket τ = 0.0214 · retrieval depth 107
**PC pipeline:** PC · CLIP ViT-B-32/laion2b_s34b_b79k on cpu · OCR tesseract
**Benchmark:** 627 items indexed on the phone (0 missing, 2 duplicate/unreadable) · 98 queries

> Indicative, not publishable: relevance is single-judge and part of the corpus is
> generated. Both columns are scored by the same code (src/pixarch/eval) on the same
> queries, qrels and depth. Phone latencies are warm, measured per search on the device.

## Retrieval — overall

| system | platform | p@1 | p@5 | mrr@10 | ndcg@10 | r_precision | recall@10 | n |
|---|---|---|---|---|---|---|---|---|
| routed | PC | 0.763 | 0.771 | 0.801 | 0.769 | 0.696 | 0.368 | 97 |
| routed | phone | 0.804 | 0.800 | 0.835 | 0.788 | 0.716 | 0.375 | 97 |
| clip_only | PC | 0.629 | 0.596 | 0.644 | 0.580 | 0.478 | 0.278 | 97 |
| clip_only | phone | 0.485 | 0.447 | 0.490 | 0.446 | 0.393 | 0.220 | 97 |
| bm25_only | PC | 0.351 | 0.344 | 0.375 | 0.345 | 0.315 | 0.155 | 97 |
| bm25_only | phone | 0.361 | 0.361 | 0.387 | 0.352 | 0.324 | 0.157 | 97 |
| filename | PC | 0.299 | 0.299 | 0.299 | 0.299 | 0.298 | 0.140 | 97 |
| filename | phone | 0.299 | 0.299 | 0.299 | 0.299 | 0.298 | 0.140 | 97 |

## Retrieval — document queries

| system | platform | p@1 | p@5 | mrr@10 | ndcg@10 | r_precision | recall@10 | n |
|---|---|---|---|---|---|---|---|---|
| routed | PC | 0.717 | 0.722 | 0.770 | 0.723 | 0.663 | 0.327 | 46 |
| routed | phone | 0.739 | 0.757 | 0.793 | 0.737 | 0.683 | 0.330 | 46 |
| clip_only | PC | 0.370 | 0.287 | 0.375 | 0.260 | 0.139 | 0.105 | 46 |
| clip_only | phone | 0.065 | 0.013 | 0.065 | 0.014 | 0.003 | 0.003 | 46 |
| bm25_only | PC | 0.717 | 0.722 | 0.770 | 0.723 | 0.663 | 0.327 | 46 |
| bm25_only | phone | 0.739 | 0.757 | 0.793 | 0.737 | 0.683 | 0.330 | 46 |
| filename | PC | 0.413 | 0.413 | 0.413 | 0.413 | 0.411 | 0.187 | 46 |
| filename | phone | 0.413 | 0.413 | 0.413 | 0.413 | 0.411 | 0.187 | 46 |

## Retrieval — scene queries

| system | platform | p@1 | p@5 | mrr@10 | ndcg@10 | r_precision | recall@10 | n |
|---|---|---|---|---|---|---|---|---|
| routed | PC | 0.804 | 0.816 | 0.828 | 0.810 | 0.725 | 0.405 | 51 |
| routed | phone | 0.863 | 0.839 | 0.873 | 0.835 | 0.745 | 0.415 | 51 |
| clip_only | PC | 0.863 | 0.875 | 0.887 | 0.869 | 0.784 | 0.434 | 51 |
| clip_only | phone | 0.863 | 0.839 | 0.873 | 0.835 | 0.745 | 0.415 | 51 |
| bm25_only | PC | 0.020 | 0.004 | 0.020 | 0.004 | 0.001 | 0.001 | 51 |
| bm25_only | phone | 0.020 | 0.004 | 0.020 | 0.004 | 0.001 | 0.001 | 51 |
| filename | PC | 0.196 | 0.196 | 0.196 | 0.196 | 0.196 | 0.098 | 51 |
| filename | phone | 0.196 | 0.196 | 0.196 | 0.196 | 0.196 | 0.098 | 51 |

## Retrieval — macro-average across buckets

| system | platform | p@1 | p@5 | mrr@10 | ndcg@10 | r_precision | recall@10 | n |
|---|---|---|---|---|---|---|---|---|
| routed | PC | 0.761 | 0.769 | 0.799 | 0.767 | 0.694 | 0.366 | 97 |
| routed | phone | 0.801 | 0.798 | 0.833 | 0.786 | 0.714 | 0.373 | 97 |
| clip_only | PC | 0.616 | 0.581 | 0.631 | 0.564 | 0.462 | 0.269 | 97 |
| clip_only | phone | 0.464 | 0.426 | 0.469 | 0.425 | 0.374 | 0.209 | 97 |
| bm25_only | PC | 0.368 | 0.363 | 0.395 | 0.364 | 0.332 | 0.164 | 97 |
| bm25_only | phone | 0.379 | 0.380 | 0.407 | 0.371 | 0.342 | 0.166 | 97 |
| filename | PC | 0.305 | 0.305 | 0.305 | 0.305 | 0.303 | 0.143 | 97 |
| filename | phone | 0.305 | 0.305 | 0.305 | 0.305 | 0.303 | 0.143 | 97 |

The `filename` row matches query words against manifest paths, which are named after
their class, so it is an inflated upper bound on the trivial floor and identical on both.

## Query routing

| platform | accuracy | misrouted |
|---|---|---|
| PC | 95.9% | 4 |
| phone | 95.9% | 4 |

Phone misroutes:

| query | text | gold | predicted |
|---|---|---|---|
| q020 | that thing from last december | document | scene |
| q094 | chest x-ray | scene | document |
| q095 | my radiograph | scene | document |
| q096 | medical x-ray image | scene | document |

## Bucket classification (document vs scene, at indexing)

| platform | accuracy | gold → predicted |
|---|---|---|
| PC | 98.6% (618/627) | errors: document→scene: 1, scene→document: 8 |
| phone | 95.9% (601/627) | errors: scene→document: 26 |

Phone hedged 62 items into both branches (margin < τ).

## Document subtype tagging

| platform | overall | synthetic | fetched |
|---|---|---|---|
| PC | 70.7% (n=287) | 100.0% (n=180) | 21.5% (n=107) |
| phone | 72.1% (n=287) | 100.0% (n=180) | 25.2% (n=107) |

## OCR accuracy (normalised text vs ground truth)

| platform | engine | n | CER | WER | synthetic CER | fetched CER |
|---|---|---|---|---|---|---|
| PC | tesseract | 258 | 0.472 | 0.597 | 0.419 | 0.593 |
| phone | mlkit-devanagari | 258 | 0.575 | 0.751 | 0.514 | 0.717 |

`bill` ground truth is extracted field values, not a page transcription, so its CER can
exceed 1.0 on either platform; compare the synthetic column for recognition quality.

## Query latency (warm)

| platform | system | scope | n | mean ms | p50 ms | p95 ms | max ms |
|---|---|---|---|---|---|---|---|
| PC | routed | all | 98 | 78.7 | 78.4 | 166.9 | 333.2 |
| PC | routed | document | 49 | 28.7 | 3.1 | 91.2 | 102.2 |
| PC | routed | scene | 49 | 128.7 | 136.1 | 192.3 | 333.2 |
| PC | clip_only | all | 98 | 76.9 | 77.4 | 92.9 | 96.7 |
| PC | clip_only | scene | 98 | 76.9 | 77.4 | 92.9 | 96.7 |
| PC | bm25_only | all | 98 | 1.3 | 0.8 | 4.1 | 5.5 |
| PC | bm25_only | document | 98 | 1.3 | 0.8 | 4.1 | 5.5 |
| PC | filename | all | 98 | 1.0 | 0.8 | 2.0 | 2.6 |
| PC | filename | filename | 98 | 1.0 | 0.8 | 2.0 | 2.6 |
| phone | routed | all | 98 | 117.9 | 119.7 | 282.0 | 328.4 |
| phone | routed | document | 49 | 46.7 | 3.3 | 163.8 | 175.9 |
| phone | routed | scene | 49 | 189.1 | 179.2 | 317.2 | 328.4 |
| phone | clip_only | all | 98 | 117.1 | 98.1 | 179.4 | 193.3 |
| phone | clip_only | scene | 98 | 117.1 | 98.1 | 179.4 | 193.3 |
| phone | bm25_only | all | 98 | 1.2 | 0.7 | 3.0 | 10.8 |
| phone | bm25_only | document | 98 | 1.2 | 0.7 | 3.0 | 10.8 |

Phone cold start: the first scene query (loads the text encoder) took **909 ms**; the image encoder loaded in 458 ms.

## Indexing throughput

Each stage is reported against its own item count (OCR only runs on document/hedged items).

| stage | PC items | PC ms/image | phone items | phone ms/image |
|---|---|---|---|---|
| ingest | 627 | 34.1 | 627 | 14.5 |
| clip_encode | 627 | 103.7 | 627 | 656.1 |
| classify | 627 | 4.6 | 627 | 0.1 |
| embed_store | 627 | 0.1 | 627 | 1.3 |
| ocr | 393 | 580.8 | 369 | 274.0 |

| measure | PC | phone |
|---|---|---|
| end-to-end ms/image | 506.6 | 833.2 |
| images/sec | 1.97 | 1.20 |
| scene ms/image | 142.6 | 671.9 |
| document ms/image | 937.8 | 1024.2 |
| projected 10,000-item gallery | 1.41 h | 2.31 h |

Phone wall-clock for the whole corpus: **526 s**; per item mean 837 ms, p50 805, p95 1372, max 1591 ms.

Phone OCR per document (ML Kit `ocr_ms`): mean 268 ms, p50 202, p95 594, max 1019 ms over 369 items.

## Storage, memory and app size

| measure | PC | phone |
|---|---|---|
| index size | 5.2 MB (627 items) | 3.3 MB (627 items) |
| bytes per item | 8,362 | 5,256 |
| projected index, 10,000 items | 84 MB | 53 MB |
| vision-language model | 151.3 M params, 605 MB fp32 | image 38.1 MB + text 64.4 MB (INT8) |
| app process memory (PSS) | — | baseline 168 MB · peak indexing 478 MB · peak querying 589 MB |
| APK size (debug) | — | 199 MB |

Phone memory is sampled every 250 ms from the app process (instrumentation runs in it),
so it includes both ONNX sessions, ML Kit and the test harness itself.
