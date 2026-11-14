# Pixarch — Baseline Report

On-device agentic retrieval over personal data. This report covers the
**baseline**: the non-agentic, single-shot system every later contribution is
measured against. All numbers come from
[`docs/results/baseline_2026-10-31.md`](results/baseline_2026-10-31.md), which
`scripts/evaluate.py` regenerates from scratch.

> **Status: indicative, not publishable.** Relevance is single-judge and part of
> the corpus is generated. The numbers are good enough to catch regressions and
> size the on-device problem; they are not good enough to support a claim.

---

## 1. Labeled Dataset Creation

No dataset exists for mixed personal-archive search, so we built one. The corpus
grew from 127 items to **629 manifest rows / 627 indexed** across **31
subtypes** — 14 document, 17 scene.

`data/raw/` is gitignored and `data/manifest.csv` is committed, carrying
`source_url`, `license` and `attribution` per row, so the corpus is reproducible
by reference rather than by redistributing images. `scripts/fetch_dataset.py`
rebuilds it.

### Sources

| bucket | subtypes | source | licence |
|---|---|---|---|
| scene | beach, mountain, trees, vehicle, party, wedding, college, food, travel, fitness, mall, planes, outdoor_games, clothes, art, transit_hub | SUN397 via HF | research use |
| scene | xray | keremberke/chest-xray-classification | CC-BY-4.0 |
| document | receipt | ICDAR-2019 SROIE | CC-BY-4.0 |
| document | bill | katanaml-org/invoices-donut-data-v1 | CC-BY-4.0 |
| document | general_doc | FUNSD | research use |
| document | handwritten | IAM Handwriting DB | research use |
| document | book | Wikimedia Commons title pages | PD / CC |
| document | aadhaar, pan, dl, passport, credit_card, debit_card, passbook, bank_statement, report_card | **synthetic** | CC0 |

### Why nine classes are synthetic

Wikimedia Commons holds **nine** usable Indian identity specimens in total —
verified by search, not assumed. Beyond those, every image of an Aadhaar card,
PAN card, passport, licence, bank passbook or payment card on the open web is a
photograph of a **real person's** document, carrying their name, address, photo
and the number itself.

That is not merely a privacy preference here, it is a property of the
repository: `manifest.csv` is committed and stores `source_url`, and
`fetch_dataset.py` reconstructs the corpus from it. Fetching such images would
publish a permanent, public index of real people's identity documents and cause
every person who clones the repo to re-download them. India's Aadhaar Act 2016
additionally restricts sharing identity information. Building the evaluation
set would have been worse still: OCR ground truth means transcribing real
Aadhaar numbers into a committed file.

Generation solves all of this and produces **better** data: exact ground truth,
balanced classes (21 PAN instead of 1), and controllable difficulty.

**Safety properties of the generated values**
- Aadhaar numbers begin with **0 or 1** — UIDAI never issues those, so no
  generated number can collide with a real one.
- Card numbers are constructed to **fail the Luhn checksum**, so none can be a
  valid payment card.
- All person, bank and school names are fictional.

There is deliberately **no "SPECIMEN" watermark**. A token appearing in every
synthetic document and no real one is a label leak straight into BM25 — the
same defect as the filename leak `schema.sql` documents removing (worth
+0.125 p@1, measured, then deleted).

The nine real Commons specimens are kept at their original indices and excluded
from the synthetic pool, so they remain a held-out realism check.

### Handwriting

Every page-level handwriting dataset on HuggingFace is gated. Only line-level
sets are reachable, and a raw IAM line is ~2467×128 — a 19:1 strip that is
neither a note a user would photograph nor anything CLIP can route. Pages are
therefore **composed from real IAM handwriting lines**: the pixels are real, only
the layout is ours, and the ground truth is the concatenated line text.

### Ground truth and judgements

- `data/ocr_groundtruth.jsonl`: **260 rows** (was 20). The 180 synthetic
  documents carry character-exact text; SROIE, FUNSD and IAM supply the rest.
- `data/queries.jsonl`: **98 queries** (was 20), ≥3 per subtype, all 31 covered.
- `data/qrels.csv`: **2,183 judgements** (was 185), expanded by subtype.

`scripts/make_qrels.py` now fails loudly if any corpus subtype has no query —
the check that would have caught half the corpus being unevaluable when the
subtype count went 15 → 31.

---

## 2. Models used for PC simulation

| role | model | footprint |
|---|---|---|
| scene embedding + bucket routing | CLIP `ViT-B-32` / `laion2b_s34b_b79k` | **151.3 M params** — 605 MB fp32, 303 MB fp16, 151 MB int8 |
| OCR | Tesseract 5, `eng+hin` | system package |
| lexical index | SQLite FTS5 + BM25 | in-index, no model |

One CLIP instance serves **two** jobs — zero-shot bucket classification at
ingestion and scene retrieval at query time — which halves what the on-device
build must budget for.

Tesseract is the engine for every number in this report. EasyOCR remains
implemented behind the same protocol as a comparator but is not used here.

---

## 3. Finalized models for the edge device

| stage | desktop | Android |
|---|---|---|
| OCR | Tesseract `eng+hin` | **ML Kit Text Recognition v2** (Latin + Devanagari) |
| embedding | CLIP ViT-B-32 fp32 | same weights, fp16/int8 |
| index | SQLite FTS5 | SQLite FTS5 (ships with Android) |

All engines sit behind `OCREngine.extract(path) -> list[TextBlock]`, so the
Android build swaps OCR by changing one config key. FTS5 was chosen partly
because it is present on Android unchanged.

**Budget implied by the measurements.** CLIP at int8 is ~151 MB of RAM, and the
index projects to 85 MB for a 10,000-item gallery (§5). Ingestion is the real
constraint: 1.56 hours for 10,000 items on a desktop GPU, which is why the
architecture assumes heavy processing runs while the phone is idle and charging.

---

## 4. Baseline Architecture

```
image ──► ingest (hash, EXIF) ──► ① CLIP zero-shot classify
                                      │
                      ┌───────────────┴───────────────┐
                 document                          scene
                      │                               │
            OCR → doctype rules              CLIP image embedding
                      │                               │
          FTS5 inverted index                   vector store
                      │                               │
                    BM25                           cosine
```

At query time a **rule-first router** picks exactly one branch; CLIP prototype
similarity is consulted only when the keyword rules abstain.

Everything lands in a single `index.db`.

**Deliberately absent**, because each is a later contribution that must not be
pre-empted by the baseline it will be measured against:
no score fusion, no agentic loop, no retry or clarification, no face branch.

Two design choices that the measurements justify:
- **`doc_type` is its own FTS column weighted 5× against body text.** Raw BM25
  over OCR text is brittle when a scan is noisy and the one distinguishing
  phrase comes back mangled.
- **No filename column.** Indexing generated names leaked the class label into
  the ranking — worth +0.125 p@1, measured, then removed.

---

## 5. Results

Corpus 627 items · 98 queries · retrieval depth 107 · CLIP on CUDA · Tesseract.

### 5.1 Retrieval

| system | p@1 | p@5 | MRR@10 | nDCG@10 | R-prec | recall@10 |
|---|---|---|---|---|---|---|
| **routed (baseline)** | **0.773** | **0.781** | **0.803** | **0.773** | **0.693** | 0.369 |
| clip_only | 0.629 | 0.596 | 0.644 | 0.580 | 0.478 | 0.278 |
| bm25_only | 0.361 | 0.355 | 0.378 | 0.349 | 0.312 | 0.157 |
| filename | 0.299 | 0.299 | 0.299 | 0.299 | 0.298 | 0.140 |

Routing beats both single-model comparators overall: **+0.144 p@1** over
CLIP-only and **+0.412** over BM25-only.

**Per bucket** (macro-averaged within each bucket, then across buckets so each
counts once regardless of query count):

| system | document p@1 | scene p@1 | macro p@1 |
|---|---|---|---|
| routed | 0.739 | 0.804 | **0.772** |
| clip_only | 0.370 | **0.863** | 0.616 |
| bm25_only | **0.739** | 0.020 | 0.379 |

The honest reading: routing's value is that it is never catastrophic. Each
single-model system collapses on the bucket it was not built for — BM25 scores
0.020 on scenes, CLIP 0.370 on documents — while routing tracks the better of
the two everywhere. On scenes alone, CLIP-only is **better** than routed
(0.863 vs 0.804), because every routing error is a pure loss.

### 5.2 Recall with ceilings

Recall@k cannot exceed `k/|relevant|`. With ~20 items per subtype, recall@10
caps at 0.48, so the raw figure badly understates performance:

| system | recall@10 | ceiling@10 | **% of achievable** |
|---|---|---|---|
| routed | 0.369 | 0.480 | **77.0%** |
| clip_only | 0.278 | 0.480 | 57.9% |
| bm25_only | 0.157 | 0.480 | 32.7% |
| filename | 0.140 | 0.480 | 29.2% |

R-precision (self-normalising, no ceiling artefact) tells the same story:
**0.693** for routed against 0.478 CLIP-only.

### 5.3 Module benchmarks

**Bucket classification (CLIP zero-shot, ingestion):** 618/627 = **98.6%**.
8 scene → document, 1 document → scene.

**Query routing (CLIP + rules):** **95.9%** — 4 of 98 misrouted. Three are the
x-ray queries: the router has no radiography vocabulary, so CLIP decides, and a
chest film reads as "a scanned document". The fourth is the intentionally
unanswerable `q020`.

**Document subtype tagging (`doctype.py`) — 70.7% over 287 documents.** This
module had never been measured.

| origin | n | accuracy |
|---|---|---|
| synthetic | 180 | **100.0%** |
| fetched | 107 | **21.5%** |

| misclassification | n | cause |
|---|---|---|
| bill → receipt | 20 | invoices carry "total"/"invoice no"; no rule distinguishes them from receipts |
| book → unknown | 20 | `book-isbn` needs isbn/chapter/copyright; title pages have none |
| handwritten → unknown | 20 | no rule exists, and OCR on handwriting is weak |
| general_doc → unknown/receipt | 20 | generic forms have no distinguishing vocabulary |
| aadhaar/pan/passport → unknown | 4 | OCR failed to recover the key phrase |

The 100% / 21.5% split is the headline: the rules are effectively **perfect on
rendered text and poor on photographs**, so the aggregate 70.7% is an artefact
of the synthetic:fetched ratio, not a property of the module. `receipt`
precision of 0.391 — recall 1.000, but 20 bills and 8 forms swept in — is the
clearest single defect.

**OCR accuracy (Tesseract, 258 items with ground truth):**

| set | n | CER | WER |
|---|---|---|---|
| all | 258 | 0.471 | 0.596 |
| excluding `bill` | 238 | **0.420** | **0.527** |
| synthetic | 180 | 0.418 | 0.507 |
| fetched | 78 | 0.592 | 0.801 |

> `bill` is **not comparable**: upstream ground truth is extracted *field
> values*, not a page transcription, so correctly reading the whole page scores
> CER > 1.0. It is a ground-truth mismatch, not an OCR failure.

The synthetic-vs-fetched gap (0.418 vs 0.592 CER) quantifies the domain gap the
generated corpus carries: rendered text is roughly 30% easier than photographs
and scans. Best subtype is `passbook` (0.187); worst comparable is
`bank_statement` (0.655).

### 5.4 Query latency

| system | scope | n | mean | p50 | p95 | max |
|---|---|---|---|---|---|---|
| routed | all | 98 | 8.0 ms | 6.8 | 13.5 | 63.3 |
| routed | document | 49 | **4.1 ms** | 1.6 | 8.5 | 46.9 |
| routed | scene | 49 | **11.9 ms** | 12.4 | 15.2 | 63.3 |
| bm25_only | — | 98 | 0.6 ms | 0.4 | 1.5 | 2.2 |

The branches are bimodal by an order of magnitude: BM25 is a SQL query
(0.6 ms), while the scene branch pays a CLIP text-encoder forward pass
(11.9 ms). Both are far below interactive thresholds — **query time is not the
on-device constraint**.

### 5.5 Ingestion throughput

| stage | items | ms/image | **images/sec** |
|---|---|---|---|
| ingest | 627 | 29.1 | 34.41 |
| clip_encode | 627 | 43.2 | 23.14 |
| classify | 627 | 26.9 | 37.22 |
| embed_store | 627 | 0.2 | 4209.58 |
| **ocr** | 393 | **738.0** | **1.35** |
| **end-to-end** | 627 | **562.0** | **1.78** |

Stages run over different denominators: `ocr` touches 393 items (287 documents
plus 106 low-confidence items routed to both branches), the rest touch all 627.

| bucket | ms/image | images/sec |
|---|---|---|
| scene | 99.4 | 10.06 |
| document | 1110.0 | 0.90 |

**OCR is 82% of total indexing time** and makes documents ~11× more expensive
than scenes, so mixed-gallery throughput depends entirely on the mix.

Per-item OCR is long-tailed — mean 691 ms, p50 478 ms, **p95 1567 ms, max
3048 ms** — so the mean understates the worst case by 4×.

**Projected: 1.56 hours to index a 10,000-item gallery** at this corpus's 46%
document ratio (linear projection, desktop GPU).

### 5.6 Storage and memory

`index.db` = **5.3 MB** for 627 items.

| component | MB | share |
|---|---|---|
| embeddings | 2.58 | 48.3% |
| ocr_blocks (+ its autoindex) | 1.54 | 28.9% |
| doc_fields | 0.39 | 7.4% |
| doc_fts (inverted index) | 0.36 | 6.7% |
| items | 0.25 | 4.7% |

| measure | bytes |
|---|---|
| per item, overall | 8,525 |
| per item, shared (row + CLIP vector) | 5,840 |
| extra per document (OCR + FTS) | 5,866 |

**Projected index for 10,000 items:** 85 MB at this corpus's ratio; 58 MB all
scene, 117 MB all document. With CLIP int8 at ~151 MB resident, the whole system
fits comfortably in a mid-range phone's budget — the storage claim holds.

---

## 6. Repository

```bash
pip install -r requirements.txt
python scripts/fetch_dataset.py --per-class 20 --per-doc 20   # rebuild corpus
python scripts/make_qrels.py                                  # queries -> qrels
python scripts/build_index.py                                 # ingest + OCR + CLIP
python scripts/evaluate.py                                    # -> docs/results/
python app/demo.py                                            # Gradio UI
pytest                                                        # 94 tests
```

| path | contents |
|---|---|
| `src/pixarch/{ingest,classify,ocr,doctype}.py` | ingestion pipeline |
| `src/pixarch/synthetic.py` | synthetic specimen generation |
| `src/pixarch/store/` | SQLite schema, FTS5, vector store |
| `src/pixarch/query/` | router, BM25, CLIP search |
| `src/pixarch/eval/` | metrics, module benchmarks, resource accounting |
| `configs/baseline.yaml` | every tunable |
| `docs/design-decisions.md` | why each choice, and the traps hit |

---

## 7. Known limitations

1. **Relevance is single-judge** and declared by subtype, not per item. The
   proposal calls for multi-judge agreement; that is outstanding.
2. **The `filename` comparator is inflated.** The fetcher names files after
   their class (`scn_beach_020.jpg`), so path matching succeeds far more often
   than on a phone, where names look like `IMG_20231205_123456.jpg`. Treat
   0.299 p@1 as a loose upper bound on the trivial floor, not the real floor.
3. **`doctype` rules do not cover `bill`, `book`, `handwritten` or
   `general_doc`** — 72 of 287 documents are mistagged. The rules were written
   when those classes had no data.
4. **No screenshots**, though the proposal names them as a gap in existing
   benchmarks (20–40% of a real gallery). Deferred.
5. **Synthetic documents are rendered, not photographed.** The 0.418 vs 0.592
   CER gap bounds the optimism this introduces; the nine held-out real
   specimens are the check.
6. **x-ray queries misroute**, and handwriting retrieval fails outright — both
   visible in §5.3 and neither yet addressed.
