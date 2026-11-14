#!/usr/bin/env python3
"""Generate Python-side reference outputs that the Android port is tested against.

Writes:
  android/app/src/test/resources/clip_tokenizer_fixture.tsv
      text<TAB>77 comma-separated token ids from open_clip's MobileCLIP2-S0 tokenizer.
      Checked on the JVM by ClipTokenizerTest (no device needed).

  android/app/src/test/resources/{router,fts_query,normalise,doctype}_fixture.tsv
      Python baseline behaviour of the rule router, the FTS5 query builder, OCR text
      normalisation and doc-type tagging (src/pixarch/...). Free text is base64 so
      tabs/newlines in OCR ground truth survive. Checked on the JVM.

  android/app/src/androidTest/assets/parity/bm25_*.tsv
      The 260 OCR ground-truth texts indexed with the desktop FTS5 schema, and the
      top-10 BM25 ranking Python's SQLite returns per query. Checked on device
      against requery's SQLite by Bm25ParityTest.

  android/app/src/androidTest/assets/parity/
      img_*.png + embeddings.tsv: reference embeddings produced by the *same* INT8 ONNX
      files the app ships, fed by OpenCLIP's own preprocessing. Any gap the on-device
      ClipParityTest sees is therefore down to Android preprocessing/tokenization alone.

Run after scripts/export_edge_models.py (it reads the installed Android assets).
Images are procedurally generated, so no dataset licence questions arise.
"""
from __future__ import annotations

import base64
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import open_clip
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "android" / "app" / "src"
ASSETS = APP / "main" / "assets"
TEST_RES = APP / "test" / "resources"
TOKENIZER_FIXTURE = TEST_RES / "clip_tokenizer_fixture.tsv"
PARITY_DIR = APP / "androidTest" / "assets" / "parity"
MODEL = "MobileCLIP2-S0"
PRETRAINED = "dfndr2b"

sys.path.insert(0, str(ROOT / "src"))
from pixarch.doctype import classify_text  # noqa: E402
from pixarch.ocr import normalise  # noqa: E402
from pixarch.query.bm25_search import to_match_query  # noqa: E402
from pixarch.query.router import QueryRouter  # noqa: E402

TOKENIZER_CASES = [
    "a photo of a beach",
    "give me my aadhaar card",
    "Find   my PAN card!!",
    "  beach photo 　 sunset at goa  ",  # Unicode whitespace collapse + strip
    "tax invoice with GST number 27AAPFU0939F1ZV",
    "photos from Dec 2023 at Goa's beach",
    "it's what they've done, we'll see, I'd go",
    "electricity bill — units consumed 345 kWh",
    "café naïve résumé",
    "आधार कार्ड दिखाओ",
    "मेरा पैन कार्ड",
    "emoji \U0001f3d6️ and \U0001f389 party",
    "<start_of_text> literal special tokens <end_of_text>",
    "supercalifragilisticexpialidocious antidisestablishmentarianism",
    "xyzzyqwv plugh frobnicate zxcvbnm",
    "the quick brown fox " * 30,  # longer than 77 tokens -> truncation, EOT last
    "",
]

PARITY_QUERIES = [
    "a photo of a beach",
    "give me my aadhaar card",
    "a screenshot of a chat",
    "red circle on a blue background",
]

# Beyond data/queries.jsonl: edge cases for routing and FTS query building.
EXTRA_QUERIES = [
    "photos of my passport", "beach party photos", "where is my", "show me my gym photos",
    "receipts", "pan", "company august", "find", "give me", "x-ray of my chest",
    "आधार कार्ड",
    "मेरा पैन कार्ड दिखाओ",
    'tax "invoice" (gst) -2023 ^total*', "my-dl:card",
    "", "   ", "the a an", "Electricity BILL March", "café receipt", "IMG_20231205_123456",
]

# Beyond data/ocr_groundtruth.jsonl: edge cases for normalise() and doc-type rules.
EXTRA_TEXTS = [
    "Hello, World!  Ünïcödé\tTABS\nnewlines",
    "½ price ² ③ Ⅻ",                      # No / Nl numerics
    "snake_case_name",
    "emoji \U0001f3d6️\U0001f389 between",
    "आधार — पहचान का "
    "प्रमाण है।",  # Devanagari + danda
    "a b　c",
    "  ...  ",
    "₹ 1,234.56 /- (Rs.)",
    "P<IND<<SURNAME<<GIVEN",
    "Copyright © 2020 ISBN 978-3-16",
    "\U0001d400\U0001d401\U0001d402 mathematical bold",       # supplementary-plane letters
    "UIDAI unique identification authority of india",
    "company in august paid gst",                               # whole-word short terms
]


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _queries() -> list[str]:
    with (ROOT / "data" / "queries.jsonl").open(encoding="utf-8") as fh:
        return [json.loads(line)["text"] for line in fh if line.strip()] + EXTRA_QUERIES


def _ocr_texts() -> list[tuple[str, str]]:
    with (ROOT / "data" / "ocr_groundtruth.jsonl").open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return [(r["item_id"], r["text"]) for r in rows] + [(f"extra_{i}", t) for i, t in enumerate(EXTRA_TEXTS)]


def _open(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8", newline="\n")


def make_images() -> dict[str, Image.Image]:
    rng = np.random.default_rng(1234)
    imgs = {}

    # Landscape photo-like: sky/sand gradient with shapes (exercises downscale + crop).
    w, h = 1280, 960
    y = np.linspace(0, 1, h)[:, None]
    arr = np.zeros((h, w, 3), np.float32)
    arr[..., 0] = 90 + 150 * y
    arr[..., 1] = 160 + 60 * y
    arr[..., 2] = 235 - 120 * y
    arr += rng.normal(0, 6, arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.ellipse((900, 80, 1080, 260), fill=(250, 220, 90))
    d.rectangle((0, 700, w, h), fill=(225, 200, 150))
    imgs["img_landscape_1280x960.png"] = img

    # Tall screenshot-like page with text lines (exercises portrait crop offset).
    w, h = 720, 1600
    img = Image.new("RGB", (w, h), (245, 245, 245))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, w, 140), fill=(30, 120, 90))
    for i in range(40):
        d.text((40, 180 + i * 34), f"Message {i}: payment of Rs {(i * 137) % 9000} received", fill=(20, 20, 20))
    imgs["img_screenshot_720x1600.png"] = img

    # Small, odd-sized image (exercises upscale and the half-pixel crop rounding).
    w, h = 301, 203
    img = Image.new("RGB", (w, h), (20, 40, 160))
    d = ImageDraw.Draw(img)
    d.ellipse((90, 40, 210, 160), fill=(220, 30, 30))
    imgs["img_small_301x203.png"] = img
    return imgs


def write_tokenizer_fixture() -> None:
    tok = open_clip.get_tokenizer(MODEL)
    with _open(TOKENIZER_FIXTURE) as f:
        for text in TOKENIZER_CASES:
            assert "\t" not in text and "\n" not in text
            ids = tok([text])[0].tolist()
            f.write(f"{text}\t{','.join(map(str, ids))}\n")
    print(f"wrote {TOKENIZER_FIXTURE.relative_to(ROOT)} ({len(TOKENIZER_CASES)} cases)")


def write_pipeline_fixtures() -> None:
    router = QueryRouter()  # no encoder: pure rules, "default" where CLIP would decide
    queries = _queries()
    with _open(TEST_RES / "router_fixture.tsv") as f:
        for q in queries:
            r = router.route(q)
            f.write(f"{_b64(q)}\t{r.branch}\t{r.confidence}\t{r.reason}\n")
    with _open(TEST_RES / "fts_query_fixture.tsv") as f:
        for q in queries:
            f.write(f"{_b64(q)}\t{_b64(to_match_query(q))}\n")
    texts = _ocr_texts()
    with _open(TEST_RES / "normalise_fixture.tsv") as f:
        for _, t in texts:
            f.write(f"{_b64(t)}\t{_b64(normalise(t))}\n")
    with _open(TEST_RES / "doctype_fixture.tsv") as f:
        for _, t in texts:
            n = normalise(t)
            doc_type, rule, weight = classify_text(n)
            f.write(f"{_b64(n)}\t{doc_type}\t{rule}\t{weight}\n")
    print(f"wrote router/fts_query fixtures ({len(queries)} queries), "
          f"normalise/doctype fixtures ({len(texts)} texts)")


def write_bm25_fixture() -> None:
    """Index OCR ground truth with the desktop schema; record Python's BM25 rankings."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE VIRTUAL TABLE doc_fts USING fts5(item_id UNINDEXED, ocr_text, doc_type, "
        "tokenize = 'porter unicode61 remove_diacritics 2')"
    )
    docs = [(i, normalise(t)) for i, t in _ocr_texts() if not i.startswith("extra_")]
    docs = [(i, n, classify_text(n)[0]) for i, n in docs]
    conn.executemany("INSERT INTO doc_fts VALUES (?,?,?)", docs)
    with _open(PARITY_DIR / "bm25_docs.tsv") as f:
        for i, n, d in docs:
            f.write(f"{i}\t{_b64(n)}\t{d}\n")
    sql = ("SELECT item_id, bm25(doc_fts, 0.0, 1.0, 5.0) AS s FROM doc_fts "
           "WHERE doc_fts MATCH ? ORDER BY s LIMIT 10")
    n_q = 0
    with _open(PARITY_DIR / "bm25_expected.tsv") as f:
        for q in _queries():
            match = to_match_query(q)
            if not match:
                continue
            try:
                rows = conn.execute(sql, (match,)).fetchall()
            except sqlite3.OperationalError:
                continue  # bm25_search.search turns a malformed MATCH into no hits
            ids = ",".join(r[0] for r in rows)
            scores = ",".join(f"{-r[1]:.6f}" for r in rows)
            f.write(f"{_b64(q)}\t{ids}\t{scores}\n")
            n_q += 1
    print(f"wrote bm25 parity fixture ({len(docs)} docs, {n_q} queries, SQLite {sqlite3.sqlite_version})")


def write_parity_fixture() -> None:
    _, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
    tok = open_clip.get_tokenizer(MODEL)
    vis = ort.InferenceSession(str(ASSETS / "models" / "mobileclip_visual_int8.onnx"))
    txt = ort.InferenceSession(str(ASSETS / "models" / "mobileclip_text_int8.onnx"))

    PARITY_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, img in make_images().items():
        img.save(PARITY_DIR / name)
        # Reload from disk so Python sees exactly the bytes the device decodes.
        x = preprocess(Image.open(PARITY_DIR / name)).unsqueeze(0).numpy()
        emb = vis.run(None, {"image": x})[0][0]
        rows.append(("image", name, emb / np.linalg.norm(emb)))
    for q in PARITY_QUERIES:
        emb = txt.run(None, {"text": tok([q]).numpy()})[0][0]
        rows.append(("text", q, emb / np.linalg.norm(emb)))

    with _open(PARITY_DIR / "embeddings.tsv") as f:
        for kind, key, emb in rows:
            f.write(f"{kind}\t{key}\t{','.join(f'{v:.7f}' for v in emb)}\n")
    print(f"wrote {PARITY_DIR.relative_to(ROOT)} ({len(rows)} reference embeddings)")


# Synthetic (CC0, generated by src/pixarch/synthetic.py) documents with exact ground
# truth; aadhaar/pan/passport carry Devanagari. Used to compare ML Kit OCR modes.
OCR_FIXTURE_IDS = [
    "doc_aadhaar_100", "doc_aadhaar_101", "doc_aadhaar_102",
    "doc_pan_100", "doc_pan_101", "doc_pan_102",
    "doc_passport_100", "doc_passport_101", "doc_passport_102",
    "doc_dl_100", "doc_dl_101",
    "doc_credit_card_000", "doc_debit_card_000",
    "doc_bank_statement_000", "doc_report_card_000",
]


def write_ocr_fixture() -> None:
    import csv
    import shutil

    gt = {}
    with (ROOT / "data" / "ocr_groundtruth.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                gt[r["item_id"]] = r["text"]
    with (ROOT / "data" / "manifest.csv").open(newline="", encoding="utf-8") as fh:
        manifest = {r["item_id"]: r for r in csv.DictReader(fh)}
    out = PARITY_DIR.parent / "ocr"
    out.mkdir(parents=True, exist_ok=True)
    with _open(out / "groundtruth.tsv") as f:
        for item_id in OCR_FIXTURE_IDS:
            row = manifest[item_id]
            assert row["origin"] == "synthetic", f"{item_id} is not synthetic; keep the fixture CC0"
            src = ROOT / "data" / "raw" / row["path"]
            if not src.exists():
                raise SystemExit(f"missing {src}; run scripts/fetch_dataset.py")
            name = f"{item_id}{src.suffix}"
            shutil.copy(src, out / name)
            f.write(f"{name}\t{row['subtype']}\t{_b64(gt[item_id])}\n")
    print(f"wrote {out.relative_to(ROOT)} ({len(OCR_FIXTURE_IDS)} synthetic documents)")


if __name__ == "__main__":
    write_tokenizer_fixture()
    write_pipeline_fixtures()
    write_bm25_fixture()
    write_ocr_fixture()
    write_parity_fixture()
