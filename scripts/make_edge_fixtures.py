#!/usr/bin/env python3
"""Generate Python-side reference outputs that the Android port is tested against.

Writes:
  android/app/src/test/resources/clip_tokenizer_fixture.tsv
      text<TAB>77 comma-separated token ids from open_clip's MobileCLIP2-S0 tokenizer.
      Checked on the JVM by ClipTokenizerTest (no device needed).

  android/app/src/androidTest/assets/parity/
      img_*.png + embeddings.tsv: reference embeddings produced by the *same* INT8 ONNX
      files the app ships, fed by OpenCLIP's own preprocessing. Any gap the on-device
      ClipParityTest sees is therefore down to Android preprocessing/tokenization alone.

Run after scripts/export_edge_models.py (it reads the installed Android assets).
Images are procedurally generated, so no dataset licence questions arise.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import open_clip
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "android" / "app" / "src"
ASSETS = APP / "main" / "assets"
TOKENIZER_FIXTURE = APP / "test" / "resources" / "clip_tokenizer_fixture.tsv"
PARITY_DIR = APP / "androidTest" / "assets" / "parity"
MODEL = "MobileCLIP2-S0"
PRETRAINED = "dfndr2b"

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
    "emoji 🏖️ and 🎉 party",
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
    TOKENIZER_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    with TOKENIZER_FIXTURE.open("w", encoding="utf-8", newline="\n") as f:
        for text in TOKENIZER_CASES:
            assert "\t" not in text and "\n" not in text
            ids = tok([text])[0].tolist()
            f.write(f"{text}\t{','.join(map(str, ids))}\n")
    print(f"wrote {TOKENIZER_FIXTURE.relative_to(ROOT)} ({len(TOKENIZER_CASES)} cases)")


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

    with (PARITY_DIR / "embeddings.tsv").open("w", encoding="utf-8", newline="\n") as f:
        for kind, key, emb in rows:
            f.write(f"{kind}\t{key}\t{','.join(f'{v:.7f}' for v in emb)}\n")
    print(f"wrote {PARITY_DIR.relative_to(ROOT)} ({len(rows)} reference embeddings)")


if __name__ == "__main__":
    write_tokenizer_fixture()
    write_parity_fixture()
