#!/usr/bin/env python3
"""Calibrate the document/scene bucket classifier for the on-device MobileCLIP2 model.

The baseline's threshold (configs/baseline.yaml: classify.confidence_threshold) was
fitted to ViT-B-32 margins. MobileCLIP2-S0 produces differently scaled margins, so the
app needs its own value, derived the same way (design-decisions.md, "Calibrating the
classifier threshold"): argmax decides the bucket, and the threshold is a narrow hedge
that sends the least-confident ~10% of items to both branches.

Steps:
  1. Prototypes = mean of normalised MobileCLIP2 text embeddings of each bucket's
     prompts (configs/baseline.yaml: edge.classify.prompts), renormalised (same maths as classify.py::BucketClassifier._prototype).
  2. Encode every corpus image with the *shipped* INT8 visual ONNX + OpenCLIP
     preprocessing, so the margins are the ones the phone will see.
  3. Report argmax accuracy vs gold_bucket and margin percentiles; pick the threshold.
  4. Write android/app/src/main/assets/models/bucket_prototypes.tsv.

    python scripts/calibrate_edge_buckets.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import onnxruntime as ort
import open_clip
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "baseline.yaml"
ASSET = ROOT / "android" / "app" / "src" / "main" / "assets" / "classify" / "bucket_prototypes.tsv"
VISUAL = ROOT / "android" / "app" / "src" / "main" / "assets" / "models" / "mobileclip_visual_int8.onnx"
MODEL, PRETRAINED = "MobileCLIP2-S0", "dfndr2b"


def prototypes(model, tokenizer, prompts: dict[str, list[str]]) -> dict[str, np.ndarray]:
    out = {}
    with torch.no_grad():
        for bucket in sorted(prompts):
            emb = model.encode_text(tokenizer(prompts[bucket]), normalize=True).numpy().mean(axis=0)
            out[bucket] = emb / max(float(np.linalg.norm(emb)), 1e-12)
    return out


def corpus(cfg: dict) -> list[tuple[str, str, Path]]:
    raw = ROOT / cfg["paths"]["raw_dir"]
    with (ROOT / cfg["paths"]["manifest"]).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [(r["item_id"], r["bucket"], raw / r["path"]) for r in rows if (raw / r["path"]).exists()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hedge", type=float, default=None,
                    help="fraction of items to route to both branches (default: config)")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    edge = cfg["edge"]["classify"]
    hedge = args.hedge if args.hedge is not None else float(edge["hedge_fraction"])
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
    model.eval()
    protos = prototypes(model, open_clip.get_tokenizer(MODEL), edge["prompts"])
    buckets = sorted(protos)
    P = np.vstack([protos[b] for b in buckets])

    items = corpus(cfg)
    if not items:
        raise SystemExit("No corpus images under data/raw -- run scripts/fetch_dataset.py first.")
    session = ort.InferenceSession(str(VISUAL))
    gold, pred, margins = [], [], []
    for n, (_, bucket, path) in enumerate(items, 1):
        x = preprocess(Image.open(path).convert("RGB"))[None].numpy()
        v = session.run(None, {"image": x})[0][0]
        s = P @ (v / np.linalg.norm(v))
        order = np.argsort(-s)
        gold.append(bucket)
        pred.append(buckets[order[0]])
        margins.append(float(s[order[0]] - s[order[1]]))
        if n % 100 == 0:
            print(f"  encoded {n}/{len(items)}")
    gold, pred, margins = np.array(gold), np.array(pred), np.array(margins)

    correct = gold == pred
    print(f"\nItems: {len(items)}   argmax accuracy: {correct.mean():.3f} ({correct.sum()}/{len(items)})")
    for g in buckets:
        for p in buckets:
            print(f"  gold {g:9s} -> {p:9s} {int(((gold == g) & (pred == p)).sum())}")
    print("\nMargin by gold bucket (min / p10 / median / max):")
    for b in buckets:
        m = margins[gold == b]
        print(f"  {b:9s} {m.min():.4f} / {np.percentile(m, 10):.4f} / {np.median(m):.4f} / {m.max():.4f}")
    wrong = margins[~correct]
    if wrong.size:
        print(f"  misclassified items: median margin {np.median(wrong):.4f}, max {wrong.max():.4f}")

    tau = float(np.quantile(margins, hedge))
    hedged = margins < tau
    rescued = int((~correct & hedged).sum())
    print(f"\nThreshold (hedge {hedge:.0%}): {tau:.4f} -> {int(hedged.sum())} items to both branches; "
          f"{rescued}/{int((~correct).sum())} misclassified items rescued by hedging")

    ASSET.parent.mkdir(parents=True, exist_ok=True)
    with ASSET.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"#threshold\t{tau:.6f}\n")
        f.write(f"#source\t{MODEL}/{PRETRAINED} prompts=configs/baseline.yaml:edge n={len(items)} "
                f"argmax_acc={correct.mean():.4f}\n")
        for b in buckets:
            f.write(f"{b}\t{','.join(f'{v:.7f}' for v in protos[b])}\n")
    print(f"wrote {ASSET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
