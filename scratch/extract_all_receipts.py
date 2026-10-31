import json
import sys
import time
from pathlib import Path

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pixarch.ocr import EasyOCREngine, run_ocr
from pixarch.doctype import classify_text

receipt_dir = ROOT / "data/raw/documents/receipt"
receipt_files = sorted(list(receipt_dir.glob("doc_receipt_*.jpg")))

print(f"Initializing EasyOCR Engine (languages=['en'], gpu=False)...")
engine = EasyOCREngine(languages=["en"], gpu=False)

results = []
# Process the receipt images
for i, path in enumerate(receipt_files):
    name = path.stem
    print(f"\n[{i+1}/{len(receipt_files)}] Processing {name} ({path.name})...")
    blocks, raw_text, norm_text, ms = run_ocr(engine, path, min_conf=30)
    dt, rule, score = classify_text(norm_text)
    
    item_res = {
        "item_id": name,
        "filename": path.name,
        "elapsed_ms": round(ms, 1),
        "block_count": len(blocks),
        "doc_type": dt,
        "rule": rule,
        "raw_text": raw_text,
        "blocks": [{"text": b.text, "conf": round(b.conf, 1), "box": list(b.box)} for b in blocks]
    }
    results.append(item_res)
    print(f"  Extracted {len(blocks)} blocks in {ms:.1f}ms | Classified as: {dt} ({rule})")
    print(f"  Snippet: {raw_text[:120]}...")

output_path = ROOT / "scratch/receipts_extracted_easyocr.json"
output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nSaved all {len(results)} receipt extraction results to {output_path}")

