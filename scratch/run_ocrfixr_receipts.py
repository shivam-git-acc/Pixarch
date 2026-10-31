import json
import time
from ocrfixr.spellcheck import spellcheck

with open('scratch/receipts_extracted_easyocr.json', 'r', encoding='utf-8') as f:
    receipts = json.load(f)

results = []

start_time = time.time()
print(f"Running OCRfixr on {len(receipts)} receipts...")

for idx, r in enumerate(receipts):
    fname = r['filename']
    raw_text = r['raw_text']
    
    # Run spellcheck
    sc = spellcheck(raw_text, return_fixes="T")
    try:
        fixed_text, fixes = sc.fix()
    except Exception as e:
        print(f"Error on {fname}: {e}")
        fixed_text = raw_text
        fixes = {}
    
    # Compute what changed and what remained the same
    # Token-level diff analysis
    raw_tokens = raw_text.split()
    fixed_tokens = fixed_text.split()
    
    is_identical = (raw_text == fixed_text)
    
    item_res = {
        "filename": fname,
        "is_identical": is_identical,
        "fixes": {str(k): v for k, v in fixes.items()},
        "original_raw_text": raw_text,
        "corrected_text": fixed_text,
        "token_count": len(raw_tokens),
    }
    results.append(item_res)
    print(f"[{idx+1}/{len(receipts)}] {fname}: Identical={is_identical}, Fixes count={len(fixes)} ({fixes})")

elapsed = time.time() - start_time
print(f"\nDone in {elapsed:.2f} seconds!")

with open('scratch/ocrfixr_receipts_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

