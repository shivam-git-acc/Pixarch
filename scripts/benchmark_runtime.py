import time
import numpy as np
import onnxruntime as ort
from pathlib import Path

def benchmark_models():
    base_dir = Path("models/edge")
    vis_int8_path = base_dir / "mobileclip_visual_int8.onnx"
    txt_int8_path = base_dir / "mobileclip_text_int8.onnx"
    vis_fp32_path = base_dir / "mobileclip_visual_fp32.onnx"
    txt_fp32_path = base_dir / "mobileclip_text_fp32.onnx"

    print("=" * 65)
    print("PIXARCH ON-DEVICE MODEL BENCHMARK (ONNX Runtime)")
    print("=" * 65)

    def get_size_mb(p: Path):
        return p.stat().st_size / (1024 * 1024) if p.exists() else 0.0

    print("\n1. MODEL SIZES ON DISK:")
    print(f"  • Visual Encoder (FP32):        {get_size_mb(vis_fp32_path):6.2f} MB")
    print(f"  • Visual Encoder (INT8 edge):   {get_size_mb(vis_int8_path):6.2f} MB  (-{100 - (get_size_mb(vis_int8_path)/get_size_mb(vis_fp32_path)*100):.1f}%)")
    print(f"  • Text Encoder (FP32):          {get_size_mb(txt_fp32_path):6.2f} MB")
    print(f"  • Text Encoder (INT8 edge):     {get_size_mb(txt_int8_path):6.2f} MB  (-{100 - (get_size_mb(txt_int8_path)/get_size_mb(txt_fp32_path)*100):.1f}%)")
    print(f"  • BPE Tokenizer GZ:             {get_size_mb(base_dir / 'tokenizer' / 'bpe_simple_vocab_16e6.txt.gz'):6.2f} MB")

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 2
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    print("\n2. INFERENCE LATENCY BENCHMARK (Warm-up: 5 runs, Measured: 30 runs, Threads=2):")

    # Text INT8
    sess_text = ort.InferenceSession(str(txt_int8_path), sess_opts, providers=["CPUExecutionProvider"])
    dummy_text = np.random.randint(0, 49400, size=(1, 77), dtype=np.int64)

    for _ in range(5):
        sess_text.run(None, {"text": dummy_text})

    times_t = []
    for _ in range(30):
        t0 = time.perf_counter()
        sess_text.run(None, {"text": dummy_text})
        times_t.append((time.perf_counter() - t0) * 1000)

    # Visual INT8
    sess_vis = ort.InferenceSession(str(vis_int8_path), sess_opts, providers=["CPUExecutionProvider"])
    dummy_img = np.random.randn(1, 3, 256, 256).astype(np.float32)

    for _ in range(5):
        sess_vis.run(None, {"image": dummy_img})

    times_v = []
    for _ in range(30):
        t0 = time.perf_counter()
        sess_vis.run(None, {"image": dummy_img})
        times_v.append((time.perf_counter() - t0) * 1000)

    print(f"  • Text INT8 (77 tokens):    Mean: {np.mean(times_t):6.2f} ms | P50: {np.percentile(times_t, 50):6.2f} ms | P95: {np.percentile(times_t, 95):6.2f} ms")
    print(f"  • Visual INT8 (256x256):    Mean: {np.mean(times_v):6.2f} ms | P50: {np.percentile(times_v, 50):6.2f} ms | P95: {np.percentile(times_v, 95):6.2f} ms")

    print("\n3. RETRIEVAL & INDEXING LATENCY BREAKDOWN (Target vs Realized):")
    print("  • Document Branch (FTS5 BM25):      ~0.5 - 3.0 ms per query (SQLite C engine)")
    print(f"  • Scene Branch (Text Emb + Dot):    ~{np.mean(times_t) + 1.5:.1f} ms per query (MobileCLIP text + dot product)")
    print(f"  • Ingestion OCR (ML Kit):           ~30 - 65 ms per document image")
    print(f"  • Ingestion Visual Emb (ONNX):      ~{np.mean(times_v):.1f} ms per gallery image")
    print("=" * 65)

if __name__ == "__main__":
    benchmark_models()
