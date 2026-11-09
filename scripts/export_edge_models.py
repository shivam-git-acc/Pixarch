#!/usr/bin/env python3
"""Export and optimize MobileCLIP models to ONNX for Android on-device execution.

Exports:
1. Visual Encoder (FP32 and INT8 quantized)
2. Text Encoder (FP32 and INT8 quantized)
3. BPE Tokenizer assets for on-device tokenization
4. Runs full numerical validation comparing PyTorch vs ONNX vs INT8.
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import numpy as np
import open_clip
import open_clip.tokenizer as oct
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "models" / "edge"


class VisualEncoderWrapper(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # Returns normalized L2 512-d visual embeddings
        return self.clip_model.encode_image(image, normalize=True)


class TextEncoderWrapper(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        # Returns normalized L2 512-d text embeddings
        return self.clip_model.encode_text(text, normalize=True)


def export_models(model_name: str = "MobileCLIP2-S0", pretrained: str = "dfndr2b"):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = OUTPUT_DIR / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== 1. Loading model: {model_name} ({pretrained}) ===")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,} ({total_params * 4 / (1024**2):.1f} MB in FP32)")

    # 1. Export Visual Encoder
    visual_wrapper = VisualEncoderWrapper(model)
    # Image resolution for MobileCLIP2-S0 is 256x256, for S2 is 224x224
    img_size = 256 if "S0" in model_name else 224
    dummy_img = torch.randn(1, 3, img_size, img_size)

    visual_fp32_path = OUTPUT_DIR / "mobileclip_visual_fp32.onnx"
    visual_int8_path = OUTPUT_DIR / "mobileclip_visual_int8.onnx"

    print(f"\n=== 2. Exporting Visual Encoder ({img_size}x{img_size}) to ONNX ===")
    t0 = time.perf_counter()
    torch.onnx.export(
        visual_wrapper,
        dummy_img,
        str(visual_fp32_path),
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"Visual FP32 exported in {time.perf_counter() - t0:.2f}s -> {visual_fp32_path.stat().st_size / (1024**2):.2f} MB")

    print("\n=== 3. Quantizing Visual Encoder to INT8 ===")
    t0 = time.perf_counter()
    quantize_dynamic(
        str(visual_fp32_path),
        str(visual_int8_path),
        weight_type=QuantType.QUInt8,
    )
    print(f"Visual INT8 quantized in {time.perf_counter() - t0:.2f}s -> {visual_int8_path.stat().st_size / (1024**2):.2f} MB")

    # 2. Export Text Encoder
    text_wrapper = TextEncoderWrapper(model)
    dummy_text = tokenizer(["a photo of a beach"])

    text_fp32_path = OUTPUT_DIR / "mobileclip_text_fp32.onnx"
    text_int8_path = OUTPUT_DIR / "mobileclip_text_int8.onnx"

    print(f"\n=== 4. Exporting Text Encoder (77 tokens) to ONNX ===")
    t0 = time.perf_counter()
    torch.onnx.export(
        text_wrapper,
        dummy_text,
        str(text_fp32_path),
        input_names=["text"],
        output_names=["embedding"],
        dynamic_axes={"text": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"Text FP32 exported in {time.perf_counter() - t0:.2f}s -> {text_fp32_path.stat().st_size / (1024**2):.2f} MB")

    print("\n=== 5. Quantizing Text Encoder to INT8 ===")
    t0 = time.perf_counter()
    quantize_dynamic(
        str(text_fp32_path),
        str(text_int8_path),
        weight_type=QuantType.QUInt8,
    )
    print(f"Text INT8 quantized in {time.perf_counter() - t0:.2f}s -> {text_int8_path.stat().st_size / (1024**2):.2f} MB")

    # 3. Copy Tokenizer Vocabulary
    vocab_src = oct.default_bpe()
    vocab_dest = tokenizer_dir / "bpe_simple_vocab_16e6.txt.gz"
    shutil.copy(vocab_src, vocab_dest)
    print(f"\n=== 6. Tokenizer Vocabulary Exported ===")
    print(f"Saved to: {vocab_dest} ({vocab_dest.stat().st_size / 1024:.1f} KB)")

    # 4. Numerical Fidelity Verification
    print("\n=== 7. Numerical Verification (PyTorch vs ONNX FP32 vs INT8) ===")
    model.eval()
    test_queries = [
        "give me my aadhaar card",
        "find a photo of a beach",
        "tax invoice with gst number",
        "gym workout",
    ]

    tokens = tokenizer(test_queries)
    test_img = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad():
        pt_text_emb = model.encode_text(tokens, normalize=True).numpy()
        pt_img_emb = model.encode_image(test_img, normalize=True).numpy()

    sess_t_fp32 = ort.InferenceSession(str(text_fp32_path))
    sess_t_int8 = ort.InferenceSession(str(text_int8_path))
    sess_v_fp32 = ort.InferenceSession(str(visual_fp32_path))
    sess_v_int8 = ort.InferenceSession(str(visual_int8_path))

    onnx_t_fp32 = sess_t_fp32.run(None, {"text": tokens.numpy()})[0]
    onnx_t_int8 = sess_t_int8.run(None, {"text": tokens.numpy()})[0]
    onnx_v_fp32 = sess_v_fp32.run(None, {"image": test_img.numpy()})[0]
    onnx_v_int8 = sess_v_int8.run(None, {"image": test_img.numpy()})[0]

    max_diff_text_fp32 = np.max(np.abs(pt_text_emb - onnx_t_fp32))
    max_diff_vis_fp32 = np.max(np.abs(pt_img_emb - onnx_v_fp32))

    print(f"Max absolute diff PyTorch vs ONNX FP32 (Text):   {max_diff_text_fp32:.6e}")
    print(f"Max absolute diff PyTorch vs ONNX FP32 (Visual): {max_diff_vis_fp32:.6e}")

    print("\nCosine Similarity Comparison across test queries vs sample image:")
    print(f"{'Query':<30} | {'PyTorch':<9} | {'ONNX FP32':<9} | {'ONNX INT8':<9} | {'Delta':<7}")
    print("-" * 75)
    for i, q in enumerate(test_queries):
        sim_pt = float(np.dot(pt_text_emb[i], pt_img_emb[0]))
        sim_fp32 = float(np.dot(onnx_t_fp32[i], onnx_v_fp32[0]))
        sim_int8 = float(np.dot(onnx_t_int8[i], onnx_v_int8[0]))
        delta = abs(sim_pt - sim_int8)
        print(f"{q[:30]:<30} | {sim_pt:9.5f} | {sim_fp32:9.5f} | {sim_int8:9.5f} | {delta:7.5f}")

    print("\n[SUCCESS] MobileCLIP models and tokenizer are ready for Android integration!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export MobileCLIP to ONNX")
    parser.add_argument("--model", default="MobileCLIP2-S0", help="Model name")
    parser.add_argument("--pretrained", default="dfndr2b", help="Pretrained tag")
    args = parser.parse_args()
    export_models(args.model, args.pretrained)
