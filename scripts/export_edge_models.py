#!/usr/bin/env python3
"""Export and optimize MobileCLIP models to ONNX for Android on-device execution.

Exports:
1. Visual Encoder (FP32 and INT8 quantized)
2. Text Encoder (FP32 and INT8 quantized)
3. BPE Tokenizer assets for on-device tokenization
4. Runs full numerical validation comparing PyTorch vs ONNX vs INT8.
5. Patches the text encoder's int64 ArgMax for ONNX Runtime Android and installs
   the models + tokenizer into android/app/src/main/assets/ (gitignored).

Usage:
    python scripts/export_edge_models.py                 # export, verify, install
    python scripts/export_edge_models.py --install-only  # reuse models/edge/*.onnx
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
ANDROID_ASSETS = ROOT / "android" / "app" / "src" / "main" / "assets"


def patch_argmax_int32(src: Path, dst: Path) -> int:
    """Rewrite int64 ArgMax as Cast(int32) -> ArgMax -> Cast(int64).

    OpenCLIP's text encoder finds the EOT position with argmax over the token ids.
    The exported ArgMax reads the int64 `text` input directly, and ONNX Runtime's
    Android build has no int64 ArgMax kernel, so the session fails to load. Token
    ids are < 49408, so int32 is lossless. Returns the number of nodes patched.
    """
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(str(src))
    int64_inputs = {
        i.name for i in model.graph.input
        if i.type.tensor_type.elem_type == TensorProto.INT64
    }
    nodes = list(model.graph.node)
    out, patched = [], 0
    for node in nodes:
        if node.op_type == "ArgMax" and node.input[0] in int64_inputs:
            cast_in = node.input[0] + "_ArgMax_int32"
            argmax_out = node.output[0] + "_int32"
            out.append(helper.make_node("Cast", [node.input[0]], [cast_in], to=TensorProto.INT32))
            out.append(helper.make_node("ArgMax", [cast_in], [argmax_out], name=node.name,
                                        **{a.name: helper.get_attribute_value(a) for a in node.attribute}))
            out.append(helper.make_node("Cast", [argmax_out], [node.output[0]], to=TensorProto.INT64))
            patched += 1
        else:
            out.append(node)
    del model.graph.node[:]
    model.graph.node.extend(out)
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))
    return patched


def install_android_assets() -> None:
    """Patch the INT8 text encoder and copy everything the app loads into assets/."""
    models_dir = ANDROID_ASSETS / "models"
    tok_dir = ANDROID_ASSETS / "tokenizer"
    models_dir.mkdir(parents=True, exist_ok=True)
    tok_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(OUTPUT_DIR / "mobileclip_visual_int8.onnx", models_dir / "mobileclip_visual_int8.onnx")
    n = patch_argmax_int32(OUTPUT_DIR / "mobileclip_text_int8.onnx", models_dir / "mobileclip_text_int8.onnx")
    if n == 0:
        raise SystemExit("Expected an int64 ArgMax in the text encoder; found none. Check the export.")
    shutil.copy(OUTPUT_DIR / "tokenizer" / "bpe_simple_vocab_16e6.txt.gz", tok_dir / "bpe_simple_vocab_16e6.txt.gz")

    # The patched graph must still match the unpatched one numerically.
    tokens = open_clip.get_tokenizer("MobileCLIP2-S0")(["a photo of a beach", "give me my aadhaar card"]).numpy()
    ref = ort.InferenceSession(str(OUTPUT_DIR / "mobileclip_text_int8.onnx")).run(None, {"text": tokens})[0]
    got = ort.InferenceSession(str(models_dir / "mobileclip_text_int8.onnx")).run(None, {"text": tokens})[0]
    diff = float(np.max(np.abs(ref - got)))
    if diff > 1e-5:
        raise SystemExit(f"ArgMax patch changed text embeddings (max diff {diff:.2e})")

    print(f"\n=== Installed Android assets ({n} ArgMax node patched, max diff {diff:.1e}) ===")
    for f in sorted(ANDROID_ASSETS.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(ROOT)}  {f.stat().st_size / (1024**2):.1f} MB")


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

    print("\n=== 3. Quantizing Visual Encoder to INT8 (MatMul/Gemm only) ===")
    # Dynamic INT8 on the Conv layers destroys FastViT: on 170 SUN397 scenes the
    # all-ops INT8 model had cosine ~0.02 to PyTorch and chance-level zero-shot
    # accuracy (0.076 vs 0.776). Quantizing only MatMul/Gemm keeps cosine > 0.998
    # with identical accuracy, at 36 MB instead of 44 MB fp32.
    t0 = time.perf_counter()
    quantize_dynamic(
        str(visual_fp32_path),
        str(visual_int8_path),
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
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

    verify_fidelity(model, preprocess, tokenizer, visual_int8_path, text_int8_path)
    print("\n[SUCCESS] MobileCLIP models and tokenizer are ready for Android integration!")


# Gates against *broken* quantization, not a quality target. Measured 2026-11-14:
#   visual MatMul-only INT8: cosine >= 0.998, zero-shot top-1 unchanged (0.776).
#   text INT8: mean cosine ~0.97, worst query 0.82 ("pan card"); on 31 subtype
#   queries x 310 corpus images MAP 0.670 -> 0.645 vs FP32 text. Accepted because
#   the FP32 text encoder is 242 MB. The all-ops INT8 visual encoder that shipped in
#   app v1 scored cosine ~0.02 -- that is what these gates exist to stop.
VISUAL_MIN_COSINE = 0.99
TEXT_MIN_COSINE = 0.80


def _fidelity_images(preprocess, n: int = 24) -> list[torch.Tensor]:
    """Real photos when the corpus is present, otherwise structured synthetic images.

    Random-noise inputs (what this script used to check with) cannot reveal a broken
    encoder: every model maps noise to a near-arbitrary vector.
    """
    from PIL import Image, ImageDraw

    photos = sorted((ROOT / "data" / "raw" / "scenes").glob("*/*"))[:: 7][:n]
    if photos:
        return [preprocess(Image.open(p).convert("RGB")) for p in photos]
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        img = Image.new("RGB", (320, 240), tuple(int(c) for c in rng.integers(0, 255, 3)))
        d = ImageDraw.Draw(img)
        for _ in range(6):
            x, y = rng.integers(0, 280), rng.integers(0, 200)
            d.ellipse((x, y, x + 40, y + 40), fill=tuple(int(c) for c in rng.integers(0, 255, 3)))
        out.append(preprocess(img))
    return out


def verify_fidelity(model, preprocess, tokenizer, visual_int8: Path, text_int8: Path) -> None:
    """Fail the export if the quantized encoders drift from PyTorch."""
    print("\n=== 8. Quantization fidelity (INT8 ONNX vs PyTorch, cosine) ===")
    images = torch.stack(_fidelity_images(preprocess))
    texts = tokenizer(["give me my aadhaar card", "find a photo of a beach",
                       "tax invoice with gst number", "gym workout", "people at a wedding"])
    with torch.no_grad():
        ref_v = model.encode_image(images, normalize=True).numpy()
        ref_t = model.encode_text(texts, normalize=True).numpy()
    sv, st = ort.InferenceSession(str(visual_int8)), ort.InferenceSession(str(text_int8))
    got_v = np.vstack([sv.run(None, {"image": x[None].numpy()})[0] for x in images])
    got_t = st.run(None, {"text": texts.numpy()})[0]
    cos_v = np.sum(ref_v * got_v / np.linalg.norm(got_v, axis=1, keepdims=True), axis=1)
    cos_t = np.sum(ref_t * got_t / np.linalg.norm(got_t, axis=1, keepdims=True), axis=1)
    print(f"visual: mean {cos_v.mean():.4f}  min {cos_v.min():.4f}  (n={len(cos_v)}, need >= {VISUAL_MIN_COSINE})")
    print(f"text:   mean {cos_t.mean():.4f}  min {cos_t.min():.4f}  (n={len(cos_t)}, need >= {TEXT_MIN_COSINE})")
    if cos_v.min() < VISUAL_MIN_COSINE or cos_t.min() < TEXT_MIN_COSINE:
        raise SystemExit("Quantized encoders diverge from PyTorch; refusing to ship them.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export MobileCLIP to ONNX")
    parser.add_argument("--model", default="MobileCLIP2-S0", help="Model name")
    parser.add_argument("--pretrained", default="dfndr2b", help="Pretrained tag")
    parser.add_argument("--install-only", action="store_true",
                        help="Skip export; patch + install existing models/edge/*.onnx into Android assets")
    parser.add_argument("--no-install", action="store_true", help="Export only; do not touch Android assets")
    args = parser.parse_args()
    if not args.install_only:
        export_models(args.model, args.pretrained)
    if not args.no_install:
        install_android_assets()
