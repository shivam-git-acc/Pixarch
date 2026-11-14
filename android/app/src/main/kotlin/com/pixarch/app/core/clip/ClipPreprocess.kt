package com.pixarch.app.core.clip

import android.graphics.Bitmap
import kotlin.math.round

/**
 * Image preprocessing matching OpenCLIP's eval transform for MobileCLIP2-S0
 * (`open_clip.get_pretrained_cfg("MobileCLIP2-S0", "dfndr2b")`):
 *
 *   Resize(256, bilinear, shortest side) -> CenterCrop(256) -> ToTensor -> Normalize(mean=0, std=1)
 *
 * MobileCLIP2 is trained WITHOUT the usual OpenAI CLIP mean/std, so pixels are just
 * scaled to [0, 1]. Applying the OpenAI constants here shifts every input and quietly
 * degrades every image embedding.
 */
object ClipPreprocess {

    const val IMAGE_SIZE = 256
    val MEAN = floatArrayOf(0f, 0f, 0f)
    val STD = floatArrayOf(1f, 1f, 1f)

    data class Geometry(val scaledW: Int, val scaledH: Int, val cropLeft: Int, val cropTop: Int)

    /**
     * torchvision semantics: the short side becomes [size] and the long side is
     * truncated (`int(size * long / short)`); the crop offset is `int(round((dim - size) / 2))`
     * with Python's round-half-to-even, which `kotlin.math.round` also uses.
     */
    fun geometry(width: Int, height: Int, size: Int = IMAGE_SIZE): Geometry {
        require(width > 0 && height > 0) { "empty image ${width}x$height" }
        val (sw, sh) = if (width <= height) {
            size to (size.toLong() * height / width).toInt()
        } else {
            (size.toLong() * width / height).toInt() to size
        }
        val left = round((sw - size) / 2.0).toInt()
        val top = round((sh - size) / 2.0).toInt()
        return Geometry(sw, sh, left, top)
    }

    /** ARGB pixels (row-major, size x size) -> NCHW float RGB, normalised. */
    fun toChw(pixels: IntArray, size: Int = IMAGE_SIZE, mean: FloatArray = MEAN, std: FloatArray = STD): FloatArray {
        val plane = size * size
        require(pixels.size == plane) { "expected $plane pixels, got ${pixels.size}" }
        val out = FloatArray(3 * plane)
        for (i in 0 until plane) {
            val p = pixels[i]
            out[i] = (((p shr 16) and 0xFF) / 255f - mean[0]) / std[0]
            out[plane + i] = (((p shr 8) and 0xFF) / 255f - mean[1]) / std[1]
            out[2 * plane + i] = ((p and 0xFF) / 255f - mean[2]) / std[2]
        }
        return out
    }

    /**
     * Resize so the short side is [size], then center-crop to size x size.
     *
     * PIL's bilinear resize (what OpenCLIP uses) antialiases on downscale; Android's
     * filtered scaling only samples a 2x2 neighbourhood. Halving step by step until the
     * image is within 2x of the target approximates PIL closely enough to keep embeddings
     * aligned (checked on device by ClipParityTest).
     */
    fun resizeAndCrop(src: Bitmap, size: Int = IMAGE_SIZE): Bitmap {
        val g = geometry(src.width, src.height, size)
        var current = src
        while (current.width / 2 >= g.scaledW && current.height / 2 >= g.scaledH) {
            val half = Bitmap.createScaledBitmap(current, current.width / 2, current.height / 2, true)
            if (current !== src) current.recycle()
            current = half
        }
        val scaled = if (current.width == g.scaledW && current.height == g.scaledH) {
            current
        } else {
            Bitmap.createScaledBitmap(current, g.scaledW, g.scaledH, true).also {
                if (current !== src) current.recycle()
            }
        }
        if (scaled.width == size && scaled.height == size) return scaled
        val cropped = Bitmap.createBitmap(scaled, g.cropLeft, g.cropTop, size, size)
        if (scaled !== src && scaled !== cropped) scaled.recycle()
        return cropped
    }
}
