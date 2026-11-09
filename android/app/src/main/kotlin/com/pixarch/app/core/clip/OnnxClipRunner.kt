package com.pixarch.app.core.clip

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.math.sqrt

/**
 * On-Device ONNX Runtime inference wrapper for MobileCLIP INT8 models.
 */
class OnnxClipRunner(
    private val context: Context,
    private val visualModelAsset: String = "models/mobileclip_visual_int8.onnx",
    private val textModelAsset: String = "models/mobileclip_text_int8.onnx",
    val imageSize: Int = 256
) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val visualSession: OrtSession
    private var textSession: OrtSession? = null

    init {
        val visualFile = copyAssetToTempFile(visualModelAsset, "mobileclip_visual.onnx")
        val opts = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setIntraOpNumThreads(2)
        }
        visualSession = env.createSession(visualFile.absolutePath, opts)

        // Eagerly initialize text session now that ArgMax int32 patch is applied
        val textFile = copyAssetToTempFile(textModelAsset, "mobileclip_text.onnx")
        textSession = env.createSession(textFile.absolutePath, opts)
    }

    @Synchronized
    private fun getTextSession(): OrtSession {
        var session = textSession
        if (session != null) return session
        val textFile = copyAssetToTempFile(textModelAsset, "mobileclip_text.onnx")
        val opts = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setIntraOpNumThreads(2)
        }
        session = env.createSession(textFile.absolutePath, opts)
        textSession = session
        return session
    }

    private fun copyAssetToTempFile(assetPath: String, tempFileName: String): File {
        val tempFile = File(context.cacheDir, tempFileName)
        val assetLength: Long = try {
            val fd = context.assets.openFd(assetPath)
            val len = fd.length
            fd.close()
            len
        } catch (_: Exception) {
            -1L
        }
        if (!tempFile.exists() || (assetLength > 0L && tempFile.length() != assetLength)) {
            context.assets.open(assetPath).use { input ->
                FileOutputStream(tempFile).use { output ->
                    input.copyTo(output)
                }
            }
        }
        return tempFile
    }

    private fun normalizeL2(vec: FloatArray): FloatArray {
        var normSq = 0f
        for (v in vec) normSq += v * v
        val norm = sqrt(maxOf(normSq, 1e-12f))
        for (i in vec.indices) vec[i] /= norm
        return vec
    }

    fun encodeText(tokens: LongArray): FloatArray {
        val session = getTextSession()
        val shape = longArrayOf(1, tokens.size.toLong())
        val tensor = OnnxTensor.createTensor(env, LongBuffer.wrap(tokens), shape)

        return try {
            tensor.use {
                val result = session.run(mapOf("text" to tensor))
                result.use {
                    val output = result[0].value as Array<FloatArray>
                    normalizeL2(output[0].clone())
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
            FloatArray(512)
        }
    }

    fun encodeImage(bitmap: Bitmap): FloatArray {
        val scaled = if (bitmap.width != imageSize || bitmap.height != imageSize) {
            Bitmap.createScaledBitmap(bitmap, imageSize, imageSize, true)
        } else {
            bitmap
        }

        val pixels = IntArray(imageSize * imageSize)
        scaled.getPixels(pixels, 0, imageSize, 0, 0, imageSize, imageSize)

        // MobileCLIP expects NCHW float32 RGB normalized with standard CLIP mean/std
        val meanR = 0.48145466f
        val meanG = 0.4578275f
        val meanB = 0.40821073f
        val stdR = 0.26862954f
        val stdG = 0.26130258f
        val stdB = 0.27577711f

        val floatBuffer = FloatBuffer.allocate(1 * 3 * imageSize * imageSize)

        // R plane
        for (p in pixels) {
            val r = ((p shr 16) and 0xFF) / 255.0f
            floatBuffer.put((r - meanR) / stdR)
        }
        // G plane
        for (p in pixels) {
            val g = ((p shr 8) and 0xFF) / 255.0f
            floatBuffer.put((g - meanG) / stdG)
        }
        // B plane
        for (p in pixels) {
            val b = (p and 0xFF) / 255.0f
            floatBuffer.put((b - meanB) / stdB)
        }
        floatBuffer.flip()

        val shape = longArrayOf(1, 3, imageSize.toLong(), imageSize.toLong())
        val tensor = OnnxTensor.createTensor(env, floatBuffer, shape)

        tensor.use {
            val result = visualSession.run(mapOf("image" to tensor))
            result.use {
                val output = result[0].value as Array<FloatArray>
                return normalizeL2(output[0].clone())
            }
        }
    }

    override fun close() {
        visualSession.close()
        textSession?.close()
        env.close()
    }
}
