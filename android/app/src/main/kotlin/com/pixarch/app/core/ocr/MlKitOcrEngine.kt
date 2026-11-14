package com.pixarch.app.core.ocr

import android.graphics.Bitmap
import com.google.android.gms.tasks.Task
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.devanagari.DevanagariTextRecognizerOptions
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import com.pixarch.app.core.storage.OcrBlockRecord
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

data class OcrResult(
    val fullText: String,
    val blocks: List<OcrBlockRecord>,
    val durationMs: Float
)

/**
 * ML Kit Text Recognition v2.
 *
 * Default [Mode.DEVANAGARI]: that recognizer reads Latin as well, so one pass covers
 * English and Hindi text. OcrModeComparisonTest, 15 synthetic documents, Galaxy A17
 * (SM-A176B), 2026-11-14:
 *
 *   mode                    latinCER  devanagariRecall  docTypeAcc  ms/img
 *   LATIN                   0.508     0.000             1.000       370
 *   DEVANAGARI              0.515     1.000             1.000       475
 *   LATIN_PLUS_DEVANAGARI   0.508     1.000             1.000       771
 *
 * [Mode.LATIN_PLUS_DEVANAGARI] keeps the Latin output plus Devanagari-only blocks from
 * the second recognizer; it costs 60% more time for no measured gain.
 */
class MlKitOcrEngine(val mode: Mode = Mode.DEVANAGARI) : AutoCloseable {

    enum class Mode { LATIN, DEVANAGARI, LATIN_PLUS_DEVANAGARI }

    private val latin: TextRecognizer? =
        if (mode != Mode.DEVANAGARI) TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS) else null
    private val devanagari: TextRecognizer? =
        if (mode != Mode.LATIN) TextRecognition.getClient(DevanagariTextRecognizerOptions.Builder().build()) else null

    suspend fun extractText(bitmap: Bitmap): OcrResult {
        val t0 = System.nanoTime()
        val image = InputImage.fromBitmap(bitmap, 0)
        val blocks = when (mode) {
            Mode.LATIN -> latin!!.process(image).await().textBlocks
            Mode.DEVANAGARI -> devanagari!!.process(image).await().textBlocks
            Mode.LATIN_PLUS_DEVANAGARI -> {
                val lat = latin!!.process(image).await().textBlocks
                val dev = devanagari!!.process(image).await().textBlocks.filter { hasDevanagari(it.text) }
                lat + dev
            }
        }
        val records = blocks.mapIndexed { idx, block ->
            val box = block.boundingBox
            OcrBlockRecord(idx, block.text, conf = null, box?.left, box?.top, box?.right, box?.bottom)
        }
        return OcrResult(
            fullText = records.joinToString("\n") { it.text },
            blocks = records,
            durationMs = (System.nanoTime() - t0) / 1e6f
        )
    }

    override fun close() {
        latin?.close()
        devanagari?.close()
    }

    companion object {
        fun hasDevanagari(s: String): Boolean = s.any { it in 'ऀ'..'ॿ' }

        private suspend fun Task<Text>.await(): Text = suspendCancellableCoroutine { cont ->
            addOnSuccessListener { cont.resume(it) }
            addOnFailureListener { cont.resumeWithException(it) }
        }
    }
}
