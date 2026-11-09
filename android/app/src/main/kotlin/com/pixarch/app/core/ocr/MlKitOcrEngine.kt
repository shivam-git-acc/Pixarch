package com.pixarch.app.core.ocr

import android.graphics.Bitmap
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
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

class MlKitOcrEngine : AutoCloseable {

    private val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

    suspend fun extractText(bitmap: Bitmap): OcrResult = suspendCancellableCoroutine { cont ->
        val t0 = System.currentTimeMillis()
        val image = InputImage.fromBitmap(bitmap, 0)

        recognizer.process(image)
            .addOnSuccessListener { visionText ->
                val elapsed = (System.currentTimeMillis() - t0).toFloat()
                val blocks = mutableListOf<OcrBlockRecord>()

                visionText.textBlocks.forEachIndexed { idx, block ->
                    val box = block.boundingBox
                    blocks.add(
                        OcrBlockRecord(
                            blockIdx = idx,
                            text = block.text,
                            conf = null,
                            x0 = box?.left,
                            y0 = box?.top,
                            x1 = box?.right,
                            y1 = box?.bottom
                        )
                    )
                }

                cont.resume(
                    OcrResult(
                        fullText = visionText.text,
                        blocks = blocks,
                        durationMs = elapsed
                    )
                )
            }
            .addOnFailureListener { exception ->
                cont.resumeWithException(exception)
            }
    }

    override fun close() {
        recognizer.close()
    }
}
