package com.pixarch.app.core.ingest

import android.content.ContentResolver
import android.content.ContentUris
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.provider.MediaStore
import com.pixarch.app.core.clip.OnnxClipRunner
import com.pixarch.app.core.doctype.DocTypeClassifier
import com.pixarch.app.core.ocr.MlKitOcrEngine
import com.pixarch.app.core.storage.ItemRecord
import com.pixarch.app.core.storage.OcrBlockRecord
import com.pixarch.app.core.storage.PixarchDatabase
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.InputStream
import java.security.MessageDigest
import java.util.UUID

data class IndexProgress(
    val indexedCount: Int,
    val totalCount: Int,
    val currentItemPath: String,
    val isComplete: Boolean = false,
    val error: String? = null
)

class GalleryIndexer(
    private val context: Context,
    private val database: PixarchDatabase,
    private val ocrEngine: MlKitOcrEngine,
    private val clipRunner: OnnxClipRunner
) {
    suspend fun indexMediaStore(
        limit: Int = 100,
        onProgress: (IndexProgress) -> Unit = {}
    ): Int = withContext(Dispatchers.IO) {
        val resolver: ContentResolver = context.contentResolver
        val projection = arrayOf(
            MediaStore.Images.Media._ID,
            MediaStore.Images.Media.DISPLAY_NAME,
            MediaStore.Images.Media.SIZE,
            MediaStore.Images.Media.DATE_ADDED
        )
        val sortOrder = "${MediaStore.Images.Media.DATE_ADDED} DESC"

        val imageUris = mutableListOf<Pair<Uri, String>>()
        resolver.query(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            projection,
            null,
            null,
            sortOrder
        )?.use { cursor ->
            val idColumn = cursor.getColumnIndexOrThrow(MediaStore.Images.Media._ID)
            val nameColumn = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DISPLAY_NAME)
            while (cursor.moveToNext() && imageUris.size < limit) {
                val id = cursor.getLong(idColumn)
                val name = cursor.getString(nameColumn) ?: "image_$id.jpg"
                val uri = ContentUris.withAppendedId(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, id)
                imageUris.add(uri to name)
            }
        }

        var processed = 0
        val total = imageUris.size

        for ((index, pair) in imageUris.withIndex()) {
            val (uri, name) = pair
            try {
                indexSingleImage(uri, name)
                processed++
            } catch (e: Exception) {
                e.printStackTrace()
            }
            onProgress(
                IndexProgress(
                    indexedCount = index + 1,
                    totalCount = total,
                    currentItemPath = name
                )
            )
        }

        onProgress(
            IndexProgress(
                indexedCount = processed,
                totalCount = total,
                currentItemPath = "Complete",
                isComplete = true
            )
        )
        processed
    }

    suspend fun indexSingleImage(uri: Uri, name: String): Boolean = withContext(Dispatchers.IO) {
        val resolver = context.contentResolver

        // 1. Calculate SHA-256
        val sha256 = calculateSha256(resolver, uri) ?: return@withContext false

        // 2. Check deduplication
        if (database.hasItemWithSha256(sha256)) {
            return@withContext false
        }

        // 3. Decode Bitmap with downsampling
        val bitmap = decodeSampledBitmapFromUri(resolver, uri, maxDimension = 1280)
            ?: return@withContext false

        val itemId = UUID.randomUUID().toString()

        try {
            // 4. ML Kit OCR
            val ocrResult = ocrEngine.extractText(bitmap)
            val ocrMs = ocrResult.durationMs

            var docType = "none"
            var docRule: String? = null
            var fullText = ""

            if (ocrResult.fullText.isNotBlank()) {
                fullText = ocrResult.fullText
                val match = DocTypeClassifier.classifyText(fullText)
                docType = match.docType
                docRule = match.ruleName
            }

            // 5. Visual MobileCLIP Embedding (INT8)
            val embedding = clipRunner.encodeImage(bitmap)

            // 6. Persist to SQLite
            val item = ItemRecord(
                itemId = itemId,
                path = uri.toString(),
                sha256 = sha256,
                width = bitmap.width,
                height = bitmap.height,
                isScreenshot = name.contains("screenshot", ignoreCase = true),
                bucketPred = if (docType != "none" && docType != DocTypeClassifier.UNKNOWN) "document" else "scene",
                bucketConf = 1.0f
            )
            database.insertItem(item)

            if (fullText.isNotBlank()) {
                database.insertDocumentOcr(
                    itemId = itemId,
                    fullText = fullText,
                    rawText = fullText,
                    docType = docType,
                    docTypeRule = docRule,
                    engine = "mlkit",
                    ocrMs = ocrMs,
                    blocks = ocrResult.blocks
                )
            }

            database.insertEmbedding(
                itemId = itemId,
                model = "mobileclip2_s0_int8",
                dim = embedding.size,
                vector = embedding
            )

            true
        } finally {
            bitmap.recycle()
        }
    }

    private fun calculateSha256(resolver: ContentResolver, uri: Uri): String? {
        return try {
            resolver.openInputStream(uri)?.use { stream ->
                val md = MessageDigest.getInstance("SHA-256")
                val buffer = ByteArray(8192)
                var bytesRead: Int
                while (stream.read(buffer).also { bytesRead = it } != -1) {
                    md.update(buffer, 0, bytesRead)
                }
                md.digest().joinToString("") { "%02x".format(it) }
            }
        } catch (e: Exception) {
            null
        }
    }

    private fun decodeSampledBitmapFromUri(resolver: ContentResolver, uri: Uri, maxDimension: Int): Bitmap? {
        return try {
            val options = BitmapFactory.Options().apply {
                inJustDecodeBounds = true
            }
            resolver.openInputStream(uri)?.use { stream ->
                BitmapFactory.decodeStream(stream, null, options)
            }

            var inSampleSize = 1
            val (width, height) = options.outWidth to options.outHeight
            if (width > maxDimension || height > maxDimension) {
                val max = maxOf(width, height)
                inSampleSize = Integer.highestOneBit(max / maxDimension)
                if (inSampleSize < 1) inSampleSize = 1
            }

            val decodeOptions = BitmapFactory.Options().apply {
                this.inSampleSize = inSampleSize
                inPreferredConfig = Bitmap.Config.ARGB_8888
            }

            resolver.openInputStream(uri)?.use { stream ->
                BitmapFactory.decodeStream(stream, null, decodeOptions)
            }
        } catch (e: Exception) {
            null
        }
    }
}
