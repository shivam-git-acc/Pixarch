package com.pixarch.app.core.ingest

import android.content.ContentResolver
import android.content.ContentUris
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import android.util.Log
import androidx.exifinterface.media.ExifInterface
import com.pixarch.app.core.classify.BucketClassifier
import com.pixarch.app.core.clip.OnnxClipRunner
import com.pixarch.app.core.doctype.DocTypeClassifier
import com.pixarch.app.core.ocr.MlKitOcrEngine
import com.pixarch.app.core.ocr.TextNormaliser
import com.pixarch.app.core.storage.ItemRecord
import com.pixarch.app.core.storage.PixarchDatabase
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.security.MessageDigest
import java.util.UUID

data class IndexProgress(
    val indexedCount: Int,
    val totalCount: Int,
    val currentItemPath: String,
    val isComplete: Boolean = false,
    val error: String? = null
)

/**
 * Receives per-stage wall time for one image, using the desktop stage names
 * (`ingest`, `clip_encode`, `classify`, `embed_store`, `ocr`) so device and PC
 * ingestion reports line up. Used by the on-device evaluation harness.
 */
fun interface StageTimer {
    fun record(stage: String, ms: Double)
}

/** One gallery image as MediaStore describes it. */
data class MediaItem(
    val uri: Uri,
    val name: String,
    val bytes: Long?,
    val dateTakenMs: Long?,
    val relativePath: String?
)

/**
 * Indexes the gallery the way `scripts/build_index.py` indexes the corpus:
 *
 *   hash + dedupe -> CLIP encode -> zero-shot bucket (margin < τ: both branches)
 *   -> store embedding for every item
 *   -> OCR -> normalise -> doc-type rules -> FTS5, ONLY for document / hedged items
 *
 * Skipping OCR on confident scene photos is where most of the indexing time goes
 * (OCR was 82% of desktop indexing time).
 */
class GalleryIndexer(
    private val context: Context,
    private val database: PixarchDatabase,
    private val ocrEngine: MlKitOcrEngine,
    private val clipRunner: OnnxClipRunner,
    private val bucketClassifier: BucketClassifier
) {
    suspend fun indexMediaStore(
        limit: Int = 100,
        onProgress: (IndexProgress) -> Unit = {}
    ): Int = withContext(Dispatchers.IO) {
        val items = queryMediaStore(limit)
        var processed = 0
        for ((index, item) in items.withIndex()) {
            try {
                if (indexSingleImage(item)) processed++
            } catch (e: Exception) {
                Log.w(TAG, "indexing failed for ${item.name}", e)
            }
            onProgress(IndexProgress(index + 1, items.size, item.name))
        }
        onProgress(IndexProgress(processed, items.size, "Complete", isComplete = true))
        processed
    }

    private fun queryMediaStore(limit: Int): List<MediaItem> {
        val projection = buildList {
            add(MediaStore.Images.Media._ID)
            add(MediaStore.Images.Media.DISPLAY_NAME)
            add(MediaStore.Images.Media.SIZE)
            add(MediaStore.Images.Media.DATE_TAKEN)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) add(MediaStore.Images.Media.RELATIVE_PATH)
        }.toTypedArray()
        val out = ArrayList<MediaItem>()
        context.contentResolver.query(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI, projection, null, null,
            "${MediaStore.Images.Media.DATE_ADDED} DESC"
        )?.use { c ->
            val idCol = c.getColumnIndexOrThrow(MediaStore.Images.Media._ID)
            val nameCol = c.getColumnIndexOrThrow(MediaStore.Images.Media.DISPLAY_NAME)
            val sizeCol = c.getColumnIndexOrThrow(MediaStore.Images.Media.SIZE)
            val takenCol = c.getColumnIndexOrThrow(MediaStore.Images.Media.DATE_TAKEN)
            val pathCol = c.getColumnIndex("relative_path")
            while (c.moveToNext() && out.size < limit) {
                val id = c.getLong(idCol)
                out.add(
                    MediaItem(
                        uri = ContentUris.withAppendedId(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, id),
                        name = c.getString(nameCol) ?: "image_$id.jpg",
                        bytes = if (c.isNull(sizeCol)) null else c.getLong(sizeCol),
                        dateTakenMs = if (c.isNull(takenCol)) null else c.getLong(takenCol).takeIf { it > 0 },
                        relativePath = if (pathCol >= 0 && !c.isNull(pathCol)) c.getString(pathCol) else null
                    )
                )
            }
        }
        return out
    }

    /**
     * @param itemId id stored in the index; the evaluation harness passes the corpus
     *   manifest id so results can be scored against the qrels.
     * @param timer optional per-stage timing sink.
     * @return true if a new item was indexed (false for duplicates / unreadable files).
     */
    suspend fun indexSingleImage(
        item: MediaItem,
        itemId: String = UUID.randomUUID().toString(),
        timer: StageTimer? = null
    ): Boolean = withContext(Dispatchers.IO) {
        val resolver = context.contentResolver
        var t = System.nanoTime()
        fun lap(stage: String) {
            val now = System.nanoTime()
            timer?.record(stage, (now - t) / 1e6)
            t = now
        }

        // Stage 0: hash + dedupe, original dimensions, EXIF.
        val sha256 = calculateSha256(resolver, item.uri) ?: return@withContext false
        if (database.hasItemWithSha256(sha256)) return@withContext false
        val (origW, origH) = readBounds(resolver, item.uri) ?: return@withContext false
        val exif = readExif(resolver, item.uri)

        val bitmap = decodeSampledBitmap(resolver, item.uri, origW, origH, maxDimension = 1280)
            ?: return@withContext false
        lap("ingest")
        try {
            // Stage 1: CLIP encode once; the same embedding classifies and indexes.
            val embedding = clipRunner.encodeImage(bitmap)
            lap("clip_encode")
            val verdict = bucketClassifier.classify(embedding)
            lap("classify")

            database.insertItem(
                ItemRecord(
                    itemId = itemId,
                    path = item.uri.toString(),
                    sha256 = sha256,
                    bytes = item.bytes,
                    width = origW,
                    height = origH,
                    captureDate = item.dateTakenMs?.let { MediaMetadata.formatDateTaken(it) } ?: exif.captureDate,
                    isScreenshot = MediaMetadata.looksLikeScreenshot(origW, origH, exif.hasCamera, item.relativePath),
                    bucketPred = verdict.bucket,
                    bucketConf = verdict.margin,
                    routedBoth = verdict.routedBoth
                )
            )
            // Stage 2b: every embedding is stored; searchVector restricts to scene/hedged.
            database.insertEmbedding(itemId, "mobileclip2_s0_int8", embedding.size, embedding)
            lap("embed_store")

            // Stage 2a: document branch only for document or hedged items.
            if (verdict.bucket == DOCUMENT || verdict.routedBoth) {
                try {
                    val ocr = ocrEngine.extractText(bitmap)
                    val norm = TextNormaliser.normalise(ocr.fullText)
                    val match = DocTypeClassifier.classifyText(norm)
                    database.insertDocumentOcr(
                        itemId = itemId,
                        fullText = norm,
                        rawText = ocr.fullText,
                        docType = match.docType,
                        docTypeRule = match.ruleName,
                        engine = "mlkit-${ocrEngine.mode.name.lowercase()}",
                        ocrMs = ocr.durationMs,
                        blocks = ocr.blocks
                    )
                } catch (e: Exception) {
                    // One bad scan must not lose the item: the scene branch still has it.
                    Log.w(TAG, "OCR failed for ${item.name}", e)
                }
                lap("ocr")
            }
            true
        } finally {
            bitmap.recycle()
        }
    }

    private data class Exif(val captureDate: String?, val hasCamera: Boolean)

    private fun readExif(resolver: ContentResolver, uri: Uri): Exif = try {
        resolver.openInputStream(uri)?.use { stream ->
            val e = ExifInterface(stream)
            Exif(
                captureDate = MediaMetadata.formatExifDate(
                    e.getAttribute(ExifInterface.TAG_DATETIME_ORIGINAL) ?: e.getAttribute(ExifInterface.TAG_DATETIME)
                ),
                hasCamera = listOf(ExifInterface.TAG_MAKE, ExifInterface.TAG_MODEL, ExifInterface.TAG_FOCAL_LENGTH)
                    .any { !e.getAttribute(it).isNullOrBlank() }
            )
        } ?: Exif(null, false)
    } catch (e: Exception) {
        Exif(null, false) // malformed EXIF is common
    }

    private fun calculateSha256(resolver: ContentResolver, uri: Uri): String? = try {
        resolver.openInputStream(uri)?.use { stream ->
            val md = MessageDigest.getInstance("SHA-256")
            val buffer = ByteArray(1 shl 16)
            while (true) {
                val n = stream.read(buffer)
                if (n < 0) break
                md.update(buffer, 0, n)
            }
            md.digest().joinToString("") { "%02x".format(it) }
        }
    } catch (e: Exception) {
        null
    }

    private fun readBounds(resolver: ContentResolver, uri: Uri): Pair<Int, Int>? = try {
        val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        resolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, opts) }
        if (opts.outWidth > 0 && opts.outHeight > 0) opts.outWidth to opts.outHeight else null
    } catch (e: Exception) {
        null
    }

    private fun decodeSampledBitmap(
        resolver: ContentResolver, uri: Uri, width: Int, height: Int, maxDimension: Int
    ): Bitmap? = try {
        var inSampleSize = 1
        while (maxOf(width, height) / (inSampleSize * 2) >= maxDimension) inSampleSize *= 2
        val opts = BitmapFactory.Options().apply {
            this.inSampleSize = inSampleSize
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        resolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, opts) }
    } catch (e: Exception) {
        null
    }

    companion object {
        private const val TAG = "GalleryIndexer"
        const val DOCUMENT = "document"
    }
}
