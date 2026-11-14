package com.pixarch.app.core.storage

import android.content.ContentValues
import android.content.Context
import android.database.Cursor
import com.pixarch.app.core.query.FtsQuery
import io.requery.android.database.sqlite.SQLiteDatabase
import io.requery.android.database.sqlite.SQLiteOpenHelper
import java.nio.ByteBuffer
import java.nio.ByteOrder

data class SearchHit(
    val itemId: String,
    val score: Float,
    val snippet: String? = null,
    val branch: String = "unknown"
)

data class ItemRecord(
    val itemId: String,
    val path: String,
    val sha256: String,
    val bytes: Long? = null,
    val width: Int? = null,
    val height: Int? = null,
    val captureDate: String? = null,
    val isScreenshot: Boolean = false,
    val bucketPred: String? = null,
    val bucketConf: Float? = null,
    val routedBoth: Boolean = false
)

data class OcrBlockRecord(
    val blockIdx: Int,
    val text: String,
    val conf: Float?,
    val x0: Int?,
    val y0: Int?,
    val x1: Int?,
    val y1: Int?
)

class PixarchDatabase(context: Context, dbName: String = "pixarch.db") :
    SQLiteOpenHelper(context, dbName, null, DATABASE_VERSION) {

    companion object {
        // v3: embeddings from app v1 came from a broken INT8 visual encoder and wrong
        // preprocessing; bumping forces onUpgrade to drop them so the gallery re-indexes.
        // v4: CLIP zero-shot buckets + routed_both, normalised OCR text, OCR only on
        // document/hedged items (Stage 1.5) -- old rows have different semantics.
        const val DATABASE_VERSION = 4

        private fun floatArrayToBlob(floats: FloatArray): ByteArray {
            val buffer = ByteBuffer.allocate(floats.size * 4).order(ByteOrder.LITTLE_ENDIAN)
            for (f in floats) {
                buffer.putFloat(f)
            }
            return buffer.array()
        }

        private fun blobToFloatArray(blob: ByteArray): FloatArray {
            val buffer = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN)
            val floats = FloatArray(blob.size / 4)
            for (i in floats.indices) {
                floats[i] = buffer.float
            }
            return floats
        }

        private fun dotProduct(a: FloatArray, b: FloatArray): Float {
            var sum = 0f
            val n = minOf(a.size, b.size)
            for (i in 0 until n) {
                sum += a[i] * b[i]
            }
            return sum
        }
    }

    override fun onConfigure(db: SQLiteDatabase) {
        super.onConfigure(db)
        db.enableWriteAheadLogging()
        db.setForeignKeyConstraintsEnabled(true)
    }

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL(
            """
            CREATE TABLE IF NOT EXISTS items (
                item_id       TEXT PRIMARY KEY,
                path          TEXT NOT NULL,
                sha256        TEXT NOT NULL,
                bytes         INTEGER,
                width         INTEGER,
                height        INTEGER,
                capture_date  TEXT,
                is_screenshot INTEGER NOT NULL DEFAULT 0,
                bucket_pred   TEXT,
                bucket_conf   REAL,
                routed_both   INTEGER NOT NULL DEFAULT 0,
                ingested_at   TEXT DEFAULT (datetime('now'))
            )
            """.trimIndent()
        )
        db.execSQL("CREATE INDEX IF NOT EXISTS idx_items_sha ON items(sha256)")
        db.execSQL("CREATE INDEX IF NOT EXISTS idx_items_pred ON items(bucket_pred)")

        db.execSQL(
            """
            CREATE TABLE IF NOT EXISTS ocr_blocks (
                item_id   TEXT NOT NULL REFERENCES items(item_id) ON DELETE CASCADE,
                block_idx INTEGER NOT NULL,
                text      TEXT NOT NULL,
                conf      REAL,
                x0 INTEGER, y0 INTEGER, x1 INTEGER, y1 INTEGER,
                PRIMARY KEY (item_id, block_idx)
            )
            """.trimIndent()
        )

        db.execSQL(
            """
            CREATE TABLE IF NOT EXISTS doc_fields (
                item_id       TEXT PRIMARY KEY REFERENCES items(item_id) ON DELETE CASCADE,
                full_text     TEXT NOT NULL,
                raw_text      TEXT,
                doc_type      TEXT,
                doc_type_rule TEXT,
                engine        TEXT,
                ocr_ms        REAL
            )
            """.trimIndent()
        )

        db.execSQL(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS doc_fts USING fts5(
                item_id UNINDEXED,
                ocr_text,
                doc_type,
                tokenize = 'porter unicode61 remove_diacritics 2'
            )
            """.trimIndent()
        )

        db.execSQL(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                item_id TEXT PRIMARY KEY REFERENCES items(item_id) ON DELETE CASCADE,
                model   TEXT NOT NULL,
                dim     INTEGER NOT NULL,
                vec     BLOB NOT NULL
            )
            """.trimIndent()
        )
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        db.execSQL("DROP TABLE IF EXISTS embeddings")
        db.execSQL("DROP TABLE IF EXISTS doc_fts")
        db.execSQL("DROP TABLE IF EXISTS doc_fields")
        db.execSQL("DROP TABLE IF EXISTS ocr_blocks")
        db.execSQL("DROP TABLE IF EXISTS items")
        onCreate(db)
    }

    fun hasItemWithSha256(sha256: String): Boolean {
        val db = readableDatabase
        val cursor = db.rawQuery("SELECT 1 FROM items WHERE sha256 = ? LIMIT 1", arrayOf(sha256))
        val exists = cursor.moveToFirst()
        cursor.close()
        return exists
    }

    fun insertItem(item: ItemRecord) {
        val db = writableDatabase
        val values = ContentValues().apply {
            put("item_id", item.itemId)
            put("path", item.path)
            put("sha256", item.sha256)
            put("bytes", item.bytes)
            put("width", item.width)
            put("height", item.height)
            put("capture_date", item.captureDate)
            put("is_screenshot", if (item.isScreenshot) 1 else 0)
            put("bucket_pred", item.bucketPred)
            put("bucket_conf", item.bucketConf)
            put("routed_both", if (item.routedBoth) 1 else 0)
        }
        db.insertWithOnConflict("items", null, values, SQLiteDatabase.CONFLICT_REPLACE)
    }

    fun insertDocumentOcr(
        itemId: String,
        fullText: String,
        rawText: String?,
        docType: String,
        docTypeRule: String?,
        engine: String = "mlkit",
        ocrMs: Float = 0f,
        blocks: List<OcrBlockRecord> = emptyList()
    ) {
        val db = writableDatabase
        db.beginTransaction()
        try {
            val fieldValues = ContentValues().apply {
                put("item_id", itemId)
                put("full_text", fullText)
                put("raw_text", rawText)
                put("doc_type", docType)
                put("doc_type_rule", docTypeRule)
                put("engine", engine)
                put("ocr_ms", ocrMs)
            }
            db.insertWithOnConflict("doc_fields", null, fieldValues, SQLiteDatabase.CONFLICT_REPLACE)

            // Re-indexing an item replaces its OCR rows instead of accumulating them
            // (FTS5 has no primary key, so CONFLICT_REPLACE would not dedupe there).
            db.delete("ocr_blocks", "item_id = ?", arrayOf(itemId))
            db.delete("doc_fts", "item_id = ?", arrayOf(itemId))

            for (b in blocks) {
                val blockValues = ContentValues().apply {
                    put("item_id", itemId)
                    put("block_idx", b.blockIdx)
                    put("text", b.text)
                    put("conf", b.conf)
                    put("x0", b.x0)
                    put("y0", b.y0)
                    put("x1", b.x1)
                    put("y1", b.y1)
                }
                db.insertWithOnConflict("ocr_blocks", null, blockValues, SQLiteDatabase.CONFLICT_REPLACE)
            }

            // Insert into FTS5 inverted index
            val ftsValues = ContentValues().apply {
                put("item_id", itemId)
                put("ocr_text", fullText)
                put("doc_type", docType)
            }
            db.insert("doc_fts", null, ftsValues)

            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
    }

    fun insertEmbedding(itemId: String, model: String, dim: Int, vector: FloatArray) {
        val db = writableDatabase
        val values = ContentValues().apply {
            put("item_id", itemId)
            put("model", model)
            put("dim", dim)
            put("vec", floatArrayToBlob(vector))
        }
        db.insertWithOnConflict("embeddings", null, values, SQLiteDatabase.CONFLICT_REPLACE)
    }

    /**
     * BM25 search over OCR text and document subtype, as `bm25_search.search` does.
     * Weights: 1.0x for OCR body text, 5.0x for doc_type tag (configs/baseline.yaml).
     */
    fun searchBm25(queryText: String, topK: Int = 10): List<SearchHit> {
        val db = readableDatabase
        val hits = mutableListOf<SearchHit>()

        val ftsQuery = FtsQuery.toMatchQuery(queryText)
        if (ftsQuery.isEmpty()) return emptyList()

        val sql = """
            SELECT item_id,
                   bm25(doc_fts, 0.0, 1.0, 5.0) AS score,
                   snippet(doc_fts, 1, '<b>', '</b>', '...', 24) AS snippet
            FROM doc_fts
            WHERE doc_fts MATCH ?
            ORDER BY score ASC
            LIMIT ?
        """.trimIndent()

        var cursor: Cursor? = null
        try {
            cursor = db.rawQuery(sql, arrayOf(ftsQuery, topK.toString()))
            while (cursor.moveToNext()) {
                val itemId = cursor.getString(0)
                // FTS5 bm25() returns negative numbers (most negative is best). Flip sign for display.
                val score = -cursor.getFloat(1)
                val snippet = if (!cursor.isNull(2)) cursor.getString(2) else null
                hits.add(SearchHit(itemId, score, snippet, branch = "document"))
            }
        } catch (e: Exception) {
            // A malformed MATCH expression yields no hits, not a crash (as in Python).
            android.util.Log.w("PixarchDatabase", "FTS query failed for [$ftsQuery]", e)
            hits.clear()
        } finally {
            cursor?.close()
        }
        return hits
    }

    /**
     * Exact cosine similarity over L2-normalised embeddings. Like the desktop engine,
     * the scene index covers only scene-bucket items plus low-confidence items that
     * were routed to both branches; documents are found through BM25.
     */
    fun searchVector(queryEmbedding: FloatArray, topK: Int = 10): List<SearchHit> {
        val db = readableDatabase
        val scored = mutableListOf<SearchHit>()

        val cursor = db.rawQuery(
            """
            SELECT e.item_id, e.vec FROM embeddings e
            JOIN items i ON i.item_id = e.item_id
            WHERE i.bucket_pred = 'scene' OR i.routed_both = 1
            """.trimIndent(),
            null
        )
        try {
            while (cursor.moveToNext()) {
                val itemId = cursor.getString(0)
                val blob = cursor.getBlob(1)
                val docVec = blobToFloatArray(blob)
                val sim = dotProduct(queryEmbedding, docVec)
                scored.add(SearchHit(itemId, sim, branch = "scene"))
            }
        } finally {
            cursor.close()
        }

        return scored.sortedByDescending { it.score }.take(topK)
    }

    fun getItemPath(itemId: String): String? {
        val db = readableDatabase
        val cursor = db.rawQuery("SELECT path FROM items WHERE item_id = ? LIMIT 1", arrayOf(itemId))
        var path: String? = null
        if (cursor.moveToFirst()) {
            path = cursor.getString(0)
        }
        cursor.close()
        return path
    }

    /** (bucket_pred, routed_both, has OCR) counts -- for diagnostics and tests. */
    fun bucketSummary(): Map<String, Int> {
        val out = LinkedHashMap<String, Int>()
        readableDatabase.rawQuery(
            """
            SELECT i.bucket_pred, i.routed_both, (d.item_id IS NOT NULL) AS has_ocr, COUNT(*)
            FROM items i LEFT JOIN doc_fields d ON d.item_id = i.item_id
            GROUP BY 1, 2, 3 ORDER BY 1, 2, 3
            """.trimIndent(),
            null
        ).use { c ->
            while (c.moveToNext()) {
                out["${c.getString(0)}|both=${c.getInt(1)}|ocr=${c.getInt(2)}"] = c.getInt(3)
            }
        }
        return out
    }

    fun getTotalItemsCount(): Int {
        return try {
            val db = readableDatabase
            val cursor = db.rawQuery("SELECT COUNT(*) FROM items", null)
            var count = 0
            if (cursor.moveToFirst()) {
                count = cursor.getInt(0)
            }
            cursor.close()
            count
        } catch (e: Exception) {
            0
        }
    }

    fun getRecentItems(limit: Int = 30): List<SearchHit> {
        val list = mutableListOf<SearchHit>()
        return try {
            val db = readableDatabase
            val cursor = db.rawQuery("SELECT item_id FROM items ORDER BY ingested_at DESC LIMIT ?", arrayOf(limit.toString()))
            while (cursor.moveToNext()) {
                list.add(SearchHit(cursor.getString(0), 0.0f, null, "gallery"))
            }
            cursor.close()
            list
        } catch (e: Exception) {
            emptyList()
        }
    }

    data class ItemDetail(
        val itemId: String,
        val path: String,
        val docType: String? = null,
        val ocrSnippet: String? = null,
        val ocrMs: Float? = null,
        val hasEmbedding: Boolean = false,
        val isScreenshot: Boolean = false
    )

    fun getItemDetail(itemId: String): ItemDetail? {
        val db = readableDatabase
        val query = """
            SELECT i.item_id, i.path, i.is_screenshot, d.doc_type, d.full_text, d.ocr_ms,
                   CASE WHEN e.item_id IS NOT NULL THEN 1 ELSE 0 END AS has_vec
            FROM items i
            LEFT JOIN doc_fields d ON i.item_id = d.item_id
            LEFT JOIN embeddings e ON i.item_id = e.item_id
            WHERE i.item_id = ?
            LIMIT 1
        """.trimIndent()

        val cursor = db.rawQuery(query, arrayOf(itemId))
        var detail: ItemDetail? = null
        if (cursor.moveToFirst()) {
            val isScreen = cursor.getInt(2) == 1
            val docType = if (!cursor.isNull(3)) cursor.getString(3) else null
            val fullText = if (!cursor.isNull(4)) cursor.getString(4) else null
            val ocrMs = if (!cursor.isNull(5)) cursor.getFloat(5) else null
            val hasVec = cursor.getInt(6) == 1

            detail = ItemDetail(
                itemId = cursor.getString(0),
                path = cursor.getString(1),
                docType = docType,
                ocrSnippet = fullText,
                ocrMs = ocrMs,
                hasEmbedding = hasVec,
                isScreenshot = isScreen
            )
        }
        cursor.close()
        return detail
    }
}
