package com.pixarch.app.eval

import android.app.ActivityManager
import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Debug
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.pixarch.app.core.classify.BucketClassifier
import com.pixarch.app.core.clip.ClipTokenizer
import com.pixarch.app.core.clip.OnnxClipRunner
import com.pixarch.app.core.ingest.GalleryIndexer
import com.pixarch.app.core.ingest.MediaItem
import com.pixarch.app.core.ingest.StageTimer
import com.pixarch.app.core.ocr.MlKitOcrEngine
import com.pixarch.app.core.query.QueryRouter
import com.pixarch.app.core.query.SearchEngine
import com.pixarch.app.core.storage.PixarchDatabase
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/**
 * Stage 3 on-device evaluation. NOT part of the normal test run: it needs the corpus
 * pushed by `scripts/run_device_eval.py`, and is skipped (passes trivially) without it.
 *
 * 1. Index every manifest image from the app's external files dir into a separate
 *    database (the user's gallery index is untouched), with per-stage timings.
 * 2. Run every query through routed / clip_only / bm25_only at the same retrieval
 *    depth the desktop evaluation uses, timing each search (warm).
 * 3. Record peak memory, model and APK sizes, index size, then write
 *    `eval_out/device_run.json` + a copy of the database for the PC-side scorer
 *    (`scripts/evaluate_device.py`), which reuses src/pixarch/eval unchanged.
 */
@RunWith(AndroidJUnit4::class)
class DeviceEvalTest {

    private val target: Context = InstrumentationRegistry.getInstrumentation().targetContext
    private val args = InstrumentationRegistry.getArguments()

    private data class Row(val itemId: String, val path: String)

    /** RFC-4180-ish CSV: quoted fields, doubled quotes, commas inside quotes. */
    private fun parseCsv(text: String): List<List<String>> {
        val rows = ArrayList<List<String>>()
        var row = ArrayList<String>()
        val field = StringBuilder()
        var quoted = false
        var i = 0
        while (i < text.length) {
            val c = text[i]
            if (quoted) {
                if (c == '"' && i + 1 < text.length && text[i + 1] == '"') { field.append('"'); i++ }
                else if (c == '"') quoted = false
                else field.append(c)
            } else when (c) {
                '"' -> quoted = true
                ',' -> { row.add(field.toString()); field.setLength(0) }
                '\n' -> { row.add(field.toString()); field.setLength(0); rows.add(row); row = ArrayList() }
                '\r' -> Unit
                else -> field.append(c)
            }
            i++
        }
        if (field.isNotEmpty() || row.isNotEmpty()) { row.add(field.toString()); rows.add(row) }
        return rows
    }

    /** Samples this process's PSS while a phase runs; keeps the peak. */
    private class MemorySampler : AutoCloseable {
        val peakPssKb = AtomicLong(0)
        private val running = AtomicBoolean(true)
        private val thread = Thread {
            while (running.get()) {
                peakPssKb.accumulateAndGet(Debug.getPss()) { a, b -> maxOf(a, b) }
                SystemClock.sleep(250)
            }
        }.apply { isDaemon = true; start() }
        override fun close() { running.set(false); thread.join(1000) }
    }

    private fun stats(xs: List<Double>): JSONObject {
        val s = xs.sorted()
        fun pick(p: Double) = s[minOf(s.size - 1, Math.round(p * (s.size - 1)).toInt())]
        return JSONObject().apply {
            if (s.isEmpty()) return@apply
            put("n", s.size); put("mean_ms", s.average()); put("p50_ms", pick(0.5))
            put("p95_ms", pick(0.95)); put("max_ms", s.last()); put("min_ms", s.first())
        }
    }

    @Test
    fun runBenchmark() = runBlocking {
        val evalDir = File(args.getString("evalDir") ?: target.getExternalFilesDir("eval")!!.path)
        val manifestFile = File(evalDir, "manifest.csv")
        if (!manifestFile.exists()) {
            Log.i(TAG, "No corpus at $evalDir -- skipping (run scripts/run_device_eval.py)")
            return@runBlocking
        }
        val depth = args.getString("depth")?.toInt() ?: 107
        val outDir = File(target.getExternalFilesDir(null), "eval_out").apply { deleteRecursively(); mkdirs() }

        val csv = parseCsv(manifestFile.readText(Charsets.UTF_8))
        val header = csv.first()
        val idCol = header.indexOf("item_id")
        val pathCol = header.indexOf("path")
        val manifest = csv.drop(1).filter { it.size > pathCol && it[idCol].isNotBlank() }
            .map { Row(it[idCol], it[pathCol]) }
        val queries = File(evalDir, "queries.jsonl").readLines(Charsets.UTF_8)
            .filter { it.isNotBlank() }.map { JSONObject(it) }

        val baselinePssKb = Debug.getPss()

        // ---- models (cold load) ------------------------------------------------
        val tLoad = SystemClock.elapsedRealtimeNanos()
        val clip = OnnxClipRunner(target)
        val visualLoadMs = (SystemClock.elapsedRealtimeNanos() - tLoad) / 1e6
        val tokenizer = ClipTokenizer(target)
        val classifier = BucketClassifier.fromAssets(target)
        val ocr = MlKitOcrEngine()

        target.deleteDatabase(DB_NAME)
        val db = PixarchDatabase(target, DB_NAME)
        val indexer = GalleryIndexer(target, db, ocr, clip, classifier)

        // ---- indexing ------------------------------------------------------------
        val stageTotals = LinkedHashMap<String, DoubleArray>() // stage -> [n, total_ms]
        val perItemMs = ArrayList<Double>()
        val timer = StageTimer { stage, ms ->
            val acc = stageTotals.getOrPut(stage) { DoubleArray(2) }
            acc[0] += 1.0; acc[1] += ms
        }
        var missing = 0; var skipped = 0
        val indexSampler = MemorySampler()
        val tIndex = SystemClock.elapsedRealtimeNanos()
        for ((n, row) in manifest.withIndex()) {
            val f = File(evalDir, "raw/${row.path}")
            if (!f.isFile) { missing++; continue }
            val t0 = SystemClock.elapsedRealtimeNanos()
            val ok = indexer.indexSingleImage(
                MediaItem(Uri.fromFile(f), f.name, f.length(), null, null),
                itemId = row.itemId, timer = timer
            )
            if (ok) perItemMs.add((SystemClock.elapsedRealtimeNanos() - t0) / 1e6) else skipped++
            if ((n + 1) % 50 == 0) Log.i(TAG, "indexed ${n + 1}/${manifest.size}")
        }
        val indexWallMs = (SystemClock.elapsedRealtimeNanos() - tIndex) / 1e6
        indexSampler.close()
        Log.i(TAG, "indexing done: ${perItemMs.size} items in ${"%.1f".format(indexWallMs / 1000)} s")

        // ---- queries ---------------------------------------------------------------
        val engine = SearchEngine(db, tokenizer, clip, classifier.prototypes)
        val querySampler = MemorySampler()
        // Cold: the first scene query loads the ~61 MB text encoder.
        val tCold = SystemClock.elapsedRealtimeNanos()
        engine.search("a photo of a beach", depth, QueryRouter.SCENE)
        val coldFirstSceneQueryMs = (SystemClock.elapsedRealtimeNanos() - tCold) / 1e6
        engine.search("aadhaar card", depth, QueryRouter.DOCUMENT) // warm the FTS path too

        val systems = linkedMapOf<String, String?>(
            "routed" to null, "clip_only" to QueryRouter.SCENE, "bm25_only" to QueryRouter.DOCUMENT
        )
        val runsJson = JSONObject()
        for ((name, force) in systems) {
            val arr = JSONArray()
            for (q in queries) {
                val t0 = SystemClock.elapsedRealtimeNanos()
                val r = engine.search(q.getString("text"), depth, force)
                val ms = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6
                arr.put(JSONObject().apply {
                    put("query_id", q.getString("query_id"))
                    put("ranked", JSONArray(r.hits.map { it.itemId }))
                    put("ms", ms)
                    put("branch", r.route.branch)
                    put("reason", r.route.reason)
                })
            }
            runsJson.put(name, arr)
        }
        querySampler.close()

        // ---- storage -----------------------------------------------------------------
        val sql = db.writableDatabase
        sql.rawQuery("PRAGMA wal_checkpoint(TRUNCATE)", null).use { it.moveToFirst() }
        val perTable = JSONObject()
        try {
            sql.rawQuery("SELECT name, SUM(pgsize) FROM dbstat GROUP BY name", null).use { c ->
                while (c.moveToNext()) perTable.put(c.getString(0), c.getLong(1))
            }
        } catch (e: Exception) {
            Log.i(TAG, "dbstat unavailable: ${e.message}")
        }
        val counts = JSONObject()
        sql.rawQuery("SELECT bucket_pred, routed_both, COUNT(*) FROM items GROUP BY 1, 2", null).use { c ->
            while (c.moveToNext()) counts.put("${c.getString(0)}|both=${c.getInt(1)}", c.getInt(2))
        }
        db.close()
        val dbFile = target.getDatabasePath(DB_NAME)
        val dbBytes = dbFile.length()
        dbFile.copyTo(File(outDir, "device_index.db"), overwrite = true)

        // ---- environment -------------------------------------------------------------
        val am = target.getSystemService(ActivityManager::class.java)
        val memInfo = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        fun assetBytes(p: String) = try { target.assets.openFd(p).use { it.length } } catch (e: Exception) { -1L }

        val out = JSONObject().apply {
            put("device", JSONObject().apply {
                put("manufacturer", Build.MANUFACTURER); put("model", Build.MODEL)
                put("sdk", Build.VERSION.SDK_INT); put("release", Build.VERSION.RELEASE)
                put("soc", if (Build.VERSION.SDK_INT >= 31) "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}" else Build.HARDWARE)
                put("abi", Build.SUPPORTED_ABIS.firstOrNull()); put("cores", Runtime.getRuntime().availableProcessors())
                put("total_ram_mb", memInfo.totalMem / 1e6)
            })
            put("config", JSONObject().apply {
                put("depth", depth); put("ocr_mode", ocr.mode.name)
                put("bucket_threshold", classifier.threshold.toDouble()); put("clip_threads", 2)
                put("clip_model", "MobileCLIP2-S0 (dfndr2b) INT8 ONNX")
            })
            put("sizes", JSONObject().apply {
                put("apk_bytes", File(target.packageCodePath).length())
                put("visual_model_bytes", assetBytes("models/mobileclip_visual_int8.onnx"))
                put("text_model_bytes", assetBytes("models/mobileclip_text_int8.onnx"))
                put("db_bytes", dbBytes); put("db_per_table", perTable)
            })
            put("index", JSONObject().apply {
                put("manifest_rows", manifest.size); put("missing_files", missing); put("skipped", skipped)
                put("indexed", perItemMs.size); put("wall_ms", indexWallMs)
                put("visual_model_load_ms", visualLoadMs)
                put("per_item", stats(perItemMs))
                put("stages", JSONObject().apply {
                    for ((stage, acc) in stageTotals) put(stage, JSONObject().put("n", acc[0].toInt()).put("total_ms", acc[1]))
                })
                put("bucket_counts", counts)
            })
            put("memory", JSONObject().apply {
                put("baseline_pss_mb", baselinePssKb / 1024.0)
                put("peak_pss_indexing_mb", indexSampler.peakPssKb.get() / 1024.0)
                put("peak_pss_querying_mb", querySampler.peakPssKb.get() / 1024.0)
            })
            put("queries", JSONObject().apply {
                put("n", queries.size); put("cold_first_scene_query_ms", coldFirstSceneQueryMs)
                put("runs", runsJson)
            })
        }
        File(outDir, "device_run.json").writeText(out.toString(2), Charsets.UTF_8)
        target.deleteDatabase(DB_NAME)
        clip.close(); ocr.close()
        Log.i(TAG, "wrote ${outDir}/device_run.json")
        assertTrue("no images were found under $evalDir/raw", perItemMs.isNotEmpty())
    }

    companion object {
        private const val TAG = "DeviceEval"
        private const val DB_NAME = "pixarch_eval.db"
    }
}
