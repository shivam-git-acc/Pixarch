package com.pixarch.app.core

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.pixarch.app.core.storage.ItemRecord
import com.pixarch.app.core.storage.PixarchDatabase
import org.junit.After
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.util.Base64
import kotlin.math.abs

/**
 * BM25 on the phone (requery SQLite) must rank like the desktop baseline (Python
 * sqlite3). Fixture: scripts/make_edge_fixtures.py::write_bm25_fixture -- the 260
 * OCR ground-truth texts, normalised + doc-typed, and Python's top-10 per query.
 */
@RunWith(AndroidJUnit4::class)
class Bm25ParityTest {

    private val target = InstrumentationRegistry.getInstrumentation().targetContext
    private val testAssets = InstrumentationRegistry.getInstrumentation().context.assets
    private lateinit var db: PixarchDatabase

    private fun b64(s: String) = String(Base64.getDecoder().decode(s), Charsets.UTF_8)
    private fun tsv(name: String) = testAssets.open(name).bufferedReader().readLines()
        .filter { it.isNotBlank() }.map { it.split('\t') }

    @Before
    fun setUp() {
        target.deleteDatabase(DB_NAME)
        db = PixarchDatabase(target, DB_NAME)
        for ((id, text, docType) in tsv("parity/bm25_docs.tsv")) {
            db.insertItem(ItemRecord(itemId = id, path = id, sha256 = id, bucketPred = "document"))
            db.insertDocumentOcr(id, fullText = b64(text), rawText = null, docType = docType, docTypeRule = null)
        }
    }

    @After
    fun tearDown() {
        db.close()
        target.deleteDatabase(DB_NAME)
    }

    @Test
    fun rankingMatchesDesktop() {
        val cases = tsv("parity/bm25_expected.tsv")
        val failures = cases.mapNotNull { row ->
            val query = b64(row[0])
            val ids = row.getOrNull(1)?.takeIf { it.isNotEmpty() }?.split(',') ?: emptyList()
            val scores = row.getOrNull(2)?.takeIf { it.isNotEmpty() }?.split(',')?.map { it.toFloat() } ?: emptyList()
            val hits = db.searchBm25(query, 10)
            // Same scores rank-by-rank; ids may only differ inside a group of tied scores.
            val sameScores = hits.size == ids.size &&
                hits.indices.all { abs(hits[it].score - scores[it]) <= 1e-3f * maxOf(1f, abs(scores[it])) }
            val sameIds = sameScores && hits.indices.groupBy { "%.3f".format(scores[it]) }.values.all { group ->
                group.map { hits[it].itemId }.toSet() == group.map { ids[it] }.toSet()
            }
            if (sameIds) null
            else "\"$query\"\n  python ${ids.zip(scores)}\n  phone  ${hits.map { it.itemId to it.score }}"
        }
        assertTrue(
            "${failures.size}/${cases.size} queries rank differently\n" + failures.take(8).joinToString("\n"),
            failures.isEmpty()
        )
    }

    companion object {
        private const val DB_NAME = "bm25_parity_test.db"
    }
}
