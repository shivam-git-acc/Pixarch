package com.pixarch.app.core

import com.pixarch.app.core.doctype.DocTypeClassifier
import com.pixarch.app.core.ocr.TextNormaliser
import com.pixarch.app.core.query.FtsQuery
import com.pixarch.app.core.query.QueryRouter
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Base64

/**
 * The Android pipeline must make the same decisions as the Python baseline.
 * Fixtures: scripts/make_edge_fixtures.py (write_pipeline_fixtures), generated from
 * data/queries.jsonl, data/ocr_groundtruth.jsonl and extra edge cases.
 */
class PipelineParityTest {

    private fun rows(name: String): List<List<String>> {
        val stream = javaClass.classLoader!!.getResourceAsStream(name)
            ?: error("missing fixture $name -- run scripts/make_edge_fixtures.py")
        return stream.bufferedReader(Charsets.UTF_8).readLines()
            .filter { it.isNotEmpty() }
            .map { it.split('\t') }
            .also { assertTrue("fixture $name is empty", it.isNotEmpty()) }
    }

    private fun b64(s: String) = String(Base64.getDecoder().decode(s), Charsets.UTF_8)

    private fun assertNoFailures(what: String, total: Int, failures: List<String>) {
        assertTrue(
            "$what: ${failures.size}/$total differ from Python\n" + failures.take(15).joinToString("\n"),
            failures.isEmpty()
        )
    }

    @Test
    fun routerRulesMatchPython() {
        val router = QueryRouter() // no encoder: Python fixture was made the same way
        val cases = rows("router_fixture.tsv")
        val failures = cases.mapNotNull { (q, branch, conf, reason) ->
            val query = b64(q)
            val r = router.route(query)
            val ok = r.branch == branch && r.reason == reason && r.confidence == conf.toFloat()
            if (ok) null else "\"$query\": python=($branch,$conf,$reason) kotlin=(${r.branch},${r.confidence},${r.reason})"
        }
        assertNoFailures("router", cases.size, failures)
    }

    @Test
    fun ftsMatchQueryMatchesPython() {
        val cases = rows("fts_query_fixture.tsv")
        val failures = cases.mapNotNull { row ->
            val query = b64(row[0])
            val expected = if (row.size > 1) b64(row[1]) else ""
            val actual = FtsQuery.toMatchQuery(query)
            if (actual == expected) null else "\"$query\": python=[$expected] kotlin=[$actual]"
        }
        assertNoFailures("fts query", cases.size, failures)
    }

    @Test
    fun ocrNormalisationMatchesPython() {
        val cases = rows("normalise_fixture.tsv")
        val failures = cases.mapNotNull { row ->
            val raw = b64(row[0])
            val expected = if (row.size > 1) b64(row[1]) else ""
            val actual = TextNormaliser.normalise(raw)
            if (actual == expected) null
            else "raw=\"${raw.take(60)}\"\n  python=\"${expected.take(80)}\"\n  kotlin=\"${actual.take(80)}\""
        }
        assertNoFailures("normalise", cases.size, failures)
    }

    @Test
    fun docTypeTaggingMatchesPython() {
        val cases = rows("doctype_fixture.tsv")
        val failures = cases.mapNotNull { row ->
            val text = b64(row[0])
            val (docType, rule, weight) = Triple(row[1], row.getOrElse(2) { "" }, row.getOrElse(3) { "0" })
            val r = DocTypeClassifier.classifyText(text)
            val ok = r.docType == docType && r.ruleName == rule && r.weight == weight.toInt()
            if (ok) null else "\"${text.take(60)}\": python=($docType,$rule) kotlin=(${r.docType},${r.ruleName})"
        }
        assertNoFailures("doctype", cases.size, failures)
    }
}
