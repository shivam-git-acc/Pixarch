package com.pixarch.app.core.ocr

import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.pixarch.app.core.doctype.DocTypeClassifier
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.util.Base64

/**
 * Measures the ML Kit OCR modes on synthetic documents with exact ground truth
 * (scripts/make_edge_fixtures.py::write_ocr_fixture): character error rate,
 * recall of Devanagari words, doc-type accuracy, and latency. Results go to logcat
 * (tag OcrModeComparison); the assertion only guards the shipped default.
 */
@RunWith(AndroidJUnit4::class)
class OcrModeComparisonTest {

    private val testAssets = InstrumentationRegistry.getInstrumentation().context.assets

    private data class Doc(val file: String, val subtype: String, val truth: String)
    private data class Score(val cer: Double, val devRecall: Double, val docTypeAcc: Double, val ms: Double)

    private fun docs(): List<Doc> = testAssets.open("ocr/groundtruth.tsv").bufferedReader().readLines()
        .filter { it.isNotBlank() }
        .map { it.split('\t') }
        .map { (f, s, t) -> Doc(f, s, String(Base64.getDecoder().decode(t), Charsets.UTF_8)) }

    private fun levenshtein(a: String, b: String): Int {
        var prev = IntArray(b.length + 1) { it }
        for (i in 1..a.length) {
            val cur = IntArray(b.length + 1)
            cur[0] = i
            for (j in 1..b.length) {
                cur[j] = minOf(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + if (a[i - 1] == b[j - 1]) 0 else 1)
            }
            prev = cur
        }
        return prev[b.length]
    }

    private fun devWords(s: String) = s.split(' ').filter { MlKitOcrEngine.hasDevanagari(it) }.toSet()

    private fun measure(mode: MlKitOcrEngine.Mode, docs: List<Doc>): Score = MlKitOcrEngine(mode).use { engine ->
        var cerSum = 0.0; var devFound = 0; var devTotal = 0; var typeOk = 0; var msSum = 0.0
        for (d in docs) {
            val bmp = testAssets.open("ocr/${d.file}").use { BitmapFactory.decodeStream(it) }
            val r = runBlocking { engine.extractText(bmp) }
            bmp.recycle()
            val got = TextNormaliser.normalise(r.fullText)
            val want = TextNormaliser.normalise(d.truth)
            // CER on the Latin part only, so Devanagari handling does not skew it.
            val latinGot = got.split(' ').filterNot { MlKitOcrEngine.hasDevanagari(it) }.joinToString(" ")
            val latinWant = want.split(' ').filterNot { MlKitOcrEngine.hasDevanagari(it) }.joinToString(" ")
            cerSum += levenshtein(latinGot, latinWant).toDouble() / maxOf(1, latinWant.length)
            val wantDev = devWords(want)
            devTotal += wantDev.size
            devFound += (wantDev intersect devWords(got)).size
            if (DocTypeClassifier.classifyText(got).docType == d.subtype) typeOk++
            msSum += r.durationMs
        }
        Score(cerSum / docs.size, if (devTotal == 0) 0.0 else devFound.toDouble() / devTotal,
            typeOk.toDouble() / docs.size, msSum / docs.size)
    }

    @Test
    fun compareModes() {
        val docs = docs()
        // Warm up both recognizers so first-call model loading doesn't skew latency.
        measure(MlKitOcrEngine.Mode.LATIN_PLUS_DEVANAGARI, docs.take(1))
        val results = MlKitOcrEngine.Mode.entries.associateWith { measure(it, docs) }
        for ((mode, s) in results) {
            Log.i(
                "OcrModeComparison",
                "%-22s n=%d  latinCER=%.3f  devanagariRecall=%.3f  docTypeAcc=%.3f  ms/img=%.0f"
                    .format(mode.name, docs.size, s.cer, s.devRecall, s.docTypeAcc, s.ms)
            )
        }
        val shipped = results.getValue(MlKitOcrEngine().mode)
        val latin = results.getValue(MlKitOcrEngine.Mode.LATIN)
        assertTrue("default mode must not tag doc types worse than Latin-only", shipped.docTypeAcc >= latin.docTypeAcc)
    }
}
