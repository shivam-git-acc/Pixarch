package com.pixarch.app.core.classify

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import kotlin.math.sqrt

class BucketClassifierTest {

    private fun unit(vararg v: Float): FloatArray {
        val n = sqrt(v.fold(0f) { s, x -> s + x * x })
        return FloatArray(v.size) { v[it] / n }
    }

    private val clf = BucketClassifier(
        mapOf("scene" to unit(0f, 1f), "document" to unit(1f, 0f)),
        threshold = 0.05f
    )

    @Test
    fun bucketsAreSortedLikePython() {
        assertEquals(listOf("document", "scene"), clf.buckets)
    }

    @Test
    fun confidentImageGoesToOneBranch() {
        val v = clf.classify(unit(1f, 0.2f))
        assertEquals("document", v.bucket)
        assertFalse(v.routedBoth)
        assertTrue(v.margin > 0.05f)
    }

    @Test
    fun lowMarginImageIsHedgedIntoBothBranches() {
        val v = clf.classify(unit(1f, 0.98f))
        assertEquals("document", v.bucket)
        assertTrue(v.margin < 0.05f)
        assertTrue(v.routedBoth)
    }

    @Test
    fun shippedPrototypesParse() {
        val clf = File("src/main/assets/" + BucketClassifier.ASSET_PATH).inputStream().use {
            BucketClassifier.parse(it)
        }
        assertEquals(listOf("document", "scene"), clf.buckets)
        assertTrue("threshold ${clf.threshold}", clf.threshold > 0f && clf.threshold < 0.1f)
        for ((b, p) in clf.prototypes) {
            assertEquals("$b dim", 512, p.size)
            val norm = sqrt(p.fold(0f) { s, x -> s + x * x })
            assertEquals("$b norm", 1f, norm, 1e-3f)
        }
    }
}
