package com.pixarch.app.core.clip

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import java.io.File

/**
 * Token-for-token parity with open_clip's MobileCLIP2-S0 tokenizer.
 * Fixture: scripts/make_edge_fixtures.py -> src/test/resources/clip_tokenizer_fixture.tsv
 */
class ClipTokenizerTest {

    companion object {
        private lateinit var tokenizer: ClipTokenizer

        @BeforeClass
        @JvmStatic
        fun load() {
            val vocab = File("src/main/assets/tokenizer/bpe_simple_vocab_16e6.txt.gz")
            tokenizer = ClipTokenizer { vocab.inputStream() }
        }

        private fun fixture(): List<Pair<String, LongArray>> {
            val stream = ClipTokenizerTest::class.java.classLoader!!
                .getResourceAsStream("clip_tokenizer_fixture.tsv")
            return stream.bufferedReader(Charsets.UTF_8).readLines()
                .filter { it.isNotEmpty() }
                .map { line ->
                    val (text, ids) = line.split('\t', limit = 2)
                    text to ids.split(',').map { it.toLong() }.toLongArray()
                }
        }
    }

    @Test
    fun loadsUncompressedVocabLikeTheApk() {
        // aapt ships the vocab gunzipped; the same bytes must give the same tokens.
        val plain = java.util.zip.GZIPInputStream(
            File("src/main/assets/tokenizer/bpe_simple_vocab_16e6.txt.gz").inputStream()
        ).use { it.readBytes() }
        val fromPlain = ClipTokenizer { plain.inputStream() }
        assertTrue(fromPlain.encode("give me my aadhaar card").contentEquals(tokenizer.encode("give me my aadhaar card")))
    }

    @Test
    fun specialTokenIdsMatchClip() {
        assertEquals(49406L, tokenizer.sotToken)
        assertEquals(49407L, tokenizer.eotToken)
    }

    @Test
    fun matchesOpenClipOnEveryFixtureCase() {
        val cases = fixture()
        assertTrue("fixture is empty", cases.isNotEmpty())
        val failures = cases.mapNotNull { (text, expected) ->
            val actual = tokenizer.encode(text)
            if (actual.contentEquals(expected)) null
            else "\"$text\"\n  expected ${expected.toList()}\n  actual   ${actual.toList()}"
        }
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    @Test
    fun everyIdFitsTheModelVocabulary() {
        // Ids >= 49408 index past the text encoder's embedding table.
        for ((text, _) in fixture()) {
            val ids = tokenizer.encode(text)
            assertTrue("out-of-vocab id for \"$text\": ${ids.max()}", ids.all { it in 0L..49407L })
        }
    }

    @Test
    fun longInputIsTruncatedWithEotLast() {
        val ids = tokenizer.encode("the quick brown fox ".repeat(40))
        assertEquals(ClipTokenizer.CONTEXT_LENGTH, ids.size)
        assertEquals(tokenizer.eotToken, ids.last())
    }
}
