package com.pixarch.app.core.clip

import android.content.Context
import java.io.BufferedReader
import java.io.InputStreamReader
import java.util.regex.Pattern
import java.util.zip.GZIPInputStream

/**
 * Android Kotlin implementation of the OpenAI/OpenCLIP SimpleTokenizer.
 * Reads bpe_simple_vocab_16e6.txt.gz from assets and converts queries into 77 token IDs.
 */
class ClipTokenizer(context: Context, vocabAssetPath: String = "tokenizer/bpe_simple_vocab_16e6.txt.gz") {

    companion object {
        const val SOT_TOKEN = 49406L
        const val EOT_TOKEN = 49407L
        const val CONTEXT_LENGTH = 77

        private val PATTERN = Pattern.compile(
            "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+",
            Pattern.CASE_INSENSITIVE
        )

        private fun bytesToUnicode(): Map<Int, Char> {
            val bs = mutableListOf<Int>()
            for (b in '!'.code..'~'.code) bs.add(b)
            for (b in '¡'.code..'¬'.code) bs.add(b)
            for (b in '®'.code..'ÿ'.code) bs.add(b)

            val cs = bs.toMutableList()
            var n = 0
            for (b in 0..255) {
                if (b !in bs) {
                    bs.add(b)
                    cs.add(256 + n)
                    n++
                }
            }
            return bs.indices.associate { bs[it] to cs[it].toChar() }
        }
    }

    private val byteEncoder = bytesToUnicode()
    private val bpeRanks = mutableMapOf<Pair<String, String>, Int>()
    private val encoder = mutableMapOf<String, Long>()

    init {
        try {
            val (rawStream, isGzipPath) = try {
                context.assets.open(vocabAssetPath) to vocabAssetPath.endsWith(".gz")
            } catch (e: Exception) {
                val fallback = vocabAssetPath.removeSuffix(".gz")
                context.assets.open(fallback) to false
            }
            rawStream.use { raw ->
                val decodedStream = if (isGzipPath) {
                    try {
                        GZIPInputStream(raw)
                    } catch (e: Exception) {
                        raw
                    }
                } else {
                    raw
                }
                BufferedReader(InputStreamReader(decodedStream, Charsets.UTF_8)).use { reader ->
                    var rank = 0
                    var line: String? = reader.readLine()
                    // Skip header line (e.g. #version: 0.2)
                    if (line != null && line.startsWith("#")) {
                        line = reader.readLine()
                    }
                    while (line != null) {
                        val trimmed = line.trim()
                        if (trimmed.isNotEmpty()) {
                            val parts = trimmed.split(" ")
                            if (parts.size == 2) {
                                bpeRanks[Pair(parts[0], parts[1])] = rank++
                            }
                        }
                        line = reader.readLine()
                    }
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }

        // Build initial vocab
        var vocabIndex = 0L
        for (v in byteEncoder.values) {
            encoder[v.toString()] = vocabIndex++
        }
        for (v in byteEncoder.values) {
            encoder["${v}</w>"] = vocabIndex++
        }
        for ((p, _) in bpeRanks) {
            val token = p.first + p.second
            if (!encoder.containsKey(token)) {
                encoder[token] = vocabIndex++
            }
        }
        encoder["<|startoftext|>"] = SOT_TOKEN
        encoder["<|endoftext|>"] = EOT_TOKEN
    }

    private fun getPairs(word: List<String>): Set<Pair<String, String>> {
        val pairs = mutableSetOf<Pair<String, String>>()
        for (i in 0 until word.size - 1) {
            pairs.add(Pair(word[i], word[i + 1]))
        }
        return pairs
    }

    private fun bpe(token: String): List<String> {
        var word = token.dropLast(1).map { it.toString() }.toMutableList()
        word.add(token.last().toString() + "</w>")

        var pairs = getPairs(word)
        if (pairs.isEmpty()) {
            return listOf(token + "</w>")
        }

        while (true) {
            val minPair = pairs.minByOrNull { bpeRanks[it] ?: Int.MAX_VALUE } ?: break
            if (!bpeRanks.containsKey(minPair)) break

            val first = minPair.first
            val second = minPair.second
            val newWord = mutableListOf<String>()
            var i = 0
            while (i < word.size) {
                val j = word.subList(i, word.size).indexOf(first)
                if (j == -1) {
                    newWord.addAll(word.subList(i, word.size))
                    break
                }
                newWord.addAll(word.subList(i, i + j))
                i += j

                if (i < word.size - 1 && word[i] == first && word[i + 1] == second) {
                    newWord.add(first + second)
                    i += 2
                } else {
                    newWord.add(word[i])
                    i += 1
                }
            }
            word = newWord
            if (word.size == 1) break
            pairs = getPairs(word)
        }
        return word
    }

    fun encode(text: String): LongArray {
        val tokens = mutableListOf<Long>()
        tokens.add(SOT_TOKEN)

        val cleanText = text.lowercase().trim()
        val matcher = PATTERN.matcher(cleanText)

        while (matcher.find()) {
            val match = matcher.group()
            val encodedBytes = match.toByteArray(Charsets.UTF_8).map { byteEncoder[it.toInt() and 0xFF] ?: it.toInt().toChar() }.joinToString("")
            for (bpeToken in bpe(encodedBytes)) {
                encoder[bpeToken]?.let { tokens.add(it) }
            }
        }

        tokens.add(EOT_TOKEN)

        val result = LongArray(CONTEXT_LENGTH) { 0L }
        val copyCount = minOf(tokens.size, CONTEXT_LENGTH)
        for (i in 0 until copyCount) {
            result[i] = tokens[i]
        }
        return result
    }
}
