package com.pixarch.app.core.query

/**
 * Port of `src/pixarch/query/bm25_search.py::to_match_query`: turns a plain-language
 * question into a safe FTS5 OR-query.
 *
 * - FTS5 syntax characters `" : ^ * ( ) -` become spaces, so user text can't be parsed
 *   as operators.
 * - Each term is quoted and terms are OR-ed (an AND over a whole question returns
 *   nothing once OCR has dropped a word). No prefix `*`: the porter tokenizer already
 *   stems, and a prefix turns "me" into "merchant", "member", ...
 * - Stopwords are dropped, but if every term is a stopword the original terms are kept
 *   so the query never silently becomes empty.
 * - Non-ASCII terms (Devanagari) are kept as-is.
 * Parity is checked by FtsQueryTest against Python-generated fixtures.
 */
object FtsQuery {

    val STOPWORDS: Set<String> = """
        a an the of for to in on at by with from and or is are was were be been
        my me mine i you your our we us this that these those it its
        give show find get me please need want looking look search
    """.trimIndent().split(Regex("\\s+")).filter { it.isNotEmpty() }.toSet()

    private val FTS_SPECIAL = Regex("[\":^*()\\-]")

    /** Python str.split(): runs of Unicode whitespace. */
    private fun splitWhitespace(s: String): List<String> {
        val out = ArrayList<String>()
        val cur = StringBuilder()
        var i = 0
        while (i < s.length) {
            val cp = s.codePointAt(i)
            i += Character.charCount(cp)
            if (Character.isWhitespace(cp) || Character.isSpaceChar(cp)) {
                if (cur.isNotEmpty()) { out.add(cur.toString()); cur.setLength(0) }
            } else {
                cur.appendCodePoint(cp)
            }
        }
        if (cur.isNotEmpty()) out.add(cur.toString())
        return out
    }

    fun toMatchQuery(text: String): String {
        val terms = splitWhitespace(FTS_SPECIAL.replace(text, " "))
        if (terms.isEmpty()) return ""
        val content = terms.filter { it.lowercase() !in STOPWORDS }
        return (content.ifEmpty { terms }).joinToString(" OR ") { "\"$it\"" }
    }
}
