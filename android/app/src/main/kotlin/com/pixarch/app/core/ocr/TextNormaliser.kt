package com.pixarch.app.core.ocr

/**
 * Port of `src/pixarch/ocr.py::normalise`, applied to OCR text before doc-type tagging
 * and FTS indexing:
 *
 *   re.sub(r"[^\w\sऀ-ॿ]", " ", text) -> re.sub(r"\s+", " ", ...) -> strip -> lower
 *
 * Python's `\w` is "alphanumeric or underscore", where alphanumeric means
 * str.isalnum(): letters plus *all* numerics (Nd, Nl, No -- e.g. "½", "²", "Ⅻ").
 * The Devanagari block U+0900-U+097F is kept whole because its vowel signs are
 * combining marks, which `\w` alone would drop. Works on code points so
 * supplementary-plane characters behave like Python's.
 * Parity is checked by TextNormaliserTest against Python-generated fixtures.
 */
object TextNormaliser {

    private fun isWordChar(cp: Int): Boolean {
        if (cp == '_'.code || Character.isLetterOrDigit(cp)) return true
        val type = Character.getType(cp)
        return type == Character.LETTER_NUMBER.toInt() || type == Character.OTHER_NUMBER.toInt()
    }

    private fun isSpace(cp: Int): Boolean = Character.isWhitespace(cp) || Character.isSpaceChar(cp)

    fun normalise(text: String): String {
        val sb = StringBuilder(text.length)
        var pendingSpace = false
        var i = 0
        while (i < text.length) {
            val cp = text.codePointAt(i)
            i += Character.charCount(cp)
            val keep = isWordChar(cp) || cp in 0x0900..0x097F
            if (!keep || isSpace(cp)) {
                pendingSpace = sb.isNotEmpty()
            } else {
                if (pendingSpace) sb.append(' ')
                pendingSpace = false
                sb.appendCodePoint(cp)
            }
        }
        return sb.toString().lowercase()
    }
}
