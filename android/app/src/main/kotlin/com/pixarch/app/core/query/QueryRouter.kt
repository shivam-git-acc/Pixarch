package com.pixarch.app.core.query

data class Route(
    val branch: String,
    val confidence: Float,
    val reason: String
)

/**
 * Stage 3 - Query Routing.
 * Keyword rules first; CLIP prototype similarity fallback when rules abstain.
 */
class QueryRouter(
    private val encodeTextFn: ((String) -> FloatArray)? = null,
    private val prototypes: Map<String, FloatArray> = emptyMap()
) {
    companion object {
        const val DOCUMENT = "document"
        const val SCENE = "scene"

        private val DOC_TERMS = setOf(
            "aadhaar", "aadhar", "pan", "passport", "licence", "license", "dl",
            "receipt", "receipts", "bill", "bills", "invoice", "statement",
            "document", "documents", "card", "id", "identity", "certificate",
            "ticket", "voucher", "payment", "tax", "gst", "book", "page", "scan"
        )

        private val DOC_PHRASES = listOf("find my", "give me my", "show me my", "where is my")

        private val SCENE_TERMS = setOf(
            "photo", "photos", "picture", "pictures", "pic", "pics", "selfie",
            "beach", "mountain", "forest", "trees", "party", "wedding", "food",
            "gym", "mall", "campus", "college", "park", "travel", "trip", "sunset"
        )

        private val WORD_REGEX = Regex("[a-z0-9]+")

        private fun dotProduct(a: FloatArray, b: FloatArray): Float {
            var sum = 0f
            val n = minOf(a.size, b.size)
            for (i in 0 until n) sum += a[i] * b[i]
            return sum
        }
    }

    fun route(queryText: String): Route {
        val lowered = queryText.lowercase()
        val tokens = WORD_REGEX.findAll(lowered).map { it.value }.toSet()

        val docHits = tokens.intersect(DOC_TERMS)
        val sceneHits = tokens.intersect(SCENE_TERMS)
        val phraseHit = DOC_PHRASES.any { lowered.contains(it) }

        if (phraseHit && docHits.isNotEmpty()) {
            return Route(DOCUMENT, 1.0f, "phrase+terms:${docHits.sorted().joinToString(",")}")
        }
        if (docHits.isNotEmpty() && sceneHits.isEmpty()) {
            return Route(DOCUMENT, 0.9f, "terms:${docHits.sorted().joinToString(",")}")
        }
        if (sceneHits.isNotEmpty() && docHits.isEmpty()) {
            return Route(SCENE, 0.9f, "terms:${sceneHits.sorted().joinToString(",")}")
        }
        if (phraseHit) {
            return Route(DOCUMENT, 0.7f, "phrase")
        }

        // Semantic fallback via prototypes if rules abstained
        if (prototypes.isNotEmpty() && encodeTextFn != null) {
            val queryEmb = encodeTextFn.invoke(queryText)
            val scored = prototypes.map { (bucket, protoVec) ->
                bucket to dotProduct(queryEmb, protoVec)
            }.sortedByDescending { it.second }

            val best = scored[0]
            val margin = if (scored.size > 1) best.second - scored[1].second else 1.0f
            return Route(best.first, margin, "clip:${"%.3f".format(best.second)}")
        }

        return Route(SCENE, 0.0f, "default")
    }
}
