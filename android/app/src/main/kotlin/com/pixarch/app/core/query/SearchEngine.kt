package com.pixarch.app.core.query

import com.pixarch.app.core.clip.ClipTokenizer
import com.pixarch.app.core.clip.OnnxClipRunner
import com.pixarch.app.core.storage.PixarchDatabase
import com.pixarch.app.core.storage.SearchHit

data class SearchResult(
    val route: Route,
    val hits: List<SearchHit>,
    val elapsedMs: Float
)

/**
 * Routes to exactly one branch -- no fusion, no retry (`src/pixarch/query/engine.py`).
 * The router's CLIP fallback uses the same document/scene prototypes as ingestion.
 */
class SearchEngine(
    private val database: PixarchDatabase,
    private val tokenizer: ClipTokenizer,
    private val clipRunner: OnnxClipRunner,
    bucketPrototypes: Map<String, FloatArray>
) {
    private val router = QueryRouter(
        encodeTextFn = { text -> clipRunner.encodeText(tokenizer.encode(text)) },
        prototypes = bucketPrototypes
    )

    fun search(query: String, topK: Int = 10, forceBranch: String? = null): SearchResult {
        val t0 = System.currentTimeMillis()
        val route = if (forceBranch != null) {
            Route(forceBranch, 1.0f, "forced")
        } else {
            router.route(query)
        }

        val hits = when (route.branch) {
            QueryRouter.DOCUMENT -> {
                database.searchBm25(query, topK)
            }
            else -> {
                val queryEmb = clipRunner.encodeText(tokenizer.encode(query))
                database.searchVector(queryEmb, topK)
            }
        }

        val elapsed = (System.currentTimeMillis() - t0).toFloat()
        return SearchResult(route, hits, elapsed)
    }
}
