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

class SearchEngine(
    private val database: PixarchDatabase,
    private val tokenizer: ClipTokenizer,
    private val clipRunner: OnnxClipRunner
) {
    private val router = QueryRouter(
        encodeTextFn = { text ->
            val tokens = tokenizer.encode(text)
            clipRunner.encodeText(tokens)
        }
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
            QueryRouter.SCENE -> {
                val tokens = tokenizer.encode(query)
                val queryEmb = clipRunner.encodeText(tokens)
                database.searchVector(queryEmb, topK)
            }
            else -> {
                database.searchBm25(query, topK)
            }
        }

        val elapsed = (System.currentTimeMillis() - t0).toFloat()
        return SearchResult(route, hits, elapsed)
    }
}
