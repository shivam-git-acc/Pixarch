package com.pixarch.app.core.classify

import android.content.Context
import java.io.InputStream

/**
 * Stage 1 -- zero-shot document/scene bucket classification, port of
 * `src/pixarch/classify.py::BucketClassifier`.
 *
 * Prototypes (mean prompt embedding per bucket) and the hedge threshold come from
 * `scripts/calibrate_edge_buckets.py`, calibrated for MobileCLIP2-S0 on the corpus.
 * Confidence is the margin between the best and second-best bucket; below the
 * threshold the item is indexed into BOTH branches instead of being guessed.
 */
class BucketClassifier(
    prototypes: Map<String, FloatArray>,
    val threshold: Float
) {
    data class Verdict(val bucket: String, val margin: Float, val routedBoth: Boolean)

    /** Sorted by name, as Python sorts `prompts`; ties go to the first bucket. */
    val buckets: List<String> = prototypes.keys.sorted()
    val prototypes: Map<String, FloatArray> = buckets.associateWith { prototypes.getValue(it) }

    init {
        require(buckets.isNotEmpty()) { "no bucket prototypes" }
        val dims = this.prototypes.values.map { it.size }.toSet()
        require(dims.size == 1) { "prototype dimensions differ: $dims" }
    }

    /** [imageEmbedding] must be L2-normalised, as OnnxClipRunner returns it. */
    fun classify(imageEmbedding: FloatArray): Verdict {
        val scores = buckets.map { b -> b to dot(imageEmbedding, prototypes.getValue(b)) }
        var best = 0
        for (i in 1 until scores.size) if (scores[i].second > scores[best].second) best = i
        val margin = if (scores.size > 1) {
            val second = scores.filterIndexed { i, _ -> i != best }.maxOf { it.second }
            scores[best].second - second
        } else 1f
        return Verdict(scores[best].first, margin, margin < threshold)
    }

    companion object {
        const val ASSET_PATH = "classify/bucket_prototypes.tsv"

        private fun dot(a: FloatArray, b: FloatArray): Float {
            require(a.size == b.size) { "embedding dim ${a.size} != prototype dim ${b.size}" }
            var s = 0f
            for (i in a.indices) s += a[i] * b[i]
            return s
        }

        /** `#threshold<TAB>τ`, other `#` lines ignored, then `bucket<TAB>comma floats`. */
        fun parse(input: InputStream): BucketClassifier {
            var threshold: Float? = null
            val protos = LinkedHashMap<String, FloatArray>()
            input.bufferedReader(Charsets.UTF_8).useLines { lines ->
                for (line in lines) {
                    if (line.isBlank()) continue
                    val parts = line.split('\t', limit = 2)
                    if (line.startsWith("#")) {
                        if (parts[0] == "#threshold") threshold = parts[1].trim().toFloat()
                        continue
                    }
                    protos[parts[0]] = parts[1].split(',').map { it.trim().toFloat() }.toFloatArray()
                }
            }
            return BucketClassifier(protos, requireNotNull(threshold) { "missing #threshold line" })
        }

        fun fromAssets(context: Context): BucketClassifier =
            context.assets.open(ASSET_PATH).use { parse(it) }
    }
}
