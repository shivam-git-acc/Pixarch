package com.pixarch.app.core.ingest

import android.content.Context
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import com.pixarch.app.PixarchApp

class GalleryIndexWorker(
    appContext: Context,
    workerParams: WorkerParameters
) : CoroutineWorker(appContext, workerParams) {

    override suspend fun doWork(): Result {
        val app = PixarchApp.get(applicationContext)
        val indexer = GalleryIndexer(
            context = applicationContext,
            database = app.database,
            ocrEngine = app.ocrEngine,
            clipRunner = app.clipRunner
        )

        return try {
            val processed = indexer.indexMediaStore(limit = 200) { progress ->
                setProgressAsync(
                    workDataOf(
                        "indexed" to progress.indexedCount,
                        "total" to progress.totalCount,
                        "current" to progress.currentItemPath
                    )
                )
            }
            Result.success(workDataOf("processed" to processed))
        } catch (e: Exception) {
            e.printStackTrace()
            Result.failure(workDataOf("error" to (e.message ?: "Unknown error")))
        }
    }
}
