package com.pixarch.app

import android.app.Application
import android.content.Context
import com.pixarch.app.core.clip.ClipTokenizer
import com.pixarch.app.core.clip.OnnxClipRunner
import com.pixarch.app.core.ocr.MlKitOcrEngine
import com.pixarch.app.core.query.SearchEngine
import com.pixarch.app.core.storage.PixarchDatabase
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

class PixarchApp : Application() {

    val appScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    val database: PixarchDatabase by lazy {
        PixarchDatabase(this)
    }

    val tokenizer: ClipTokenizer by lazy {
        ClipTokenizer(this)
    }

    val clipRunner: OnnxClipRunner by lazy {
        OnnxClipRunner(this)
    }

    val ocrEngine: MlKitOcrEngine by lazy {
        MlKitOcrEngine()
    }

    val searchEngine: SearchEngine by lazy {
        SearchEngine(database, tokenizer, clipRunner)
    }

    override fun onCreate() {
        super.onCreate()
        // Pre-warm database and tokenizer off the main thread
        appScope.launch(Dispatchers.IO) {
            try {
                database.writableDatabase
                tokenizer
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    companion object {
        fun get(context: Context): PixarchApp {
            return context.applicationContext as PixarchApp
        }
    }
}
