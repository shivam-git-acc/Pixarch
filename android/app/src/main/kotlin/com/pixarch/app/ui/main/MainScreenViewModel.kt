package com.pixarch.app.ui.main

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkInfo
import androidx.work.WorkManager
import com.pixarch.app.PixarchApp
import com.pixarch.app.core.ingest.GalleryIndexWorker
import com.pixarch.app.core.query.QueryRouter
import com.pixarch.app.core.query.Route
import com.pixarch.app.core.query.SearchResult
import com.pixarch.app.core.storage.PixarchDatabase
import com.pixarch.app.core.storage.SearchHit
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

enum class BranchFilter(val displayName: String, val branchKey: String?) {
    AUTO("Auto Router", null),
    DOCUMENT("Document (FTS5)", QueryRouter.DOCUMENT),
    SCENE("Scene (MobileCLIP)", QueryRouter.SCENE)
}

data class UiSearchHit(
    val hit: SearchHit,
    val path: String?,
    val detail: PixarchDatabase.ItemDetail?
)

data class PixarchUiState(
    val query: String = "",
    val selectedFilter: BranchFilter = BranchFilter.AUTO,
    val isSearching: Boolean = false,
    val isIndexing: Boolean = false,
    val indexingStatusText: String = "",
    val totalIndexed: Int = 0,
    val currentRoute: Route? = null,
    val searchElapsedMs: Float = 0f,
    val results: List<UiSearchHit> = emptyList(),
    val activeInspectorDetail: PixarchDatabase.ItemDetail? = null,
    val errorMessage: String? = null
)

class MainScreenViewModel(application: Application) : AndroidViewModel(application) {

    private val app = PixarchApp.get(application)
    private val workManager = WorkManager.getInstance(application)

    private val _uiState = MutableStateFlow(PixarchUiState())
    val uiState: StateFlow<PixarchUiState> = _uiState.asStateFlow()

    init {
        loadInitialState()
        observeWorkManager()
    }

    private fun loadInitialState() {
        viewModelScope.launch(Dispatchers.IO) {
            val count = app.database.getTotalItemsCount()
            val recent = app.database.getRecentItems(limit = 40)
            val uiHits = recent.map { hit ->
                val detail = app.database.getItemDetail(hit.itemId)
                UiSearchHit(
                    hit = hit,
                    path = detail?.path,
                    detail = detail
                )
            }
            _uiState.update {
                it.copy(
                    totalIndexed = count,
                    results = uiHits
                )
            }
        }
    }

    private fun observeWorkManager() {
        viewModelScope.launch {
            workManager.getWorkInfosForUniqueWorkFlow("GalleryIndexWork").collect { workInfoList ->
                val workInfo = workInfoList.firstOrNull() ?: return@collect
                when (workInfo.state) {
                    WorkInfo.State.RUNNING -> {
                        val current = workInfo.progress.getString("current") ?: "Scanning..."
                        val indexed = workInfo.progress.getInt("indexed", 0)
                        val total = workInfo.progress.getInt("total", 0)
                        _uiState.update {
                            it.copy(
                                isIndexing = true,
                                indexingStatusText = if (total > 0) "Indexing $indexed/$total ($current)" else "Scanning gallery..."
                            )
                        }
                    }
                    WorkInfo.State.SUCCEEDED -> {
                        _uiState.update {
                            it.copy(
                                isIndexing = false,
                                indexingStatusText = "Indexing completed"
                            )
                        }
                        loadInitialState()
                    }
                    WorkInfo.State.FAILED -> {
                        _uiState.update {
                            it.copy(
                                isIndexing = false,
                                indexingStatusText = "Indexing failed"
                            )
                        }
                    }
                    else -> Unit
                }
            }
        }
    }

    private var searchJob: Job? = null

    fun onQueryChanged(newQuery: String) {
        _uiState.update { it.copy(query = newQuery) }
        searchJob?.cancel()
        if (newQuery.isBlank()) {
            loadInitialState()
        } else {
            searchJob = viewModelScope.launch {
                delay(300)
                executeSearch(newQuery, _uiState.value.selectedFilter)
            }
        }
    }

    fun onFilterSelected(filter: BranchFilter) {
        _uiState.update { it.copy(selectedFilter = filter) }
        val currentQ = _uiState.value.query
        searchJob?.cancel()
        if (currentQ.isNotBlank()) {
            executeSearch(currentQ, filter)
        }
    }

    fun onSearchSubmitted(query: String) {
        searchJob?.cancel()
        if (query.isBlank()) {
            loadInitialState()
            return
        }
        executeSearch(query, _uiState.value.selectedFilter)
    }

    private fun executeSearch(query: String, filter: BranchFilter) {
        viewModelScope.launch {
            _uiState.update { it.copy(isSearching = true, errorMessage = null) }
            try {
                val result: SearchResult = withContext(Dispatchers.Default) {
                    app.searchEngine.search(
                        query = query,
                        topK = 20,
                        forceBranch = filter.branchKey
                    )
                }

                val uiHits = withContext(Dispatchers.IO) {
                    result.hits.map { hit ->
                        val detail = app.database.getItemDetail(hit.itemId)
                        UiSearchHit(
                            hit = hit,
                            path = detail?.path,
                            detail = detail
                        )
                    }
                }

                _uiState.update {
                    it.copy(
                        isSearching = false,
                        currentRoute = result.route,
                        searchElapsedMs = result.elapsedMs,
                        results = uiHits
                    )
                }
            } catch (e: Exception) {
                e.printStackTrace()
                _uiState.update {
                    it.copy(
                        isSearching = false,
                        errorMessage = "Search failed: ${e.message}"
                    )
                }
            }
        }
    }

    fun onHitClicked(hit: UiSearchHit) {
        _uiState.update { it.copy(activeInspectorDetail = hit.detail) }
    }

    fun onDismissInspector() {
        _uiState.update { it.copy(activeInspectorDetail = null) }
    }

    fun startIndexing() {
        val request = OneTimeWorkRequestBuilder<GalleryIndexWorker>().build()
        workManager.enqueueUniqueWork(
            "GalleryIndexWork",
            ExistingWorkPolicy.REPLACE,
            request
        )
        _uiState.update {
            it.copy(
                isIndexing = true,
                indexingStatusText = "Starting gallery indexing..."
            )
        }
    }
}
