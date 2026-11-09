package com.pixarch.app.ui.main

import junit.framework.TestCase.assertEquals
import junit.framework.TestCase.assertFalse
import org.junit.Test

class MainScreenViewModelTest {
  @Test
  fun testDefaultUiState() {
    val state = PixarchUiState()
    assertEquals("", state.query)
    assertEquals(BranchFilter.AUTO, state.selectedFilter)
    assertFalse(state.isSearching)
    assertFalse(state.isIndexing)
    assertEquals(0, state.totalIndexed)
    assertEquals(emptyList<UiSearchHit>(), state.results)
  }

  @Test
  fun testBranchFilters() {
    assertEquals("Auto Router", BranchFilter.AUTO.displayName)
    assertEquals("Document (FTS5)", BranchFilter.DOCUMENT.displayName)
    assertEquals("Scene (MobileCLIP)", BranchFilter.SCENE.displayName)
    assertEquals("document", BranchFilter.DOCUMENT.branchKey)
    assertEquals("scene", BranchFilter.SCENE.branchKey)
    assertEquals(null, BranchFilter.AUTO.branchKey)
  }
}
