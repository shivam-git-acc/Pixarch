package com.pixarch.app.ui.main

import androidx.activity.ComponentActivity
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithText
import org.junit.Before
import org.junit.Rule
import org.junit.Test

/** Smoke test for [MainScreen]: it composes and shows the search field. */
class MainScreenTest {

  @get:Rule val composeTestRule = createAndroidComposeRule<ComponentActivity>()

  @Before
  fun setup() {
    composeTestRule.setContent { MainScreen() }
  }

  @Test
  fun searchField_isShown() {
    composeTestRule
      .onNodeWithText("Search photos or documents", substring = true)
      .assertExists()
  }
}
