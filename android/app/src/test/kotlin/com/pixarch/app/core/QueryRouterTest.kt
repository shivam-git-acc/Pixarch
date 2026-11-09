package com.pixarch.app.core

import com.pixarch.app.core.query.QueryRouter
import org.junit.Assert.assertEquals
import org.junit.Test

class QueryRouterTest {

    private val router = QueryRouter()

    @Test
    fun testDocumentPossessivePhrase() {
        val route = router.route("give me my aadhaar card")
        assertEquals(QueryRouter.DOCUMENT, route.branch)
        assertEquals(1.0f, route.confidence)
    }

    @Test
    fun testSceneTerms() {
        val route = router.route("find a photo of a beach")
        assertEquals(QueryRouter.SCENE, route.branch)
        assertEquals(0.9f, route.confidence)
    }

    @Test
    fun testDocumentTerms() {
        val route = router.route("tax invoice with gst number")
        assertEquals(QueryRouter.DOCUMENT, route.branch)
        assertEquals(0.9f, route.confidence)
    }

    @Test
    fun testGymWorkoutPhotos() {
        val route = router.route("photos from gym workout")
        assertEquals(QueryRouter.SCENE, route.branch)
        assertEquals(0.9f, route.confidence)
    }
}
