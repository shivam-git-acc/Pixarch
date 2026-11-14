package com.pixarch.app.core.ingest

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.TimeZone

class MediaMetadataTest {

    @Test
    fun screenshotNeedsScreenSizeAndNoCameraExif() {
        assertTrue(MediaMetadata.looksLikeScreenshot(1080, 2400, hasCameraExif = false, relativePath = "Pictures/"))
        assertTrue(MediaMetadata.looksLikeScreenshot(2400, 1080, hasCameraExif = false, relativePath = null))
        assertFalse(MediaMetadata.looksLikeScreenshot(1080, 2400, hasCameraExif = true, relativePath = "DCIM/Camera/"))
        assertFalse(MediaMetadata.looksLikeScreenshot(4000, 3000, hasCameraExif = false, relativePath = null))
    }

    @Test
    fun screenshotsFolderIsADirectSignal() {
        assertTrue(MediaMetadata.looksLikeScreenshot(999, 777, hasCameraExif = false, relativePath = "Pictures/Screenshots/"))
        assertTrue(MediaMetadata.looksLikeScreenshot(999, 777, hasCameraExif = true, relativePath = "DCIM/Screenshots/"))
    }

    @Test
    fun datesUseThePythonFormat() {
        assertEquals("2023-12-05 12:34:56", MediaMetadata.formatExifDate("2023:12:05 12:34:56"))
        assertNull(MediaMetadata.formatExifDate("2023"))
        assertEquals(
            "2023-12-05 12:34:56",
            MediaMetadata.formatDateTaken(1701779696000L, TimeZone.getTimeZone("UTC"))
        )
    }
}
