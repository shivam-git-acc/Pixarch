package com.pixarch.app.core.clip

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class ClipPreprocessTest {

    /** Expected values from torchvision's Resize(256) + CenterCrop(256) arithmetic. */
    @Test
    fun geometryMatchesTorchvision() {
        val cases = mapOf(
            (1280 to 960) to ClipPreprocess.Geometry(341, 256, 42, 0),
            (720 to 1600) to ClipPreprocess.Geometry(256, 568, 0, 156),
            (301 to 203) to ClipPreprocess.Geometry(379, 256, 62, 0), // 61.5 rounds half-to-even
            (256 to 256) to ClipPreprocess.Geometry(256, 256, 0, 0),
            (1000 to 257) to ClipPreprocess.Geometry(996, 256, 370, 0),
            (4000 to 3000) to ClipPreprocess.Geometry(341, 256, 42, 0),
        )
        for ((dims, expected) in cases) {
            assertEquals("for ${dims.first}x${dims.second}", expected, ClipPreprocess.geometry(dims.first, dims.second))
        }
    }

    @Test
    fun mobileClip2UsesUnitRangeWithoutMeanStd() {
        assertArrayEquals(floatArrayOf(0f, 0f, 0f), ClipPreprocess.MEAN, 0f)
        assertArrayEquals(floatArrayOf(1f, 1f, 1f), ClipPreprocess.STD, 0f)
    }

    @Test
    fun toChwProducesPlanarRgbInUnitRange() {
        // 2x2 image: orange, white, black, pure blue (ARGB)
        val pixels = intArrayOf(0xFFFF8000.toInt(), 0xFFFFFFFF.toInt(), 0xFF000000.toInt(), 0xFF0000FF.toInt())
        val chw = ClipPreprocess.toChw(pixels, size = 2)
        val r = floatArrayOf(1f, 1f, 0f, 0f)
        val g = floatArrayOf(128 / 255f, 1f, 0f, 0f)
        val b = floatArrayOf(0f, 1f, 0f, 1f)
        assertArrayEquals(r + g + b, chw, 1e-6f)
    }
}
