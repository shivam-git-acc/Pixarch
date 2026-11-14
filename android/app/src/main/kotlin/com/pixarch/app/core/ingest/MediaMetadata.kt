package com.pixarch.app.core.ingest

import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

/**
 * Port of the metadata half of `src/pixarch/ingest.py`: capture date and the
 * screenshot heuristic. Recorded as features only -- never used to route (baseline).
 */
object MediaMetadata {

    /** Same table as ingest.py `_SCREEN_SIZES` (both orientations). */
    private val SCREEN_SIZES: Set<Pair<Int, Int>> = listOf(
        1080 to 1920, 1080 to 2340, 1080 to 2400, 1080 to 2160, 1440 to 3200,
        1440 to 2560, 1170 to 2532, 1284 to 2778, 828 to 1792, 750 to 1334,
        720 to 1280, 1125 to 2436, 1242 to 2688, 1179 to 2556, 1290 to 2796,
        1920 to 1080, 2560 to 1440, 3840 to 2160, 1366 to 768, 1536 to 864,
    ).flatMap { (w, h) -> listOf(w to h, h to w) }.toSet()

    /**
     * ingest.py: no camera EXIF and pixel-exact to a known screen size. On a phone the
     * MediaStore folder is a direct signal, so `Screenshots/` also counts.
     */
    fun looksLikeScreenshot(width: Int, height: Int, hasCameraExif: Boolean, relativePath: String?): Boolean {
        if (relativePath != null && relativePath.contains("screenshot", ignoreCase = true)) return true
        if (hasCameraExif) return false
        return (width to height) in SCREEN_SIZES
    }

    /** MediaStore DATE_TAKEN (epoch ms, UTC) -> "YYYY-MM-DD HH:MM:SS" in device local time. */
    fun formatDateTaken(epochMs: Long, zone: TimeZone = TimeZone.getDefault()): String =
        SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).apply { timeZone = zone }.format(Date(epochMs))

    /** EXIF "YYYY:MM:DD HH:MM:SS" -> "YYYY-MM-DD HH:MM:SS", as ingest.py::read_exif does. */
    fun formatExifDate(raw: String?): String? {
        if (raw == null || raw.length < 10) return null
        return (raw.substring(0, 10).replace(':', '-') + raw.substring(10)).trim()
    }
}
