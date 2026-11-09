package com.pixarch.app.core

import com.pixarch.app.core.doctype.DocTypeClassifier
import org.junit.Assert.assertEquals
import org.junit.Test

class DocTypeClassifierTest {

    @Test
    fun testAadhaarDetection() {
        val text = "GOVERNMENT OF INDIA UNIQUE IDENTIFICATION AUTHORITY OF INDIA ENROLMENT NO"
        val result = DocTypeClassifier.classifyText(text)
        assertEquals("aadhaar", result.docType)
        assertEquals("aadhaar-authority", result.ruleName)
        assertEquals(3, result.weight)
    }

    @Test
    fun testPanDetection() {
        val text = "INCOME TAX DEPARTMENT GOVT OF INDIA PERMANENT ACCOUNT NUMBER CARD"
        val result = DocTypeClassifier.classifyText(text)
        assertEquals("pan", result.docType)
        assertEquals("pan-incometax", result.ruleName)
    }

    @Test
    fun testPassportDetection() {
        val text = "REPUBLIC OF INDIA PASSPORT TYPE P COUNTRY CODE IND"
        val result = DocTypeClassifier.classifyText(text)
        assertEquals("passport", result.docType)
    }

    @Test
    fun testDrivingLicenceDetection() {
        val text = "UNION OF INDIA DRIVING LICENCE TRANSPORT DEPARTMENT LMV"
        val result = DocTypeClassifier.classifyText(text)
        assertEquals("dl", result.docType)
    }

    @Test
    fun testReceiptDetection() {
        val text = "TAX INVOICE STORE 102 ITEMS TOTAL 450.00 CASH 500 CHANGE 50.00"
        val result = DocTypeClassifier.classifyText(text)
        assertEquals("receipt", result.docType)
    }

    @Test
    fun testBillDetection() {
        val text = "ELECTRICITY BILL UNITS CONSUMED 320 METER READING PREVIOUS 1050"
        val result = DocTypeClassifier.classifyText(text)
        assertEquals("bill", result.docType)
    }

    @Test
    fun testWholeWordBoundaries() {
        // "company" contains "pan" as substring, but shouldn't trigger whole-word pan rule
        val text1 = "General company announcement for all employees"
        val result1 = DocTypeClassifier.classifyText(text1)
        assertEquals(DocTypeClassifier.UNKNOWN, result1.docType)

        // "august" contains "gst" as substring, but shouldn't trigger gst rule
        val text2 = "Monthly newsletter dated august 15th"
        val result2 = DocTypeClassifier.classifyText(text2)
        assertEquals(DocTypeClassifier.UNKNOWN, result2.docType)
    }
}
