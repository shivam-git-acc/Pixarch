package com.pixarch.app.core.doctype

import java.util.regex.Pattern

data class Rule(
    val docType: String,
    val name: String,
    val anyOf: List<String> = emptyList(),
    val allOf: List<String> = emptyList(),
    val weight: Int = 1
)

data class ClassificationResult(
    val docType: String,
    val ruleName: String,
    val weight: Int
)

/**
 * Stage 2a - document subtype tagging from OCR text.
 * Rule-based on purpose, matching the Pixarch baseline design.
 */
object DocTypeClassifier {
    const val UNKNOWN = "unknown"

    private val RULES = listOf(
        Rule("aadhaar", "aadhaar-authority", weight = 3,
            anyOf = listOf("unique identification authority", "uidai", "आधार", "भारतीय विशिष्ट पहचान प्राधिकरण")),
        Rule("aadhaar", "aadhaar-keyword", weight = 2, anyOf = listOf("aadhaar", "aadhar")),
        Rule("pan", "pan-incometax", weight = 3,
            anyOf = listOf("permanent account number", "income tax department", "आयकर विभाग")),
        Rule("pan", "pan-keyword", weight = 1, anyOf = listOf("pan card")),
        Rule("passport", "passport-republic", weight = 3,
            allOf = listOf("passport"), anyOf = listOf("republic of india", "भारत गणराज्य")),
        Rule("passport", "passport-mrz", weight = 3,
            anyOf = listOf("p<ind", "type passport", "passport no")),
        Rule("passport", "passport-keyword", weight = 1, anyOf = listOf("passport")),
        Rule("dl", "dl-transport", weight = 3,
            anyOf = listOf("driving licence", "driving license", "transport department", "international driving permit")),
        Rule("dl", "dl-cov", weight = 2, anyOf = listOf("cov issue date", "lmv", "mcwg")),
        Rule("receipt", "receipt-total", weight = 2,
            anyOf = listOf("total", "subtotal", "grand total", "cash", "change")),
        Rule("receipt", "receipt-tax", weight = 3,
            anyOf = listOf("gst", "tax invoice", "invoice no", "receipt no", "bill no")),
        Rule("credit_card", "credit-card-phrase", weight = 3, anyOf = listOf("credit card")),
        Rule("debit_card", "debit-card-phrase", weight = 3, anyOf = listOf("debit card")),
        Rule("credit_card", "card-face", weight = 1,
            allOf = listOf("valid thru"), anyOf = listOf("cardholder", "card holder")),
        Rule("passbook", "passbook-keyword", weight = 3,
            anyOf = listOf("passbook", "pass book", "savings bank account")),
        Rule("bank_statement", "statement-header", weight = 3,
            anyOf = listOf("statement of account", "account statement", "opening balance", "closing balance")),
        Rule("bank_statement", "statement-columns", weight = 2,
            allOf = listOf("debit", "credit"), anyOf = listOf("balance")),
        Rule("report_card", "report-card-phrase", weight = 3,
            anyOf = listOf("report card", "progress report", "marks obtained", "academic session")),
        Rule("bill", "bill-utility", weight = 3,
            anyOf = listOf("electricity bill", "utility bill", "units consumed", "meter reading", "consumer no", "billing period", "previous reading", "sanctioned load")),
        Rule("book", "book-isbn", weight = 3, anyOf = listOf("isbn", "chapter", "copyright ©"))
    )

    private fun isAscii(str: String): Boolean = str.all { it.code < 128 }

    private fun isPresent(term: String, text: String): Boolean {
        return if (term.length <= 4 && isAscii(term)) {
            val pattern = Pattern.compile("\\b" + Pattern.quote(term) + "\\b", Pattern.CASE_INSENSITIVE)
            pattern.matcher(text).find()
        } else {
            text.contains(term, ignoreCase = true)
        }
    }

    fun classifyText(normalisedText: String): ClassificationResult {
        if (normalisedText.isBlank()) {
            return ClassificationResult(UNKNOWN, "", 0)
        }

        val lowered = normalisedText.lowercase()
        var best = ClassificationResult(UNKNOWN, "", 0)

        for (rule in RULES) {
            if (rule.allOf.isNotEmpty() && !rule.allOf.all { isPresent(it, lowered) }) {
                continue
            }
            if (rule.anyOf.isNotEmpty() && !rule.anyOf.any { isPresent(it, lowered) }) {
                continue
            }
            if (rule.anyOf.isEmpty() && rule.allOf.isEmpty()) {
                continue
            }
            if (rule.weight > best.weight) {
                best = ClassificationResult(rule.docType, rule.name, rule.weight)
            }
        }
        return best
    }
}
