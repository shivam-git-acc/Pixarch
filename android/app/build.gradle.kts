plugins {
  alias(libs.plugins.android.application)
  alias(libs.plugins.compose.compiler)
  alias(libs.plugins.kotlin.serialization)
}

android {
  namespace = "com.pixarch.app"
  compileSdk = 37
  defaultConfig {
    applicationId = "com.pixarch.app"
    minSdk = 26
    targetSdk = 37
    versionCode = 1
    versionName = "1.0"
  }

  buildTypes {
    release {
      isMinifyEnabled = false
      proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
    }
  }
  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
  }
  buildFeatures {
    compose = true
    aidl = false
    buildConfig = false
    shaders = false
  }

  androidResources {
    noCompress += listOf("onnx", "gz")
  }

  packaging {
    resources {
      excludes += "/META-INF/{AL2.0,LGPL2.1}"
    }
  }
}

// The ONNX models are generated (and gitignored), so a fresh clone has none and the
// app would crash at start-up. Fail the build early with the fix instead.
val checkEdgeModels by tasks.registering {
  val assets = layout.projectDirectory.dir("src/main/assets")
  val required = listOf(
    "models/mobileclip_visual_int8.onnx",
    "models/mobileclip_text_int8.onnx",
    "tokenizer/bpe_simple_vocab_16e6.txt.gz",
    "classify/bucket_prototypes.tsv", // committed; regenerate: scripts/calibrate_edge_buckets.py
  ).map { assets.file(it).asFile }
  doLast {
    val missing = required.filterNot { it.isFile }
    if (missing.isNotEmpty()) {
      throw GradleException(
        "Missing edge model assets:\n" + missing.joinToString("\n") { "  $it" } +
          "\nGenerate them from the repo root with:\n  python scripts/export_edge_models.py"
      )
    }
  }
}
tasks.named("preBuild") { dependsOn(checkEdgeModels) }

dependencies {
  val composeBom = platform(libs.androidx.compose.bom)
  implementation(composeBom)
  androidTestImplementation(composeBom)

  // Core Android dependencies
  implementation(libs.androidx.core.ktx)
  implementation(libs.androidx.lifecycle.runtime.ktx)
  implementation(libs.androidx.activity.compose)

  // Arch Components
  implementation(libs.androidx.lifecycle.runtime.compose)
  implementation(libs.androidx.lifecycle.viewmodel.compose)

  // Compose
  implementation(libs.androidx.compose.ui)
  implementation(libs.androidx.compose.ui.tooling.preview)
  implementation(libs.androidx.compose.material3)
  implementation(libs.androidx.compose.material.icons.extended)
  // Tooling
  debugImplementation(libs.androidx.compose.ui.tooling)
  // Instrumented tests
  androidTestImplementation(libs.androidx.compose.ui.test.junit4)
  debugImplementation(libs.androidx.compose.ui.test.manifest)

  // Local tests: jUnit, coroutines, Android runner
  testImplementation(libs.junit)
  testImplementation(libs.kotlinx.coroutines.test)

  // Instrumented tests: jUnit rules and runners
  androidTestImplementation(libs.androidx.test.core)
  androidTestImplementation(libs.androidx.test.ext.junit)
  androidTestImplementation(libs.androidx.test.runner)
  androidTestImplementation(libs.androidx.test.espresso.core)

  // Navigation
  implementation(libs.androidx.navigation3.ui)
  implementation(libs.androidx.navigation3.runtime)
  implementation(libs.androidx.lifecycle.viewmodel.navigation3)

  // On-device ML Kit OCR (Latin + Devanagari)
  implementation(libs.mlkit.text.recognition)
  implementation(libs.mlkit.text.recognition.devanagari)

  // On-device ONNX Runtime for MobileCLIP visual and text embeddings
  implementation(libs.onnxruntime.android)

  // WorkManager for background image indexing
  implementation(libs.androidx.work.runtime.ktx)

  // Coil for async image rendering
  implementation(libs.coil.compose)

  // EXIF capture date + camera tags (screenshot heuristic)
  implementation(libs.androidx.exifinterface)

  // Bundled SQLite with FTS5 support
  implementation(libs.requery.sqlite.android)
}
