plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.example.artesting"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.artesting"
        minSdk = 30
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
}

dependencies {

    implementation(libs.appcompat)
    implementation(libs.material)
    implementation(libs.activity)
    implementation(libs.constraintlayout)
    implementation(libs.sceneform.ux)
    testImplementation(libs.junit)
    androidTestImplementation(libs.ext.junit)
    androidTestImplementation(libs.espresso.core)
//    implementation("com.google.ar.sceneform.ux:sceneform-ux:1.15.0")
//    implementation("io.github.sceneview:sceneview:2.2.1")
    implementation("com.gorisse.thomas.sceneform:sceneform:1.23.0")
//    implementation("com.github.sceneview.sceneform-android:sceneform:-SNAPSHOT")
    implementation("com.google.ar:core:1.33.0")
    implementation("com.mapbox.mapboxsdk:mapbox-android-navigation:0.42.6")
    implementation("com.mapbox.mapboxsdk:mapbox-android-core:1.3.0")
    implementation("com.mapbox.mapboxsdk:mapbox-android-vision-ar:4.4.0")

}
