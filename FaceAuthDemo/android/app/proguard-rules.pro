# Add project specific ProGuard rules here.
# By default, the flags in this file are appended to flags specified
# in /usr/local/Cellar/android-sdk/24.3.3/tools/proguard/proguard-android.txt
# You can edit the include path and order by changing the proguardFiles
# directive in build.gradle.
#
# For more details, see
#   http://developer.android.com/guide/developing/tools/proguard.html

# Add any project specific keep options here:
# Ignore missing TensorFlow Lite GPU classes
-dontwarn org.tensorflow.lite.gpu.**

# ---------------------------------------------------------------------------
# Rules below are INACTIVE while enableProguardInReleaseBuilds = false in
# build.gradle. They exist so minification can be re-enabled without repeating
# the failure documented there: R8 renamed VisionCameraFaceDetectorPlugin to
# n4.g and removed VisionCameraFaceDetectorPluginKt, which silently killed face
# detection in release builds only (debug builds never minify, so they worked).
#
# These are written from the reflective-entrypoint requirements of each library
# and have NOT been verified on-device. Test enroll AND verify on a real phone
# before trusting them — the app boots and the camera previews either way, so
# only a full run proves anything.
# ---------------------------------------------------------------------------

# VisionCamera frame processor plugins are looked up reflectively by the string
# name passed to initFrameProcessorPlugin(), so the plugin classes and their
# registering packages must survive renaming.
-keep class com.mrousavy.camera.** { *; }
-keep class * extends com.mrousavy.camera.frameprocessors.FrameProcessorPlugin { *; }

# react-native-vision-camera-face-detector: registers the "detectFaces" plugin.
-keep class com.visioncamerafacedetector.** { *; }

# vision-camera-resize-plugin: registers the "resize" plugin.
-keep class com.visioncameraresizeplugin.** { *; }

# ML Kit face detection loads its detector and model via registrars/reflection.
-keep class com.google.mlkit.** { *; }
-keep class com.google.android.gms.internal.mlkit_vision_face.** { *; }
-keep class com.google.android.gms.vision.** { *; }
-dontwarn com.google.mlkit.**

# TensorFlow Lite: the interpreter and its native bindings are reached via JNI.
-keep class org.tensorflow.lite.** { *; }
-dontwarn org.tensorflow.**

# react-native-worklets-core installs JSI bindings from native.
-keep class com.worklets.** { *; }
