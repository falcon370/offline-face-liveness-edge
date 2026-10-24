<!--
READ-ONLY — verbatim transcription of the original brief in ../hackathon_doc7.pdf.
Do not edit to reflect implementation status, progress, or opinions — this file
exists so the exact original requirements are never lost or paraphrased as the
rest of the docs evolve. Track how the solution meets these requirements in
DELIVERABLES.md instead; link back here for the source text.
-->

# Hackathon 7.0 — Original Requirements

> Verbatim from [`../hackathon_doc7.pdf`](../hackathon_doc7.pdf). See
> [DELIVERABLES.md](./DELIVERABLES.md) for how the solution maps to each clause.

## Title of Hackathon

Develop a mobile based secure offline facial recognition and liveness detection
system for remote locations.

## The Objective

To develop a highly accurate, lightweight, and entirely offline facial
recognition and liveness detection algorithm that can be seamlessly integrated
into the existing Datalake 3.0 app, ensuring uninterrupted operations in
zero-network zones.

## The Problem Statement

> "How can we accurately and securely authenticate field personnel using facial
> recognition and liveness detection on standard mid-range mobile devices
> without any active internet connection, while ensuring the AI model remains
> lightweight and seamlessly integrates with a React Native application on both
> Android and iOS devices?"

## Target Audience

Open to IT Students, AI/ML Enthusiasts, App Developers, Tech Professionals, and
Startups.

## Technical Constraints & Specifications

1. **Framework Compatibility:** Must be fully compatible with React Native,
   supporting cross-functional deployment on both Android and iOS.
2. **Model Footprint:** The AI model must be extremely lightweight to avoid
   bloating the Datalake app package. The target size is ~20 MB (the smaller,
   the better).
3. **Processing Speed:** Time taken to recognize a face and verify liveness
   must be < 1 second on standard mid-range devices.
4. **Hardware Requirements:** Must function smoothly without requiring
   high-end GPUs. Minimum supported OS: Android 8.0+ and iOS 12+ on devices
   with a minimum of 3GB RAM.
5. **Accuracy Threshold:** The facial recognition accuracy must be > 95%. The
   model must be trained to recognize diverse Indian demographics and function
   reliably in varying outdoor lighting conditions (e.g., harsh sunlight, low
   light, shadows).
6. **Open-Source Technologies only:** Solution should use only open-source
   technologies if at all any third-party code is used and source-code of the
   working prototype should be shared. No licenses should be required
   additionally.

## Mandatory Deliverables

1. **Working Prototype with Source Code:** A functional cross-platform
   prototype (Android + iOS) built in React Native demonstrating the offline
   capability.
   a. **Offline Liveness Detection:** The solution must include basic offline
      anti-spoofing measures (e.g., requiring the user to blink, smile, or
      turn their head slightly) to prevent attendance fraud via photographs or
      screens.
   b. **Sync & Purge Mechanism:** The solution should have the scope for sync
      with AWS server after network connectivity is restored (local data to be
      purged).
2. **Presentation and Technical Documentation:** The solution should be well
   presented in a pptx. / .pdf file along with clear technical documentation
   detailing the model architecture, integration steps and performance
   benchmarks.

- Submission Open Date: 22.05.2026
- Submission Closure Date: 05.06.2026

## Evaluation Criteria

1. **Innovation Level (30 Marks):** Efficiency of the edge AI model,
   compression techniques used to keep the size under 20MB, and effectiveness
   of offline liveness detection.
2. **Feasibility (30 Marks):** Ease of integration into the existing Datalake
   3.0 React Native architecture and performance on mid-range devices
   (speed < 1 sec).
3. **Scalability & Sustainability (20 Marks):** Reliability of the
   offline-to-online sync/purge mechanism and adaptability to diverse
   lighting/demographics.
4. **Presentation & Documentation (20 Marks):** Clarity of the source code,
   integration guides, and final presentation to the evaluation committee.
