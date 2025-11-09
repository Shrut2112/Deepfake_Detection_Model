# Deepfake Detection Model — Detailed Description

## Problem Overview
Detecting deepfake images in low-resolution, diverse-object datasets is a significant challenge in computer vision. Unlike typical facial deepfake datasets, this task involves images of various objects (e.g., animals, vehicles) at a small size (32x32 pixels). The absence of facial cues and the low quality of samples require a specialized feature extraction and modeling pipeline. Our goal is to robustly distinguish real and fake images as labeled in the proprietary dataset, matching the expected output format for hackathon evaluation.

Solution Pipeline:

   ![WhatsApp Image 2025-11-09 at 18 15 40_c3bebcd3](https://github.com/user-attachments/assets/012e53ba-014c-4b5d-a215-f3e3c8fc64e1)

## Data Preprocessing Steps
- *Image Characteristics:* All images are originally sized at 32x32 pixels and are low quality by design. To avoid losing vital frequency features, images for the frequency branch are used at their original size.
- *EfficientNet Branch Preprocessing:* Images are resized to 224x224 pixels to match EfficientNetB1's requirements for spatial feature extraction. This resizing is carefully controlled to minimize semantic loss.
- *Frequency Feature Branch:* Images are converted to grayscale for Discrete Fourier Transform (DFT) analysis, which yields magnitude and phase spectra for frequency domain feature extraction.
- *Data Augmentation:* Each sample is augmented threefold using horizontal flips, rotations (±5%), translations (±5%), brightness adjustment (±10%), and contrast variation (±10%). These augmentations enhance generalization and increase sample diversity without disrupting brittle features in small images.

## Feature Extraction Techniques
- *Frequency Domain Analysis:* Magnitude and phase spectra are computed using DFT, allowing the model to capture subtle manipulation artifacts in the frequency domain. Studies have shown that deepfake images often have unnatural or smooth frequency distributions.[3]
- *Error Level Analysis (ELA):* ELA extracts compression inconsistencies, highlighting areas in images where digital tampering may have occurred. ELA complements frequency analysis by revealing spatial artifacts invisible to pixel-level methods.
- *Spatial Deep Features:* EfficientNetB1 (pretrained on ImageNet) is used to extract semantic and structural features from the upsampled images. This robust CNN architecture provides high-level spatial embeddings relevant to object classification (beyond faces).[2][4]
- *Multi-Branch Fusion:* Frequency spectra, ELA maps, and spatial features are processed in parallel branches and then concatenated for joint analysis. This complementary fusion improves artifact detection across image types, aligning with findings from low-res deepfake detection research.[1][6]

## Model Architecture, Hyperparameters & Training
- *Dual-Branch Neural Network:*
  - Branch 1: EfficientNetB1 (frozen weights) processes 224x224 RGB images, followed by dense layers (256 → 64 units, ReLU), dropout (0.3, 0.4), and L2 regularization (0.01).
  - Branch 2: Custom CNN processes 32x32 frequency domain inputs (DFT, ELA); three Conv2D layers (32, 64, 128 filters), max pooling, batch normalization, global average pooling, dense layer (256 units), dropout (0.3).
  - Outputs from both branches are concatenated, then passed through Gaussian noise (std=0.2) and dense layers (256 → 64 units, ReLU + dropout) to the final sigmoid classifier.
- *Training Strategy:*
  - Hyperparams: Dropout in dense layers (0.3, 0.4), L2 regularization (0.01), Gaussian noise std (0.2), learning rate (0.0001), label smoothing (0.05).
  - Optimizer: Adam
  - Loss: Binary cross-entropy with label smoothing
  - Training: Early stopping and validation split (train/val: 80/20)
 
 
![WhatsApp Image 2025-11-09 at 18 15 40_45728017](https://github.com/user-attachments/assets/cdbc8aba-89cc-42d3-8b44-7e17a74de321)


## Reasoning Behind Approach
- *Frequency anomalies* are well-documented in deepfake images, especially in low-res settings. DFT-based magnitude and phase spectra expose these differences, critical for detecting non-face manipulations.[3]
- *ELA* addresses digital tampering beyond pixel changes by exposing compression artifacts, a useful signal in compressed or low-res images.
- *EfficientNetB1* extracts high-level object semantics and is robust in transfer learning contexts for non-facial datasets.[4][2]
- *Fusion of frequency and spatial features* improves accuracy, generalizability, and robustness, particularly when small image sizes would otherwise cripple single-modality detectors.

## Challenges & Solutions
- *Small Image Size/Low Quality:* Detailed information is sparse, so the pipeline preserves original image resolution for frequency analysis, while upsampling for spatial feature extraction is limited to required branches only.[6][3]
- *Feature Loss in Augmentation:* Data augmentation was carefully limited to transformations known to preserve frequency signals while improving generalization (flips, small rotations, translations, brightness/contrast changes).
- *Overfitting Risk:* Regularization using dropout, Gaussian noise, L2 penalties, and label smoothing was critical, as validated by rigorous hyperparameter tuning.

## Model Evaluation Results
- *Training Accuracy:* 97.34%
- *Validation Accuracy:* 95.00%
- *Optimized Threshold using Roc Curve:* 0.509
- *Validation Metrics:*
  | Class     | Precision | Recall | F1-Score | Support |
  |-----------|-----------|--------|----------|---------|
  | Fake (0)  | 0.95      | 0.95   | 0.95     | 637     |
  | Real (1)  | 0.95      | 0.95   | 0.95     | 643     |
  | *Overall* |   —       |   —    | 0.95     | 1280    |
- The model demonstrates high accuracy and balance across true positive and true negative rates, backing up efficacy on unseen data.

![WhatsApp Image 2025-11-09 at 18 16 20_e9d2937b](https://github.com/user-attachments/assets/756dd2b6-26a1-4217-8892-2cb1aae51017)


![WhatsApp Image 2025-11-09 at 18 16 20_28cfd89e](https://github.com/user-attachments/assets/5ff3778b-468c-4ba0-aaac-9f791ec0d60c)


## References to Related Work
- Studies confirm optimized preprocessing and fusion of frequency/spatial features improves accuracy for low-res deepfake detection.[1][2][4][6][3]
- Literature supports DFT-, DCT-, and ELA-based approaches for non-face, artifact-level forgery detection.[5][3]
- EfficientNetB1 recognized for adaptability to diverse domains in transfer learning.[2][4]

## Final Notes & Recommendations
- This pipeline demonstrates robust, scalable deepfake detection for varied objects at low resolutions, outperforming single-feature models and avoiding common pitfalls documented in recent surveys.[7]
- Further enhancements could look at advanced augmentations (MixUp, CutMix), regularization variants (DropConnect, spatial dropout), or integration with transformer architectures for real-time detection.

***
