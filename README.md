# facial-stress-cues-detection-music-recommendation
A computer vision–based system that detects facial stress cues from eyebrow motion and facial emotion to support adaptive music recommendation.

# Eyebrow-Based Stress Detection using Computer Vision

This repository contains the implementation of a **computer vision–based stress indicator** that combines **eyebrow motion analysis** with **facial emotion recognition** to support **adaptive music recommendation**.

This project was developed as a **research capstone** with primary emphasis on **classical computer vision techniques**, using pretrained deep learning models only as supporting components.

---

## Project Overview

Stress is often reflected through **subtle facial movements**, particularly in the **eyebrow region**, which may not be fully captured by emotion recognition alone.

This project proposes a **hybrid stress estimation framework** where:

- **Facial emotion recognition** provides emotional context
- **Eyebrow motion analysis** captures fine-grained stress cues
- Both signals are fused into a **continuous stress index**
- The final stress index is used to **recommend appropriate music**

---

## Objectives

- Extract eyebrow-based stress cues using classical computer vision
- Analyze facial landmarks to quantify eyebrow motion and tension
- Fuse eyebrow-based stress with emotion-derived stress
- Validate the approach using the **RAVDESS dataset**
- Demonstrate real-time stress detection using a **live webcam**
- Apply the stress indicator to **adaptive music recommendation**

---
## Methods

- Facial landmarks: MediaPipe Face Mesh  
- Emotion recognition: Pretrained Mini-Xception  
- Stress estimation: Handcrafted eyebrow motion features  
- Visualization: RAVDESS dataset and live webcam output  

---

## Technologies

Python, OpenCV, MediaPipe, NumPy, Pandas, Matplotlib, TensorFlow/Keras

---

## Disclaimer

This project is for academic demonstration purposes only and is not a medical stress assessment tool.

---

## Author

Lianna Fabay and Samuel Ocenar  
University of Santo Tomas
Research Capstone – Computer Vision

