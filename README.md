# 🚨 VigilantAI: Autonomous CCTV Threat Recognition Engine

An AI-powered automated video surveillance system designed to detect **Normal**, **Violence**, and **Weaponized** activity in CCTV video feeds using deep learning and real-time temporal smoothing.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Dataset Structure](#-dataset-structure)
- [Model Training & Performance](#-model-training--performance)
- [Installation & Setup](#-installation--setup)
- [Running the Streamlit Web Application](#-running-the-streamlit-web-application)
- [Project Directory Structure](#-project-directory-structure)
- [License](#-license)

---

## 🔍 Overview

Manual monitoring of municipal and private CCTV networks can lead to delayed incident response times during violent altercations or armed threats. This project addresses automated surveillance by extracting frame features from raw CCTV footage and running classification inference via **YOLOv8 Medium (`yolov8m-cls`)**.

To eliminate false-alarm flickering caused by camera shake, fast motion, or lighting shifts, the inference engine employs a **10-frame sliding window averaging algorithm** (majority voting across temporal sequences) to deliver stable real-time alert classifications.

---

## ✨ Key Features

- **Multi-Class Scene Classification:** Real-time categorization into `Normal`, `Violence`, and `Weaponized` categories.
- **High-Resolution Feature Processing:** Model trained at $416 \times 416$ spatial resolution to preserve fine object details like knives, handguns, and hand movements.
- **Sliding-Window Temporal Smoothing:** Averages predicted class probabilities across consecutive frames to prevent rapid flickering between categories.
- **Interactive Streamlit Web Dashboard:** Upload CCTV video clips, observe live color-coded overlay alerts, adjust confidence thresholds on the fly, and view post-analysis probability charts via Plotly.
- **Google Drive Integration:** Automatic checkpointing and evaluation metrics saving directly to mounted cloud storage.

---

## 🛠 System Architecture
┌─────────────────┐      ┌──────────────────────────┐      ┌──────────────────────────┐
│ Raw CCTV Video  │ ───► │ OpenCV Frame Extraction  │ ───► │ YOLOv8m Classification   │
│ Stream (.mp4)   │      │ (Sampling Stride = 25)   │      │ (Input Size: 416x416)    │
└─────────────────┘      └──────────────────────────┘      └────────────┬─────────────┘
│
┌─────────────────┐      ┌──────────────────────────┐                   │
│ Streamlit UI /  │ ◄─── │  10-Frame Sliding Window │ ◄────────────────┘
│ Analytics Chart │      │   Probability Smoothing  │
└─────────────────┘

## 📂 Dataset Structure

The model is trained on the **Smart City Violence Detection (SCVD)** dataset. Videos are sub-sampled at a stride of **25 frames** (~1 frame per second at 25 FPS) to reduce background redundancy and avoid data leakage across train/test splits.
- **Dataset Source:** [Smart City Violence Detection (SCVD) Dataset on Kaggle][https://www.kaggle.com/datasets/toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd]
- **Data Preprocessing:** Videos were sub-sampled at a stride of **25 frames** (~1 frame per second at 25 FPS) to eliminate frame-to-frame background redundancy and prevent data leakage across train/test splits.
 
 ```text
    SCVD_Project/
    └── SCVD-extracted/
        ├── train/
        │   ├── Normal/        (*.jpg)
        │   ├── Violence/      (*.jpg)
        │   └── Weaponized/    (*.jpg)
        └── test/
            ├── Normal/        (*.jpg)
            ├── Violence/      (*.jpg)
            └── Weaponized/    (*.jpg)```
```

📊 Model Training & Performance
The pipeline utilizes Ultralytics YOLOv8 Medium Classification (yolov8m-cls.pt) trained on an NVIDIA T4 GPU in Google Colab.Hyperparameters:Base Weights: yolov8m-cls.pt Image Resolution (imgsz): $416 \times 416$ Epochs: 35 Batch Size: 16 Optimizer: Auto (SGD / AdamW with LR Warmup)Evaluated Metrics (Saved in SCVD-test):Overall Accuracy: Computed across extracted test frames.Macro Precision, Recall & F1-Score: Accounted equally across Normal, Violence, and Weaponized classes.Confusion Matrix: Automatically rendered and saved to Google Drive.⚡ Installation & SetupPrerequisitesPython 3.10 or higherGit1. Clone the RepositoryBashgit clone [https://github.com/your-username/smart-city-violence-detection.git](https://github.com/your-username/smart-city-violence-detection.git)
  
```bash
    cd smart-city-violence-detection
```

2. Create and Activate Virtual Environment
  Bash
  # Windows
  
  ```bash
  python -m venv venv
  venv\Scripts\activate
```
  # Linux / macOS
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
3. Install Dependencies
```Bash
    pip install -r requirements.txt
```

🚀 Running the Streamlit Web Application
Place your trained model file (best.pt) in the project root folder.

Launch the Streamlit server:
```Bash
    streamlit run app.py
```
Open your browser and navigate to http://localhost:8501.

Upload any .mp4, .avi, or .mov CCTV clip to monitor live alerts and sequence analytics.

📋 Requirements (requirements.txt)
 ```Plaintext
      streamlit>=1.28.0
      ultralytics>=8.0.0
      opencv-python-headless>=4.8.0
      numpy>=1.24.0
      pandas>=2.0.0
      plotly>=5.15.0
      scikit-learn>=1.2.0
```
        
