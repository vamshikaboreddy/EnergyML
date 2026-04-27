# ⚡ Energy-Aware Machine Learning (EnergyML)

## 📌 Overview
This project trains EfficientNet-based deep learning models while tracking and optimizing energy consumption using CodeCarbon.

---

## 🛠 Setup Instructions

### 1. Install Anaconda
Download and install from:  
https://www.anaconda.com/

---

### 2. Create Virtual Environment (PowerShell)
conda create -n mlenv python=3.10 -y

---

### 3. Activate Environment
conda activate mlenv

---

### 4. Install Dependencies
pip install tensorflow==2.15.1 numpy<2.0.0 protobuf>=3.20.3,<5.0 wrapt>=1.11.0,<1.15 tensorflow-datasets==4.8.3 tensorflow-metadata==1.14.0 tensorflow-model-optimization==0.8.0 absl-py>=1.2,<2.0 tf-keras codecarbon psutil matplotlib pandas scipy

---

## 🚀 How to Run

### Train Model
python train.py

### Analyze Results
python analyze.py

---

## 📊 Outputs
- results/plots/ → Graphs and comparisons  
- results/FINAL_REPORT.txt → Summary report  

---

## 🧠 Key Features
- Energy-aware model training  
- Carbon emission tracking using CodeCarbon  
- EfficientNet-based architecture  
- Performance vs energy comparison  

---

## ⚠️ Notes
- Model files (.keras, .tflite) are not included due to large size  
- GPU recommended for faster training (optional)  

---

## 🛠 Tech Stack
- Python  
- TensorFlow  
- EfficientNet  
- CodeCarbon  
- NumPy, Pandas, Matplotlib  
