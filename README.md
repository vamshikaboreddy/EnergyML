\# ⚡ Energy-Aware Machine Learning (EnergyML)



\## 📌 Overview

This project focuses on training deep learning models (EfficientNet-based) while tracking and optimizing energy consumption using CodeCarbon.



\---



\## 🛠 Complete Setup Guide (From Scratch)



\### 1. Install Anaconda

Download and install from:

https://www.anaconda.com/



\---



\### 2. Create Virtual Environment (PowerShell)

conda create -n mlenv python=3.10 -y



\---



\### 3. Activate Environment

conda activate mlenv



\---



\### 4. Install Dependencies

pip install tensorflow==2.15.1 numpy<2.0.0 protobuf>=3.20.3,<5.0 wrapt>=1.11.0,<1.15 tensorflow-datasets==4.8.3 tensorflow-metadata==1.14.0 tensorflow-model-optimization==0.8.0 absl-py>=1.2,<2.0 tf-keras codecarbon psutil matplotlib pandas scipy



\---



\## 🚀 How to Run the Project



\### Train Model

python train.py



\### Analyze Results

python analyze.py



\---



\## 📊 Outputs

\- results/plots/ → Graphs and comparisons

\- results/FINAL\_REPORT.txt → Summary report



\---



\## ⚙️ Git Setup (Reference)



git init  

git remote add origin https://github.com/vamshikaboreddy/EnergyML.git  

git add .  

git commit -m "Initial commit"  

git branch -M main  

git push -u origin main  



\---



\## ⚠️ Important Notes

\- Model files (.keras, .tflite) are not included due to large size

\- Use .gitignore to avoid pushing large files

\- GPU is recommended for faster training



\---



\## 🧠 Key Features

\- Energy-aware model training

\- Carbon emission tracking using CodeCarbon

\- EfficientNet-based architecture

\- Performance vs energy comparison



\---



\## 🛠 Tech Stack

\- Python

\- TensorFlow

\- EfficientNet

\- CodeCarbon

\- NumPy, Pandas, Matplotlib

