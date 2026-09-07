# 🏥 Real-Time Sepsis Early Detection & Clinical Dashboard System

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-Machine_Learning-orange.svg)

## 📌 Project Overview
A production-ready, low-latency machine learning microservice designed to predict Sepsis in ICU patients. This system utilizes a **Multimodal Architecture**, analyzing both numerical physiological vitals and NLP-processed clinical observation notes to provide real-time diagnostic alerts.

Built as our B.Tech CSE Final Year Project at **BBS College of Engineering and Technology, Prayagraj (AKTU)**.

## 🚀 Core Capabilities
* **1. High-Speed Prediction Engine:** Powered by FastAPI and LightGBM, delivering risk analysis in **< 50ms**.
* **2. Basic Explainable AI (XAI):** Transparently highlights specific physiological risk contributors (e.g., Critical Blood Pressure) to build clinical trust.
* **3. NLP-Driven Clinical Integration:** A smart keyword-matching module that parses unstructured doctor notes (e.g., "shivering", "pale", "confused") to dynamically adjust real-time risk scores.
* **4. Smart Action Protocols:** An automated recommendation engine that translates high-risk alerts into immediate bedside protocols (e.g., "Initiate IV Fluids", "Start Oxygen Therapy").

## 🧠 Model Benchmarking
We conducted a comparative study between **Gradient Boosted Trees (LightGBM)** and **Deep Learning (Google TabNet)**. LightGBM was selected as the core engine due to its superior inference speed, low resource footprint, and high accuracy, making it ideal for hospital and edge server deployment.

## ⚙️ Tech Stack
* **Frontend UI:** Streamlit
* **Backend API:** FastAPI, Uvicorn, Pydantic
* **Machine Learning Pipeline:** LightGBM, Scikit-learn, Pandas

## 💻 How to Run Locally

**Step 1: Clone the repository & Install dependencies**
```bash
git clone [https://github.com/yourusername/sepsis-advanced-system.git](https://github.com/yourusername/sepsis-advanced-system.git)
cd sepsis-advanced-system
pip install -r requirements.txt