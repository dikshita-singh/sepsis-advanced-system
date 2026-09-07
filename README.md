# Real-Time Sepsis Early Detection & Clinical Dashboard System

An end-to-end, real-time Sepsis Early Detection System powered by a **LightGBM Classifier**. The system processes clinical vitals and NLP-processed observation notes, classifies patient risk for Sepsis with high confidence, serves predictions via a FastAPI backend, and streams analytics to an interactive Streamlit web dashboard.

---

## 🔗 Architecture Overview

```text
+-------------------+       +-------------------+       +-------------------+
|   Clinical Data   |       |  FastAPI Backend  |       |   Streamlit UI    |
| (Vitals & Notes)  | ----> |  (ML Prediction)  | ----> | (Risk Dashboard)  |
+-------------------+       +-------------------+       +-------------------+
```

---

## 🧠 Machine Learning Model & Rationale

### Model Used
**LightGBM Classifier** (`lightgbm.LGBMClassifier`) trained on ICU patient clinical data to predict the early onset of sepsis.

### Why LightGBM?
The choice of LightGBM over other machine learning algorithms (such as deep learning or standard decision trees) is grounded in key domain requirements for clinical monitoring:

1. **Low Latency & High Throughput Inference:** Real-time patient monitoring requires instantaneous prediction without introducing pipeline latency. LightGBM performs rapid inference, requiring sub-millisecond evaluation times crucial for ICU environments.
2. **Handling Complex Tabular Data:** Medical data contains mixed feature types (numerical vitals and categorical data). LightGBM handles tabular data exceptionally well and manages missing clinical values natively.
3. **High Accuracy with Gradient Boosting:** It builds trees sequentially to correct previous errors, yielding high confidence scores and minimizing false negatives in critical healthcare scenarios.

---

## ✨ Key Features

* **Live Data Processing:** Asynchronous capture and processing of patient vitals and clinical notes.
* **Automated ML Classification:** Evaluates every data point against the pre-trained LightGBM model.
* **Persistent Event Logging:** Stores patient risk scores, prediction labels, and confidence metrics securely.
* **REST API Backend:** Lightweight FastAPI service serving prediction endpoints, statistics, and risk triggers.
* **Clinical Web Dashboard:** Interactive UI built with Streamlit featuring live charts, real-time polling, patient risk metrics, and dynamic security/medical alert indicators.

---

## 📂 Project Structure

```text
├── api/
│   ├── app.py                 # FastAPI application and endpoints
│   ├── schemas.py             # Pydantic models for data validation
│   └── utils.py               # API helper functions
├── data/
│   └── raw/                   # Raw clinical datasets (ignored in git)
├── models/
│   └── sepsis_model.pkl       # Pre-trained LightGBM model
├── src/                       # Source code for data processing
├── dashboard.py               # Streamlit frontend application
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 🚀 Quickstart Guide

**1. Clone the repository**
```bash
git clone [https://github.com/dikshita-singh/sepsis-advanced-system.git](https://github.com/dikshita-singh/sepsis-advanced-system.git)
cd sepsis-advanced-system
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the FastAPI Backend**
```bash
uvicorn api.app:app --reload
```

**4. Run the Streamlit Dashboard (Open a new terminal)**
```bash
streamlit run dashboard.py
```
