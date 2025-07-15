# 🤖 Finory AI Categorizer  
> **Smart Transaction Categorization – Baseline ML Model**  

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)  [![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange)](https://jupyter.org/)  [![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)](https://xgboost.ai/)  [![Status](https://img.shields.io/badge/Status-Baseline%20Ready-yellow)]()  

Finory AI Categorizer is a **baseline machine learning model** that classifies financial transactions (e.g., vendor → category) using **XGBoost**, synthetic training data, and preprocessing pipelines.  

This model will be integrated into the **Finory App**, providing **smart transaction categorization** that improves over time with real user data.

---

## 🚀 Features  

✅ **Synthetic transaction dataset (5,000 rows)** for initial training  
✅ **Feature engineering** – log-transformed amounts, date features, grouped vendors  
✅ **Baseline Models** – Logistic Regression, Random Forest, and XGBoost  
✅ **Hyperparameter Tuning** for XGBoost → improved accuracy **~61%**  
✅ **Saved Tuned Model + Label Encoders** for future predictions  
✅ `load_and_predict.ipynb` to easily load the model and test predictions  
✅ Designed for **future fine-tuning** with real-world Finory App data  

---

## 📂 Project Structure  

```
finory-ai-categorizer/
├── data/
│   ├── synthetic_finory_transactions.csv        # Raw synthetic dataset
│   └── synthetic_finory_preprocessed.csv        # Preprocessed dataset
│
├── models/
│   ├── finory_baseline_xgb.joblib               # Tuned baseline XGBoost model
│   └── finory_label_encoders.joblib             # Saved label encoders
│
├── notebooks/
│   ├── 01_data_generation.ipynb                 # Synthetic data creation
│   ├── 02_data_exploration.ipynb                # Exploratory Data Analysis (EDA)
│   ├── 03_data_preprocessing.ipynb              # Feature engineering + encoding
│   ├── 04_model_training.ipynb                  # Baseline ML models
│   ├── 05_hyperparameter_tuning.ipynb           # XGBoost tuning → ~61% accuracy
│   ├── load_and_predict.ipynb                   # Load tuned model & predict
│   └── vendor_mapping.py                        # Vendor → grouped vendor logic
│
├── requirements.txt                             # Dependencies
└── README.md                                    # This file
```

---

## 📊 Baseline Results  

| Model                 | Accuracy | Notes |
|-----------------------|----------|-------|
| Logistic Regression   | ~43%     | Weak baseline |
| Random Forest         | ~52%     | Slight improvement |
| XGBoost (Default)     | ~53%     | Best baseline |
| **XGBoost (Tuned)**   | **~61%** | After hyperparameter tuning |

✅ Tuned XGBoost chosen as the **initial model** for Finory App  
✅ Will be retrained later with **real-world transaction data**  

---

## 🔧 Quickstart  

1️⃣ **Clone this repo**  

```bash
git clone https://github.com/<your-username>/finory-ai-categorizer.git
cd finory-ai-categorizer
```

2️⃣ **Install dependencies**  

```bash
pip install -r requirements.txt
```

3️⃣ **Run notebooks**  

Start Jupyter and open any notebook:  

```bash
jupyter lab
```

For example:  
- `03_data_preprocessing.ipynb` → preprocess raw data  
- `04_model_training.ipynb` → train baseline models  
- `05_hyperparameter_tuning.ipynb` → tune XGBoost  

4️⃣ **Load the tuned model for predictions**  

```python
import joblib

# Load model & encoders
model = joblib.load("models/finory_baseline_xgb.joblib")
encoders = joblib.load("models/finory_label_encoders.joblib")

# Predict on new transaction samples
predictions = model.predict(X_new)
```

Or simply open `load_and_predict.ipynb` for a demo.

---

## 🔮 Future Improvements  

✅ Collect **real transaction data** from Finory App  
✅ Retrain & fine-tune model with live data  
✅ Add **deep learning (LSTM/NLP)** for merchant name understanding  
✅ Deploy as a **REST API** or lightweight **on-device ML model**  
✅ Continuous learning loop (active learning with user corrections)

---

## 🤝 Contributing  

Want to improve the baseline? Fork the repo, make changes, and submit a PR!  

---

## 📜 License  

This project is proprietary and intended for integration into the Finory app.  
**All rights reserved.**  

You may not copy, modify, merge, publish, distribute, sublicense, or sell copies of this software without explicit written permission from the author.  

**Author:** Philip Haapala  
**Part of:** AI Portfolio Projects for Smart Finance Applications 
