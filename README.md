# Finory AI Module: Smart Transaction Categorization

## Overview
This project is part of the **Finory AI Module**, designed to automatically categorize financial transactions based on vendor, amount, payment method, and other metadata.  
It uses a synthetic dataset of 5,000 transactions to simulate real-world financial data and trains machine learning models for **multi-class classification**.

---

## Project Structure

```
├── data/
│   └── synthetic_finory_transactions.csv   # Synthetic dataset
├── notebooks/
│   ├── 01_generate_data.ipynb              # Generates synthetic transaction data
│   ├── 02_data_exploration.ipynb           # Exploratory data analysis
│   ├── 03_data_preprocessing.ipynb         # Data cleaning, encoding, and feature engineering
│   ├── 04_model_training.ipynb             # Baseline model training and evaluation
│   ├── 05_hyperparameter_tuning.ipynb      # Hyperparameter tuning for Random Forest & XGBoost
│   └── load_and_predict.ipynb              # Example of loading a saved model for predictions
├── models/
│   └── best_xgb_model.pkl                  # Best tuned XGBoost model saved for later use
├── requirements.txt                        # Dependencies
└── README.md                               # Project documentation
```

---

## Objective
The goal is to **predict the category of a financial transaction** using features like vendor, amount, payment method, and date metadata.

- **Problem Type:** Multi-class classification  
- **Target:** `category`  
- **Example Categories:** Groceries, Electronics, DiningOut, Transportation, etc.

---

## Dataset

**Synthetic dataset with 5,000 samples** containing:
- `transaction_id`
- `vendor`
- `amount`
- `category` (target)
- `date`
- `payment_method`
- `note`

Additional engineered features:
- `day_of_week`
- `month`
- `amount_log` (normalized transaction amount)

---

## Phases

1. **Data Generation** → Created synthetic financial transactions with realistic distributions.
2. **Data Exploration** → Visualized category and vendor distributions.
3. **Data Preprocessing** → Encoded vendors, categories, and payment methods. Extracted date features and normalized transaction amounts.
4. **Model Training** → Trained multiple models (Logistic Regression, Random Forest, XGBoost).
5. **Hyperparameter Tuning** → Improved Random Forest and XGBoost accuracy via GridSearchCV.
6. **Model Saving/Loading** → Saved the best-tuned model for future predictions.

---

## 📊 Model Performance Summary  

| Model                      | Test Accuracy | Notes |
|----------------------------|--------------:|-------|
| Logistic Regression        | ~48%          | Baseline linear model |
| Random Forest (default)    | ~55%          | Non-tuned |
| XGBoost (default)          | ~53%          | Non-tuned |
| **Random Forest (tuned)**  | **60%**       | Best tuned parameters via GridSearchCV |
| **XGBoost (tuned)**        | **59%**       | Best tuned parameters via GridSearchCV |

✅ **Best Models:** Tuned Random Forest & Tuned XGBoost both achieved ~60% accuracy.  

---

## Next Steps

- **Vendor Grouping & Category Merging** → Reduce class imbalance and noise.
- **Incorporate real-world transaction data** to improve generalization.
- **Deploy model as an API** for integration into the Finory app.

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

1. **Clone the repo**
```bash
git clone <your-repo-url>
cd finory-ai-module
```

2. **Generate synthetic data**
```bash
jupyter notebook notebooks/01_generate_data.ipynb
```

3. **Explore & preprocess**
```bash
jupyter notebook notebooks/02_data_exploration.ipynb
jupyter notebook notebooks/03_data_preprocessing.ipynb
```

4. **Train models**
```bash
jupyter notebook notebooks/04_model_training.ipynb
```

5. **Run hyperparameter tuning**
```bash
jupyter notebook notebooks/05_hyperparameter_tuning.ipynb
```

6. **Load saved model for predictions**
```bash
jupyter notebook notebooks/load_and_predict.ipynb
```

---

## Saving & Loading the Best Model

After hyperparameter tuning, we saved the **best XGBoost model** like this:

```python
import joblib
joblib.dump(best_xgb, "../models/best_xgb_model.pkl")
```

Later, you can load it for predictions:

```python
best_model = joblib.load("../models/best_xgb_model.pkl")
predictions = best_model.predict(X_new)
```

---

## Key Takeaways

- Synthetic data provides a realistic sandbox for **multi-class categorization**.
- Feature engineering (vendor encoding, date extraction, amount normalization) is critical.
- Hyperparameter tuning can boost accuracy by **~7%** compared to default models.
- This module will later integrate into **Finory**, enabling AI-driven smart categorization.

---

**Author:** Philip Haapala  
**Part of:** AI Portfolio Projects for Smart Finance Applications 
