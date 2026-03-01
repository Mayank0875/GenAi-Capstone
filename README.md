# House Price Prediction System

## 📌 Project Overview

The **House Price Prediction System** is an end-to-end Machine Learning project designed to predict house prices using structured real estate data.

This project demonstrates the complete ML lifecycle:

- Data Cleaning  
- Feature Engineering  
- Data Preprocessing  
- Model Training (XGBoost)  
- Hyperparameter Tuning  
- Model Evaluation  
- Deployment using Streamlit  

The system is modular, scalable, and production-ready.

---

## Objectives

- Predict house prices using regression techniques  
- Improve performance through feature engineering  
- Use **XGBoost Regressor** for strong predictive accuracy  
- Apply **GridSearchCV** for hyperparameter tuning  
- Deploy an interactive web app using **Streamlit**  

---

## System Workflow

```
Raw Data
   ↓
Data Cleaning (Outlier Removal + Feature Engineering)
   ↓
Preprocessing (Scaling + Encoding)
   ↓
XGBoost Model Training
   ↓
Model Evaluation (R², MSE, RMSE)
   ↓
Model Saved as house_price_model.pkl
   ↓
Streamlit App for Real-Time Prediction
```

---

## Technology Stack

- Python  
- Pandas  
- NumPy  
- Scikit-Learn  
- XGBoost  
- Joblib  
- Streamlit  

---

## Model Performance

**Final Model:** XGBoost Regressor  

- **R² Score:** 0.776  
- **MSE:** 1.04 × 10¹⁰  
- **RMSE:** ≈ 102,000  

The model explains approximately **77.6% of the variance** in house prices.

---

## Project Structure

```
House-Price-Prediction/
│
├── data/
│   └── dataset.csv
│
├── model.py
├── app.py
├── house_price_model.pkl
├── requirements.txt
└── README.md
```

---

## Installation & Usage

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python model.py
```

This will generate:

- `house_price_model.pkl`

### Run the Streamlit App

```bash
streamlit run app.py
```

---

## Features

- End-to-end ML pipeline  
- Clean and modular code structure  
- Hyperparameter tuning with GridSearchCV  
- Production-ready model serialization  
- Interactive UI for real-time predictions  

---

## Conclusion

This project showcases a complete machine learning lifecycle from data preprocessing to deployment. It reflects strong practical understanding of regression modeling and real-world structured data prediction.

---

## Author

**Mayank Gupta**  