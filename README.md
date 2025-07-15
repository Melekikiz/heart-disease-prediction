# 🫀 Heart Disease Prediction with Logistic Regression

This project demonstrates a complete machine learning workflow for predicting the likelihood of heart disease based on clinical data.

## 📌 Project Overview

Using the `heart.csv` dataset, we built a Logistic Regression model to classify whether a patient is likely to have heart disease (`1`) or not (`0`). The dataset includes features like age, gender, chest pain type, cholesterol levels, and more.

We followed all key machine learning steps:
- Data loading and preprocessing
- One-hot encoding of categorical variables
- Feature scaling with `StandardScaler`
- Logistic Regression model training
- Evaluation using accuracy, confusion matrix, and classification report
- Data visualization with `matplotlib` and `seaborn`

---

## 🔍 Dataset

- Source: [Kaggle - Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- Rows: 918 samples
- Features: 11 input features + 1 target variable (`HeartDisease`)

---

## 📊 Sample Visualizations

- Age distribution by heart disease
- Gender distribution by heart disease
- Chest pain type analysis
  
Visualizations were created using `seaborn` and `matplotlib`.

---

## 📈 Model Performance

- **Accuracy:** ~85%
- **Precision & Recall:** Balanced performance across both classes
- **Confusion Matrix:** Shows solid separation of positive/negative classes

---

## ⚙️ Tools & Libraries

- Python 3
- Pandas, NumPy
- scikit-learn
- seaborn, matplotlib





