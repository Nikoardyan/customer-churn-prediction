# 📊 Customer Churn Prediction (Telco Industry)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-0.24.2-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📖 Project Overview
Customer churn is a critical issue in the telecommunications industry, where retaining customers is often more cost-effective than acquiring new ones. This project focuses on building a machine learning model to predict customer churn, enabling businesses to take proactive measures to retain their customers and reduce revenue loss.

---

## ❓ Business Problem
Telecommunication companies face significant challenges in retaining customers due to high competition and customer switching behavior. The goal of this project is to predict which customers are likely to churn (stop using the service) and identify key factors contributing to churn. This will help businesses implement targeted retention strategies and improve customer satisfaction.

---

## 🎯 Objectives
1. Predict whether a customer will churn based on their demographic and service usage data.
2. Identify key factors that influence customer churn.
3. Provide actionable business recommendations to reduce churn rates.

---

## 📂 Dataset Information
The dataset used for this project is the **Telco Customer Churn Dataset**, which contains information about customer demographics, account details, and service usage. Below are the key details:

- **Number of rows**: 7,043
- **Number of features**: 21
- **Target variable**: `Churn` (Yes/No)
- **Key features**:
  - `tenure`: Number of months the customer has stayed with the company.
  - `MonthlyCharges`: The amount charged to the customer monthly.
  - `Contract`: Type of contract (Month-to-month, One year, Two year).

---

## 🛠️ Data Preprocessing Steps
To prepare the data for modeling, the following steps were performed:
- **Handling missing values**: Missing values in the `TotalCharges` column were replaced with the median value.
- **Encoding categorical variables**: One-hot encoding was applied to convert categorical variables into numerical format.
- **Feature scaling**: StandardScaler was used to normalize numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) to ensure all features are on the same scale.
- **Handling class imbalance**: The dataset was imbalanced, with fewer customers labeled as "Churn". To address this, **SMOTE (Synthetic Minority Oversampling Technique)** was used to balance the dataset.

---

## 🔍 Exploratory Data Analysis (EDA)
Key insights from the data:
- Customers with shorter **tenure** are more likely to churn.
- Customers with **higher monthly charges** are more likely to churn.
- Customers with **month-to-month contracts** are more likely to churn compared to those with longer-term contracts.

Visualizations such as histograms, boxplots, and count plots were used to uncover these patterns.

---

## 🤖 Modeling
Two machine learning models were implemented and compared:
1. **Logistic Regression**: A simple and interpretable model for binary classification.
2. **Random Forest**: An ensemble learning method that combines multiple decision trees to improve prediction accuracy.

---

## 📊 Model Evaluation
The models were evaluated using the following metrics:
- **Accuracy**: Overall correctness of the model.
- **Precision**: Proportion of correctly predicted positive cases out of all predicted positives.
- **Recall**: Proportion of actual positive cases correctly identified.
- **F1-Score**: Harmonic mean of precision and recall.
- **AUC (Area Under Curve)**: Measures the ability of the model to distinguish between classes.

| Metric              | Logistic Regression | Random Forest |
|---------------------|---------------------|---------------|
| **Accuracy**        | 80%                | 83%           |
| **Precision**       | 78%                | 81%           |
| **Recall**          | 72%                | 76%           |
| **F1-Score**        | 75%                | 78%           |
| **AUC**             | 84%                | 88%           |

---

## 🌟 Why Random Forest Was Chosen
While Logistic Regression is simple and interpretable, **Random Forest** outperformed it in all evaluation metrics, particularly in terms of **AUC** and **Recall**. Random Forest's ability to handle non-linear relationships and its robustness to overfitting made it the ideal choice for this project.

---

## 💡 Business Recommendations
Based on the analysis and model results, the following actions are recommended to reduce customer churn:
1. **Focus on high-risk customers**: Target customers with low tenure and high monthly charges for retention campaigns.
2. **Promote long-term contracts**: Offer incentives for customers to switch from month-to-month contracts to annual or bi-annual contracts.
3. **Improve customer service**: Address customer complaints and provide better support to increase satisfaction and reduce churn.

---

## 💰 Potential Financial Impact
By reducing customer churn, the company can:
- Increase customer lifetime value (CLV).
- Reduce costs associated with acquiring new customers.
- Improve overall revenue and profitability.

For example, reducing churn by just **5%** could result in a significant increase in annual revenue, depending on the average customer lifetime value.

---

## 🛠️ Tech Stack
- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Seaborn, Matplotlib, Scikit-Learn, Imbalanced-learn
- **Tools**: Google Colab, Jupyter Notebook

---

## 📂 Project Structure