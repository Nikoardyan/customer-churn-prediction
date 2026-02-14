

from google.colab import drive
drive.mount('/content/drive')



import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

## Load dataset

df = pd.read_csv('/content/sample_data/1702184567307-WA_FnUseC_TelcoCustomerChurn.csv')
df.head(5)

df.info()

## Split Dataset

df = df.drop('customerID', axis=1)


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')


df.isnull().sum()


df = df.dropna()


df = pd.get_dummies(df, drop_first=True)


X = df.drop('Churn_Yes', axis=1)
y = df['Churn_Yes']


## Train

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


## EDA



sns.countplot(x='Churn', data=df)
plt.title("Distribusi Churn")
plt.show()


num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

df[num_cols].hist(figsize=(10,6))
plt.show()



plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
sns.boxplot(x='Churn', y='tenure', data=df)

plt.subplot(1,3,2)
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)

plt.subplot(1,3,3)
sns.boxplot(x='Churn', y='TotalCharges', data=df)

plt.tight_layout()
plt.show()




### Modeling (Gunakan lebih min 2 model dan bandingkan hasil evaluasinya)

df_model = df.copy()


df_model['TotalCharges'] = pd.to_numeric(df_model['TotalCharges'], errors='coerce')



df_model['TotalCharges'].fillna(df_model['TotalCharges'].median(), inplace=True)



df_model.drop('customerID', axis=1, inplace=True)


df_model['Churn'] = df_model['Churn'].map({'No':0, 'Yes':1})


df_model = pd.get_dummies(df_model, drop_first=True)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train_scaled, y_train)

y_pred_log = log_model.predict(X_test_scaled)
y_prob_log = log_model.predict_proba(X_test_scaled)[:,1]

print("Logistic Regression Report")
print(classification_report(y_test, y_pred_log))
print("AUC:", roc_auc_score(y_test, y_prob_log))


rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:,1]

print("Random Forest Report")
print(classification_report(y_test, y_pred_rf))
print("AUC:", roc_auc_score(y_test, y_prob_rf))


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(name, y_test, y_pred, y_prob):
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_prob))

evaluate_model("Logistic Regression", y_test, y_pred_log, y_prob_log)
evaluate_model("Random Forest", y_test, y_pred_rf, y_prob_rf)




## Test

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)



log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_res, y_train_res)


y_pred_log = log_model.predict(X_test_scaled)
y_prob_log = log_model.predict_proba(X_test_scaled)[:,1]


### Data Preprocessing

### Evaluation

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_log))

print("\nAUC Score:")
print(roc_auc_score(y_test, y_prob_log))

