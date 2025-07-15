import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv("heart.csv")


df_encoded = pd.get_dummies(df, drop_first=True)


X = df_encoded.drop("HeartDisease", axis=1)
y = df_encoded["HeartDisease"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)


print(" Accuracy Score:", accuracy_score(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))



plt.figure(figsize=(10, 6))
sns.histplot(data=df, x="Age", hue="HeartDisease", bins=30, kde=True, palette="Set2")
plt.title("Age Distribution by Heart Disease")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()


plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Sex", hue="HeartDisease", palette="Set1")
plt.title("Heart Disease by Gender")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.show()


plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="ChestPainType", hue="HeartDisease", palette="coolwarm")
plt.title("Heart Disease by Chest Pain Type")
plt.xlabel("Chest Pain Type")
plt.ylabel("Count")
plt.show()
