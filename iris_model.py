import pandas as pd
import numpy as py 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('Iris.csv')  

print(df.head())
print(df.describe())
print(df.info())

print(df.isnull().sum())


X = df.drop('Species', axis=1)  
y = df['Species']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)


y_pred_log_reg = log_reg.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log_reg))


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)


y_pred_knn = knn.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))


print("Confusion Matrix for Logistic Regression:")
print(confusion_matrix(y_test, y_pred_log_reg))
print("\n")

print("Confusion Matrix for KNN:")
print(confusion_matrix(y_test, y_pred_knn))
print("\n")

print("Confusion Matrix for Decision Tree:")
print(confusion_matrix(y_test, y_pred_dt))
print("\n")

print("Classification Report for Logistic Regression:")
print(classification_report(y_test, y_pred_log_reg))

print("Classification Report for KNN:")
print(classification_report(y_test, y_pred_knn))

print("Classification Report for Decision Tree:")
print(classification_report(y_test, y_pred_dt))


cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)
cm_knn = confusion_matrix(y_test, y_pred_knn)
cm_dt = confusion_matrix(y_test, y_pred_dt)


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sns.heatmap(cm_log_reg, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
axes[0].set_title('Logistic Regression')

sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', ax=axes[1], cbar=False)
axes[1].set_title('KNN')

sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', ax=axes[2], cbar=False)
axes[2].set_title('Decision Tree')

plt.tight_layout()
plt.show()

