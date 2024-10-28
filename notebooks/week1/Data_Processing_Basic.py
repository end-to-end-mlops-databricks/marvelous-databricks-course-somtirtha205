# Databricks notebook source
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# COMMAND ----------

data = pd.read_csv("/Volumes/mlops_dev/hotel_reservation/data/Data.csv")
print(data)

# COMMAND ----------

data.info()

# COMMAND ----------

data.describe()

# COMMAND ----------

print(data.isnull().sum().sum())

# COMMAND ----------

print(data.isna().sum().sum())

# COMMAND ----------

df = data.drop("Booking_ID", axis=1)
X = df.drop("booking_status", axis=1)
y = df["booking_status"]
y = pd.get_dummies(y, columns="booking_status", drop_first=True)
y.set_axis(["booking_status"], axis="columns", inplace=True)

# COMMAND ----------

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# COMMAND ----------

numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# COMMAND ----------

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# COMMAND ----------

X.info()

# COMMAND ----------

y.info()

# COMMAND ----------

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# COMMAND ----------

model = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier(n_estimators=1000, random_state=42))]
)

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# COMMAND ----------

model.fit(X_train, y_train)

# COMMAND ----------

y_pred = model.predict(X_test)

# COMMAND ----------

f1 = f1_score(y_test, y_pred, average="weighted")
cm = confusion_matrix(y_test, y_pred)

# COMMAND ----------

print(f"F1 Score: {f1}")
print(f"Confusion Matrix: {cm}")

# COMMAND ----------

plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Reservation")
plt.ylabel("Actual Reservation")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# COMMAND ----------

feature_importance = model.named_steps["classifier"].feature_importances_
feature_names = model.named_steps["preprocessor"].get_feature_names_out()

# COMMAND ----------

plt.figure(figsize=(10, 6))
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx[-10:].shape[0]) + 0.5
plt.barh(pos, feature_importance[sorted_idx[-10:]])
plt.yticks(pos, feature_names[sorted_idx[-10:]])
plt.title("Top 10 Feature Importance")
plt.tight_layout()
plt.show()
