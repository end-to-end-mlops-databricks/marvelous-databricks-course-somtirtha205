# Databricks notebook source
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# COMMAND ----------

data = pd.read_csv("/Volumes/mlops_dev/hotel_reservation/data/Data.csv")
display(data)

# COMMAND ----------

data.info()

# COMMAND ----------

data.describe()

# COMMAND ----------

print(data.isnull().sum().sum())

# COMMAND ----------

print(data.isna().sum().sum())

# COMMAND ----------

df = data.drop('Booking_ID', axis=1)
X = df.drop('booking_status', axis=1)
y = df['booking_status']
y = pd.get_dummies(y, columns="booking_status", drop_first=True)
y.set_axis(["booking_status"], axis="columns", inplace=True)

# COMMAND ----------

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# COMMAND ----------

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# COMMAND ----------

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# COMMAND ----------

X.info()

# COMMAND ----------

y.info()

# COMMAND ----------

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# COMMAND ----------


