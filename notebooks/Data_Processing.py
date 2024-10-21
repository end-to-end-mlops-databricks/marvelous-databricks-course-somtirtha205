# Databricks notebook source
import pandas as pd

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


