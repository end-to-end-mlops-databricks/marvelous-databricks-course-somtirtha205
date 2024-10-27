# Databricks notebook source
# MAGIC %pip install /Volumes/mlops_dev/hotel_reservation/data/hotel_reservation-0.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import yaml

from hotel_reservation.data_processor import DataProcessor
from hotel_reservation.reservation_model import ReservationModel
from hotel_reservation.utils import plot_feature_importance, visualize_results

# COMMAND ----------

# Load configuration
with open("../../project_config.yml", "r") as file:
    config = yaml.safe_load(file)

print("Configuration loaded:")
print(yaml.dump(config, default_flow_style=False))

# COMMAND ----------

# Initialize DataProcessor
data_processor = DataProcessor("/Volumes/mlops_dev/hotel_reservation/data/Data.csv", config)

# COMMAND ----------

# Preprocess the data
data_processor.preprocess_data()

# COMMAND ----------

# Split the data
X_train, X_test, y_train, y_test = data_processor.split_data()

# COMMAND ----------

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# COMMAND ----------

# Initialize and train the model
model = ReservationModel(data_processor.preprocessor, config)
model.train(X_train, y_train)

# COMMAND ----------

# Evaluate the model
f1, cm = model.evaluate(X_test, y_test)
print(f"F1 Score: {f1}")
print(f"Confusion Matrix: {cm}")

# COMMAND ----------

## Visualizing Results
y_pred = model.predict(X_test)
visualize_results(cm)

# COMMAND ----------

## Feature Importance
feature_importance, feature_names = model.get_feature_importance()
plot_feature_importance(feature_importance, feature_names)
