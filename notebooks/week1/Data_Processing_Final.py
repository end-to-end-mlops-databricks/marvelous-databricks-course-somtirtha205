# Databricks notebook source
# MAGIC %pip install ../hotel_reservation-0.0.1-py3-none-any.whl --force-reinstall

# COMMAND ----------

# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import SparkSession

from hotel_reservation.config import ProjectConfig
from hotel_reservation.data_processor import DataProcessor
from hotel_reservation.reservation_model import ReservationModel
from hotel_reservation.utils import plot_feature_importance, visualize_results

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# Load configuration
config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

# COMMAND ----------

# Initialize DataProcessor
data_processor = DataProcessor(config)

# COMMAND ----------

# Extract configuration details
num_features = config.num_features
cat_features = config.cat_features
target = config.target
parameters = config.parameters
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------

# Preprocess the data
data_processor.preprocess_data("/Volumes/mlops_dev/hotel_reservation/data/Data.csv", spark)

# COMMAND ----------

# Split the data
train_set, test_set = data_processor.split_data()

# COMMAND ----------

# Save to Catalog
data_processor.save_to_catalog(train_set=train_set, test_set=test_set, spark=spark)

# Load training and testing sets from Databricks tables
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

X_train = train_set[num_features + cat_features]
y_train = train_set[target]

X_test = test_set[num_features + cat_features]
y_test = test_set[target]

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
