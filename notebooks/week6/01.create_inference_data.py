# Databricks notebook source
# MAGIC %md
# MAGIC ## Generate synthetic datasets for inference

# COMMAND ----------

# MAGIC %pip install /Volumes/mlops_dev/hotel_reservation/data/hotel_reservation-0.0.1-py3-none-any.whl

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Loading tables

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import col
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
import numpy as np
from pyspark.sql import SparkSession

from hotel_reservation.config import ProjectConfig

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# Load configuration
config = ProjectConfig.from_yaml(config_path="../../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name
pipeline_id = config.pipeline_id
num_features = config.num_features
cat_features = config.cat_features
parameters = config.parameters
target = config.target

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get the most important features using random forest model

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn import set_config

# Define features and target (adjust columns accordingly)
features = train_set[num_features + cat_features]
target = train_set["booking_status"]

# Set output to pandas dataframe
set_config(transform_output="pandas")

# Create preprocessing steps for numeric and categorical data
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
    ]
)

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features),
    ]
)

# Train a Random Forest model
model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier(**parameters))])
model.fit(features, target)
clf = model[-1]

# Identify the most important features
data = list(zip(clf.feature_names_in_, clf.feature_importances_))
feature_importances = pd.DataFrame(data, columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)

print("Top 5 important features:")
print(feature_importances.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate 2 synthetic datasets, similar distribution to the existing data and skewed

# COMMAND ----------

from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_object_dtype
import numpy as np
import pandas as pd

def create_synthetic_data(df, drift=False, num_rows=1000):
    synthetic_data = pd.DataFrame()

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]) and column != "Booking_ID":
            synthetic_data[column] = np.random.randint(df[column].min(), df[column].max() + 1, num_rows)

        elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
            synthetic_data[column] = np.random.choice(
                df[column].unique(), num_rows, p=df[column].value_counts(normalize=True)
            )

        elif isinstance(df[column].dtype, pd.CategoricalDtype) or isinstance(df[column].dtype, pd.StringDtype):
            synthetic_data[column] = np.random.choice(
                df[column].unique(), num_rows, p=df[column].value_counts(normalize=True)
            )
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            min_date, max_date = df[column].min(), df[column].max()
            if min_date < max_date:
                synthetic_data[column] = pd.to_datetime(np.random.randint(min_date.value, max_date.value, num_rows))
            else:
                synthetic_data[column] = [min_date] * num_rows

        else:
            synthetic_data[column] = np.random.choice(df[column], num_rows)

    new_ids = []
    i = max(existing_ids) + 1 if existing_ids else 1
    while len(new_ids) < num_rows:
        if i not in existing_ids:
            new_ids.append(str(i))  # Convert numeric ID to string
        i += 1
    synthetic_data["Booking_ID"] = ["INN" + id for id in new_ids]

    if drift:
        # Skew the top features to introduce drift
        top_features = ["lead_time", "avg_price_per_room"]  # Select top 2 features
        for feature in top_features:
            if feature in synthetic_data.columns:
                synthetic_data[feature] = synthetic_data[feature] * 1.5

    return synthetic_data

# Generate and visualize fake data

combined_set = pd.concat([train_set, test_set], ignore_index=True)
existing_ids = set(int(id) for id in combined_set["Booking_ID"].str[3:8].astype(int))

synthetic_data_normal = create_synthetic_data(train_set,  drift=False, num_rows=1000)
synthetic_data_skewed = create_synthetic_data(train_set, drift=True, num_rows=1000)

print(synthetic_data_normal.dtypes)
print(synthetic_data_normal.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add timestamp

# COMMAND ----------

synthetic_normal_df = spark.createDataFrame(synthetic_data_normal)
synthetic_normal_df_with_ts = synthetic_normal_df.withColumn(
    "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
)

synthetic_normal_df_with_ts.write.mode("append").saveAsTable(
    f"{catalog_name}.{schema_name}.inference_set_normal"
)

synthetic_skewed_df = spark.createDataFrame(synthetic_data_skewed)
synthetic_skewed_df_with_ts = synthetic_skewed_df.withColumn(
    "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
)

synthetic_skewed_df_with_ts.write.mode("append").saveAsTable(
    f"{catalog_name}.{schema_name}.inference_set_skewed"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write to feature table

# COMMAND ----------

import time
from databricks.sdk import WorkspaceClient

workspace = WorkspaceClient()

#write into feature table; update online table
spark.sql(f"""
    INSERT INTO {catalog_name}.{schema_name}.hotel_features
    SELECT Booking_ID, lead_time, avg_price_per_room, no_of_special_requests
    FROM {catalog_name}.{schema_name}.inference_set_normal
""")

#write into feature table; update online table
spark.sql(f"""
    INSERT INTO {catalog_name}.{schema_name}.hotel_features
    SELECT Booking_ID, lead_time, avg_price_per_room, no_of_special_requests
    FROM {catalog_name}.{schema_name}.inference_set_skewed
""")
  
update_response = workspace.pipelines.start_update(
    pipeline_id=pipeline_id, full_refresh=False)
while True:
    update_info = workspace.pipelines.get_update(pipeline_id=pipeline_id, 
                            update_id=update_response.update_id)
    state = update_info.update.state.value
    if state == 'COMPLETED':
        break
    elif state in ['FAILED', 'CANCELED']:
        raise SystemError("Online table failed to update.")
    elif state == 'WAITING_FOR_RESOURCES':
        print("Pipeline is waiting for resources.")
    else:
        print(f"Pipeline is in {state} state.")
    time.sleep(30)
