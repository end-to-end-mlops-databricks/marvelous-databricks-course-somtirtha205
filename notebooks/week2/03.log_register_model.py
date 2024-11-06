# Databricks notebook source
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hotel_reservation.config import ProjectConfig

# COMMAND ----------

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# Load configuration
config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

# COMMAND ----------

# Extract configuration details
num_features = config.num_features
cat_features = config.cat_features
target = config.target
parameters = config.parameters
catalog_name = config.catalog_name
schema_name = config.schema_name


# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# Load training and testing sets from Databricks tables
train_set_spark = spark.table(f"{catalog_name}.{schema_name}.train_set")
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()


# COMMAND ----------

# Split features and target
X_train = train_set[num_features + cat_features]
y_train = train_set[target]

X_test = test_set[num_features + cat_features]
y_test = test_set[target]

# COMMAND ----------

# Create preprocessing steps for numeric and categorical data
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features),
    ]
)

# COMMAND ----------

model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier(**parameters))])

# COMMAND ----------

mlflow.set_experiment(experiment_name="/Shared/hotel-reservation")
git_sha = "3055af355f360ba5784ae7037f3260a70331f702"

# COMMAND ----------

# Start an MLflow run to track the training process
with mlflow.start_run(
    tags={"git_sha": f"{git_sha}", "branch": "week2"},
) as run:
    run_id = run.info.run_id

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate the model performance
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
    cm_display.plot()
    plt.savefig("confusion-matrix.png")
    plt.show()

    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix: {cm}")

    # Log parameters, metrics, and the model to MLflow
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_params(parameters)
    mlflow.log_metric("f1 score", f1)
    mlflow.log_artifact("confusion-matrix.png")
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    dataset = mlflow.data.from_spark(train_set_spark, table_name=f"{catalog_name}.{schema_name}.train_set", version="0")
    mlflow.log_input(dataset, context="training")

    mlflow.sklearn.log_model(sk_model=model, artifact_path="RandomForest", signature=signature)
