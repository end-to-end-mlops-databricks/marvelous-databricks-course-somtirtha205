# Databricks notebook source
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()


def load_data(filepath):
    """
    Load the data from the given filepath.
    """
    df = spark.read.csv(filepath, header=True, inferSchema=True).toPandas()

    return df


filepath = "/Volumes/mlops_dev/hotel_reservation/data/Data.csv"
df = load_data(filepath)

# COMMAND ----------


def preprocess_data(df, target_column="booking_status", column="Booking_ID"):
    """
    Preprocess the data.

    Args:
    df (pandas.DataFrame): The input dataframe
    target_column (str): The name of the target column (default is 'booking_status')

    Returns:
    X (pandas.DataFrame): The feature dataframe
    y (pandas.Series): The target series
    preprocessor (ColumnTransformer): The preprocessing pipeline
    """

    # Separate features and target
    y = df[target_column]
    X = df.drop([column, target_column], axis=1)

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

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
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    print("Features shape:", X.shape)
    print("Target shape:", y.shape)

    return X, y, preprocessor


X, y, preprocessor = preprocess_data(df)

# COMMAND ----------


def train_and_evaluate_model(X, y, preprocessor, test_size=0.2, random_state=42, n_estimators=1000, max_depth=25):
    # Create a pipeline with a preprocessor and a classifier
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state),
            ),
        ]
    )

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Fit the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)

    # Print the metrics
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix: {cm}")

    return model, f1, cm, X_train, X_test, y_train, y_test, y_pred


model, f1, cm, X_train, X_test, y_train, y_test, y_pred = train_and_evaluate_model(X, y, preprocessor)

# COMMAND ----------


def plot_actual_vs_predicted(
    y_test, y_pred, xlabel="Predicted Reservation", ylabel="Actual Reservation", title="Confusion Matrix"
):
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()


plot_actual_vs_predicted(y_test, y_pred)

# COMMAND ----------


def plot_feature_importance(model, top_n=10, title="Top Feature Importance", figsize=(10, 6)):
    # Extract feature importance and feature names
    feature_importance = model.named_steps["classifier"].feature_importances_
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    # Plot feature importance
    plt.figure(figsize=figsize)
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(top_n) + 0.5
    plt.barh(pos, feature_importance[sorted_idx[-top_n:]])
    plt.yticks(pos, feature_names[sorted_idx[-top_n:]])
    plt.title(title)
    plt.tight_layout()
    plt.show()


plot_feature_importance(model)
