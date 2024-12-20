import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


class DataProcessor:
    def __init__(self, config):
        self.df = None
        self.config = config
        self.preprocessor = None
        self.labelencoder = None

    def preprocess_data(self, filepath, spark):
        # Load Data
        self.df = spark.read.csv(filepath, header=True, inferSchema=True).toPandas()

        # Create preprocessing steps for numeric and categorical data
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.config.num_features),
                ("cat", categorical_transformer, self.config.cat_features),
            ]
        )

        self.labelencoder = LabelEncoder()
        self.df[self.config.target] = self.labelencoder.fit_transform(self.df[self.config.target])

    def split_data(self, test_size=0.2, random_state=42):
        """Split the DataFrame (self.df) into training and test sets."""
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame, spark: SparkSession):
        """Save the train and test sets into Databricks tables."""

        catalog_name = self.config.catalog_name
        schema_name = self.config.schema_name

        train_set_with_timestamp = spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("overwrite").saveAsTable(f"{catalog_name}.{schema_name}.train_set")

        test_set_with_timestamp.write.mode("overwrite").saveAsTable(f"{catalog_name}.{schema_name}.test_set")

        spark.sql(
            f"ALTER TABLE {catalog_name}.{schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        spark.sql(
            f"ALTER TABLE {catalog_name}.{schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
