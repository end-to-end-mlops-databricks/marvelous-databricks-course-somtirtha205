import logging
import yaml
from databricks.connect import DatabricksSession

from src.hotel_reservation.config import ProjectConfig
from src.hotel_reservation.data_processor import DataProcessor
from src.hotel_reservation.reservation_model import ReservationModel
from src.hotel_reservation.utils import plot_feature_importance, visualize_results

#Spark Session
spark = DatabricksSession.builder.profile("adb-1846957892648178").getOrCreate()

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load configuration
config = ProjectConfig.from_yaml(config_path="project_config.yml")

print("Configuration loaded:")
print(yaml.dump(config, default_flow_style=False))

# Extract configuration details
num_features = config.num_features
cat_features = config.cat_features
target = config.target
parameters = config.parameters
catalog_name = config.catalog_name
schema_name = config.schema_name

# Initialize DataProcessor
data_processor = DataProcessor(r"data\Data.csv", config)
logger.info("DataProcessor initialized.")

# Preprocess the data
data_processor.preprocess_data()
logger.info("Data preprocessed.")

# Split the data
train_set, test_set = data_processor.split_data()

# Save to Catalog
data_processor.save_to_catalog(train_set=train_set, test_set=test_set, spark=spark)

# Load training and testing sets from Databricks tables
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

# Split the data
X_train = train_set[num_features + cat_features]
y_train = train_set[target]

X_test = test_set[num_features + cat_features]
y_test = test_set[target]

logger.info("Data split into training and test sets.")
logger.debug(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# Initialize and train the model
model = ReservationModel(data_processor.preprocessor, config)
model.train(X_train, y_train)
logger.info("Model training completed.")

# Evaluate the model
f1, cm = model.evaluate(X_test, y_test)
logger.info(f"Model evaluation completed: f1={f1}, cm={cm}")

## Visualizing Results
y_pred = model.predict(X_test)
visualize_results(cm)
logger.info("Results visualization completed.")

## Feature Importance
feature_importance, feature_names = model.get_feature_importance()
plot_feature_importance(feature_importance, feature_names)
logger.info("Feature importance plot generated.")
