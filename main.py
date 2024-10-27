import logging

import yaml

from src.hotel_reservation.data_processor import DataProcessor
from src.hotel_reservation.reservation_model import ReservationModel
from src.hotel_reservation.utils import plot_feature_importance, visualize_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Load configuration
with open("project_config.yml", "r") as file:
    config = yaml.safe_load(file)

print("Configuration loaded:")
print(yaml.dump(config, default_flow_style=False))


# Initialize DataProcessor
data_processor = DataProcessor(r"data\Data.csv", config)
logger.info("DataProcessor initialized.")

# Preprocess the data
data_processor.preprocess_data()
logger.info("Data preprocessed.")

# Split the data
X_train, X_test, y_train, y_test = data_processor.split_data()
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
