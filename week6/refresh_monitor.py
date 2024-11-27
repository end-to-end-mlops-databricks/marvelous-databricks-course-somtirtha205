import argparse

from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, IntegerType, StringType, StructField, StructType

from hotel_reservation.config import ProjectConfig

spark = DatabricksSession.builder.getOrCreate()
workspace = WorkspaceClient()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)

args = parser.parse_args()
root_path = args.root_path

# Load configuration
config_path = f"{root_path}/files/project_config.yml"
config = ProjectConfig.from_yaml(config_path=config_path)
catalog_name = config.catalog_name
schema_name = config.schema_name

inf_table = spark.sql(f"SELECT * FROM {catalog_name}.{schema_name}.`hotel-reservation-model-serving-fe_payload`")

request_schema = StructType(
    [
        StructField(
            "dataframe_records",
            ArrayType(
                StructType(
                    [
                        StructField("no_of_adults", IntegerType(), True),
                        StructField("no_of_children", IntegerType(), True),
                        StructField("no_of_weekend_nights", IntegerType(), True),
                        StructField("no_of_week_nights", IntegerType(), True),
                        StructField("required_car_parking_space", IntegerType(), True),
                        StructField("arrival_year", IntegerType(), True),
                        StructField("arrival_month", IntegerType(), True),
                        StructField("arrival_date", IntegerType(), True),
                        StructField("repeated_guest", IntegerType(), True),
                        StructField("no_of_previous_cancellations", IntegerType(), True),
                        StructField("no_of_previous_bookings_not_canceled", IntegerType(), True),
                        StructField("type_of_meal_plan", StringType(), True),
                        StructField("room_type_reserved", StringType(), True),
                        StructField("market_segment_type", StringType(), True),
                        StructField("Booking_ID", StringType(), True),
                    ]
                )
            ),
            True,
        )
    ]
)

response_schema = StructType(
    [
        StructField("predictions", ArrayType(IntegerType()), True),
        StructField(
            "databricks_output",
            StructType(
                [StructField("trace", StringType(), True), StructField("databricks_request_id", StringType(), True)]
            ),
            True,
        ),
    ]
)

inf_table_parsed = inf_table.withColumn("parsed_request", F.from_json(F.col("request"), request_schema))

inf_table_parsed = inf_table_parsed.withColumn("parsed_response", F.from_json(F.col("response"), response_schema))

df_exploded = inf_table_parsed.withColumn("record", F.explode(F.col("parsed_request.dataframe_records")))


df_final = df_exploded.select(
    F.from_unixtime(F.col("timestamp_ms") / 1000).cast("timestamp").alias("timestamp"),
    "timestamp_ms",
    "databricks_request_id",
    "execution_time_ms",
    F.col("record.Booking_ID").alias("Booking_ID"),
    F.col("record.no_of_adults").alias("no_of_adults"),
    F.col("record.no_of_children").alias("no_of_children"),
    F.col("record.no_of_weekend_nights").alias("no_of_weekend_nights"),
    F.col("record.no_of_week_nights").alias("no_of_week_nights"),
    F.col("record.required_car_parking_space").alias("required_car_parking_space"),
    F.col("record.arrival_year").alias("arrival_year"),
    F.col("record.arrival_month").alias("arrival_month"),
    F.col("record.arrival_date").alias("arrival_date"),
    F.col("record.repeated_guest").alias("repeated_guest"),
    F.col("record.no_of_previous_cancellations").alias("no_of_previous_cancellations"),
    F.col("record.no_of_previous_bookings_not_canceled").alias("no_of_previous_bookings_not_canceled"),
    F.col("record.type_of_meal_plan").alias("type_of_meal_plan"),
    F.col("record.room_type_reserved").alias("room_type_reserved"),
    F.col("record.market_segment_type").alias("market_segment_type"),
    F.col("parsed_response.predictions")[0].alias("prediction"),
    F.lit("hotel_reservation_model_fe").alias("model_name"),
)


test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")
inference_set_normal = spark.table(f"{catalog_name}.{schema_name}.inference_set_normal")
inference_set_skewed = spark.table(f"{catalog_name}.{schema_name}.inference_set_skewed")

inference_set = inference_set_normal.union(inference_set_skewed)

df_final_with_status = (
    df_final.join(test_set.select("Booking_ID", "booking_status"), on="Booking_ID", how="left")
    .withColumnRenamed("booking_status", "booking_status_test")
    .join(inference_set.select("Booking_ID", "booking_status"), on="Booking_ID", how="left")
    .withColumnRenamed("booking_status", "booking_status_inference")
    .select("*", F.coalesce(F.col("booking_status_test"), F.col("booking_status_inference")).alias("booking_status"))
    .drop("booking_status_test", "booking_status_inference")
    .withColumn("booking_status", F.col("booking_status").cast("int"))
    .withColumn("prediction", F.col("prediction").cast("int"))
    .dropna(subset=["booking_status", "prediction"])
)

hotel_features = spark.table(f"{catalog_name}.{schema_name}.hotel_features")

df_final_with_features = df_final_with_status.join(hotel_features, on="Booking_ID", how="left")

df_final_with_features = df_final_with_features.withColumn(
    "avg_price_per_room", F.col("avg_price_per_room").cast("double")
)

df_final_with_features.write.format("delta").mode("append").saveAsTable(
    f"{catalog_name}.{schema_name}.model_monitoring"
)

workspace.quality_monitors.run_refresh(table_name=f"{catalog_name}.{schema_name}.model_monitoring")
