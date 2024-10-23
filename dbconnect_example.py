from databricks.connect import DatabricksSession

spark = DatabricksSession.builder.profile("adb-1846957892648178").getOrCreate()
df = spark.read.table("samples.nyctaxi.trips")
df.show(5)
