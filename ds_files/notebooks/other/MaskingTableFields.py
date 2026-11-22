# Databricks notebook source
# MAGIC %md
# MAGIC # DO NOT MODIFY THIS PART AND LEAVE AS IT IS
# MAGIC Fixed section to make MLops working

# COMMAND ----------

# DBTITLE 1,Set notebook extensions
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# DBTITLE 1,Set working directory
import os
import sys
from databricks.sdk.runtime import dbutils

notebook_path = "/Workspace/" + os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)
sys.path.append("../..")
os.chdir(notebook_path)

# COMMAND ----------

# DBTITLE 1,Install requirements
# MAGIC %pip install -r ../../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Set working directory
import os
import sys
from databricks.sdk.runtime import dbutils

notebook_path = "/Workspace/" + os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)
sys.path.append("../..")
os.chdir(notebook_path)

# COMMAND ----------

# MAGIC %md
# MAGIC # EDITABLE SECTION
# MAGIC Here you can put your code, starting from imports

# COMMAND ----------

# DBTITLE 1,Import libraries
import logging
import mlflow
import json
import pandas as pd
from datetime import date, datetime, timedelta
from pyspark.sql.functions import lit, current_timestamp
from pyspark.sql import functions as F, SparkSession

spark: SparkSession = SparkSession.builder.getOrCreate()

from src.utils.input_validation import is_iso_date
from src.config.config_utils import Config

pd.set_option("display.max_columns", None)

logger_azure = logging.getLogger("azure")
logger_azure.setLevel(logging.ERROR)

# COMMAND ----------

# DBTITLE 1,Notebook arguments
# --- Set notebook arguments --- #
# Reset widgets
dbutils.widgets.removeAll()

# Notebook Environment
dbutils.widgets.text("env", "dev", "Environment Name")  # prod-sbx
dbutils.widgets.dropdown("is_cicd", "false", ["true", "false"], "Is run by CICD")

# date_and_time = datetime.now() # sballa l'orario di 2 ore perchè usa il riferimento UTC
timezone = "Europe/Amsterdam"
date_and_time = spark.sql("SELECT current_timestamp() as current_time").withColumn(
    "current_time", F.from_utc_timestamp("current_time", timezone)
)
date_and_time = date_and_time.collect()[0][0]

print("Current date and time:", date_and_time.strftime("%Y-%m-%d %H:%M:%S"), "at the {t} timezone".format(t=timezone))
dbutils.widgets.text(
    "snapshot_date",
    str(date_and_time.date()),
    label="Snapshot date for the run",
)

# --- Get notebook arguments ---#
env = dbutils.widgets.get("env")
is_cicd = bool(dbutils.widgets.get("is_cicd") == "true")
snapshot_date_str = dbutils.widgets.get("snapshot_date")

# Validate input
if not is_iso_date(snapshot_date_str):
    raise ValueError(f"Invalid snapshot_date: {snapshot_date_str}")

list_date_to_check = pd.date_range(start="2025-06-01", end=date.today()).strftime("%Y-%m-%d").tolist()
list_date_to_check[-1]

# COMMAND ----------

# DBTITLE 1,Retrieve config values
# Retrieve config values
config = Config(config_path="../../src/config/", name_list=["config_setup.yml"])
config_tests = Config(config_path="../../src/config/config_tests.yml")

catalog = config.get_config_value("environment.{env}.catalog".format(env=env))
schema = config.get_config_value("environment.{env}.schema".format(env=env))
path = f"{catalog}.{schema}."
print(path)

# Get fields to be masked of each table
list_of_table_types = [
    "targets",
    "metrics",
    "customer_bases",
    "master_training",
    "master_training_final",
    "master_inference",
    "master_inference_versioned",
    "inference_monitoring",
]
diz_tab = []
for tab in list_of_table_types:
    diz_tab = diz_tab + config.get_config_value(tab)
diz_tab = {d["name"]: d["fields_to_mask"] for d in diz_tab}
diz_tab

# COMMAND ----------

# MAGIC %md
# MAGIC # MASKING FIELDS

# COMMAND ----------

# DBTITLE 1,SET MASK
print(path)

for tab, cols in diz_tab.items():
    if cols:
        print(f"{tab} -> n° {len(cols)} fields to be masked:")
        for col in cols:
            print(col, end=", ")

            query = """
                ALTER TABLE  {table_path}
                ALTER COLUMN {col} SET MASK p_ind_core_ai.metadata.dp_mask_str USING COLUMNS ('GG.Appl.DataPlatform1850.Adb.Contatto-All.ENINET');
            """.format(
                table_path=path + tab,
                col=col,
            )

            spark.sql(query)
        print("\n")

# COMMAND ----------

# DBTITLE 1,DROP MASK
# print(path)

# for tab, cols in diz_tab.items():
#     if cols:
#         print(f"{tab} -> n° {len(cols)} fields to be masked:")
#         for col in cols:
#             print(col, end=", ")

#             query = """
#                 ALTER TABLE  {table_path}
#                 ALTER COLUMN {col} DROP MASK;
#             """.format(
#                 table_path=path + tab,
#                 col=col,
#             )

#             spark.sql(query)
#         print("\n")
