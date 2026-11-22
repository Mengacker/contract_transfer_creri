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
dbutils.widgets.text("env", "prod-dpp", "Environment Name")  # dev, prod-sbx
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
config = Config(config_path="../../src/config/", name_list=["config_setup.yml", "config_model.yml"])
config_tests = Config(config_path="../../src/config/config_tests.yml")

tracking_uri = config.get_config_value("mlflow_config.tracking_uri")

experiment_path = config.get_config_value("environment.dev.experiment_path".format(env=env))

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
    "inference_monitoring",
]
diz_tab = []
for tab in list_of_table_types:
    diz_tab = diz_tab + config.get_config_value(tab)
diz_tab = {d["name"]: d["fields_to_mask"] for d in diz_tab}
diz_tab

# COMMAND ----------

# MAGIC %md
# MAGIC # CHECK MASKED FIELDS

# COMMAND ----------

# DBTITLE 1,Elaboration
diz_results = {}
for tab, cols in diz_tab.items():
    if cols:
        print(f"{tab} -> n° {len(cols)} fields to be masked:")
        query = """
            SELECT {fields}
            FROM {table_path}
            LIMIT 100
            """.format(
            table_path=path + tab, fields=",".join(cols)
        )

        df = spark.sql(query)

        diz_count = {}
        for c in cols:
            print(c, end=", ")
            # Measure length of a column
            df = df.withColumn(f"{c}_len", F.length(c))
            df = df.withColumn(f"{c}_flg_len_32", F.col(f"{c}_len") != 32)
            df = df.drop(f"{c}_len")

            # Aggregate count of True values for the flag column
            df_count = df.agg(F.count_if(F.col(f"{c}_flg_len_32")).alias(f"true_count"))
            df = df.drop(f"{c}_flg_len_32")

            diz_count[f"{c}_count_no_len32"] = df_count.collect()[0]["true_count"]

            # Measure bit length of a column
            df = df.withColumn(f"{c}_bit", F.bit_length(c))
            # df = df.withColumn(f"{c}_flg_bit_128", F.col(f"{c}_bit") != 128)
            df = df.withColumn(f"{c}_flg_bit_256", F.col(f"{c}_bit") != 256)

            # Aggregate count of True values for the flag column
            df_count = df.agg(F.count_if(F.col(f"{c}_flg_bit_256")).alias(f"true_count"))
            df = df.drop(f"{c}_flg_bit_256")

            diz_count[f"{c}_count_no_bit256"] = df_count.collect()[0]["true_count"]
        print("\n")
        diz_results.update({tab: diz_count})

# COMMAND ----------

diz_results

# COMMAND ----------

# DBTITLE 1,Visual check on a dataframe
df_results = pd.DataFrame(diz_results)

dfsum = (df_results.sum() > 0).reset_index()
wcols = dfsum.loc[dfsum[0], "index"].values
df_results = df_results[wcols]

dfsum = (df_results.T.sum() > 0).reset_index()
wrows = dfsum.loc[dfsum[0], "index"].values
df_results = df_results.T[wrows].T

df_results

# COMMAND ----------

# DBTITLE 1,Do the TEST
l_fail_test = []
for tab, diz in diz_results.items():
    for col, val in diz.items():
        if val > 0:
            l_fail_test.append([tab, col, val])

assert len(l_fail_test) == 0, f"There are columns not masked: {l_fail_test}"

# COMMAND ----------
