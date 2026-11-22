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
sys.path.append("..")
os.chdir(notebook_path)

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Set working directory
import os
import sys
from databricks.sdk.runtime import dbutils

notebook_path = "/Workspace/" + os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)
sys.path.append("..")
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
from datetime import datetime, timedelta
from pyspark.sql.functions import lit, current_timestamp
from pyspark.sql import functions as F

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
dbutils.widgets.text("env", "dev", "Environment Name")
dbutils.widgets.dropdown("is_cicd", "false", ["true", "false"], "Is run by CICD")

config_setup = Config("../src/config/config_setup.yml")
dev_experiment_path = config_setup.get_config_value("environment.dev.experiment_path")
dbutils.widgets.text("experiment_path", dev_experiment_path, "Experiment path name read by asset workflow file yaml")

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

print(f"env: {env}")

# COMMAND ----------

# DBTITLE 1,day before
day_before = (date_and_time.date() - timedelta(days=1)).isoformat()  # the day before
day_before

# COMMAND ----------

# DBTITLE 1,Retrive config params
# Retrieve config values
config_setup = Config("../src/config/config_setup.yml")
catalog = config_setup.get_config_value("environment.{env}.catalog".format(env=env))
schema = config_setup.get_config_value("environment.{env}.schema".format(env=env))

# Payload table
endpoint_payload_table = config_setup.get_config_value("inference_monitoring")[0]["name"]
endpoint_payload_path = f"{catalog}.{schema}.{endpoint_payload_table}"
print(f"{endpoint_payload_path}   table of the inference payload output")

# Output table
output_result_table = config_setup.get_config_value("inference_monitoring")[1]["name"]
output_result_path = f"{catalog}.{schema}.{output_result_table}"
print(f"{output_result_path}   table of the final inference output")

# COMMAND ----------

# DBTITLE 1,payload columns

# OLD PAYLOAD TABLE COLUMNS
# payload_date='date'
# payload_pk = 'databricks_request_id'
# payload_cols = [
#     # 'databricks_request_id', # PK
#     # 'date'
#     "client_request_id", "timestamp_ms", "status_code", "sampling_fraction", "execution_time_ms"
#     # ,'request', 'response'
#     # , 'request_metadata'
#     ]

# NEW PAYLOAD TABLE COLUMNS
payload_date='request_date'
payload_pk = 'databricks_request_id'
payload_cols = [
    # 'databricks_request_id',
    # 'request_date',
    'client_request_id', 'request_time', 'status_code', 'sampling_fraction', 'execution_duration_ms'
    # ,'request', 'response'
    # , 'logging_error_codes', 'served_entity_id', 'requester' # nuovi campi
    # , 'request_metadata'
    ]

final_payload_pk = 'request_PODPDR_id'

# COMMAND ----------

# DBTITLE 1,bad request allert
query = """
WITH tab as (
            SELECT *
            FROM {table_path}
            WHERE
            response LIKE "%BAD_REQUEST%"
            and
            {payload_date} = '{checking_date}'
)
SELECT {payload_date}, response, COUNT(*) as NR_LOGS
FROM tab
GROUP BY {payload_date}, response
""".format(
    checking_date=day_before,
    table_path=endpoint_payload_path,
    payload_date=payload_date
)

df_errors = spark.sql(query)
df_errors = df_errors.withColumn("response", df_errors["response"].cast("string"))
df_errors = df_errors.toPandas().sort_values(by=[payload_date, "NR_LOGS"], ascending=[False, False])
nr_errors = df_errors.shape[0]
print(nr_errors, "n° of errors")
if nr_errors > 0:
    display(df_errors)

# COMMAND ----------

# DBTITLE 1,scoring model out of service allert
query = """
WITH tab as (
            SELECT *,
            CAST(substring_index(substring_index(response, '"score":', -1), '}}', 1) AS DOUBLE) AS score
            FROM {table_path}
            WHERE
            response NOT LIKE "%BAD_REQUEST%"
            and
            {payload_date} = '{checking_date}'
)
SELECT *
, CASE
  WHEN ( score > 0 AND score < 1) THEN True
  WHEN score = -1 THEN True
  WHEN ( score = 0 OR score = 1) THEN False
  ELSE NULL
  END AS flg_scoring_model
FROM tab
""".format(
    checking_date=day_before,
    table_path=endpoint_payload_path,
    payload_date = payload_date
)

df_score = spark.sql(query)
df_score = df_score.withColumn("response", df_score["response"].cast("string"))
df_score = df_score.toPandas().sort_values(by=[payload_date, "flg_scoring_model", "score"], ascending=[False, False, False])
nr_scoring_request = sum([1 for x in df_score["flg_scoring_model"] if x == True])
nr_deterministic_request = sum([1 for x in df_score["flg_scoring_model"] if x == False])
print(nr_scoring_request, "n° of scoring request (", round(nr_scoring_request / df_score.shape[0] * 100, 2), "%)")
print(
    nr_deterministic_request,
    "n° of deterministic request (",
    round(nr_deterministic_request / df_score.shape[0] * 100, 2),
    "%)",
)
print(df_score.shape[0], "n° of all request")
df_score

# COMMAND ----------

# DBTITLE 1,unique request_PODPDR_id allert
if nr_scoring_request > 0:
    query = """
    WITH tab as (
                SELECT *
                FROM {table_path}
                WHERE
                {payload_date} = '{checking_date}'
    )
    SELECT *
    FROM tab
    """.format(
        checking_date=day_before,
        table_path=output_result_path,
        payload_date= payload_date
    )
    df_results = spark.sql(query)
    dtypes_spark_map = df_results.dtypes
    df_results = df_results.toPandas()
    nr_col = df_results.shape[1]
    nr_row = df_results.shape[0]
    nr_id = df_results[final_payload_pk].nunique()
    print(nr_row, "n° rows,", nr_col, "n° columns")
    print(f"{nr_row} n° of rows == {nr_id} n° of {final_payload_pk}")
    df_results = df_results.sort_values(by=final_payload_pk)
    df_results
else:
    print("No scoring request found.")

# COMMAND ----------

# DBTITLE 1,assert errors
if nr_scoring_request > 0:
    assert all(
        [nr_errors == 0, nr_scoring_request > 0, nr_deterministic_request > 0, nr_row == nr_id]
    ), f"""Unexpected values found:
    n° of errors 'BAD REQUEST': {nr_errors} = 0 ?
    n° of model scoring request: {nr_scoring_request} > 0 ?  {round(nr_scoring_request/df_score.shape[0]*100,1)}% > 0% ?
    n° of model deterministic request: {nr_deterministic_request} > 0 ? {round(nr_deterministic_request/df_score.shape[0]*100,1)}% > 0% ?
    n° of duplicated id request: {nr_row-nr_id} = 0 ?
    """
else:
    assert all(
        [nr_errors == 0, nr_scoring_request == 0, nr_deterministic_request/df_score.shape[0] == 1]
    ), f"""Unexpected values found:
    n° of errors 'BAD REQUEST': {nr_errors} = 0 ?
    n° of model scoring request: {nr_scoring_request} = 0 ?  {round(nr_scoring_request/df_score.shape[0]*100,1)}% = 0% ?
    n° of model deterministic request: {nr_deterministic_request}, {round(nr_deterministic_request/df_score.shape[0]*100,1)}% = 100% ?
    """

print("No problems found.")
