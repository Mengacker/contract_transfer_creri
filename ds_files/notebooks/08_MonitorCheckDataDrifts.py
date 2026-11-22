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
import pandas as pd
from datetime import date, datetime, timedelta
from pyspark.sql.functions import lit, current_timestamp
from pyspark.sql import functions as F

from src.utils.input_validation import is_iso_date
from src.config.config_utils import Config

pd.set_option("display.max_columns", None)

logger_azure = logging.getLogger("azure")
logger_azure.setLevel(logging.ERROR)

# COMMAND ----------

# DBTITLE 1,Notebook arguments
# Reset widgets
dbutils.widgets.removeAll()

# Set notebook arguments
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

# Get notebook arguments
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

# DBTITLE 1,general config values
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

# Data configurations
config_data = Config("../src/config/config_data.yml")
input_data = config_data.get_config_value("input_data")
input_features = config_data.get_config_value("input_features")
target_features = config_data.get_config_value("target_features")
engineered_fs = config_data.get_config_value("engineered_features")
# not_predictors = config_data.get_config_value("not_predictors")

# Reshape data informations
input_data.update(input_data.pop("Forniture", None)[0])  # remove "forniture" nesting
input_features = {c: d2 for m, d1 in input_features.items() for c, d2 in d1.items()}  # remove "metrics" nesting
engineered_fs = {c: d2 for m, d1 in engineered_fs.items() for c, d2 in d1.items()}  # remove "ante/post" nesting
target_features = {
    c: d2 for m, d1 in target_features.items() for c, d2 in d1.items()
}  # remove "training/serving" nesting

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

# DBTITLE 1,check active models
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

df_payload = spark.sql(query)
df_payload = df_payload.withColumn("response", df_payload["response"].cast("string"))
df_payload = df_payload.toPandas().sort_values(by=[payload_date, "flg_scoring_model", "score"], ascending=[False, False, False])
nr_scoring_request = sum([1 for x in df_payload["flg_scoring_model"] if x == True])
nr_deterministic_request = sum([1 for x in df_payload["flg_scoring_model"] if x == False])
print(nr_scoring_request, "n° of scoring request (", round(nr_scoring_request / df_payload.shape[0] * 100, 2), "%)")
print(
    nr_deterministic_request,
    "n° of deterministic request (",
    round(nr_deterministic_request / df_payload.shape[0] * 100, 2),
    "%)",
)
print(df_payload.shape[0], "n° of all request")
del df_payload

# COMMAND ----------

# DBTITLE 1,load data
if nr_scoring_request > 0:
    table_path=output_result_path
    print('The scoring model is active,', end=' ')
else:
    table_path=endpoint_payload_path
    print('The scoring model is not active.')

print(f"so the table to check is '{table_path}'.")

query = """
WITH tab as (
            SELECT *
            FROM {table_path}
            WHERE
            response NOT LIKE "%BAD_REQUEST%"
            and {payload_date} = '{checking_date}'
)
SELECT *
FROM tab
""".format(
    checking_date=day_before,
    table_path=table_path,
    payload_date = payload_date
)
df_results = spark.sql(query)
dtypes_spark_map = df_results.dtypes
df_results = df_results.toPandas()
col_results = list(df_results.columns)
print('Shape:',df_results.shape[0], "n° rows,", df_results.shape[1], "n° columns")
df_results.head()

# COMMAND ----------

# DBTITLE 1,get valorization maps
# Get the maps of admitted values by datatype
float_map = {}
boolean_map = {}
categories_map = {}
strings_map = {}
dates_map = {}
for diz in [input_data, input_features, engineered_fs, target_features]:
    float_map.update({c: d["values"] for c, d in diz.items() if d["pandas_type"] == "float64"})
    boolean_map.update({c: d["values"] for c, d in diz.items() if d["pandas_type"] == "bool"})
    categories_map.update(
        {c: d["values"] for c, d in diz.items() if d["pandas_type"] == "category" if type(d["values"]) == list}
    )
    strings_map.update(
        {c: d["values"] for c, d in diz.items() if d["pandas_type"] == "category" if type(d["values"]) == str}
    )
    strings_map.update({c: d["values"] for c, d in diz.items() if d["pandas_type"] == "string"})
    dates_map.update({c: d["values"] for c, d in diz.items() if d["pandas_type"] == "date"})

col_list = (
    list(float_map.keys())
    + list(boolean_map.keys())
    + list(categories_map.keys())
    + list(strings_map.keys())
    + list(dates_map.keys())
)

print("Float:", len(float_map), "columns")
print("Booleans:", len(boolean_map), "columns")
print("Categories:", len(categories_map), "columns")
print("Strings:", len(strings_map), "columns")
print("Dates:", len(dates_map), "columns")
print("Total:", len(col_list), "columns")

# Check not managed columns
not_managed = []
for diz in [input_data, input_features, engineered_fs, target_features]:
    not_managed = not_managed + [c for c in diz.keys() if c not in col_list]
assert not_managed == [], f"Columns not managed: {not_managed}"

# COMMAND ----------

# DBTITLE 1,check floats
not_expected_floats = {}
not_exp = []
df_err = None
for col in df_results.columns:
    if col in float_map.keys():
        values = float_map[col]

        try:
            df_results[col] = df_results[col].astype(float)

            if "min" in values:
                ind = df_results[col] < values["min"]
                if sum(ind) > 0:
                    not_exp = not_exp + df_results.loc[ind, col].values

            if "max" in values:
                ind = df_results[col] > values["max"]
                if sum(ind) > 0:
                    print(col, values)
                    not_exp = not_exp + df_results.loc[ind, col].values

            if "null" in values:
                if values["null"] == "false":
                    ind = df_results[col].isnull()
                    if sum(ind) > 0:
                        not_exp = not_exp + (sum(ind), "n° of null")
        except Exception as e:
            print(f"An error occurred with the column {col}\n")

        if not_exp != []:
            not_expected_floats[col] = not_exp
            ind = df_results[col].isin(not_exp)
            perc_err = round(100 * ind.sum() / df_results.shape[0], 1)
            print(f"Unexpected values in column {col}, error cases: {not_exp}")
            print(f"The number of row with unexpected values is {ind.sum()}, the {perc_err}% of the total\n")

            if df_err is None:
                df_err = df_results.loc[ind, :].copy()
                df_err.insert(0, "FIELD_WITH_ERROR", col)
            else:
                df = df_results.loc[ind, :].copy()
                df["FIELD_WITH_ERROR"] = col
                df_err = pd.concat([df_err, df], ignore_index=True)

if not_expected_floats == {}:
    print("All right")
else:
    df_err_grouped = df_err.groupby(final_payload_pk)["FIELD_WITH_ERROR"].apply(list).reset_index()
    df_err_grouped = df_err_grouped.merge(
        df_err.copy().astype(str).drop(columns=["FIELD_WITH_ERROR"]).drop_duplicates(),
        on=final_payload_pk,
        how="left",
    )
    display(df_err_grouped)

# COMMAND ----------

# DBTITLE 1,check booleans
not_expected_booleans = {}
not_exp = []
df_err = None
for col in df_results.columns:
    if col in boolean_map.keys():
        values = [v for v in boolean_map[col]]
        if "" in values:
            values.append(None)

        assert [x for x in values if x not in [True, False, "", None]] == []
        try:
            not_exp = [x for x in df_results[col].unique().tolist() if x not in values]
        except Exception as e:
            print(f"An error occurred with the column {col}\n")

        if not_exp != []:
            not_expected_booleans[col] = not_exp
            ind = df_results[col].isin(not_exp)
            perc_err = round(100 * ind.sum() / df_results.shape[0], 1)
            print(f"Unexpected values in column {col}, error cases: {not_exp}")
            print(f"The number of row with unexpected values is {ind.sum()}, the {perc_err}% of the total\n")

            if df_err is None:
                df_err = df_results.loc[ind, :].copy()
                df_err.insert(0, "FIELD_WITH_ERROR", col)
            else:
                df = df_results.loc[ind, :].copy()
                df["FIELD_WITH_ERROR"] = col
                df_err = pd.concat([df_err, df], ignore_index=True)

if not_expected_booleans == {}:
    print("All right")
else:
    df_err_grouped = df_err.groupby(final_payload_pk)["FIELD_WITH_ERROR"].apply(list).reset_index()
    df_err_grouped = df_err_grouped.merge(
        df_err.copy().astype(str).drop(columns=["FIELD_WITH_ERROR"]).drop_duplicates(),
        on=final_payload_pk,
        how="left",
    )
    display(df_err_grouped)

# COMMAND ----------

# DBTITLE 1,check categories
not_expected_categories = {}
not_exp = []
df_err = None
for col in df_results.columns:
    if col in categories_map.keys():
        values = categories_map[col]
        try:
            values = [v.lower() for v in values]
            not_exp = [x for x in df_results[col].unique().tolist() if x.lower() not in values]
        except Exception as e:
            print(f"An error occurred with the column {col}\n")

        if not_exp != []:
            not_expected_categories[col] = not_exp
            ind = df_results[col].isin(not_exp)
            perc_err = round(100 * ind.sum() / df_results.shape[0], 1)
            print(f"Unexpected values in column {col}, error cases: {not_exp}")
            print(f"The number of row with unexpected values is {ind.sum()}, the {perc_err}% of the total\n")

            if df_err is None:
                df_err = df_results.loc[ind, :].copy()
                df_err.insert(0, "FIELD_WITH_ERROR", col)
            else:
                df = df_results.loc[ind, :].copy()
                df["FIELD_WITH_ERROR"] = col
                df_err = pd.concat([df_err, df], ignore_index=True)

if not_expected_categories == {}:
    print("All right")
else:
    df_err_grouped = df_err.groupby(final_payload_pk)["FIELD_WITH_ERROR"].apply(list).reset_index()
    df_err_grouped = df_err_grouped.merge(
        df_err.copy().astype(str).drop(columns=["FIELD_WITH_ERROR"]).drop_duplicates(),
        on=final_payload_pk,
        how="left",
    )
    display(df_err_grouped)

# COMMAND ----------

# DBTITLE 1,check string
not_expected_string = {}
not_exp = []
df_err = None
for col in df_results.columns:
    if col in strings_map.keys():
        values = strings_map[col]

        try:
            not_exp = [x for x in df_results[col].unique().tolist() if type(x) != str]
        except Exception as e:
            print(f"An error occurred with the column {col}\n")

        if not_exp != []:
            not_expected_string[col] = not_exp
            ind = df_results[col].isin(not_exp)
            perc_err = round(100 * ind.sum() / df_results.shape[0], 1)
            print(f"Unexpected values in column {col}, error cases: {not_exp}")
            print(f"The number of row with unexpected values is {ind.sum()}, the {perc_err}% of the total\n")

            if df_err is None:
                df_err = df_results.loc[ind, :].copy()
                df_err.insert(0, "FIELD_WITH_ERROR", col)
            else:
                df = df_results.loc[ind, :].copy()
                df["FIELD_WITH_ERROR"] = col
                df_err = pd.concat([df_err, df], ignore_index=True)

if not_expected_string == {}:
    print("All right")
else:
    df_err_grouped = df_err.groupby(final_payload_pk)["FIELD_WITH_ERROR"].apply(list).reset_index()
    df_err_grouped = df_err_grouped.merge(
        df_err.copy().astype(str).drop(columns=["FIELD_WITH_ERROR"]).drop_duplicates(),
        on=final_payload_pk,
        how="left",
    )
    display(df_err_grouped)

# COMMAND ----------

# DBTITLE 1,check dates
not_expected_dates = {}
not_exp = []
df_err = None
for col in df_results.columns:
    if col in dates_map.keys():
        values = dates_map[col]

        try:
            not_exp = [x for x in df_results[col].unique().tolist() if type(x) != date]
        except Exception as e:
            print(f"An error occurred with the column {col}\n")

        if not_exp != []:
            not_expected_dates[col] = not_exp
            ind = df_results[col].isin(not_exp)
            perc_err = round(100 * ind.sum() / df_results.shape[0], 1)
            print(f"Unexpected values in column {col}, error cases: {not_exp}")
            print(f"The number of row with unexpected values is {ind.sum()}, the {perc_err}% of the total\n")

            if df_err is None:
                df_err = df_results.loc[ind, :].copy()
                df_err.insert(0, "FIELD_WITH_ERROR", col)
            else:
                df = df_results.loc[ind, :].copy()
                df["FIELD_WITH_ERROR"] = col
                df_err = pd.concat([df_err, df], ignore_index=True)

if not_expected_dates == {}:
    print("All right")
else:
    df_err_grouped = df_err.groupby(final_payload_pk)["FIELD_WITH_ERROR"].apply(list).reset_index()
    df_err_grouped = df_err_grouped.merge(
        df_err.copy().astype(str).drop(columns=["FIELD_WITH_ERROR"]).drop_duplicates(),
        on=final_payload_pk,
        how="left",
    )
    display(df_err_grouped)

# COMMAND ----------

# DBTITLE 1,asserts
assert all(
    d == {}
    for d in [
        not_expected_floats,
        not_expected_booleans,
        not_expected_string,
        not_expected_dates,
        not_expected_categories,
    ]
), f"""Unexpected values found:
Floats: {not_expected_floats}
Booleans: {not_expected_booleans}
Strings: {not_expected_string}
Dates: {not_expected_dates}
Categories: {not_expected_categories}
"""
print("No problems found")
