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
from datetime import date, datetime

# from pyspark.sql.functions import lit #, current_timestamp, quarter
from pyspark.sql import functions as F

from src.utils.input_validation import is_iso_date
from src.monitoring.utils_monit import monthly_table_aggregator
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

first_consuntivation_date = "2025-06-01" # OLD model
first_consuntivation_date = "2026-04-01" # NEW model
list_date_to_check = pd.date_range(start=first_consuntivation_date, end=date.today()).strftime("%Y-%m-%d").tolist()
list_date_to_check[-1]

# COMMAND ----------

# DBTITLE 1,Retrive config params
# Retrieve config values
config_setup = Config("../src/config/config_setup.yml")
catalog = config_setup.get_config_value("environment.{env}.catalog".format(env=env))
schema = config_setup.get_config_value("environment.{env}.schema".format(env=env))

# Payload table
endpoint_payload_table = config_setup.get_config_value("inference_monitoring")[0]["name"]
endpoint_payload_path = f"{catalog}.{schema}.{endpoint_payload_table}"
print(f"{endpoint_payload_path}   table to read for the payload log input")

# Output table
output_result_table = config_setup.get_config_value("inference_monitoring")[1]["name"]
output_result_path = f"{catalog}.{schema}.{output_result_table}"
print(f"{output_result_path}   table to write the output results")

# COMMAND ----------

# MAGIC %md
# MAGIC # Payload

# COMMAND ----------

endpoint_payload_path

# COMMAND ----------

# DBTITLE 1,payload group by day date
query = """
WITH tab as (
            SELECT *
            , regexp_replace(split_part(split_part(request, r'CodiceFiscale', 2), r'",', 1), '[^A-Za-z0-9]', '') AS CF
            , regexp_replace(split_part(split_part(request, r'PartitaIva', 2), r'",', 1), '[^A-Za-z0-9]', '') AS PIVA
            , regexp_replace(split_part(split_part(request, r'POD/PDR', 2), r',',1), '[^A-Za-z0-9]', '') AS POD_PDR
            , regexp_replace(split_part(split_part(request, r'POD/PDR', 3), r',',1), '[^A-Za-z0-9]', '') AS POD_PDR_BIS
            , regexp_count(request, 'POD/PDR') AS N_PODPDR

            , CASE
                WHEN response LIKE "%BAD_REQUEST%" THEN "BAD_REQUEST"
                WHEN response LIKE "%Semaforo Rosso%" THEN "RED"
                WHEN response LIKE "%Semaforo Giallo%" THEN "YELLOW"
                WHEN response LIKE "%Semaforo Verde%" THEN "GREEN"
            END AS `RESPONSE_CATEGORY`
            , CASE
                WHEN response LIKE "%BAD_REQUEST%" THEN "BAD_REQUEST"
                WHEN response LIKE "%Semaforo%" AND response LIKE '%"score": 0.%' THEN "SCORING"
                WHEN response LIKE "%Semaforo%" AND response LIKE '%"score": -1%' THEN "SCORING" -- actually NO ONE model has used
                WHEN response LIKE "%Semaforo%" AND response LIKE '%"score": 1%' THEN "DETERMINISTIC"
                WHEN response LIKE "%Semaforo%" AND response LIKE '%"score": 0%' THEN "DETERMINISTIC"
                ELSE ""
            END AS `USED_MODEL`
            FROM {table_path}
            WHERE
            request_date >= '{first_consuntivation_date}'

            AND request NOT RLIKE 'XXXXXX00X00X000X|MRNDVD89C22H501B|12345678910|ebb1868a42d1e45efc7a8f0d862acc61|01234567891011'
)
, tab2 AS (
    SELECT *
        , databricks_request_id AS PK_REQUEST
        , concat_ws('_', IF(CF != "", CF, PIVA), POD_PDR, POD_PDR_BIS, N_PODPDR) AS PK_VOLTURA
        , IF(CF != "", CF, PIVA) AS COD_FISCALE_PIVA_CLIIN
    FROM tab
)
, aggr AS (
    SELECT request_date, RESPONSE_CATEGORY, USED_MODEL
        , COUNT(DISTINCT PK_REQUEST) as NR_REQUEST
        , COUNT(DISTINCT PK_VOLTURA) as NR_VOLTURE
    FROM tab2
    GROUP BY request_date, RESPONSE_CATEGORY, USED_MODEL
    ORDER BY request_date, RESPONSE_CATEGORY, USED_MODEL
)
SELECT *
FROM aggr
""".format(
    table_path=endpoint_payload_path,  # output_result_path
    first_consuntivation_date = first_consuntivation_date
)

dfg_payload = spark.sql(query).toPandas()

dfg_payload["request_date"] = pd.to_datetime(dfg_payload["request_date"])
missing_dates = [d for d in list_date_to_check if d not in dfg_payload["request_date"].dt.strftime("%Y-%m-%d").values]
print("Missing dates:", missing_dates)

ind = dfg_payload["RESPONSE_CATEGORY"] != "GREEN"
dfg_payload.loc[ind, :].tail(7)
# display(dfg_payload[["CF", "PIVA", "POD_PDR", "COD_FISCALE_PIVA_CLIIN", "PK_VOLTURA", "PK_REQUEST","request"]])

# COMMAND ----------

# DBTITLE 1,payload group by month
dfg_payload_monthly = monthly_table_aggregator(
    dfg_payload.copy(),
    date="request_date",
    nr="NR_VOLTURE", #"NR_REQUEST",
    model={"USED_MODEL": ["DETERMINISTIC", "SCORING"]},
    colors={"RESPONSE_CATEGORY": ["RED", "YELLOW", "GREEN", "BAD_REQUEST"]},
)

dfg_payload_monthly

# COMMAND ----------

# MAGIC %md
# MAGIC # Output result

# COMMAND ----------

output_result_path

# COMMAND ----------

# DBTITLE 1,output result group by day date
query = """
WITH tab as (
            SELECT *
            , regexp_replace(split_part(split_part(request, r'CodiceFiscale', 2), r'",', 1), '[^A-Za-z0-9]', '') AS CF
            , regexp_replace(split_part(split_part(request, r'PartitaIva', 2), r'",', 1), '[^A-Za-z0-9]', '') AS PIVA
            , regexp_replace(split_part(split_part(request, r'POD/PDR', 2), r',',1), '[^A-Za-z0-9]', '') AS POD_PDR_FIRST
            , regexp_replace(split_part(split_part(request, r'POD/PDR', 3), r',',1), '[^A-Za-z0-9]', '') AS POD_PDR_BIS
            , regexp_count(request, 'POD/PDR') AS N_PODPDR

            FROM {table_path}
            WHERE
            request_date >= '2025-05-13' -- data di messa in produzione
)
, tab2 AS (
    SELECT *
        , databricks_request_id AS PK_REQUEST
        , concat_ws('_', IF(CF != "", CF, PIVA), POD_PDR_FIRST, POD_PDR_BIS, N_PODPDR) AS PK_VOLTURA
    FROM tab
)
, agg AS (
    SELECT request_date, USED_MODEL, esitoSemaforico
        , COUNT(DISTINCT PK_REQUEST) as NR_REQUEST
        , COUNT(DISTINCT PK_VOLTURA) as NR_VOLTURE
    FROM tab2
    GROUP BY request_date, USED_MODEL, esitoSemaforico
    ORDER BY request_date,  USED_MODEL
)
SELECT *
FROM agg
""".format(
    table_path=output_result_path
)

dfg_output = spark.sql(query).toPandas()

dfg_output["request_date"] = pd.to_datetime(dfg_output["request_date"])
missing_dates = [d for d in list_date_to_check if d not in dfg_output["request_date"].dt.strftime("%Y-%m-%d").values]
print("Missing dates:", missing_dates)

ind = dfg_output["esitoSemaforico"] != "Semaforo Verde"

dfg_output["esitoSemaforico"] = dfg_output["esitoSemaforico"].apply(lambda x: x.split("Semaforo ")[-1])
dfg_output.rename(columns={"esitoSemaforico": "RESPONSE_CATEGORY"}, inplace=True)

print(
    "La data odierna non viene conteggiata nella tabella di output dei risultati perchè quest'ultima viene elaborata quotidianamente il giorno successivo!"
)
dfg_output.loc[ind, :].tail(7)

# COMMAND ----------

# DBTITLE 1,output result group by day month
dfg_output_monthly = monthly_table_aggregator(
    dfg_output.copy(),
    date="request_date",
    nr="NR_VOLTURE", #"NR_REQUEST",
    model={"USED_MODEL": ["DETERMINISTIC", "SCORING", "NO ONE"]},
    colors={"RESPONSE_CATEGORY": ["Rosso", "Giallo", "Verde", "Errore"]},
)

print("Il 19 Marzo 2026 è stato fatto rilascio in produzione P3 che ha portato la messa in esercizio del nuovo model, e solo al 30 Marzo 2026 è stato sistemato il monit_daily job che registra i dati giornalieri nella tabella degli output results, pertanto è stato registrato 1 solo giorno.\n")
print("La data dalla quale si inizia a consuntivare è", first_consuntivation_date)
print("\nMancano invece i record registrati nella giornata odierna perchè ancora non sono stati elaborati")

dfg_output_monthly

# COMMAND ----------

# MAGIC %md
# MAGIC # Consuntivations

# COMMAND ----------

# DBTITLE 1,saving parameters tracking
# Test set informations to retrieve by the risk_classification.png barplot in the experiment evaluation_test_run
date_col = "barplot_snap_date"

savings_parameters = {
    "2025-03": {  # MONTH OF REFERENCE
        date_col: "2025-04-01",
        "red_predicted_cases": 28,
        "yellow_predicted_cases": 32,
        "red_hp_saving": 39181,
        "yellow_hp_saving": 33005,  # questo valore è stato stimato male
    },
    "2025-05": {
        date_col: "2025-06-01",
        "red_predicted_cases": 21,
        "yellow_predicted_cases": 27,
        "red_hp_saving": 28435,
        "yellow_hp_saving": 5766,
    },
    "2025-06": {
        date_col: "2025-07-01",
        "red_predicted_cases": 24,
        "yellow_predicted_cases": 32,
        "red_hp_saving": 21244,
        "yellow_hp_saving": 5612,
    },
    "2025-08": {
        date_col: "2025-09-01",
        "red_predicted_cases": 16,
        "yellow_predicted_cases": 11,
        "red_hp_saving": 21603,
        "yellow_hp_saving": 14200,
    },
    "2025-09": {
        date_col: "2025-10-01",
        "red_predicted_cases": 19,
        "yellow_predicted_cases": 40,
        "red_hp_saving": 21684,
        "yellow_hp_saving": 20633,
    },
    "2025-12": {
        date_col: "2026-01-01",
        "red_predicted_cases": 18,
        "yellow_predicted_cases": 19,
        "red_hp_saving": 18271,
        "yellow_hp_saving": 41705,
    },
    "2026-03": {
        date_col: "2026-04-01",
        "red_predicted_cases": 15 ,
        "yellow_predicted_cases": 20,
        "red_hp_saving": 13992,
        "yellow_hp_saving": 45060,
    },
    "2026-05": { # in anticipo rispetto alla fine del 2° quarto del 2026
        date_col: "2026-06-01",
        "red_predicted_cases": 27 ,
        "yellow_predicted_cases": 11,
        "red_hp_saving": 27376,
        "yellow_hp_saving": 4573,
    },
    "2026-06": {
        date_col: "2026-07-01",
        "red_predicted_cases": 24 ,
        "yellow_predicted_cases": 26,
        "red_hp_saving": 23258,
        "yellow_hp_saving": 8656,
    },
}
df_savings_parameters = pd.DataFrame.from_dict(savings_parameters, orient="index")
df_savings_parameters.index.name = "YEAR_MONTH"


# Saving rates calculations
df_savings_parameters["red_saving_rate"] = (
    df_savings_parameters["red_hp_saving"] / df_savings_parameters["red_predicted_cases"]
)
df_savings_parameters["yellow_saving_rate"] = (
    df_savings_parameters["yellow_hp_saving"] / df_savings_parameters["yellow_predicted_cases"]
)

df_savings_parameters.reset_index(inplace=True)

# Add missing YEAR_MONTH rows from 2025-01 to the last present in the dataframe
all_months = pd.date_range("2025-01-01", df_savings_parameters["YEAR_MONTH"].max() + "-01", freq="MS").strftime("%Y-%m")
df_savings_parameters = df_savings_parameters.set_index("YEAR_MONTH").reindex(all_months).reset_index().rename(columns={"index": "YEAR_MONTH"})
df_savings_parameters['barplot_snap_date'] = df_savings_parameters['YEAR_MONTH'].str[:7]

df_savings_parameters["QUARTER"] = pd.to_datetime(df_savings_parameters["YEAR_MONTH"] + "-01").dt.to_period("Q")

df_savings_parameters.fillna(method='bfill', inplace=True)

# # Group by quarter of the year
# df_savings_parameters = df_savings_parameters.loc[df_savings_parameters.groupby("QUARTER")[date_col].idxmax()]
df_savings_parameters

# COMMAND ----------

# DBTITLE 1,monthly consuntivation elaboration
dfg = dfg_payload_monthly.copy()
# dfg = dfg_output_monthly.copy()

dfg["QUARTER"] = pd.to_datetime(dfg["YEAR_MONTH"] + "-01").dt.to_period("Q")

df_savings = dfg.merge(df_savings_parameters, how="left", on=["YEAR_MONTH","QUARTER"]
                       )

df_savings["RED_SAVING"] = (df_savings["red_saving_rate"] * df_savings["RED"]).round(2)
df_savings["YELLOW_SAVING"] = (df_savings["yellow_saving_rate"] * df_savings["YELLOW"]).round(2)
df_savings["TOTAL_SAVING"] = (df_savings["RED_SAVING"].fillna(0) + df_savings["YELLOW_SAVING"].fillna(0)).round(1)

df_savings

# COMMAND ----------

# Dataframe to visualize
df_savings[
    [
        "QUARTER",
        "red_saving_rate",
        "yellow_saving_rate",
        "YEAR_MONTH",
        "RED",
        "YELLOW",
        "RED_SAVING",
        "YELLOW_SAVING",
        "TOTAL_SAVING",
    ]
]

# COMMAND ----------

# DBTITLE 1,quarterly consuntivation elaboration
df_savings_quarter = (
    df_savings.groupby("QUARTER")
    .agg(
        {
            "YEAR_MONTH": "count",
            "NR_VOLTURE": "sum", #"NR_REQUEST": "sum",
            "DETERMINISTIC": "sum",
            "SCORING": "sum",
            "BAD_REQUEST": "sum",
            "GREEN": "sum",
            "RED": "sum",
            "YELLOW": "sum",
            "RED_SAVING": "sum",  # lambda x: x.sum() if x.notna().all() else float('nan'),
            "YELLOW_SAVING": "sum",  # lambda x: x.sum() if x.notna().all() else float('nan')
            "TOTAL_SAVING": "sum",
        }
    )
    .reset_index()
    .rename(columns={"YEAR_MONTH": "NR_RECORDED_MONTHS"})
)

df_savings_quarter = df_savings_quarter.round(1)
df_savings_quarter

# COMMAND ----------

# MAGIC %md
# MAGIC # Pick red and yellow cases POST-release

# COMMAND ----------

# DBTITLE 1,red cases
# Get the red cases
ind = dfg_output["RESPONSE_CATEGORY"] == "Rosso"
nr_red_cases = dfg_output.loc[ind, "NR_VOLTURE"].sum()
print(nr_red_cases, "number of red cases")
dfg_output.loc[ind, :]

# COMMAND ----------

# DBTITLE 1,yellow cases
# Get the yellow cases
ind = dfg_output["RESPONSE_CATEGORY"] == "Giallo"
nr_yellow_cases_tot = dfg_output.loc[ind, "NR_VOLTURE"].sum()
print(nr_yellow_cases_tot, "whole number of yellow cases")

# Get the yellow cases of Scoring Model
dfg_yellow = dfg_output.loc[ind, :]

ind = dfg_yellow["USED_MODEL"] == "SCORING"
nr_yellow_cases_scoring = dfg_yellow.loc[ind, "NR_VOLTURE"].sum()
print(nr_yellow_cases_scoring, "number of yellow cases of Scoring model")
print(nr_yellow_cases_tot - nr_yellow_cases_scoring, "number of yellow cases of Deterministic model")

dfg_yellow

# COMMAND ----------

# MAGIC %md
# MAGIC # Deterministic's yellow cases ANTE-release

# COMMAND ----------

endpoint_payload_path

# COMMAND ----------

# DBTITLE 1,yellow deterministic cases
query = """
WITH tab as (
            SELECT *
            , regexp_replace(split_part(split_part(request, r'CodiceFiscale', 2), r'",', 1), '[^A-Za-z0-9]', '') AS CF
            , regexp_replace(split_part(split_part(request, r'PartitaIva', 2), r'",', 1), '[^A-Za-z0-9]', '') AS PIVA
            , regexp_replace(split_part(split_part(request, r'POD/PDR', 2), r',',1), '[^A-Za-z0-9]', '') AS POD_PDR
            , regexp_replace(split_part(split_part(request, r'POD/PDR', 3), r',',1), '[^A-Za-z0-9]', '') AS POD_PDR_BIS
            , regexp_count(request, 'POD/PDR') AS N_PODPDR

            , CASE
                WHEN response LIKE "%BAD_REQUEST%" THEN "BAD_REQUEST"
                WHEN response LIKE "%Semaforo Rosso%" THEN "RED"
                WHEN response LIKE "%Semaforo Giallo%" THEN "YELLOW"
                WHEN response LIKE "%Semaforo Verde%" THEN "GREEN"
            END AS `RESPONSE_CATEGORY`
            , CASE
                WHEN response LIKE "%BAD_REQUEST%" THEN "BAD_REQUEST"
                WHEN response LIKE "%Semaforo%" AND response LIKE '%"score": 0.%' THEN "SCORING"
                WHEN response LIKE "%Semaforo%" AND response LIKE '%"score": -1%' THEN "SCORING" -- actually NO ONE model has used
                WHEN response LIKE "%Semaforo%" AND response LIKE '%"score": 1%' THEN "DETERMINISTIC"
                WHEN response LIKE "%Semaforo%" AND response LIKE '%"score": 0%' THEN "DETERMINISTIC"
                ELSE ""
            END AS `USED_MODEL`
            FROM {table_path}
            WHERE
            response NOT LIKE "%Semaforo Verde%"
            AND
            response NOT LIKE "%BAD_REQUEST%"
            AND
            --request_date <= '2025-05-13' -- OLD data di messa in esercizio
            request_date <= '2026-03-30' -- data di fix monit daily job dopo la messa in esercizio del '2026-03-19'

            AND request NOT RLIKE 'XXXXXX00X00X000X|MRNDVD89C22H501B|12345678910|ebb1868a42d1e45efc7a8f0d862acc61|01234567891011'
)
, tab2 AS (
    SELECT *
        , databricks_request_id AS PK_REQUEST
        , concat_ws('_', IF(CF != "", CF, PIVA), POD_PDR, POD_PDR_BIS, N_PODPDR) AS PK_VOLTURA
        , IF(CF != "", CF, PIVA) AS COD_FISCALE_PIVA_CLIIN
    FROM tab
)
, aggr AS (
    SELECT request_date, RESPONSE_CATEGORY, USED_MODEL
        , COUNT(DISTINCT PK_REQUEST) as NR_REQUEST
        , COUNT(DISTINCT PK_VOLTURA) as NR_VOLTURE
    FROM tab2
    GROUP BY request_date, RESPONSE_CATEGORY, USED_MODEL
    ORDER BY request_date, RESPONSE_CATEGORY, USED_MODEL
)
SELECT *
FROM aggr
""".format(
    table_path=endpoint_payload_path
)

dfg_ante_release = spark.sql(query).toPandas()
dfg_ante_release
