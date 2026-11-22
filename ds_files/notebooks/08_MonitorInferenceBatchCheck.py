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
from datetime import date, datetime, timedelta
from pyspark.sql.functions import lit, current_timestamp
from pyspark.sql import functions as F

from src.serving.batch_inference import batch_inference
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

# DBTITLE 1,Retrive config params
# Retrieve config values
config_setup = Config("../src/config/config_setup.yml")
catalog = config_setup.get_config_value("environment.{env}.catalog".format(env=env))
schema = config_setup.get_config_value("environment.{env}.schema".format(env=env))

# Payload table
endpoint_payload_table = config_setup.get_config_value("inference_monitoring")[0]["name"]
endpoint_payload_path = f"{catalog}.{schema}.{endpoint_payload_table}"
print(f"{endpoint_payload_path}   table of the payload logs")

# Final Output table
output_result_table = config_setup.get_config_value("inference_monitoring")[1]["name"]
output_result_path = f"{catalog}.{schema}.{output_result_table}"
print(f"{output_result_path}   table of the final output")

# COMMAND ----------

# DBTITLE 1,input data
query = """
WITH tab as (
            SELECT *
            FROM {table_path}
)
SELECT *
FROM tab
""".format(
    table_path=output_result_path,
)

df_input = spark.sql(query)
df_input = df_input.toPandas()
df_input.rename(
    columns={"esitoSemaforico": "esitoSemaforico_rt", "score": "score_rt", "USED_MODEL": "USED_MODEL_rt"}, inplace=True
)
print(df_input.shape)
df_input.head(2)

# COMMAND ----------

# DBTITLE 1,list predictors
# Check the presence of predictors
predictors = [  # predittori al 19-set-2025
    "FLG_COMTY_GAS",
    "FLG_COGNOME_UGUALE",
    "FLG_EMAIL_UGUALE",
    "FLG_TELEFONO_UGUALE",
    "FLG_COMUNE_UGUALE",
    "FLG_PROV_UGUALE",
    "FLG_REGIONE_UGUALE",
    "FLG_NAZ_UGUALE",
    "CLI_AGE_RANGE_18-30",
    "CLI_AGE_RANGE_30-40",
    "CLI_AGE_RANGE_40-50",
    "CLI_AGE_RANGE_50-60",
    "CLI_AGE_RANGE_60-70",
    "CLI_AGE_RANGE_70-80",
    "CLI_AGE_RANGE_80-200",
    "ENCOD_MACROSEGMENTO",
    "ENCOD_MACROSEGMENTO_CLIOUT",
    "IND_FORN_CAP_CLIOUT",
    "IND_FORN_COMUNE_CLIOUT",
    "IND_FORN_COD_PROV_CLIOUT",
    "IND_FORN_REGIONE_CLIOUT",
    "CLI_PROV_CLIOUT",
    "CLI_COM_CLIOUT",
    "CLI_REG_CLIOUT",
    "ENCOD_SDF_CANLE_ATTIVAZ_ATT",
    "PDF_STATO_CONTO_CRM_LAST",
    "PDF_AGING_SCAD_RATE_LAST",
    "PDF_NR_FATTURE_INSOLUTE_LAST",
    "PDF_NR_RATE_INSOLUTE_LAST",
    "PDF_INSOLUTO_LORDO_LAST",
    "PDF_INSOLUTO_NETTO_LAST",
    "PDF_INSOLUTO_CANONE_TV_LAST",
    "PDF_INSOLUTO_R_LAST",
    "PDF_NR_FORN_MOROSE_ALL",
    "PDF_AGING_SCAD_RATE_ALL",
    "PDF_NR_FATTURE_INSOLUTE_ALL",
    "PDF_NR_RATE_INSOLUTE_ALL",
    "PDF_INSOLUTO_LORDO_ALL",
    "PDF_INSOLUTO_NETTO_ALL",
    "PDF_INSOLUTO_CANONE_TV_ALL",
    "PDF_INSOLUTO_R_ALL",
    "PDF_INSOLUTO_6_MM_ALL",
    "PDF_INSOLUTO_1_ANNI_ALL",
    "PDF_INSOLUTO_2_ANNI_ALL",
    "PDF_INSOLUTO_3_ANNI_ALL",
    "PDF_INSOLUTO_5_ANNI_ALL",
    "PDF_INSOLUTO_PIU_ANNI_ALL",
    "CLI_NR_CC_FORN_MOROSE",
    "CLI_AGING_SCAD_RATE_OLDER",
    "CLI_NR_FATTURE_INSOLUTE",
    "CLI_NR_RATE_INSOLUTE",
    "CLI_INSOLUTO_LORDO",
    "CLI_INSOLUTO_NETTO",
    "CLI_INSOLUTO_CANONE_TV",
    "CLI_INSOLUTO_NETTO_R",
    "CLI_INSOLUTO_6_MM",
    "CLI_INSOLUTO_1_ANNI",
    "CLI_INSOLUTO_2_ANNI",
    "CLI_INSOLUTO_3_ANNI",
    "CLI_INSOLUTO_5_ANNI",
    "CLI_INSOLUTO_PIU_ANNI",
    "CLI_NR_CC_FORN_MOROSE_CLIOUT",
    "CLI_AGING_SCAD_RATE_OLDER_CLIOUT",
    "CLI_NR_FATTURE_INSOLUTE_CLIOUT",
    "CLI_NR_RATE_INSOLUTE_CLIOUT",
    "CLI_INSOLUTO_LORDO_CLIOUT",
    "CLI_INSOLUTO_NETTO_CLIOUT",
    "CLI_INSOLUTO_CANONE_TV_CLIOUT",
    "CLI_INSOLUTO_NETTO_R_CLIOUT",
    "CLI_INSOLUTO_6_MM_CLIOUT",
    "CLI_INSOLUTO_1_ANNI_CLIOUT",
    "CLI_INSOLUTO_2_ANNI_CLIOUT",
    "CLI_INSOLUTO_3_ANNI_CLIOUT",
    "CLI_INSOLUTO_5_ANNI_CLIOUT",
    "CLI_INSOLUTO_PIU_ANNI_CLIOUT",
    "AGING_METER_ATTIV_SWIN",
]
predictors_to_eng = [
    "FLG_COGNOME_UGUALE",
    "FLG_EMAIL_UGUALE",
    "FLG_TELEFONO_UGUALE",
    "ENCOD_MACROSEGMENTO",
    "AGING_METER_ATTIV_SWIN",
]
print("The predictors that will be engineered in real-time during inference:", predictors_to_eng)
missing_predictors = [c for c in predictors if c not in df_input.columns if c not in predictors_to_eng]
assert len(missing_predictors) == 0, print("missing_predictors:", missing_predictors)

# Check the presence of the columns to do the feature engineering in the serving model
cols_to_eng_missing_cols = [
    "TELEFONO",
    "TELEFONO_CLIOUT",
    "EMAIL",
    "EMAIL_CLIOUT",
    "COGNOME",
    "COGNOME_CLIOUT",
    "DATA_VOLTURA",
    "MACROSEGMENTO",
    "SegmentoVolturante",
    "ENCOD_SDF_CANLE_ATTIVAZ_ATT",
    "CanaleAcquisizione",
    "DATA_ULT_LETTURA_ATTIV_SWINPWR_CLIOUT",
    "DATA_ULT_LETTURA_ATTIV_SWINGAS_CLIOUT",
]  #  "DATA_ULT_LETTURA_ATTIV_SWIN",
print("Columns will be used to do feature engineering in the real-time inference:", cols_to_eng_missing_cols)
df_input[cols_to_eng_missing_cols]

# COMMAND ----------

# MAGIC %md
# MAGIC # Execute Inference

# COMMAND ----------

run_id, results, predictors, score_thresholds = batch_inference(
    data=df_input, model_use_mode="ab-test", is_cicd=is_cicd, env=env  # "ab-test" or "deterministic" or "scoring"
)
run_id

# COMMAND ----------

print(score_thresholds)
print(predictors)
results

# COMMAND ----------

# MAGIC %md
# MAGIC # Compare Output

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

date_col = "DATA_VOLTURA"
month_col = "year_month_VOLTURA"

results[month_col] = pd.to_datetime(results[date_col]).dt.strftime("%Y-%m")

# COMMAND ----------

results[
    [
        "request_PODPDR_id",
        # 'POD_PDR','COD_FISCALE_PIVA',
        # 'DATA_VOLTURA','year_month_VOLTURA',
        "USED_MODEL",
        "esitoSemaforico",
        "score",
        # 'USED_MODEL_rt','esitoSemaforico_rt','score_rt',
        # 'response',
        # 'ENCOD_MACROSEGMENTO','ENCOD_SDF_CANLE_ATTIVAZ_ATT'
    ]
].to_csv("results.csv", index=False)

# COMMAND ----------

# field = 'USED_MODEL'
# field_rt = field+'_rt'

# df = results[[month_col, field_rt, field]].copy()

# df = df.melt(id_vars=[month_col], value_vars=[field_rt, field], var_name='ModelType', value_name='Model')

# plt.figure(figsize=(12,6))
# ax = sns.countplot(data=df, x=month_col, hue='Model', palette='Set2', dodge=True)
# plt.xlabel(month_col)
# plt.ylabel('Count')
# plt.title(field_rt+' vs '+ field+'_rt')
# plt.legend(title=field)
# plt.xticks(rotation=45)
# plt.tight_layout()

# # Add count labels on top of bars
# for p in ax.patches:
#     height = p.get_height()
#     if height > 0:
#         ax.annotate(f'{int(height)}',
#                     (p.get_x() + p.get_width() / 2., height),
#                     ha='center', va='bottom', fontsize=10, color='black', rotation=0, xytext=(0, 3), textcoords='offset points')

# plt.show()

# COMMAND ----------

# field = 'esitoSemaforico'
# field_rt = field+'_rt'
# month_col = 'year_month_VOLTURA'

# df = results[[month_col, field_rt, field]].copy()

# ind = df['esitoSemaforico'] != 'Semaforo Verde'
# df = df.loc[ind, :]

# df = df.melt(id_vars=[month_col], value_vars=[field_rt, field], var_name='ModelType', value_name='Model')

# plt.figure(figsize=(12,6))
# ax = sns.countplot(data=df, x=month_col, hue='Model', palette='Set2', dodge=True)
# plt.xlabel(month_col)
# plt.ylabel('Count')
# plt.title(field_rt+' vs '+ field+'_rt')
# plt.legend(title=field)
# plt.xticks(rotation=45)
# plt.tight_layout()

# # Add count labels on top of bars
# for p in ax.patches:
#     height = p.get_height()
#     if height > 0:
#         ax.annotate(f'{int(height)}',
#                     (p.get_x() + p.get_width() / 2., height),
#                     ha='center', va='bottom', fontsize=10, color='black', rotation=0, xytext=(0, 3), textcoords='offset points')

# plt.show()
