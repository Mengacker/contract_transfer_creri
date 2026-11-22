# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook initialization and setting

# COMMAND ----------

# DBTITLE 1,Set notebook extensions
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# DBTITLE 1,Set working directory
import os
import sys
from databricks.sdk.runtime import dbutils

notebook_path = "/Workspace" + os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get() #.replace("/notebooks", "")

)
sys.path.append("..")
os.chdir(notebook_path)
print(notebook_path)

# COMMAND ----------

# DBTITLE 1,Install requirements
# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import sys
from databricks.sdk.runtime import dbutils

notebook_path = "/Workspace" + os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get() #.replace("/notebooks", "")

)
sys.path.append("..")
os.chdir(notebook_path)
print(notebook_path)

# COMMAND ----------

# DBTITLE 1,Import  class from a notebook
# MAGIC %run ./materialization_utils/materialization_utils

# COMMAND ----------

# DBTITLE 1,Import libraries
import logging
from pyspark.sql import DataFrame, functions as F
from datetime import datetime
import pandas as pd
from pandas import DataFrame as pandasDataFrame
from typing import Dict
import json
import ast
import hashlib

from src.masters.inferencing_masters import materializationInferenceMasterVolturante, materializationInferenceMasterVolturato
from src.data_preparation.alter_table import alter_table

from src.config.config_utils import Config

# COMMAND ----------

# DBTITLE 1,Notebook arguments
# --- Set notebook arguments --- #
# Reset widgets
dbutils.widgets.removeAll()

# Notebook Environment
dbutils.widgets.text("env", "prod-dpp", "Environment Name")  # dev, prod-sbx, prod-dpp
dbutils.widgets.dropdown("is_cicd", "true", ["true", "false"], "Is run by CICD")

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

dbutils.widgets.text("RunId", "", "RunId")
dbutils.widgets.text("cod_pipelineID", "", "cod_pipelineID")
dbutils.widgets.text("cod_productName", "", "cod_productName")

# --- Get notebook arguments ---#
env = dbutils.widgets.get("env")
is_cicd = bool(dbutils.widgets.get("is_cicd") == "true")
snapshot_date = dbutils.widgets.get("snapshot_date")

RunId = dbutils.widgets.get("RunId")
cod_pipelineID = dbutils.widgets.get("cod_pipelineID")
cod_productName = dbutils.widgets.get("cod_productName")

print("env:", env)
print("is_cicd", is_cicd)
print("snapshot_date:", snapshot_date)
print("RunId:", RunId)
print("cod_pipelineID:", cod_pipelineID)
print("cod_productName:", cod_productName)

# COMMAND ----------

# DBTITLE 1,Instantiate objects
general_utils = GeneralUtils(spark)
metric_materializer = MetricMaterializer(spark)

# feature_engineering_master = FeatureEngineeringMaster(spark)
logger = general_utils.set_logging()

materialization_inference_master_volturante = materializationInferenceMasterVolturante(spark, logger)
materialization_inference_master_volturato = materializationInferenceMasterVolturato(spark, logger)

snapshot_date = general_utils.date_to_start_month(snapshot_date)
print("snapshot_date:", snapshot_date)

# COMMAND ----------

# DBTITLE 1,New Retrieve config values
# Retrieve config values
config = Config(config_path="../src/config/", name_list=["config_setup.yml","config_data.yml"])

catalog = config.get_config_value("environment.{env}.catalog".format(env=env))
schema = config.get_config_value("environment.{env}.schema".format(env=env))
dp_path = f"{catalog}.{schema}."
print("catalog.schema:",dp_path,"\n")

encode_maps = {
    "SDF_CANLE_ATTIVAZ_ATT": config.get_config_value("input_features.MetricaSdfAttivazioneAttivi.SDF_CANLE_ATTIVAZ_ATT_CLIIN.encoding"),
    "MACROSEGMENTO": config.get_config_value("input_features.CustomBase.MACROSEGMENTO.encoding")
                }

tab_cliout_dp_path = config.get_config_value("master_inference")[0]['name']
tab_cliin_dp_path = config.get_config_value("master_inference")[1]['name']

tab_cliout_dp_path = dp_path+tab_cliout_dp_path
tab_cliin_dp_path = dp_path+tab_cliin_dp_path
print("tab_cliout_dp_path:", tab_cliout_dp_path)
print("tab_cliin_dp_path:",tab_cliin_dp_path)
print("")

if env == 'prod-sbx':
    oltab_cliout_dp_path = config.get_config_value("serving_endpoint.oltab_volturato.name_for_test")
    oltab_cliin_dp_path = config.get_config_value("serving_endpoint.oltab_volturante.name_for_test")
else:
    oltab_cliout_dp_path = config.get_config_value("serving_endpoint.oltab_volturato.name")
    oltab_cliin_dp_path = config.get_config_value("serving_endpoint.oltab_volturante.name")

oltab_cliout_dp_path = dp_path+oltab_cliout_dp_path
oltab_cliin_dp_path = dp_path+oltab_cliin_dp_path
print("oltab_cliout_dp_path:",oltab_cliout_dp_path)
print("oltab_cliin_dp_path:",oltab_cliin_dp_path)

# COMMAND ----------

# DBTITLE 1,Parameters choice
flg_lastSnap = 1
version = 1 # class_obj.get_version()

# COMMAND ----------

# MAGIC %md
# MAGIC # Inference master CLI-IN

# COMMAND ----------

# DBTITLE 1,Elaborate CLI-IN
df_CLIIN = materialization_inference_master_volturante.get_dataframe_as_of(snapshot_date, catalog, schema, general_utils, metric_materializer)
print("Shape:", (df_CLIIN.count(), len(df_CLIIN.columns)))
display(df_CLIIN.limit(5))

# COMMAND ----------

# DBTITLE 1,Postprocessing CLI-IN
# df_CLIIN = metric_materializer.column_renaming(df_CLIIN, logger)

schema_dati = metric_materializer.remove_tech_columns(spark.read.table(tab_cliin_dp_path)).schema
schema_map = {field.name: field.dataType for field in schema_dati.fields}
common_cols, df_only_cols = metric_materializer.get_schema_fields(schema_map, df_CLIIN)
df_CLIIN = df_CLIIN.select(*common_cols)
df_CLIIN = metric_materializer.cast_columns(common_cols, schema_map, df_CLIIN)

volturante_configs = [x for x in config.get_config_value('master_inference') if "volturante" in x['name']][0]
metrics_used_list_volturante = metric_materializer.get_metrics_used(config, volturante_configs)
hash_data_version_volturante = metric_materializer.get_met_version(catalog, schema, metrics_used_list_volturante, volturante_configs['class_name'], 1, snapshot_date)

logger.info(f"Generating tech columns")
df_CLIIN = metric_materializer.add_tech_columns(
                df_CLIIN,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                RunId,
                hash_data_version_volturante,
                flg_lastSnap,
                ['COD_FISCALE_PIVA_CLIIN'],
                snapshot_date)
print("Shape:", (df_CLIIN.count(), len(df_CLIIN.columns)))
display(df_CLIIN.limit(5))

# COMMAND ----------

# DBTITLE 1,write data CLI-IN
if is_cicd:
    df_CLIIN.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(tab_cliin_dp_path)
    print("Table written", tab_cliin_dp_path)
else:
    print("Skipping table write of", tab_cliin_dp_path)

# COMMAND ----------

# DBTITLE 1,write versioned data CLI-IN
if is_cicd:
       try:
              metric_versioning = metric_materializer.metric_versioning(catalog, schema, tab_cliin_dp_path+"_ver", snapshot_date, "versioned")
              logger.info(f"{metric_versioning}")
       except Exception as e:
              logger.error(f"Error during metric versioning: {e}")
              pass
       df_CLIIN.write.mode("append").option("mergeSchema", "true").saveAsTable(tab_cliin_dp_path+"_ver")
       print("Table written", tab_cliin_dp_path+"_ver")
else:
       print("Skipping table write of", tab_cliin_dp_path+"_ver")

# COMMAND ----------

# MAGIC %md
# MAGIC # Inference master CLI-OUT

# COMMAND ----------

# DBTITLE 1,Elaborate CLI-OUT
df_CLIOUT  = materialization_inference_master_volturato.get_dataframe_as_of(snapshot_date,
                                                                            catalog,
                                                                            schema,
                                                                            encode_maps,
                                                                            general_utils,
                                                                            metric_materializer)
print("Shape:", (df_CLIOUT.count(), len(df_CLIOUT.columns)))
display(df_CLIOUT.limit(5))

# COMMAND ----------

count_not_null = df_CLIOUT.filter(F.col("PDF_INSOLUTO_LORDO_ALL").isNotNull()).count()
print("Number of not null values in PDF_INSOLUTO_LORDO_ALL:", count_not_null)
count_not_null = df_CLIOUT.filter(F.col("CLI_INSOLUTO_LORDO").isNotNull()).count()
print("Number of not null values in CLI_INSOLUTO_LORDO:", count_not_null)

# COMMAND ----------

# DBTITLE 1,Postprocessing CLI-OUT
logger.info(f"Casting fields for {tab_cliout_dp_path}")
schema_dati = metric_materializer.remove_tech_columns(spark.read.table(tab_cliout_dp_path)).schema
schema_map = {field.name: field.dataType for field in schema_dati.fields}
common_cols, df_only_cols = metric_materializer.get_schema_fields(schema_map, df_CLIOUT)
df_CLIOUT = df_CLIOUT.select(*common_cols)
df_CLIOUT = metric_materializer.cast_columns(common_cols, schema_map, df_CLIOUT)

volturato_configs = [x for x in config.get_config_value('master_inference') if "volturato" in x['name']][0]
metrics_used_list_volturato = metric_materializer.get_metrics_used(config, volturato_configs)
hash_data_version_volturato = metric_materializer.get_met_version(catalog, schema, metrics_used_list_volturato, volturato_configs['class_name'], 1, snapshot_date)

logger.info(f"Generating tech columns")
df_CLIOUT = metric_materializer.add_tech_columns(
                df_CLIOUT,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                RunId,
                hash_data_version_volturato,
                flg_lastSnap,
                ['POD_PDR'],
                snapshot_date)
print("Shape:", (df_CLIOUT.count(), len(df_CLIOUT.columns)))
display(df_CLIOUT.limit(5))

# COMMAND ----------

# DBTITLE 1,write data CLI-OUT
if is_cicd:
    df_CLIOUT.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(tab_cliout_dp_path)
    print("Table written", tab_cliout_dp_path)
else:
    print("Skipping table write of", tab_cliout_dp_path)

# COMMAND ----------

# DBTITLE 1,write versioned data CLI-OUT
if is_cicd:
       try:
              # spark.sql(f"DELETE FROM {tab_cliout_dp_path}_ver where snapshot_date = '{snapshot_date}'")
              metric_versioning = metric_materializer.metric_versioning(catalog, schema, tab_cliout_dp_path+"_ver", snapshot_date, "versioned")
              logger.info(f"{metric_versioning}")
       except Exception as e:
              logger.error(f"Error during metric versioning: {e}")
              pass
       df_CLIOUT.write.mode("append").option("mergeSchema", "true").saveAsTable(tab_cliout_dp_path+"_ver")
       print("Table written", tab_cliout_dp_path+"_ver")
else:
       print("Skipping table write of", tab_cliout_dp_path+"_ver")

# COMMAND ----------

# MAGIC %md
# MAGIC # PREPARE SYNC-TABLES (EX ONLINE-TABLES)

# COMMAND ----------

# DBTITLE 1,ALTER  DELTA TABLES
tables_dict = {
    "POD_PDR": {"delta_table": tab_cliout_dp_path, "sync_table": oltab_cliout_dp_path}
    ,
    "COD_FISCALE_PIVA_CLIIN": {"delta_table": tab_cliin_dp_path, "sync_table": oltab_cliin_dp_path}
}

for pk_col, diz in tables_dict.items():
    tab = diz['delta_table']
    oltab = diz['sync_table']
    print(
        f"\n...............................ALTER DELTA TABLE: {tab}......................................"
    )
    print(oltab, 'name of the sync-table (ex on-line table)')
    l_messages = alter_table(pk_col=[pk_col], table_path=tab)
    [print(f'\n{m}\n') for m in l_messages]

# COMMAND ----------

# DBTITLE 1,REFRESH SYNC-TABLE
# for pk_col, diz in tables_dict.items():
#     tab = diz['delta_table']
#     oltab = diz['sync_table']

#     oltab_dp_path = f"'{oltab}'"
#     spark.sql(f"CALL REFRESH_TABLE({oltab_dp_path})")

# COMMAND ----------

dbutils.notebook.exit("Success!!")

# COMMAND ----------

# MAGIC %md
# MAGIC # CODE FOR MANUAL FIX

# COMMAND ----------

# DBTITLE 1,Tabs list
# tab_lst = [
#     "creri.creri_met_morositapregressacliente",
#     "creri.creri_met_morositapregressapodpdr",
#     "creri.creri_met_dataultletturaattivswinpwr",
#     "creri.creri_met_dataultletturaattivswingas",
#     "creri.creri_met_metricapdfpodpdrgtwv",
#     "creri.creri_met_clienteattributi",
#     "creri.creri_met_accounttocodfiscalepiva",
#     "creri.creri_met_sdfattivazioneattivi"
# ]

# COMMAND ----------

# DBTITLE 1,DELETE snapshot_date
# for tab in tab_lst:
#     spark.sql(f"DELETE FROM {tab} WHERE snapshot_date = '2026-03-01'")

# COMMAND ----------

# DBTITLE 1,READ snapshot_date from p3 to p1
# for tab in tab_lst:
#     spark.read.table(f"p_dpp_core_ai.{tab}").where("snapshot_date = '2026-03-01'").write.mode("append").option("mergeSchema", "true").saveAsTable(f"p_ind_core_ai.{tab}")

# COMMAND ----------

# DBTITLE 1,check metriche per snapshot_date
# for tab in tab_lst:
#     print(f"{tab}")
#     max_date = (
#         spark.read.table(f"p_ind_core_ai.{tab}")
#         .select(F.max("snapshot_date").alias("max_date"))
#         .first()["max_date"]
#     )
#     print(f"Max date: {max_date}")
#     print(spark.read.table(f"p_ind_core_ai.{tab}").where(f"snapshot_date = '{max_date}' and flg_lastSnap = 1").count())

# COMMAND ----------

spark.read.table(tab_cliin_dp_path).display()

# COMMAND ----------

spark.read.table(tab_cliout_dp_path).display()
