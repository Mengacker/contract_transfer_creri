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

# DBTITLE 1,Set working directory
import os
import sys
from databricks.sdk.runtime import dbutils

notebook_path = "/Workspace" + os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)
sys.path.append("..")
os.chdir(notebook_path)
print(notebook_path)

# COMMAND ----------

# DBTITLE 1,Import libraries
import logging
from pyspark.sql import functions as F
from datetime import datetime
import hashlib

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

# DBTITLE 1,Retrieve config values
# Retrieve config values
config = Config(config_path="../src/config/", name_list=["config_setup.yml"])

catalog = config.get_config_value("environment.{env}.catalog".format(env=env))
schema = config.get_config_value("environment.{env}.schema".format(env=env))
dp_path = f"{catalog}.{schema}."
print("catalog.schema:",dp_path,"\n")

if env == 'prod-sbx':
    oltab_cliout_dp_path = config.get_config_value("serving_endpoint.oltab_volturato.name_for_test")
    oltab_cliin_dp_path = config.get_config_value("serving_endpoint.oltab_volturante.name_for_test")
else:
    oltab_cliout_dp_path = config.get_config_value("serving_endpoint.oltab_volturato.name")
    oltab_cliin_dp_path = config.get_config_value("serving_endpoint.oltab_volturante.name")

oltab_cliout_dp_path = dp_path+oltab_cliout_dp_path
oltab_cliin_dp_path = dp_path+oltab_cliin_dp_path
print("Sync-tables (ex online-tables):")
print("- oltab_cliout_dp_path:",oltab_cliout_dp_path)
print("- oltab_cliin_dp_path:",oltab_cliin_dp_path)

# COMMAND ----------

# MAGIC %md
# MAGIC # PREPARE SYNC-TABLES (EX ONLINE-TABLES)

# COMMAND ----------

# DBTITLE 1,REFRESH SYNC-TABLES
sync_table_1 = oltab_cliout_dp_path
sync_table_2 = oltab_cliin_dp_path

print("#################################################################################################")
print("#################################################################################################")
print("#################################################################################################")
print("-------------------------------------------------------------------------------------------------")
print("------- INSERIRE QUI IL CODICE PER AVVIARE LA PIPELINE DI AGGIORNAMENTO DELLE SYNC-TABLES -------")
print("-------------------------------------------------------------------------------------------------")
print("#################################################################################################")
print("#################################################################################################")
print("#################################################################################################")


# COMMAND ----------

# DBTITLE 1,exit
dbutils.notebook.exit("Success!!")
