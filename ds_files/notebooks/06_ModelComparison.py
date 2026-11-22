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

# DBTITLE 1,Install requirements
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
from datetime import datetime
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.utils.input_validation import is_iso_date
from src.data_preparation.model_pre_processing import modeling_pre_processer
from src.deployment.model_comparison import compare_models

from src.config.config_utils import Config

import logging

logger = logging.getLogger("py4j")
logger.setLevel(logging.ERROR)

# COMMAND ----------

# DBTITLE 1,Notebook arguments
# Reset widgets
dbutils.widgets.removeAll()

# Get job's task arguments
# training_run_id = dbutils.jobs.taskValues.get("ModelTraining", "training_run_id", debugValue=0)

# Set notebook widgets
dbutils.widgets.text("env", "dev", "Environment Name")
dbutils.widgets.dropdown("is_cicd", "true", ["true", "false"], "Is run by CICD")

config_setup = Config("../src/config/config_setup.yml")
dev_experiment_path = config_setup.get_config_value("environment.dev.experiment_path")
dbutils.widgets.text("experiment_path", dev_experiment_path, "Experiment path name read by asset workflow file yaml")

# date_and_time = datetime.now() # sballa l'orario di 2 ore perchè usa il riferimento UTC
timezone = "Europe/Amsterdam"
date_and_time = spark.sql("SELECT current_timestamp() as current_time").withColumn(
    "current_time", F.from_utc_timestamp("current_time", timezone)
)
date_and_time = date_and_time.collect()[0][0]
dbutils.widgets.text(
    "snapshot_date",
    str(date_and_time.date()),
    label="Snapshot date for the run",
)
# dbutils.widgets.dropdown(
#     "macrosegmento", "ALL", ["ALL", "RESIDENZIALI", "MICROBUSINESS"], "Macrosegmento to filter data to use"
# )

# Get notebook arguments
# training_run_id = dbutils.widgets.get("training_run_id")
env = dbutils.widgets.get("env")
is_cicd = bool(dbutils.widgets.get("is_cicd") == "true")
experiment_path = dbutils.widgets.get("experiment_path")
# model_alias = dbutils.widgets.get("model_alias")
# macrosegmento = dbutils.widgets.get("macrosegmento")
snapshot_date_str = dbutils.widgets.get("snapshot_date")
snapshot_date_str = snapshot_date_str[:-2] + "01"

# Validate input
if not is_iso_date(snapshot_date_str):
    raise ValueError(f"Invalid snapshot_date: {snapshot_date_str}")

print(env, "env")
print(is_cicd, "is_cicd")
print("Current date and time:", date_and_time.strftime("%Y-%m-%d %H:%M:%S"), "at the {t} timezone".format(t=timezone))
print(snapshot_date_str)

# COMMAND ----------

print(dev_experiment_path)
print(experiment_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load master and data preparation

# COMMAND ----------

# DBTITLE 1,Retrieve config values
# Retrieve config values
config_setup = Config("../src/config/config_setup.yml")

master_name = config_setup.get_config_value("master_training_final")[0]["name"]
catalog = config_setup.get_config_value("environment.{env}.catalog".format(env=env))

if env == "dev-old":
    schema = config_setup.get_config_value("environment.{env}.schema_atum".format(env=env))
    table_full_name = f"{catalog}.{schema}.vw_{master_name}"
else:
    schema = config_setup.get_config_value("environment.{env}.schema".format(env=env))
    table_full_name = f"{catalog}.{schema}.{master_name}"

print(f"table_full_name: '{table_full_name}'\n")

# COMMAND ----------

# DBTITLE 1,Pre-processing
y_test, X_test, WHOLE_master = modeling_pre_processer(
    table_catalog_path=table_full_name,
    utility_set=["test"],
    oversampling=False,
    # params={"macrosegmento": macrosegmento},
)

df_test = pd.concat([y_test, X_test], axis=1).reset_index(drop=True)

target = y_test.name
predictors = list(X_test.columns)
print(f"'{target}' is the target field\n")
print(f"n°{len(predictors)} predictors: {predictors}\n")

extra_features = [e for e in list(WHOLE_master.columns) if e not in predictors + [target]]
print(f"n°{len(extra_features)} extra_features: {extra_features}\n")

# COMMAND ----------

WHOLE_master.sort_values(by="PDF_INSOLUTO_LORDO_ALL", ascending=False)

# COMMAND ----------

WHOLE_master["snapshot_date"].unique()

# COMMAND ----------

df_test.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Comparison and deployment

# COMMAND ----------

# df_test[target]=df_test[target].astype(int) # from boolean to integer

run_id, validator_report = compare_models(
    eval_data=df_test, target=target, predictors=predictors, experiment_path=experiment_path, env=env
)
run_id

# COMMAND ----------

validator_report

# COMMAND ----------

validator_report["params"]
