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
from pyspark.sql import functions as F

from src.config.config_utils import Config

from src.utils.input_validation import is_iso_date

from src.data_preparation.model_pre_processing import modeling_pre_processer
from src.model.model_tuning import find_best_params

# COMMAND ----------

# DBTITLE 1,Notebook arguments
# Reset widgets
dbutils.widgets.removeAll()

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
env = dbutils.widgets.get("env")
is_cicd = bool(dbutils.widgets.get("is_cicd") == "true")
experiment_path = dbutils.widgets.get("experiment_path")
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
# MAGIC ## Load master and Prepare data

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
y_full_train, X_full_train, WHOLE_master = modeling_pre_processer(
    table_catalog_path=table_full_name,
    utility_set=["train", "calib"],
    oversampling=False,
    # params={"macrosegmento": macrosegmento},
)

# COMMAND ----------

WHOLE_master.sort_values(by="PDF_INSOLUTO_LORDO_ALL", ascending=False)

# COMMAND ----------

WHOLE_master["snapshot_date"].unique()

# COMMAND ----------

y_full_train.info()

# COMMAND ----------

X_full_train.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tuning

# COMMAND ----------

run_id = find_best_params(
    X_full_train,
    y_full_train,  # .astype(int),
    experiment_path,
)
run_id

# COMMAND ----------

# Set a parameter of the job's task that will be re-used in the next task
dbutils.jobs.taskValues.set("tuning_run_id", run_id)
run_id

# COMMAND ----------
