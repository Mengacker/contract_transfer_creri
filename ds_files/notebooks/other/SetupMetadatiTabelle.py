# Databricks notebook source
# DBTITLE 1,Set notebook extensions
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# DBTITLE 1,Install requirements
import os
import sys
from databricks.sdk.runtime import dbutils

notebook_path = "/Workspace/" + os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)
sys.path.append("..")
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

# DBTITLE 1,Import libraries
import pandas as pd

from src.data_preparation.alter_table import alter_table
from src.config.config_utils import Config

# COMMAND ----------

# MAGIC %md
# MAGIC # SPECIFICA PK NELLE TABELLE

# COMMAND ----------

# DBTITLE 1,Retrieve config values
# Retrieve config values
config = Config(config_path="../../src/config/", name_list=["config_setup.yml"])
config_tests = Config(config_path="../../src/config/config_tests.yml")

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
tables_pk = {d["name"]: d["primary_key"][0] for d in diz_tab}
tables_description = {d["name"]: d["description"] for d in diz_tab}
tables_pk

# COMMAND ----------

tables_description

# COMMAND ----------

# DBTITLE 1,alter_table to set PK
# Try to set the primary key in the tables
schema = "data"
catalog = "p_sbx_data"

l_done_ok = []
l_NO_done = []
for tab, pk in tables_pk.items():
    path = f"{catalog}.{schema}.{tab}"
    try:
        alter_table(pk_col=[pk], table_path=path)
        l_done_ok.append(path)
    except Exception as e:
        l_NO_done.append(path)
        print(path)
        print(f"Error on table {tab}: {e}")

assert len(l_NO_done) == 0, l_NO_done

# COMMAND ----------

# DBTITLE 1,comment description on tables
# Retrieve the dictionary of table names and descriptions
# tables_description = {v["name"]: v["description"] for k, v in tables.items() if type(v) != list}
# tables_description.update({v["name"]: v["description"] for k, l in tables.items() if type(l) == list for v in l})
# tables_description

schema='data'
catalog='p_sbx_data'

l_done_ok = []
l_NO_done = []
for tab, description in tables_description.items():
    path = f"{catalog}.{schema}.{tab}"
    try:
        spark.sql(f"""
        COMMENT ON TABLE {path} IS '{description}'
        """)

        l_done_ok.append(path)
    except Exception as e:
        l_NO_done.append(path)
        print(path)
        print(f"Error on table {tab}: {e}")

assert len(l_NO_done)==0, l_NO_done

# COMMAND ----------

from mlflow.tracking import MlflowClient

schema = "data"
catalog = "p_sbx_data"
ogg = "creri_model_serving"
path = f"{catalog}.{schema}.{ogg}"
description = "Prova"

client = MlflowClient()
client.update_registered_model(name=path, description=description)
