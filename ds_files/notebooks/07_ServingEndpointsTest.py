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

notebook_path = "/Workspace" + os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)
sys.path.append("..")
os.chdir(notebook_path)

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
print(notebook_path)

# COMMAND ----------

# MAGIC %md
# MAGIC # EDITABLE SECTION
# MAGIC Here you can put your code, starting from imports

# COMMAND ----------

# DBTITLE 1,Import libraries
import logging
import mlflow
import mlflow.deployments
from pyspark.sql import SparkSession, functions as F
import pandas as pd
import json

from src.utils.input_validation import is_iso_date
from src.config.config_utils import Config

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# COMMAND ----------

# DBTITLE 1,Notebook arguments
# Reset widgets
dbutils.widgets.removeAll()

# Set notebook widgets

dbutils.widgets.text("env", "dev", "Environment Name")

# date_and_time = datetime.now() # sballa l'orario di 2 ore perchè usa il riferimento UTC
timezone = "Europe/Amsterdam"

spark: SparkSession = SparkSession.builder.getOrCreate()
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
snapshot_date_str = dbutils.widgets.get("snapshot_date")
snapshot_date_str = snapshot_date_str[:-2] + "01"

# Validate input
if not is_iso_date(snapshot_date_str):
    raise ValueError(f"Invalid snapshot_date: {snapshot_date_str}")

snapshot_date_str
print("env:", env)
print("snapshot_date:", snapshot_date_str)

# COMMAND ----------

# DBTITLE 1,Retrive config params
# Retrive config params
config = Config(config_path="../src/config/", name_list=["config_setup.yml", "config_model.yml"])
config_tests = Config("../src/config/config_tests.yml")


# Endpoints for inference masters
sep_mstr_cliin = config.get_config_value("serving_endpoint.ep_mstr_volturante")["name"]
sep_mstr_cliout = config.get_config_value("serving_endpoint.ep_mstr_volturato")["name"]

# Endpoint for scoring model
if env == "prod-sbx":
    sep_model = config.get_config_value("serving_endpoint.ep_scoring")['name_for_test']
else:
    sep_model = config.get_config_value("serving_endpoint.ep_scoring")['name']

# Test request
json_test_sep_infer_mstr_cliin = config_tests.get_config_value("serving_endpoint.json_test_request_cfpiva_ep")
json_test_sep_infer_mstr_cliout = config_tests.get_config_value("serving_endpoint.json_test_request_podpdr_ep")
json_test_sep_model = config_tests.get_config_value("serving_endpoint.json_test_request_model_ep")

print("sep_mstr_cliin:", sep_mstr_cliin)
print("sep_mstr_cliout:", sep_mstr_cliout)
print("sep_model:", sep_model)

# COMMAND ----------

# MAGIC %md
# MAGIC # EXECUTE SERVING ENDPOINT TESTS

# COMMAND ----------

# DBTITLE 1,Get client's endpoints
# Get the client's endpoints
client = mlflow.deployments.get_deploy_client("databricks")
l_error_message = []

list_endpoints = client.list_endpoints()
for sep in list_endpoints:
    if "creri" in sep["name"]:
        print("ENDPOINT:   ", sep["name"], "----------------------------------------------------------")
        try:
            if len(sep["description"]) > 0:
                print("Description:", sep["description"])
        except Exception as e:
            print(e)
            continue
        try:
            print(sep["config"])
        except Exception as e:
            print(e)
            continue
        print("\n")

# COMMAND ----------

# DBTITLE 1,TEST sep of udf inference master cli IN

print("TEST serving endpoint of inference master incoming customer:", sep_mstr_cliin, "\n")
print(client.get_endpoint(sep_mstr_cliin))

try:
    response = client.predict(endpoint=sep_mstr_cliin, inputs=json_test_sep_infer_mstr_cliin)
    df_infer_mstr_cliin = pd.DataFrame(response["outputs"])
    df_infer_mstr_cliin = df_infer_mstr_cliin[["COD_FISCALE_PIVA_CLIIN"] + df_infer_mstr_cliin.columns[:-1].tolist()]
    logger.info(
        f"✅ The response of serving endpoint '{sep_mstr_cliin}' is for these COD_FISCALE_PIVA_CLIIN:{df_infer_mstr_cliin['COD_FISCALE_PIVA_CLIIN'].values.tolist()}"
    )
    print(f"✅ The endpoint's response:\n{df_infer_mstr_cliin}")

except Exception as e:
    error_message= f"❌ Error in the serving endpoint '{sep_mstr_cliin}' is:\n {e}"
    logger.error(error_message)
    l_error_message.append(error_message)


# COMMAND ----------

l_error_message

# COMMAND ----------

# DBTITLE 1,TEST sep of udf inference master cli OUT
print("TEST serving endpoint of inference master outgoing customer:", sep_mstr_cliout)
print(client.get_endpoint(sep_mstr_cliout))

try:
    response = client.predict(endpoint=sep_mstr_cliout, inputs=json_test_sep_infer_mstr_cliout)
    df_infer_mstr_cliout = pd.DataFrame(response["outputs"])
    df_infer_mstr_cliout = df_infer_mstr_cliout[["POD_PDR"] + df_infer_mstr_cliout.columns[:-1].tolist()]

    logger.info(
        f"✅ The response of serving endpoint '{sep_mstr_cliout}' is for these POD_PDR:{df_infer_mstr_cliout['POD_PDR'].values.tolist()}"
    )
    print(f"✅ The endpoint's response:\n{df_infer_mstr_cliout}")
except Exception as e:
    error_message= f"❌ Error in serving endpoint '{sep_mstr_cliout}' is:\n {e}"
    logger.error(error_message)
    l_error_message.append(error_message)

# COMMAND ----------

l_error_message

# COMMAND ----------

# DBTITLE 1,TEST sep of serving model
print("TEST serving endpoint of model:", sep_model)
print(client.get_endpoint(sep_model))

try:
    sep_model_responses = []
    for i, j in enumerate(json_test_sep_model):
        diz_input = json.loads(j["dataframe_records"][0]["INPUT_JSON"])
        cfpiva = [cc for cc in ["CodiceFiscale", "PartitaIva"] if cc in list(diz_input.keys())]
        cfpiva = [diz_input[cp] for cp in cfpiva]
        podpdr = [diz["POD/PDR"] for diz in diz_input["Forniture"]]

        response = client.predict(endpoint=sep_model, inputs=j)
        response["predictions"].update({"POD/PDR": podpdr, "COD_FISCALE_PIVA_CLIIN": cfpiva})
        sep_model_responses.append(response["predictions"])

        logger.info(f"✅ The response n° {i} of serving endpoint '{sep_model}' is: {response}")

    df_infer_model = pd.DataFrame(sep_model_responses)
    print(f"✅ The endpoint's response:\n{sep_model_responses}")

except Exception as e:
    error_message= f"❌ Error in serving endpoint '{sep_model}' is:\n {e}"
    logger.error(error_message)
    l_error_message.append(error_message)

# COMMAND ----------

l_error_message

# COMMAND ----------

assert len(l_error_message)==0, [print(f"\n{m}\n") for m in l_error_message]

# COMMAND ----------

df_infer_model
