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
import os ###
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
import mlflow
import mlflow.deployments
from pyspark.sql import functions as F
from datetime import datetime

from src.utils.input_validation import is_iso_date
from src.serving.databricks_endpoint import DatabricksEndpoint
from src.serving.mlflow_publish_model_deterministic import mlflow_publish_model

from src.config.config_utils import Config

import logging

logger = logging.getLogger("py4j")
logger.setLevel(logging.ERROR)
logger_azure = logging.getLogger("azure")
logger_azure.setLevel(logging.ERROR)

# COMMAND ----------

# DBTITLE 1,Notebook arguments
# Reset widgets
dbutils.widgets.removeAll()

# Set notebook widgets
dbutils.widgets.text("databricks_host", "https://adb-...azuredatabricks.net/", "Databricks host")
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
    label="Date from which to take training date interval",
)


# Get notebook arguments
databricks_host = dbutils.widgets.get("databricks_host").removesuffix("/")

env = dbutils.widgets.get("env")
is_cicd = bool(dbutils.widgets.get("is_cicd") == "true")
experiment_path = dbutils.widgets.get("experiment_path")
snapshot_date_str = dbutils.widgets.get("snapshot_date")

# Validate input
if not is_iso_date(snapshot_date_str):
    raise ValueError(f"Invalid snapshot_date: {snapshot_date_str}")

print("databricks_host:", databricks_host)
print("env:", env)
print("is_cicd:", is_cicd)
print("dev_experiment_path:", dev_experiment_path)
print("experiment_path:", experiment_path)
print("snapshot_date:", snapshot_date_str)

# COMMAND ----------

# DBTITLE 1,Retrive config params
# Get the token from the notebook's context
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# Retrive config params
config_setup = Config("../src/config/config_setup.yml")
config_model = Config("../src/config/config_model.yml")
config_tests = Config("../src/config/config_tests.yml")

tracking_uri = config_model.get_config_value("mlflow_config.tracking_uri")

if env == "dev":
    databricks_host = (
        config_setup.get_config_value("environment.{env}.databricks_host".format(env=env)).removesuffix("/")
    )

catalog = config_setup.get_config_value("environment.{env}.catalog".format(env=env))
schema = config_setup.get_config_value("environment.{env}.schema".format(env=env))

# Model
model_name = config_setup.get_config_value("modeling.model_deterministic")["name"]
dp_path_model = f"{catalog}.{schema}.{model_name}"

# Payload table
endpoint_payload_table = config_setup.get_config_value("inference_monitoring")[0]["name"]

# Endpoint of model
if env == "prod-sbx":
    model_endpoint_name = config_setup.get_config_value("serving_endpoint.ep_scoring")['name_for_test']
else:
    model_endpoint_name = config_setup.get_config_value("serving_endpoint.ep_scoring")['name']

# Test request
json_test_model_endpoint = config_tests.get_config_value("serving_endpoint.json_test_request_model_ep")
json_test_model_endpoint = json_test_model_endpoint[0]


print("databricks_host:", databricks_host)
print("catalog:", catalog)
print("schema:", schema)
print("model_name:", model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC # Execute operations

# COMMAND ----------

# DBTITLE 1,Deploy new Deterministic model in catalog
# Deploy the Deterministic model in unity catalog
run_id = mlflow_publish_model(is_cicd=is_cicd, env=env, experiment_path=experiment_path)
run_id

# COMMAND ----------

print("Model path:  ",dp_path_model)
print("Endpoint of the model:  ",model_endpoint_name)
print("Model's endpoint payload table:  ",endpoint_payload_table)

# COMMAND ----------

# DBTITLE 1,Update exposed model in the endpoint
# MLflow client
mlflow.set_registry_uri(tracking_uri)  # ("databricks-uc")
client = mlflow.tracking.MlflowClient()
model_version = client.get_model_version_by_alias(dp_path_model, "deterministic").version
print("model_version:", model_version)

# Define the endpoint config, check: https://docs.databricks.com/api/workspace/servingendpoints
data_json = {
    "name": model_endpoint_name,
    "config": {
        # "served_models": [ # OLD CODE
        #     {
        #         "model_name": dp_path_model,
        #         "model_version": model_version,
        #         "workload_size": "Small",
        #         "scale_to_zero_enabled": False,
        #     }
        # ]
        "served_entities": [
            {
                "entity_name": dp_path_model,
                "entity_version": model_version,
                "workload_size": "Small",
                "scale_to_zero_enabled": False,
            }
        ]
        # , "auto_capture_config": # OLD CODE TO BE COMMENTED BECAUSE CURRENTLY IT BROKES THE ENDPOINT UPDATE
        # {
        # "catalog_name": catalog, # THE PAYLOAD TABLE HAVE TO BE FIXED MANUALLY AND NOT WITH SDK CODE
        # "schema_name": schema,
        # "table_name_prefix": endpoint_payload_table.replace("_payload",""),  # remove "_payload" suffix because the endpoint will re-add it automatically
        # "enabled": True
        # }
    },
}

model_endpoint = DatabricksEndpoint(
    auth_token=token,
    databricks_host=databricks_host,
    endpoint_name=model_endpoint_name,
    entity_type = 'model'
)
print("endpoint_name:",model_endpoint.endpoint_name)
print("databricks_host:",model_endpoint.databricks_host)
print("request_endpoint_url:", model_endpoint.request_endpoint_url)
print("invocation_endpoint_url:", model_endpoint.invocation_endpoint_url)

model_endpoint.create_update_endpoint(json_config=data_json)
logger.info(f"✅ The model's endpoint created or updated:{model_endpoint_name}")

model_endpoint.wait_for_config_applied()
logger.info(f"✅ Model endpoint is ready:{model_endpoint_name}")

# COMMAND ----------

# DBTITLE 1,Test the model's endpoint response

# Test the endpoint's responses
client = mlflow.deployments.get_deploy_client("databricks")
response = client.predict(endpoint=model_endpoint_name, inputs=json_test_model_endpoint)
logger.info(f"✅ The endpoint's response:{type(response)}")

# COMMAND ----------

# DBTITLE 1,Assert to check errors
assert isinstance(response['predictions'], dict), "ERROR: The response doesn't contain the predictions"
assert response['predictions']['score'] is not None, "ERROR: The response doesn't contain the score"
assert "Non Disponibile" not in str(response['predictions']['esitoSemaforico']), "ERROR: The response doesn't contain the 'esitoSemaforico'"
