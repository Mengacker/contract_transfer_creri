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

from src.utils.input_validation import is_iso_date
from src.serving.mlflow_publish_model_scoring import mlflow_publish_model
from src.serving.databricks_endpoint import DatabricksEndpoint

from src.config.config_utils import Config

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# COMMAND ----------

# DBTITLE 1,Notebook arguments
# Reset widgets
dbutils.widgets.removeAll()

# Set notebook widgets
dbutils.widgets.text("databricks_host", "https://adb-...azuredatabricks.net/", "Databricks host")
dbutils.widgets.text("env", "dev", "Environment Name")
dbutils.widgets.dropdown("is_cicd", "true", ["true", "false"], "Is run by CICD")


config_setup = Config("../src/config/config_setup.yml")
dev_experiment_path = config_setup.get_config_value("environment.dev.experiment_path")
dbutils.widgets.text("experiment_path", dev_experiment_path,"Experiment path name read by asset workflow file yaml")

# date_and_time = datetime.now() # sballa l'orario di 2 ore perchè usa il riferimento UTC
timezone = "Europe/Amsterdam"
# spark: SparkSession = SparkSession.builder.getOrCreate()
date_and_time = spark.sql("SELECT current_timestamp() as current_time").withColumn(
    "current_time", F.from_utc_timestamp("current_time", timezone)
)
# spark.stop()
date_and_time = date_and_time.collect()[0][0]

print("Current date and time:", date_and_time.strftime("%Y-%m-%d %H:%M:%S"), "at the {t} timezone".format(t=timezone))
dbutils.widgets.text(
    "snapshot_date",
    str(date_and_time.date()),
    label="Snapshot date for the run",
)


# Get notebook arguments
databricks_host = dbutils.widgets.get("databricks_host").removesuffix("/")

env = dbutils.widgets.get("env")
is_cicd = bool(dbutils.widgets.get("is_cicd") == "true")
experiment_path = dbutils.widgets.get("experiment_path")
snapshot_date_str = dbutils.widgets.get("snapshot_date")
snapshot_date_str = snapshot_date_str[:-2] + "01"

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
    databricks_host = config_setup.get_config_value("environment.{env}.databricks_host".format(env=env)).removesuffix("/")

catalog = config_setup.get_config_value("environment.{env}.catalog".format(env=env))
schema = config_setup.get_config_value("environment.{env}.schema".format(env=env))

# Inference master tables
master_volturante_name = config_setup.get_config_value("master_inference")[1]['name']
master_volturato_name = config_setup.get_config_value("master_inference")[0]['name']
dp_path_mstr_cli_in = f"{catalog}.{schema}.{master_volturante_name}"
dp_path_mstr_cli_out = f"{catalog}.{schema}.{master_volturato_name}"

# UDF to serve in endpoints
udf_spec_cli_in = config_setup.get_config_value("serving_endpoint.udf_spec_volturante")['name']
udf_spec_cli_out = config_setup.get_config_value("serving_endpoint.udf_spec_volturato")['name']
dp_path_udf_spec_cli_in = f"{catalog}.{schema}.{udf_spec_cli_in}"
dp_path_udf_spec_cli_out = f"{catalog}.{schema}.{udf_spec_cli_out}"

# Scoring model
model_name = config_setup.get_config_value("modeling.model_serving")['name']
dp_path_model = f"{catalog}.{schema}.{model_name}"

# Endpoints for inference masters
ep_mstr_cli_in = config_setup.get_config_value("serving_endpoint.ep_mstr_volturante")['name']
ep_mstr_cli_out = config_setup.get_config_value("serving_endpoint.ep_mstr_volturato")['name']

# Endpoint for scoring model
if env == "prod-sbx":
    model_endpoint_name = config_setup.get_config_value("serving_endpoint.ep_scoring")['name_for_test']
else:
    model_endpoint_name = config_setup.get_config_value("serving_endpoint.ep_scoring")['name']

# Payload table
endpoint_payload_table = config_setup.get_config_value("inference_monitoring")[0]['name']

# Test request
json_test_oltab_cliin_cfpiva = config_tests.get_config_value("serving_endpoint.json_test_request_cfpiva_ep")
json_test_oltab_cliout_podpdr = config_tests.get_config_value("serving_endpoint.json_test_request_podpdr_ep")
json_test_model_endpoint = config_tests.get_config_value("serving_endpoint.json_test_request_model_ep")
json_test_model_endpoint = json_test_model_endpoint[0]

print("databricks_host:", databricks_host)
print("catalog:", catalog)
print("schema:", schema)
print("model_name:", model_name)

# COMMAND ----------

print("Delta tables, udf feature spec and endpoint of the two inference master dataset:")
print(dp_path_mstr_cli_in)
print(dp_path_mstr_cli_out)
print(dp_path_udf_spec_cli_in)
print(dp_path_udf_spec_cli_out)
print(ep_mstr_cli_in)
print(ep_mstr_cli_out)
print("\nServing endpoint of the model:", model_endpoint_name)

# COMMAND ----------

# MAGIC %md
# MAGIC # Execute operations

# COMMAND ----------

# DBTITLE 1,Test udf inference masters endpoints
# NO NEED TO UPDATE THE EXPOSED UDFS (OF INFERENCE MASTERS) IN THE ENDPOINTS

# Data Platform's tables and their keys to create online
online_table_dict = {
    "COD_FISCALE_PIVA_CLIIN": {
        "tab__path": dp_path_mstr_cli_in,
        "spec_path": dp_path_udf_spec_cli_in,
        "ep_name": ep_mstr_cli_in,
        "test_data": json_test_oltab_cliin_cfpiva,
    },
    "POD_PDR": {
        "tab__path": dp_path_mstr_cli_out,
        "spec_path": dp_path_udf_spec_cli_out,
        "ep_name": ep_mstr_cli_out,
        "test_data": json_test_oltab_cliout_podpdr,
    },
}
error_ep_infer_mstr = []
for pk_col, diz in online_table_dict.items():
    table_path = diz["tab__path"]
    spec_path = diz["spec_path"]
    endpoint_name = diz["ep_name"]
    json_test_endpoint_onlinetable = diz["test_data"]

    # OLD CODE USED AT DEVELOPING PHASE IN SBX_DATA
    # json_config = {
    #     "name": endpoint_name,
    #     "config": {
    #         "served_entities": [
    #             {
    #                 "entity_name": spec_path,
    #                 "scale_to_zero_enabled": False,  # True,
    #                 "workload_size": "Small",
    #             }
    #         ]
    #     },
    # }
    # # Instance of the endpoint # OLD CODE USED AT DEVELOPING PHASE IN SBX_DATA
    # online_table_ep = DatabricksEndpoint(
    #     auth_token=token,
    #     databricks_host=databricks_host,
    #     endpoint_name=endpoint_name,
    #     entity_type = 'udf'
    # )
    # # Create or update the endpoint # OLD CODE USED AT DEVELOPING PHASE IN SBX_DATA
    # try:
    #     online_table_ep.create_update_endpoint(json_config=json_config)
    #     logger.info(f"✅ Online table endpoint done:{endpoint_name}")
    #     # Wait till the endpoint is ready
    #     online_table_ep.wait_for_config_applied()
    # except Exception as e:
    #     error = f"❌ Error creating/updating endpoint {endpoint_name}: {e}"
    #     logger.warning(error)
    #     error_ep_infer_mstr.append(error)

    # TEST THE UDFS ENDPOINT RESPONSES
    try:
        client = mlflow.deployments.get_deploy_client("databricks")
        response = client.predict(endpoint=endpoint_name, inputs=json_test_endpoint_onlinetable)
        message = f"✅ The endpoint's response:{response}"
        logger.info(message)
        print(message,'\n')
    except Exception as e:
        error = f"❌ Error in the client.predict of the endpoint {endpoint_name}: {e}"
        logger.warning(error)
        error_ep_infer_mstr.append(error)

# COMMAND ----------

print("is_cicd", is_cicd)
print("env", env)
print("experiment_path", experiment_path)
print("databricks_host", databricks_host)

# COMMAND ----------

# DBTITLE 1,Deploy new Serving model in catalog

# Deploy the Serving model in unity catalog
run_id, model_info, model = mlflow_publish_model(is_cicd=is_cicd, env=env, experiment_path=experiment_path, token=token, databricks_host=databricks_host)

# COMMAND ----------

# print(run_id, "run_id")
# print(model_info.run_id, "model_info.run_id")
# print(model_info.model_uuid, "model_info.model_uuid")
# print("model_info.model_uri:", model_info.model_uri)
# print("registered_model_version:", model_info.registered_model_version)
# print("model_info.flavors:", model_info.flavors)
# print("model_info.utc_time_created:", model_info.utc_time_created)
# print("model_info.metadata:", model_info.metadata)

try:
    print("model_info.model_uri:", model_info.model_uri)
    print("SIGNATURE:\n", model_info.signature)
    print("run_id", run_id)
except Exception as e:
    print("model_info:",model_info)

# COMMAND ----------

print("Model path:  ",dp_path_model)
print("Endpoint of the model:  ",model_endpoint_name)
print("Model's endpoint payload table:  ",endpoint_payload_table)

# COMMAND ----------

# DBTITLE 1,Update exposed model in the endpoint
# MLflow client
mlflow.set_registry_uri(tracking_uri)  # ("databricks-uc")
client = mlflow.tracking.MlflowClient()
model_version = client.get_model_version_by_alias(dp_path_model, "serving_champion").version
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

# DBTITLE 1,Test the model's endpoint responses
# Test the endpoint's responses
client = mlflow.deployments.get_deploy_client("databricks")
response = client.predict(endpoint=model_endpoint_name, inputs=json_test_model_endpoint)
logger.info(f"✅ Endpoint's test response:{response}")

# COMMAND ----------

# DBTITLE 1,Assert to check errors
assert isinstance(response['predictions'], dict), "ERROR: The response doesn't contain the predictions"
assert response['predictions']['score'] is not None, "ERROR: The response doesn't contain the score"
assert "Non Disponibile" not in str(response['predictions']['esitoSemaforico']), "ERROR: The response doesn't contain the 'esitoSemaforico'"
assert len(error_ep_infer_mstr) == 0, "ERROR: There were errors with the udf inference masters endpoint, check the logs for more details"
