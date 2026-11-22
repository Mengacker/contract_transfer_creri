# Databricks notebook source
# MAGIC %md
# MAGIC # THIS NOTEBOOK HAVE TO BE USED WITH THE COMPUTE CLUSTER "IT":
# MAGIC "IT...DATATUILLE" OR "IT...DATA"

# COMMAND ----------

# DBTITLE 1,Install requirements
# MAGIC %pip install databricks-sdk -U
# MAGIC %pip install databricks-feature-engineering
# MAGIC %pip install mlflow-skinny[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Get ENV by notebook argument
# Reset widgets
dbutils.widgets.removeAll()

# Set notebook widgets
dbutils.widgets.text("env", "dev", "Environment Name")

# Get notebook arguments
env = dbutils.widgets.get("env")

# COMMAND ----------

# DBTITLE 1,Import configurations
# Riferimenti inseriti in hardcoding
catalog = "p_sbx_data"
schema = "data"
mstr_path_cli_in = "creri_infer_mstr_volturante"
mstr_path_cli_out = "creri_infer_mstr_volturato"
udf_spec_cli_in = "creri_udf_infer_mstr_volturante"
udf_spec_cli_out = "creri_udf_infer_mstr_volturato"

# Codice come dovrebbe essere:
# from config.config_utils import Config

# config_setup = Config("src/config/config_setup.yml")
# catalog = config_setup.get_config_value("environment.{env}.catalog".format(env=env))
# schema = config_setup.get_config_value("environment.{env}.schema".format(env=env))

# mstr_path_cli_in = config_setup.get_config_value("master_inference.table_master_volturante")
# mstr_path_cli_out = config_setup.get_config_value("master_inference.table_master_volturato")
# udf_spec_cli_in = config_setup.get_config_value("serving_endpoint.udf_spec_volturante")
# udf_spec_cli_out = config_setup.get_config_value("serving_endpoint.udf_spec_volturato")

mstr_path_cli_in = f"{catalog}.{schema}.{mstr_path_cli_in}"
mstr_path_cli_out = f"{catalog}.{schema}.{mstr_path_cli_out}"
udf_spec_cli_in = f"{catalog}.{schema}.{udf_spec_cli_in}"  # PATH OF UDF SPEC FUNCTION
udf_spec_cli_out = f"{catalog}.{schema}.{udf_spec_cli_out}"  # PATH OF UDF SPEC FUNCTION

print(env, catalog, schema)
print(mstr_path_cli_in, mstr_path_cli_out)
print(udf_spec_cli_in, udf_spec_cli_out)

# COMMAND ----------

# DBTITLE 1,feature_spec_creator
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup


def feature_spec_creator(pk_col: str, table_path: str, spec_path: str) -> None:
    """Function to create a feature spec to serve the feature on the endpoint.
    The feature spec automatically uses the online table associated with the Delta table."""

    # The feature spec function can be created only with the cluster "IT..."
    w = WorkspaceClient()
    fe = FeatureEngineeringClient()
    try:
        w.functions.get(name=spec_path)
        feature_spec_exists = True
    except NotFound:
        feature_spec_exists = False

    if not feature_spec_exists:
        fe.create_feature_spec(
            name=spec_path,
            features=[FeatureLookup(table_name=table_path, lookup_key=pk_col)],
        )
        print(f"UDF feature spec function '{spec_path}' created to lookup the online table '{table_path}'")
    else:
        # The feature spec function no needs to be refreshed, it is a function, not data
        print(f"UDF feature spec function '{spec_path}' already exists and it lookups the online table '{table_path}'")

# COMMAND ----------

# DBTITLE 1,EXECUTION
# TRASFORM D.P. TABLES INTO FEATURE STORE'S ON-LINE TABLES

online_table_dict = {
    "COD_FISCALE_PIVA_CLIIN": {
        "tab__path": mstr_path_cli_in,
        "spec_path": udf_spec_cli_in,
    },
    "POD_PDR": {"tab__path": mstr_path_cli_out, "spec_path": udf_spec_cli_out},
}
for pk_col, val_dict in online_table_dict.items():
    feature_spec_creator(pk_col=pk_col, table_path=val_dict["tab__path"], spec_path=val_dict["spec_path"])

# COMMAND ----------

# DBTITLE 1,delete old spec
# from databricks.sdk import WorkspaceClient
# from databricks.feature_engineering import FeatureEngineeringClient

# w = WorkspaceClient()
# fe = FeatureEngineeringClient()

# feature_spec_name = [
#     f"{catalog}.{schema}.antifrode_volture_cf_piva_cli_morosita_pregressa_spec"
#     ,
#     f"{catalog}.{schema}.antifrode_volture_pod_pdr_master_volturato_spec"
# ]

# for speack in feature_spec_name:
#     try:
#         w.functions.get(name=speack)
#         fe.delete_feature_spec(name=speack)
#         print(f"Feature spec function {speack} deleted")
#     except NotFound:
#         print(f"Feature spec function {speack} does not exist")
#     print("only if necessary")
