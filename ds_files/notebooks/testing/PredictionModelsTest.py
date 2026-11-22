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
sys.path.append("../..")
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
import requests

from src.utils.input_validation import is_iso_date
from src.model.model_utils import load_deployed_model, get_version_mapped_to_alias
from src.serving.batch_inference_scoring_model import batch_inference
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
dbutils.widgets.text("env", "prod-dpp", "Environment Name")  # dev, prod-sbx, prod-dpp
dbutils.widgets.dropdown("is_cicd", "false", ["true", "false"], "Is run by CICD")

config_setup = Config("../../src/config/config_setup.yml")

# experiment_path = config_setup.get_config_value("environment.dev.experiment_path")
experiment_path = "/Shared/DataScience/Exp/creri/dev_creri_exp"

dbutils.widgets.text("experiment_path", experiment_path, "Experiment path name read by asset workflow file yaml")

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
experiment_path = dbutils.widgets.get("experiment_path")
snapshot_date_str = dbutils.widgets.get("snapshot_date")

print(f"experiment_path: {experiment_path}\n")

# Validate input
if not is_iso_date(snapshot_date_str):
    raise ValueError(f"Invalid snapshot_date: {snapshot_date_str}")


# --- snapshot_date of the first_day_of_week ---#
snapshot_first_day_of_week = datetime.today() - timedelta(days=datetime.today().weekday())
snapshot_first_day_of_week = snapshot_first_day_of_week.strftime("%Y-%m-%d")
print(f"The first date of the current week is :{snapshot_first_day_of_week}\n")

# COMMAND ----------

# DBTITLE 1,Retrieve config values
# Retrieve config values
config = Config(config_path="../../src/config/", name_list=["config_setup.yml", "config_model.yml"])
config_tests = Config(config_path="../../src/config/config_tests.yml")

tracking_uri = config.get_config_value("mlflow_config.tracking_uri")

catalog = config.get_config_value("environment.{env}.catalog".format(env=env))
schema = config.get_config_value("environment.{env}.schema".format(env=env))
print(f"{catalog}.{schema}")

master_train_final = config.get_config_value("master_training_final")[0]["name"]
master_train_final = f"{catalog}.{schema}.{master_train_final}"

model_scoring = config.get_config_value("modeling.model_scoring")["name"]
model_deterministic = config.get_config_value("modeling.model_deterministic")["name"]
model_serving = config.get_config_value("modeling.model_serving")["name"]


# Test request
input_scoring_model = config_tests.get_config_value("scoring_model.final_training_master")
input_serving_model = config_tests.get_config_value("serving_model.final_training_master")
# input_serving_model = config_tests.get_config_value("serving_endpoint.json_test_request_model_ep")
input_deterministic_model = config_tests.get_config_value("deterministic_model.json_test_podpdr_yellow_list")

# COMMAND ----------

# DBTITLE 1,Pre-processing
master = spark.sql(f"""SELECT * FROM {master_train_final}""").toPandas()
print(master.groupby("UTILITY_SET").agg({"PK_PROPOSAL_LINE_ITEM": "count"}))
print("TOTAL numbers of records: ", master.shape)

master = master.dropna(subset=["COD_FISCALE_PIVA_CLIIN", "POD_PDR"])
print("PK NOT NULL numbers of records: ", master.shape)

master.sort_values(by="snapshot_date", ascending=False, inplace=True)
print("\nsnapshot dates:\n", master["snapshot_date"].astype(str).unique())
master.sort_values(by="PDF_INSOLUTO_LORDO_ALL", ascending=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # EXECUTE INFERENCE-CALL TESTS

# COMMAND ----------

# DBTITLE 1,TEST Scoring model
alias = "champion"
target = "FLG_CLI_MOROSO"
print(f"'{alias}' is the model's alias that is going to be evaluated\n")

df = master.copy()
### df = master.loc[master['FLG_INSOLUTO'],:].reset_index(drop=True).copy()

# Filter to selected test cases for Scoring model
l_pk = [pk for d in input_scoring_model + input_serving_model for k, v in d.items() for pk in v if type(v) == list]
ind_pk = df["PK_PROPOSAL_LINE_ITEM"].isin(l_pk)
df = df.loc[ind_pk, :].reset_index(drop=True)

df = df.loc[df[target].notnull(), :].reset_index(drop=True).copy()
df[target] = df[target].astype(int)  # from boolean to integer

print(f"experiment_path: {experiment_path}\n")
run_id, df_scoring, score_model = batch_inference(
    df_input=df,
    target=target,
    model_alias=alias,
    is_cicd=is_cicd,
    experiment_path=experiment_path,
    env=env,
    params={"config_path": "../../src/config"},
)
print(f"\nThe run_id is {run_id}\n")

print(df_scoring.shape, end=" == ")
d_pk = [
    {"case_description": d["description"], k: pk}
    for d in input_scoring_model + input_serving_model
    for k, v in d.items()
    for pk in v
    if type(v) == list
]
df_scoring = df_scoring.merge(pd.DataFrame(d_pk), how="left", on="PK_PROPOSAL_LINE_ITEM")
print(df_scoring.shape)

cols = [
    "FLG_CLI_MOROSO",
    "score",
    "BINARY_PREDICTION",
    "esitoSemaforico",
    "USED_MODEL",
    "case_description",
    "PK_PROPOSAL_LINE_ITEM",
    "POD_PDR",
    "MACROSEGMENTO",
    "MACROSEGMENTO_CLIIN",
    "PDF_INSOLUTO_LORDO_ALL",
    "MANCATI_INCASSI_RATE",
    "PERC_MANCATI_INCASSI_RATE",
    "CLI_INSOLUTO_LORDO",
    "CLI_INSOLUTO_LORDO_CLIIN",
    "AGING_METER_ATTIV_SWIN",
    "COD_FISCALE_PIVA",
    "COD_FISCALE_PIVA_CLIIN",
    "SDF_CANLE_ATTIVAZ_ATT_CLIIN",
    "UTILITY_SET",
    "DATA_VOLTURA",
    "snapshot_date",
]
df_scoring[cols]

# COMMAND ----------

# DBTITLE 1,df_scoring.groupby
print("The red and yellow cases at inference time may not be the same color at training time!")

df_scoring.groupby(["USED_MODEL", "esitoSemaforico"]).agg({"PK_PROPOSAL_LINE_ITEM": "count", "score": "max"}).rename(
    columns={"PK_PROPOSAL_LINE_ITEM": "n_count_cases", "score": "max_score"}
).sort_values(by="max_score", ascending=False)

# COMMAND ----------

# DBTITLE 1,prepare test data for serving model
df = df_scoring[
    [
        "PK_PROPOSAL_LINE_ITEM",
        "case_description",
        "POD_PDR",
        "COD_FISCALE_PIVA_CLIIN",
        "COGNOME_CLIIN",
        "EMAIL_CLIIN",
        "TELEFONO_CLIIN",
        "MACROSEGMENTO_CLIIN",
        "ENCOD_SDF_CANLE_ATTIVAZ_ATT_CLIIN",
    ]
].copy()
df["COGNOME_CLIIN"] = df["COGNOME_CLIIN"].astype(str)


# Filter to true positive cases at training time
# df = df.loc[df_scoring['FLG_CLI_MOROSO'] == 1,:].reset_index(drop=True).copy() # 16 gialli, 0 rossi
# df = df.loc[df_scoring['FLG_INSOLUTO'] == 1,:].reset_index(drop=True).copy()  # 69 gialli, 2 rossi (però 89k casi analizzati)
# cash = 1000
# df = df.loc[(df_scoring["CLI_INSOLUTO_LORDO"] > cash) | (df_scoring["CLI_INSOLUTO_LORDO_CLIIN"] > cash) | (df_scoring["PDF_INSOLUTO_LORDO_ALL"] > cash),:].reset_index(drop=True).copy() # 8 gialli, 0 rossi
# gg = 365
# df = df.loc[(df_scoring["PDF_AGING_SCAD_RATE_ALL"] > gg) | (df_scoring["PDF_AGING_SCAD_RATE_LAST"] > gg),:].reset_index(drop=True).copy()  # 9 gialli, 0 rossi

# FIlter to selected test cases for Serving model
l_pk = [pk for d in input_scoring_model + input_serving_model for k, v in d.items() for pk in v if type(v) == list]
# l_pk = [pk for d in input_serving_model for k, v in d.items() for pk in v if type(v) == list]
ind_pk = df["PK_PROPOSAL_LINE_ITEM"].isin(l_pk)
df = df.loc[ind_pk, :].reset_index(drop=True)

templ_str = '{"SegmentoVolturante": *MACROSEGMENTO_CLIIN*, "CodiceFiscale": *COD_FISCALE_CLIIN*, "PartitaIva":*PIVA_CLIIN* ,"NomeCognome": *NOMECOGNOME_CLIIN*, "CanaleAcquisizione": *ENCOD_SDF_CANLE_ATTIVAZ_ATT_CLIIN*, "MailVolturante": *EMAIL_CLIIN*, "CellulareVolturante": *TELEFONO_CLIIN*, "Forniture": [{"POD/PDR": *POD_PDR* }]}'

# Prepare input for serving model endpoint
field_input = "INPUT_JSON"
input_serving_model_prep = []
for idx, row in df.iterrows():
    row = row.to_dict()
    row["NOMECOGNOME_CLIIN"] = "Test|" + row["COGNOME_CLIIN"]
    row.pop("COGNOME_CLIIN")

    if row["COD_FISCALE_PIVA_CLIIN"] and len(row["COD_FISCALE_PIVA_CLIIN"]) == 16:
        row["COD_FISCALE_CLIIN"] = row["COD_FISCALE_PIVA_CLIIN"]
        row["PIVA_CLIIN"] = ""
    elif (row["COD_FISCALE_PIVA_CLIIN"]) and (len(row["COD_FISCALE_PIVA_CLIIN"]) == 11):
        row["COD_FISCALE_CLIIN"] = ""
        row["PIVA_CLIIN"] = row["COD_FISCALE_PIVA_CLIIN"]
    row.pop("COD_FISCALE_PIVA_CLIIN")

    # Prepare input for serving model endpoint
    input_str = templ_str
    for k, v in row.items():
        input_str = input_str.replace(f"*{k}*", f'"{v}"')
    input_diz = {
        "dataframe_records": [
            {
                "INPUT_JSON": input_str,
            },
        ],
    }

    # Add some field to be used for tracking
    input_diz["PK_PROPOSAL_LINE_ITEM"] = row["PK_PROPOSAL_LINE_ITEM"]
    input_diz["case_description"] = row["case_description"]
    input_diz["POD_PDR"] = row["POD_PDR"]
    input_serving_model_prep.append(input_diz)
print(f"n°{len(input_serving_model_prep)} requests for the scoring model endpoint\n")

# Prepare input for online tables endpoints
input_pk_oltab_sep = {"COD_FISCALE_PIVA_CLIIN": {"dataframe_records": list()}, "POD_PDR": {"dataframe_records": list()}}
print(f"n°{len(input_pk_oltab_sep)} requests for the online table endpoints:")
for field_input, diz in input_pk_oltab_sep.items():
    list_input = []
    for idx, row in df.iterrows():
        row = row.to_dict()
        list_input.append({field_input: row[field_input]})
    input_pk_oltab_sep[field_input].update({"dataframe_records": list_input})
    print(f" - n°{len(list_input)} of {field_input} keys")

# COMMAND ----------

print(catalog)

# COMMAND ----------

# DBTITLE 1,load serving model
alias = "serving_champion"
catalog_model_path = f"{catalog}.{schema}.{model_serving}"
print(catalog_model_path)

# Load model
model_version = get_version_mapped_to_alias(
    model_name=catalog_model_path,
    alias=alias,
    registry_uri=tracking_uri,
)
print(f"\nModel version: {model_version}\n")

model_uri = "models:/{model_name}@{alias}".format(model_name=catalog_model_path, alias=alias)
print(f"\nModel URI: {model_uri}\n")

# MlFlow settings
mlflow.set_registry_uri(tracking_uri)
mlflow.set_experiment(experiment_path)
experiment_id = mlflow.get_experiment_by_name(experiment_path).experiment_id
client = mlflow.MlflowClient()

# Load model with mlflow
with mlflow.start_run(
    experiment_id=experiment_id,
) as run:
    run_id = run.info.run_id

    mlflow_wrap = load_deployed_model(model_uri=model_uri, as_spark_udf=False, result_type="long")
    wrapped_model = mlflow_wrap.unwrap_python_model()
    print("\nType of custom model", type(wrapped_model), "\n")

    raw_model = wrapped_model.model
    print("\nType of raw model", type(raw_model), "\n")

    # Get model's attributes
    token_ep = wrapped_model.token_ep
    host = wrapped_model.host
    print("\nDatabricks host:\n", host, "\n")
    tables_endpoints_url = wrapped_model.tables_endpoints_url
    print(f"\ntables_endpoints_url:\n")
    for k, v in tables_endpoints_url.items():
        tables_endpoints_url[k] = "https://" + v
        print(f"{k}:\n{v}\n")

    predictors = wrapped_model.predictors
    print("\nPredictors:\n", predictors, "\n")

    mlflow.end_run()

# COMMAND ----------

# DBTITLE 1,TEST call-request to udf inference tables endpoint
response = {}
for pk, endpoint_url in tables_endpoints_url.items():
    input_keys = input_pk_oltab_sep[pk]
    print(f"endpoint_url: {endpoint_url}")

    # Query endpoint of the online tables
    response[pk] = requests.post(
        url=endpoint_url,
        headers={"Authorization": f"Bearer {token_ep}"},
        json=input_keys,
        timeout=60,  # TIMEOUT, parameter expressed in seconds
    )
    try:
        print(f"{pk} online table dataframe shape: {pd.DataFrame(response[pk].json()['outputs']).shape}")
    except Exception as e:
        print(f"Error: {response[pk].json()}")
response

# COMMAND ----------

# DBTITLE 1,CHECK infer_mstr_volturato
pk = "POD_PDR"
df_infer_mstr_volturato = pd.DataFrame(response[pk].json()["outputs"])

snap_dates = [x for x in df_infer_mstr_volturato["snapshot_date"].unique().tolist() if x != None]
snap_dates.sort(reverse=True)
print(f"\nThe snapshot date:{snap_dates}\n")
# assert snap_dates[0] == snapshot_first_day_of_week, f"The snapshot date is not the first day of the present week:{snap_dates[0]}"
if snap_dates[0] != snapshot_first_day_of_week:
    print(f"The snapshot date is not the first day of the present week: {snapshot_first_day_of_week}")

ind = df_infer_mstr_volturato.drop(columns=[pk]).notnull().any(axis=1)
df_infer_mstr_volturato.loc[ind, :]

# COMMAND ----------

# DBTITLE 1,CHECK infer_mstr_volturante
pk = "COD_FISCALE_PIVA_CLIIN"
df_infer_mstr_volturante = pd.DataFrame(response[pk].json()["outputs"])

snap_dates = [x for x in df_infer_mstr_volturante["snapshot_date"].unique().tolist() if x != None]
snap_dates.sort(reverse=True)
print(f"\nThe snapshot date:{snap_dates}\n")
# assert snap_dates[0] == snapshot_first_day_of_week, f"The snapshot date is not the first day of the present week:{snap_dates[0]}"
if snap_dates[0] != snapshot_first_day_of_week:
    print(f"The snapshot date is not the first day of the present week: {snapshot_first_day_of_week}")

ind = df_infer_mstr_volturante.drop(columns=[pk]).notnull().any(axis=1)
df_infer_mstr_volturante.loc[ind, :]

# COMMAND ----------

# DBTITLE 1,TEST Serving model by pyfunc predict method
input_sep = [x for x in input_serving_model_prep]
print(f"\n n°{len(input_sep)} of request\n")

# PREDICTIONS
serving_results = []
error_cases = {}
for i, request_case in enumerate(input_sep):
    print(f"\ncase: {request_case}\n")
    try:
        cc = []
        for j, c in enumerate(["PK_PROPOSAL_LINE_ITEM", "case_description", "POD_PDR"]):
            cc.append(request_case.pop(c))
            request_case[c] = cc[j]
        print(f"PK:{cc[0]}")
    except:
        cc = [None, None, None]

    df_input = pd.DataFrame(request_case["dataframe_records"])

    try:
        result = wrapped_model.predict(user_input=df_input)
        result.update({"result at training time": cc[1], "POD_PDR": cc[2], "PK_PROPOSAL_LINE_ITEM": cc[0]})
        print(f"\nResult of call n°{i}:", result, "\n")
        serving_results.append(result)
    except Exception as e:
        error_cases[i] = request_case
        print(f"\nError of call n°{i} case: {request_case} Error: {e}\n")

df_serving = pd.DataFrame(serving_results).sort_values(by="score", ascending=False).reset_index(drop=True)
df_serving.head(20)

# COMMAND ----------

# DBTITLE 1,requests error cases
print(f"n°{len(error_cases)} of error cases: {error_cases}")

# COMMAND ----------

# DBTITLE 1,df_serving.groupby
print("The red and yellow cases at training time may not be the same color at inference time!")
df_serving.groupby(["esitoSemaforico"]).agg({"PK_PROPOSAL_LINE_ITEM": "count", "score": "max"}).rename(
    columns={"PK_PROPOSAL_LINE_ITEM": "n_count_cases", "score": "max_score"}
).sort_values(by="max_score", ascending=False)

# COMMAND ----------

# DBTITLE 1,TEST Deterministic model
alias = "deterministic"
catalog_model_path = f"{catalog}.{schema}.{model_deterministic}"
print(catalog_model_path)

# Load model
model_version = get_version_mapped_to_alias(
    model_name=catalog_model_path,
    alias=alias,
    registry_uri=tracking_uri,
)
print(f"Model version: {model_version}")

model_uri = "models:/{model_name}@{alias}".format(
    model_name=catalog_model_path, alias=alias
)  # = model_uri_by_runid = model_uri_by_alias
print(f"Model URI: {model_uri}")

# MlFlow settings
mlflow.set_registry_uri(tracking_uri)
mlflow.set_experiment(experiment_path)
experiment_id = mlflow.get_experiment_by_name(experiment_path).experiment_id
client = mlflow.MlflowClient()

# Load model with mlflow
with mlflow.start_run(
    experiment_id=experiment_id,
) as run:
    run_id = run.info.run_id

    mlflow_wrap = load_deployed_model(model_uri=model_uri, as_spark_udf=False, result_type="long")
    wrapped_model = mlflow_wrap.unwrap_python_model()

    print("\nType of custom model", type(wrapped_model), "\n")

    yellow_list = wrapped_model.yellow_list
    print("\nYellow_list:", len(yellow_list), "of POD/PDR \n")

    # Insert yellow list cases in the input request
    y_cases = str([{"POD/PDR": pp} for pp in yellow_list])
    y_cases = y_cases.replace("'", '"')
    input_str = '{"Forniture": [{"POD/PDR": *POD_PDR* }]}'
    input_str = input_str.replace('[{"POD/PDR": *POD_PDR* }]', y_cases)
    input_diz = {
        "dataframe_records": [
            {
                "INPUT_JSON": input_str,
            },
        ],
    }
    input_deterministic_model.append(input_diz)

    # PREDICTIONS
    deterministic_results = []
    for i, case in enumerate(input_deterministic_model):
        # for i, case in enumerate(input_serving_model_prep):
        df_input = pd.DataFrame(case["dataframe_records"])

        result = wrapped_model.predict(user_input=df_input)
        print(f"\nResult of call n°{i}:", result, "\n")
        deterministic_results.append(result)
    mlflow.end_run()


df_deterministic = pd.DataFrame(deterministic_results)
df_deterministic
