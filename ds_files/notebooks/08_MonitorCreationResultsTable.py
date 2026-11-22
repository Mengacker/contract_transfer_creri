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
from databricks.sdk.runtime import dbutils, display

notebook_path = "/Workspace" + os.path.dirname(
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

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from pyspark.sql.functions import lit, current_timestamp
from pyspark.sql import functions as F

from src.utils.input_validation import is_iso_date
from src.monitoring.utils_monit import request_json_to_pandas, spark_schema_by_pandas
from src.config.config_utils import Config

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# logger_azure = logging.getLogger("azure")
# logger_azure.setLevel(logging.ERROR)

pd.set_option("display.max_columns", None)

# COMMAND ----------

# DBTITLE 1,Notebook arguments
# Reset widgets
dbutils.widgets.removeAll()

# Set notebook arguments
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
    label="Snapshot date for the run",
)

# Get notebook arguments
env = dbutils.widgets.get("env")
is_cicd = bool(dbutils.widgets.get("is_cicd") == "true")
snapshot_date_str = dbutils.widgets.get("snapshot_date")

# Validate input
if not is_iso_date(snapshot_date_str):
    raise ValueError(f"Invalid snapshot_date: {snapshot_date_str}")

# COMMAND ----------

# DBTITLE 1,Retrive config params
# Retrieve config values
config_setup = Config("../src/config/config_setup.yml")
catalog = config_setup.get_config_value("environment.{env}.catalog".format(env=env))
schema = config_setup.get_config_value("environment.{env}.schema".format(env=env))

# Output table
output_result_table = config_setup.get_config_value("inference_monitoring")[1]["name"]
output_result_path = f"{catalog}.{schema}.{output_result_table}"
print(f"{output_result_path}   table to write the output results")

# Payload table
endpoint_payload_table = config_setup.get_config_value("inference_monitoring")[0]["name"]
endpoint_payload_path = f"{catalog}.{schema}.{endpoint_payload_table}"
print(f"{endpoint_payload_path}   table to read for the payload log input")

# Inference Master tables
if env == "dev-old":
    master_schema = config_setup.get_config_value("environment.{env}.schema_atum".format(env=env))
else:
    master_schema = config_setup.get_config_value("environment.{env}.schema".format(env=env))

master_volturante_name = config_setup.get_config_value("master_inference")[1]["name"]
master_volturato_name = config_setup.get_config_value("master_inference")[0]["name"]
master_volturante_path = f"{catalog}.{master_schema}.{master_volturante_name}"
master_volturato_path = f"{catalog}.{master_schema}.{master_volturato_name}"
print(f"{master_volturante_path}   table to read for the master volturante")
print(f"{master_volturato_path}   table to read for the master volturato")

# Data configurations
config_data = Config("../src/config/config_data.yml")
input_data = config_data.get_config_value("input_data")
input_features = config_data.get_config_value("input_features")
target_features = config_data.get_config_value("target_features.serving")
engineered_fs = config_data.get_config_value("engineered_features")
not_predictors = config_data.get_config_value("not_predictors")

# Reshape data informations
features_for_engineering = list(
    set([c for m, d1 in engineered_fs.items() for e, d2 in d1.items() for c in d2["origin_fields"]])
)
features_not_for_inference = not_predictors + features_for_engineering
input_data.update(input_data.pop("Forniture", None)[0])  # remove "forniture" nesting
input_features = {c: d2 for m, d1 in input_features.items() for c, d2 in d1.items()}  # remove "metrics" nesting
engineered_fs = {c: d2 for m, d1 in engineered_fs.items() for c, d2 in d1.items()}  # remove "ante/post" nesting


# Get configured Pyspark datatype maps
config_datatype_map = {}
for ff in [input_data, input_features, target_features, engineered_fs]:
    config_datatype_map.update(
        {
            c: d1["pyspark_type"]
            for c, d1 in ff.items()
            if type(d1) == dict and "pyspark_type" in d1.keys() and d1["values"] != "not_used"
        }
    )

print(f"\n{len(config_datatype_map)} n° of columns from config_data.yml")

# COMMAND ----------

# DBTITLE 1,Get the days to save
# days = [(date_and_time.date() - timedelta(days=i + 1)).isoformat() for i in range(0, 2)] # Some days before
today = date_and_time.date().isoformat()
print('today:', today)
day_before = (date_and_time.date() - timedelta(days=1)).isoformat()  # the day before
days = [day_before]
print('day_before:', day_before)
days_of_week = [
    (date_and_time.date() - timedelta(days=date_and_time.date().weekday()) + timedelta(days=i)).isoformat()
    for i in range(0, date_and_time.date().weekday())
]
days = days_of_week +[day_before]
days = list(set(days)) # to drop duplicated dates
print('days:', days)

# Check if the output results table not contain records of the day before
query = """
SELECT *
FROM {table_path}
WHERE DATA_VOLTURA IN ({lista})
""".format(
    table_path=output_result_path, lista=", ".join("'{}'".format(x) for x in days)
)
df_actual = spark.sql(query).toPandas()

days_not_to_save = pd.to_datetime(df_actual['request_date']).dt.strftime('%Y-%m-%d').unique()
print('days_not_to_save:', days_not_to_save)

days_to_save = [d for d in days if d not in days_not_to_save]
print('days_to_save:', days_to_save)

# assert df_actual.shape[0] == 0, f"The {output_result_path} table already contains records for the days: {days}"
assert len(days_to_save) > 0, f"The {output_result_path} table already contains records for the days: {days_to_save}"

# COMMAND ----------

# MAGIC %md
# MAGIC # LOADING

# COMMAND ----------

# DBTITLE 1,payload columns
# OLD PAYLOAD TABLE COLUMNS
# payload_date='date'
# payload_pk = 'databricks_request_id'

# NEW PAYLOAD TABLE COLUMNS
payload_date='request_date'
payload_pk = 'databricks_request_id'

new_cols = ['logging_error_codes', 'requester', 'served_entity_id'] # nuovi campi

map_new_to_old_cols = {
    "databricks_request_id": "databricks_request_id",
    "request_date":"date",
    "client_request_id": "client_request_id",
    "timestamp_ms": "request_time",
    "status_code": "status_code",
    "sampling_fraction": "sampling_fraction",
    "execution_time_ms": "execution_duration_ms",
    "request": "request",
    "response": "response",
    "request_metadata": "request_metadata" # campo non più esistente e da ricreare
}
string_cols = ["client_request_id", "timestamp_ms", "status_code", "sampling_fraction", "execution_time_ms"]

# COMMAND ----------

# DBTITLE 1,load dataset of logs
query = """
SELECT *
FROM {table_path}
WHERE {payload_date} IN ({lista})
""".format(
    table_path=endpoint_payload_path, lista=", ".join("'{}'".format(x) for x in days_to_save),
    payload_date = payload_date
)
df_payload = spark.sql(query)

# Create the "request_metadata" map column with the new columns
map_col = F.create_map(
    F.lit("logging_error_codes"), F.col("logging_error_codes").cast("string"),
    F.lit("requester"), F.col("requester"),
    F.lit("served_entity_id"), F.col("served_entity_id")
)
df_payload = df_payload.withColumn("request_metadata", map_col)

# Renaming new columns to the old name
for old, new in map_new_to_old_cols.items():
    if new != old:
        df_payload = df_payload.withColumnRenamed(new, old)

# Select only the columns used in the first version of the final monitoring table
df_payload = df_payload.select(*list(map_new_to_old_cols.keys()))

for c in string_cols:
    df_payload = df_payload.withColumn(c, df_payload[c].cast("string"))
payload_dtypes = df_payload.dtypes
df_payload = df_payload.toPandas()
output_fields = list(df_payload.columns)
nr_tot = df_payload.shape[0]
nr_id = df_payload[payload_pk].nunique()
if nr_tot != nr_id:
    print(f"{nr_tot} n° of rows != {nr_id} n° of {payload_pk} -> Ci sono dei DUPLICATI!!!!")

    # Drop duplicated rows
    df_payload = df_payload.drop_duplicates(subset=[payload_pk], keep="first").reset_index(drop=True)
    nr_tot = df_payload.shape[0]
    nr_id = df_payload[payload_pk].nunique()
    print(f"{nr_tot} n° of rows =={nr_id} n° of {payload_pk} duplicati eliminati....")
    assert (
        nr_tot == nr_id
    ), f"There are duplicated {payload_pk} {nr_id} != {nr_tot} n° of rows in the antifrode_volture_prod_payload table"
else:
    print(nr_tot, "n° of rows ==", nr_id, "n° of", payload_pk," -> non ci sono duplicati.")
df_payload.head()

# 15-16-17 agosto 2025 non ci sono state volture

# COMMAND ----------

df_payload.groupby([payload_date]).agg({payload_pk: ["count", "nunique"]}).tail(10)

# COMMAND ----------

# DBTITLE 1,reshape dataset of logs
# Initializations
df = df_payload.copy()

request_cols = list(input_data.keys())
response_cols = ["esitoSemaforico", "score"]
l_df = []

# Iterations to transform the request and response json to dataframes and join them
if nr_tot == 0:
    print("There aren't contract transfers in the days considered, so the dataset is empty!")
else:
    for i in range(0, nr_tot):
        id = df[payload_pk][i]

        # Extract the fields in the json of request field
        j_request = df["request"][i]
        try:
            df_1 = request_json_to_pandas(json_string=j_request)
            df_1[payload_pk] = id
            # Complete the dataset with missing columns in dict format
            dd1 = {k: v for k, v in df_1.to_dict().items()}
            dd1.update(
                {k: {row: "" for row, v in dd1["POD/PDR"].items()} for k in request_cols if k not in df_1.columns}
            )
        except:
            dd1 = {k: {0: ""} for k in request_cols}  # dictionary with 1 empty row
            dd1[payload_pk] = {0: id}
        df_request = pd.DataFrame(dd1)

        # Extract the fields in the json of response field
        j_response = df["response"][i]
        try:
            dd2 = json.loads(j_response)["predictions"]
            dd2[payload_pk] = id
            dd2 = {k: {0: v} for k, v in dd2.items()}
            # Complete the dataset with missing columns in dict format
            dd2.update({k: {row: "" for row, v in dd2["score"].items()} for k in response_cols if k not in dd2.keys()})
        except:
            dd2 = {k: {0: ""} for k in response_cols}  # dictionary with 1 empty row
            dd2[payload_pk] = {0: id}
        df_response = pd.DataFrame(dd2)

        # Join the two dataframes
        df_2 = df_response.merge(df_request, on=payload_pk, how="outer")
        df_2 = df_2[[payload_pk] + response_cols + request_cols]

        l_df.append(df_2)

    df_jsons = pd.concat(l_df)
    print(df_jsons.shape, "shape")

# COMMAND ----------

# DBTITLE 1,elaborate specific columns
if nr_tot == 0:
    l_PODPDR = []
    l_CFPIVA = []
    print("There aren't contract transfers in the days considered, so the dataset is empty!")
else:
    df_2 = df_jsons.copy()

    # Elaborate specific columns
    col_cf_piva = [cc for cc in ["CodiceFiscale", "PartitaIva"] if cc in df_2.columns]
    if len(col_cf_piva) == 2:
        df_2.insert(0, "COD_FISCALE_PIVA", df_2["CodiceFiscale"].fillna(df_2["PartitaIva"]))
    else:
        df_2.insert(0, "COD_FISCALE_PIVA", df_2[col_cf_piva[0]])
    ind = df_2["COD_FISCALE_PIVA"] == ""
    if (sum(ind) > 0) & (len(col_cf_piva) > 1):
        df_2.loc[ind, "COD_FISCALE_PIVA"] = df_2.loc[ind, col_cf_piva[-1]]

    df_2["MACROSEGMENTO"] = df_2["SegmentoVolturante"].astype(str)
    df_2["ENCOD_SDF_CANLE_ATTIVAZ_ATT"] = df_2["CanaleAcquisizione"].astype(str)

    if "CellulareVolturante" in df_2.columns:
        df_2["TELEFONO"] = df_2["CellulareVolturante"]
    else:
        df_2["TELEFONO"] = ""

    if "MailVolturante" in df_2.columns:
        df_2["EMAIL"] = df_2["MailVolturante"]
    else:
        df_2["EMAIL"] = ""
    ind = df_2["EMAIL"].isin(["", "nan"])
    if (sum(ind) > 0) & ("PECVolturante" in df_2.columns):
        df_2.loc[ind, "EMAIL"] = df_2.loc[ind, "PECVolturante"]

    if "NomeCognome" in df_2.columns:
        df_2["COGNOME"] = df_2["NomeCognome"].astype(str)
        df_2["COGNOME"] = df_2["COGNOME"].apply(lambda x: x.split("|")[-1])
        if "RagioneSociale" in df_2.columns:
            ind = (df_2["COGNOME"] == "") | (df_2["COGNOME"] == "nan")
            df_2.loc[ind, "COGNOME"] = df_2.loc[ind, "RagioneSociale"]
    else:
        df_2["COGNOME"] = ""

    ind = df_2["COGNOME"].isin(["", "nan"])
    if (sum(ind) > 0) & ("RagioneSociale" in df_2.columns):
        df_2.loc[ind, "COGNOME"] = df_2.loc[ind, "RagioneSociale"]

    for cc in ["COD_FISCALE_PIVA", "COGNOME", "MACROSEGMENTO", "ENCOD_SDF_CANLE_ATTIVAZ_ATT"]:
        df_2[cc] = df_2[cc].str.upper()
    for cc in ["EMAIL"]:
        df_2[cc] = df_2[cc].str.lower()

    df_2.rename(
        columns={
            "POD/PDR": "POD_PDR",
            "COD_FISCALE_PIVA": "COD_FISCALE_PIVA_CLIIN",
            "COGNOME": "COGNOME_CLIIN",
            "MACROSEGMENTO": "MACROSEGMENTO_CLIIN",
            "ENCOD_SDF_CANLE_ATTIVAZ_ATT": "ENCOD_SDF_CANLE_ATTIVAZ_ATT_CLIIN",
            "TELEFONO": "TELEFONO_CLIIN",
            "EMAIL": "EMAIL_CLIIN",
        },
        inplace=True,
    )

    # Merge with the payload dataset
    df_logs = df_payload.merge(df_2, on=payload_pk, how="inner")

    # Insert important columns
    df_logs.insert(
        0,
        "USED_MODEL",
        df_logs["score"].apply(lambda x: "NO ONE" if x == -1 else "DETERMINISTIC" if x in [0, 1] else "SCORING"),
    )
    df_logs.insert(0, "DATA_VOLTURA", df_logs[payload_date])
    df_logs.insert(0, "request_PODPDR_id", df_logs[payload_pk] + "+" + df_logs["POD_PDR"])

    # Get the datatypes map of the request and response dataframe
    request_response_dtypes = list(df_2.dtypes.replace("object", "string").items())
    request_response_dtypes.remove((payload_pk, "string"))
    logs_dtypes = (
        [("request_PODPDR_id", "string"), ("DATA_VOLTURA", "date"), ("USED_MODEL", "string")]
        + payload_dtypes
        + request_response_dtypes
    )
    output_fields = ["request_PODPDR_id", "USED_MODEL"] + output_fields

    # Get the lists of key columns
    l_PODPDR = df_logs["POD_PDR"].values
    l_CFPIVA = df_logs["COD_FISCALE_PIVA_CLIIN"].values

    print(df_logs.shape, df_logs[payload_pk].nunique(), df[payload_pk].nunique())
    assert (
        df_payload[payload_pk].nunique() == df_logs[payload_pk].nunique()
    ), f"There are duplicated {payload_pk} in the payload log table"

# COMMAND ----------

# casi_test = ['23597f81-7e4e-4394-bf71-305bd1fa32f6','57518656-711c-4277-bec6-97b73f48ce39']
# ind = df_logs[payload_pk].isin(casi_test)
# df_logs.loc[ind,:]

# COMMAND ----------

# DBTITLE 1,load master cli. morosita pregressa
print(master_volturante_path)
# partition_date = "DATA_SNAPSHOT" # OLD
partition_date = "snapshot_date" # NEW
pk_col="COD_FISCALE_PIVA_CLIIN"

if len(l_CFPIVA) == 0:
    print(f"There aren't contract transfers in the days considered, so the list of {pk_col} is empty!")
else:
    query = """
    SELECT *
    FROM {table_path}
    WHERE {pk_col} IN ({lista})
    AND flg_lastSnap = 1
    """.format(
        table_path=master_volturante_path,
        lista=", ".join("'{}'".format(x) for x in l_CFPIVA),
        pk_col=pk_col
    )

    df_inference_cfpiva = spark.sql(query)
    df_inference_cfpiva = df_inference_cfpiva.drop(partition_date)
    morosita_pregressa_dtypes = df_inference_cfpiva.dtypes

    df_inference_cfpiva = df_inference_cfpiva.toPandas()
    nr_tot = df_inference_cfpiva.shape[0]
    nr_id = df_inference_cfpiva[pk_col].nunique()
    print(df_inference_cfpiva.shape[1], "n° columns")
    print(f"{nr_tot} n° of rows == {nr_id} n° of {pk_col}")
    assert (
        nr_tot == nr_id
    ), f"There are duplicated {pk_col} {nr_id} != {nr_tot} n° of rows in the {payload_pk} table"
    df_inference_cfpiva.head(3)

# COMMAND ----------

# DBTITLE 1,load master pod pdr master volturato
print(master_volturato_path)
# partition_date = "DATA_SNAPSHOT" # OLD
partition_date = "snapshot_date" # NEW
pk_col="POD_PDR"

if len(l_PODPDR) == 0:
    print(f"There aren't contract transfers in the days considered, so the list of {pk_col} is empty!")
else:
    query = """
    SELECT *
    FROM {table_path}
    WHERE {pk_col} IN ({lista})
    AND flg_lastSnap = 1
    """.format(
        table_path=master_volturato_path,
        lista=", ".join("'{}'".format(x) for x in l_PODPDR),
        pk_col=pk_col
    )

    df_inference_podpdr = spark.sql(query)
    df_inference_podpdr = df_inference_podpdr.withColumn(
        "DATA_SNAPSHOT_TABLES", df_inference_podpdr[partition_date].cast("date")
    )

    master_volturato_dtypes = df_inference_podpdr.dtypes

    df_inference_podpdr = df_inference_podpdr.toPandas()
    nr_tot = df_inference_podpdr.shape[0]
    nr_id = df_inference_podpdr[pk_col].nunique()
    print(f"{nr_tot} n° of rows == {nr_id} n° of {pk_col}")
    assert nr_tot == nr_id

# COMMAND ----------

# MAGIC %md
# MAGIC # CLEANING

# COMMAND ----------

# DBTITLE 1,join datasets
if "df_logs" not in locals():
    print("There aren't contract transfers in the days considered, so the dataset is empty!")
else:
    # Two left join
    for df in [df_logs, df_inference_cfpiva, df_inference_podpdr]:
        print(df.shape)

    df_result = df_logs.merge(df_inference_cfpiva, on="COD_FISCALE_PIVA_CLIIN", how="left")
    df_result = df_result.merge(df_inference_podpdr, on="POD_PDR", how="left")

    col_results = list(df_result.columns)

    print(
        df_result.shape,
        df_result["POD_PDR"].nunique(),
        "unique POD_PDR, ",
        df_result["COD_FISCALE_PIVA_CLIIN"].nunique(),
        "unique COD_FISCALE_PIVA_CLIIN",
    )

# COMMAND ----------

# DBTITLE 1,Fix datatypes map and select columns
if "df_result" not in locals():
    print("There aren't contract transfers in the days considered, so the dataset is empty!")
else:
    df = df_result.copy()

    # Remove two duplicated element in the datatype maps
    try:
        morosita_pregressa_dtypes.remove(("COD_FISCALE_PIVA_CLIIN", "string"))
    except ValueError:
        pass
    try:
        master_volturato_dtypes.remove(("POD_PDR", "string"))
    except ValueError:
        pass
    actual_datatype_maplist = logs_dtypes + morosita_pregressa_dtypes + master_volturato_dtypes

    # Check and replace datatypes in the map
    cols_configured = []
    not_configured = []
    other_features = []
    for ii, (col, dtype) in enumerate(actual_datatype_maplist):
        col2 = col  # .split("_CLIIN")[0]  # to manage "CLIIN" similar fields
        if col2 in config_datatype_map.keys():
            cols_configured.append(col)
            if config_datatype_map[col2] != "":
                if dtype != config_datatype_map[col2]:
                    # Replace field's datatype in the list-map
                    actual_datatype_maplist[ii] = (col, config_datatype_map[col2])
                else:
                    continue
            else:
                # list the unmanaged pyspark_type field in the file config.yml
                not_configured.append(col)
        else:
            if col not in features_not_for_inference + output_fields:
                other_features.append(col)
            else:
                not_configured.append(col)

    # Selecting columns
    cols_to_save = [c for c in df.columns if c in cols_configured + output_fields + other_features]
    actual_datatype_maplist = [(col, dtype) for col, dtype in actual_datatype_maplist if col in cols_to_save]

    df_fixed = df[cols_to_save].copy()
    # df_fixed = df.copy()
    print(f"{df.shape} dataframe shape before selecting columns\n")
    print(f"{df_fixed.shape} afterwards\n")
    print(f"{len(actual_datatype_maplist)} n° of columns in the actual_datatype_maplist\n")

# COMMAND ----------

# DBTITLE 1,Spark dataframe trasformation
partition_date = "snapshot_date"

if "df_fixed" not in locals():
    print("There aren't contract transfers in the days considered, so the dataset is empty!")
else:
    df = df_fixed.copy()
    print(df.shape, "shape Pandas")

    df, schema = spark_schema_by_pandas(df_pandas=df, datatypes_tuple_list=actual_datatype_maplist)

    for c in df.columns:
        try: # fix special case
            if any(isinstance(x, list) for x in df['FlagVulnerabilityVolturante']):
                df['FlagVulnerabilityVolturante']=df['FlagVulnerabilityVolturante'].apply(lambda s: ','.join(s) if isinstance(s, list) else s)
        except Exception as e:
            print(e)
            logger.warning(f"Error in the conversion of 'FlagVulnerabilityVolturante': {e}")

    df_spark = spark.createDataFrame(df, schema=schema)

    # Insert "snapshot_date" OF RESULTS TABLE CREATION
    df_spark = df_spark.withColumn(partition_date, lit(datetime.now().date()))
    dtypes_new_map = df_spark.dtypes

    print(f"({df_spark.count()}, {len(df_spark.columns)}) shape Pyspark")
    display(df_spark.limit(5))

# COMMAND ----------

# DBTITLE 1,check duplicated columns
if "dtypes_new_map" not in locals():
    print("There aren't contract transfers in the days considered, so the dataset is empty!")
else:
    cols_new, dtypes_new = zip(*dtypes_new_map)
    cols_new_nodup, dtypes_new_nodup = zip(*dict(dtypes_new_map).items())

    assert len(cols_new) == len(
        cols_new_nodup
    ), f"There are duplicated columns in the datatype schema of new Pyspark dataframe to save"

# COMMAND ----------

# MAGIC %md
# MAGIC # WRITING AND CHECKING

# COMMAND ----------

# DBTITLE 1,write data in catalog
partition_date = "snapshot_date"
if is_cicd:
    if "df_spark" not in locals():
        print("There aren't contract transfers in the days considered, so the new dataset is empty!")
    else:
        df_to_save = df_spark

        dd = day_before.replace("-", "")
        save_path = output_result_path
        print(save_path)

        df_to_save.withColumn(partition_date, F.col(partition_date).cast("string")).write.mode("append").option(
            "mergeSchema", "true"
        ).saveAsTable(save_path)
        display(df_to_save.limit(5))

# COMMAND ----------

# DBTITLE 1,check saved table
check_path = output_result_path
# check_path = output_result_path #+'_13-14-15_may_2025'
print(check_path)
query = """
WITH tab as (
            SELECT *
            FROM {table_path}
            WHERE {payload_date} IN ({lista})
)
SELECT *
FROM tab
""".format(
    table_path=check_path, lista=", ".join("'{}'".format(x) for x in days),
    payload_date= payload_date
)

df_sql = spark.sql(query)
check_dtype_map = df_sql.dtypes
print(df_sql.count(), "n° rows,", len(df_sql.columns), "n° columns\n")

df_sql = df_sql.toPandas()
all_cols_saved = list(df_sql.columns)

df_sql_group = (
    df_sql.groupby([partition_date, "DATA_VOLTURA"]).agg({"request_PODPDR_id": ["count", "nunique"]}).reset_index()
).sort_values(["DATA_VOLTURA", partition_date])
df_sql_group.columns = [partition_date, "DATA_VOLTURA", "count", "nunique"]
df_sql_group["difference"] = df_sql_group["count"] - df_sql_group["nunique"]

print("The total numbers of request_PODPDR_id are:")
print(df_sql_group[["count", "nunique", "difference"]].sum())
df_sql_group

# 15-16-17 agosto 2025 non ci sono state volture
