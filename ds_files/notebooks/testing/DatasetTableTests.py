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
import pandas as pd
from datetime import date, datetime, timedelta

from pyspark.sql import functions as F, SparkSession

spark: SparkSession = SparkSession.builder.getOrCreate()

from src.utils.input_validation import is_iso_date
from src.config.config_utils import Config

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COMMAND ----------

# DBTITLE 1,Notebook arguments
# --- Set notebook arguments --- #
# Reset widgets
dbutils.widgets.removeAll()

# Notebook Environment
dbutils.widgets.text("env", "prod-dpp", "Environment Name")  # dev, prod-sbx

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
snapshot_date_str = dbutils.widgets.get("snapshot_date")


# Validate input
if not is_iso_date(snapshot_date_str):
    raise ValueError(f"Invalid snapshot_date: {snapshot_date_str}")

list_date_to_check = pd.date_range(start="2025-06-01", end=date.today()).strftime("%Y-%m-%d").tolist()
print('list_date_to_check:',list_date_to_check)
print('\nEnviroment:', env)

# COMMAND ----------

# DBTITLE 1,Retrieve config values
# Retrieve config values
config = Config(config_path="../../src/config/", name_list=["config_setup.yml", "config_model.yml"])
config_tests = Config(config_path="../../src/config/config_tests.yml")

catalog = config.get_config_value("environment.{env}.catalog".format(env=env))
schema = config.get_config_value("environment.{env}.schema".format(env=env))
path = f"{catalog}.{schema}."
print(path)

# COMMAND ----------

# DBTITLE 1,Set variables
diz_tab_key = {
    "creri_met_targetcontofattclientimorosi": "COD_CONTO_NETA",
    "creri_met_metricaflgvolturelordenette": "PK_PROPOSAL_LINE_ITEM",
    "creri_met_pontesdfproposal": "FK_PROPOSAL_LINE_ITEM",
    "creri_met_morositapregressapodpdr": "POD_PDR",
    "creri_met_morositapregressacliente": "COD_FISCALE_PIVA",
    "creri_met_accounttocodfiscalepiva": "PK_ACCOUNT",
    "creri_met_sdfattivazioneattivi": "PK_PDF",
    "creri_met_clienteattributi": "PK_ACCOUNT",
    "creri_met_metricapdfpodpdrgtwv": "COD_PDF",
    "creri_met_dataultletturaattivswingas": "COD_PDR",
    "creri_met_dataultletturaattivswinpwr": "COD_POD",
    "creri_train_cb_volturato": "PK_PROPOSAL_LINE_ITEM",
    "creri_train_cb_volturante": "PK_PROPOSAL_LINE_ITEM",
    "creri_train_mstr_volturato": "PK_PROPOSAL_LINE_ITEM",
    "creri_train_mstr_volturante": "PK_PROPOSAL_LINE_ITEM",
    "creri_train_mstr_credit_check_volture": "PK_PROPOSAL_LINE_ITEM",
    "creri_infer_mstr_volturato": "POD_PDR",
    "creri_infer_mstr_volturante": "COD_FISCALE_PIVA_CLIIN",
    # DA CONTROLLARE IL GIORNO DOPO:
    # 'creri_out_infer_payload': 'databricks_request_id', --group by date
    # 'creri_out_infer_final': 'request_PODPDR_id' -- group by DATA_VOLTURA
}
l_warnings = []

# COMMAND ----------

# DBTITLE 1,Set snapshot dates

delta_months = 15  # + 1  # chosed 15 month ago plus 1 month for the skillet
snapshot_date_today = date.today().strftime("%Y-%m-%d")
current_snapshot_date = snapshot_date_today[:-2] + "01"
date_today = datetime.strptime(snapshot_date_today, "%Y-%m-%d")
date_month_ago = date_today - timedelta(days=delta_months * 365.25 / 12)
end_date = "-".join([str(date_month_ago.year), str(date_month_ago.month), "1"])

end_date = datetime.strptime(end_date, "%Y-%m-%d")
start_date = datetime.strptime("2022-07-01", "%Y-%m-%d")

# ---------------------------------- snapshot_date_train_cb_mstr_list
snapshot_date_train_cb_mstr_list = []
current_date = start_date
while current_date <= end_date:
    snapshot_date_train_cb_mstr_list.append(current_date.strftime("%Y-%m-%d"))
    current_date += timedelta(days=31)  # Move to the next month
    current_date = str(current_date).split(" ")[0][:-2] + "01"  # Reset to the first day of the month
    current_date = datetime.strptime(current_date, "%Y-%m-%d")
snapshot_date_train_cb_mstr_list.reverse()

# ---------------------------------- snapshot_date_metrics_list
end_date = datetime.strptime(current_snapshot_date, "%Y-%m-%d")
snapshot_date_metrics_list = []
current_date = start_date
while current_date <= end_date:
    snapshot_date_metrics_list.append(current_date.strftime("%Y-%m-%d"))
    current_date += timedelta(days=31)  # Move to the next month
    current_date = str(current_date).split(" ")[0][:-2] + "01"  # Reset to the first day of the month
    current_date = datetime.strptime(current_date, "%Y-%m-%d")
snapshot_date_metrics_list.reverse()

# ---------------------------------- snapshot_date of the first_day_of_week
snapshot_first_day_of_week = datetime.today() - timedelta(days=datetime.today().weekday())
snapshot_first_day_of_week = snapshot_first_day_of_week.strftime("%Y-%m-%d")

mondays = pd.date_range(start="2026-06-22", end=snapshot_date_today, freq='W-MON').strftime("%Y-%m-%d").tolist()
mondays = mondays + ['2026-05-18']

print(f"Today is {snapshot_date_today}.")
print(f"The current monthly snapshot date is {current_snapshot_date}.")
print(f"The current weekly snapshot date is {snapshot_first_day_of_week}.")
print(f"\nThe list of snapshot date for the materialized metrics are in number {len(snapshot_date_metrics_list)} months.")
print(f"\nThe list of snapshot date for the training are in number {len(snapshot_date_train_cb_mstr_list)}, the list is:\n",snapshot_date_train_cb_mstr_list)

print(f"\nThe list of snapshot date for the weekly materialized tables are: {mondays}")

# COMMAND ----------

# MAGIC %md
# MAGIC # CHECK MATERIALIZED TABLES

# COMMAND ----------

# DBTITLE 1,TEST duplicates
print(path)
perc_threshold = 1  # %
date_col = "snapshot_date"
l_df = []
i = 0
for table, key_col in diz_tab_key.items():
    print(i, table)
    i += 1
    query = """
        WITH tab AS (
            SELECT *, CONCAT({key}, {date_col}) AS concat_key
            FROM {table_path}
            )
        SELECT COUNT(*) AS n_rows, count(distinct concat_key) AS n_distinct_keys, COUNT(*) - count(distinct concat_key) AS n_duplicates
        FROM tab
        WHERE flg_lastSnap = 1
        """.format(
        table_path=path + table, key=key_col, date_col=date_col
    )

    df = spark.sql(query)
    df = df.toPandas()

    df["path"] = path
    df["table"] = table
    df["key_col"] = key_col

    df["perc_duplicates"] = df["n_duplicates"].values[0] / df["n_distinct_keys"].values[0]

    if df["perc_duplicates"].values[0] > perc_threshold / 100:
        df[f"TEST_{perc_threshold}%_duplicates"] = True
        mess = f"The {table} has the {key_col} column with more than {perc_threshold}% of duplicates"
        l_warnings.append(mess)
        logger.warning(mess)
    else:
        df[f"TEST_{perc_threshold}%_duplicates"] = False

    df = df[
        [
            "path",
            "table",
            "key_col",
            f"TEST_{perc_threshold}%_duplicates",
            "n_duplicates",
            "n_rows",
            "n_distinct_keys",
            "perc_duplicates",
        ]
    ]

    l_df.append(df)

df_dup = pd.concat(l_df)
df_dup

# COMMAND ----------

# DBTITLE 1,Grouped tables by snapshot date
print(path)
date_col = "snapshot_date"
l_df_monthly = []
i = 0
for table, key_col in diz_tab_key.items():
    print(i, table)
    i += 1

    query = """
        SELECT {date_col}, dat_startDate
        , COUNT(*) AS n_rows, count(distinct {key}) AS n_distinct_keys, COUNT(*) - count(distinct {key}) AS n_duplicates
        FROM {table_path}
        WHERE flg_lastSnap = 1
        group by {date_col}, dat_startDate
        """.format(
        table_path=path + table, key=key_col, date_col=date_col
    )

    df = spark.sql(query).toPandas()
    df["path"] = path
    df["table"] = table
    df["key_col"] = key_col
    df["snapshot_date"] = df["snapshot_date"].astype(str)
    df["dat_startDate"] = pd.to_datetime(df["dat_startDate"], format='ISO8601').dt.date.astype(str)
    df = df[["path", "table", "key_col", "snapshot_date", "dat_startDate", "n_rows", "n_distinct_keys", "n_duplicates"]]

    l_df_monthly.append(df)


df_monthly = pd.concat(l_df_monthly)
df_monthly

# COMMAND ----------

# DBTITLE 1,TEST snapshot dates
df_snap = df_dup[["path", "table"]].copy()
df_snap.set_index("table", inplace=True)
i = 0
for table, key_col in diz_tab_key.items():
    print(i, table)
    df = l_df_monthly[i].copy()
    i += 1

    l_snap = list(df["snapshot_date"].values)

    # --------------------------------------------------------- missing snapshot dates
    if ("creri_met" in table) and ("target" not in table):
        miss_snap = [d for d in snapshot_date_metrics_list if d not in l_snap]
        if ("creri_met_morositapregressa" in table) and (snapshot_first_day_of_week not in l_snap):
            miss_snap = miss_snap + [snapshot_first_day_of_week]
    elif ("creri_train" in table) or ("target" in table):
        miss_snap = [d for d in snapshot_date_train_cb_mstr_list if d not in l_snap]
    elif "creri_infer" in table:
        miss_snap = current_snapshot_date if current_snapshot_date not in l_snap else []
    else:
        raise Exception(f"Table {table} not recognized")

    if len(miss_snap) > 0:
        df_snap.loc[table, "TEST_missing_snapshot"] = True
        mess = f"The {table} has missing snapshot dates: {miss_snap}."
        l_warnings.append(mess)
        logger.warning(mess)
    else:
        df_snap.loc[table, "TEST_missing_snapshot"] = False

    df_snap.loc[table, "miss_snap_num"] = len(miss_snap)
    df_snap.loc[table, "miss_snap"] = str(miss_snap)

    # --------------------------------------------------------- excessive snapshot dates
    if ("creri_met" in table) and ("target" not in table):
        excess_snap = [d for d in l_snap if d not in snapshot_date_metrics_list]
        if "creri_met_morositapregressa" in table:
            excess_snap = [d for d in excess_snap if d != snapshot_first_day_of_week if d not in mondays]
    elif ("creri_train" in table) or ("target" in table):
        excess_snap = [d for d in l_snap if d not in snapshot_date_train_cb_mstr_list]
    elif "creri_infer" in table:
        excess_snap = [d for d in l_snap if d not in [current_snapshot_date]]
    else:
        raise Exception(f"Table {table} not recognized")

    if len(excess_snap) > 0:
        df_snap.loc[table, "TEST_excessive_snapshot"] = True
        mess = f"The {table} has excessive snapshot dates: {excess_snap}."
        l_warnings.append(mess)
        logger.warning(mess)
    else:
        df_snap.loc[table, "TEST_excessive_snapshot"] = False

    df_snap.loc[table, "excess_snap_num"] = len(excess_snap)
    df_snap.loc[table, "excess_snap"] = str(excess_snap)

df_snap.reset_index()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT snapshot_date, dat_startDate
# MAGIC   , count(*) as total_rows
# MAGIC   , sum(case when CLI_INSOLUTO_LORDO is not null then 1 else 0 end) as not_null_CLI_INSOLUTO_LORDO
# MAGIC   , sum(case when PDF_INSOLUTO_LORDO_ALL is not null then 1 else 0 end) as not_null_PDF_INSOLUTO_LORDO_ALL
# MAGIC FROM p_dpp_core_ai.creri.creri_infer_mstr_volturato
# MAGIC WHERE flg_lastSnap = 1
# MAGIC GROUP BY snapshot_date, dat_startDate

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT snapshot_date, dat_startDate
# MAGIC   , count(*) as total_rows
# MAGIC   , sum(case when CLI_INSOLUTO_LORDO_CLIIN is not null then 1 else 0 end) as not_null_CLI_INSOLUTO_LORDO
# MAGIC FROM p_dpp_core_ai.creri.creri_infer_mstr_volturante
# MAGIC WHERE flg_lastSnap = 1
# MAGIC GROUP BY snapshot_date, dat_startDate

# COMMAND ----------

# DBTITLE 1,TEST target values in train master
print(path)
perc_threshold = 80  # %
denominatore = 100
diz = {
    "creri_train_mstr_volturante": "FLG_CLI_MOROSO",
    "creri_train_mstr_credit_check_volture": "FLG_CLI_MOROSO",
}
l_df = []
for table, col in diz.items():
    print(table)
    query = """
            SELECT snapshot_date, COUNT(*) AS n_rows, SUM(CASE WHEN {col} IS NOT NULL THEN 1 ELSE 0 END) AS n_not_null
            FROM {table_path}
            WHERE flg_lastSnap = 1
            GROUP BY snapshot_date
            ORDER BY snapshot_date DESC
            """.format(
        table_path=path + table, col=col
    )
    df = spark.sql(query)
    df = df.toPandas()
    df["snapshot_date"] = df["snapshot_date"].astype(str)

    df["perc_not_null"] = df["n_not_null"] / df["n_rows"]
    df[f"TEST_{perc_threshold}%_not_null"] = df["perc_not_null"] < perc_threshold / denominatore
    # Filter out the rows with perc_null = False
    df = df[df[f"TEST_{perc_threshold}%_not_null"]]

    if df.shape[0] > 0:
        mess = f"The {table} has the {col} column with less than {perc_threshold}% of NOT null values in the snapshot dates {df['snapshot_date'].values}, so the training master is builted with a wrong target dataset."
        l_warnings.append(mess)
        logger.warning(mess)

    df["column"] = col
    df["table"] = table
    df["path"] = path

    df = df[
        [
            "path",
            "table",
            "column",
            "snapshot_date",
            "n_rows",
            "n_not_null",
            "perc_not_null",
            f"TEST_{perc_threshold}%_not_null",
        ]
    ]

    l_df.append(df)

df_target_in_train_mstr = pd.concat(l_df, axis=0)

if df_target_in_train_mstr.shape[0] == 0:
    print(f"\nThere are no warnings for the target in the training master datasets ^.^\n")

df_target_in_train_mstr

# COMMAND ----------

# DBTITLE 1,TEST join of cb
print(path)
perc_threshold = 90  # %
denominatore = 100
l_df = []
i = 0
diz = {
    "creri_train_cb_volturato": ["PK_PROPOSAL_LINE_ITEM", "PK_PDF", "POD_PDR", "COD_FISCALE_PIVA"],
    "creri_train_cb_volturante": ["PK_PROPOSAL_LINE_ITEM", "PK_PDF", "POD_PDR", "COD_FISCALE_PIVA"],
}
for table, key_cols in diz.items():
    print(i, table)

    sub_l_df = [pd.DataFrame({"path": [path], "table": [table]})]
    for j, k in enumerate(key_cols):
        query = """
            SELECT COUNT(*) AS n_r{j}ws, count_if({key} IS NOT NULL) AS n_not_null_{key}
            FROM {table_path}
            WHERE flg_lastSnap = 1
            """.format(
            table_path=path + table, key=k, j=j
        )
        df = spark.sql(query)
        df = df.toPandas()
        df[f"perc_not_null_{k}"] = df[f"n_not_null_{k}"].values[0] / df[f"n_r{j}ws"].values[0]

        if df[f"perc_not_null_{k}"].values[0] < perc_threshold / denominatore:
            df[f"TEST_{perc_threshold}%_not_null_{k}"] = True
            mess = f"The {table} has the {k} column with less than {perc_threshold}% of NOT null values, so the join has not worked correctly"
            l_warnings.append(mess)
            logger.warning(mess)
        else:
            df[f"TEST_{perc_threshold}%_not_null_{k}"] = False

        sub_l_df.append(df)
    df = pd.concat(sub_l_df, axis=1)

    df = df[
        ["path", "table", "n_r0ws"]
        + [f"TEST_{perc_threshold}%_not_null_{k}" for k in key_cols]
        + ["n_not_null_" + k for k in key_cols]
        + ["perc_not_null_" + k for k in key_cols]
    ]

    l_df.append(df.set_index("table").T.reset_index().rename(columns={"index": f"columns_{i}"}))
    i += 1

df_joins_cb = pd.concat(l_df, axis=1)

df_joins_cb

# COMMAND ----------

# DBTITLE 1,TEST join train master
print(path)
perc_threshold = 1  # %
denominatore = 100
l_df = []
i = 0
diz = {
    "creri_train_mstr_credit_check_volture": [
        "PK_PROPOSAL_LINE_ITEM",
        "FLG_CLI_MOROSO",
        "POD_PDR",
        "POD_PDR_CLIIN",
        "SDF_CANLE_ATTIVAZ_ATT_CLIIN",
        "CLI_INSOLUTO_LORDO_CLIIN",
        "CLI_AGE",
        "CLI_INSOLUTO_LORDO",
        "PDF_INSOLUTO_LORDO_ALL",
        "DATA_ULT_LETTURA_ATTIV_SWINPWR",
        "DATA_ULT_LETTURA_ATTIV_SWINGAS",
    ],
    "creri_train_mstr_volturato": [
        "PK_PROPOSAL_LINE_ITEM",
        "POD_PDR",
        "CLI_AGE",
        "PDF_INSOLUTO_LORDO_ALL",
        "CLI_INSOLUTO_LORDO",
        "DATA_ULT_LETTURA_ATTIV_SWINGAS",
        "DATA_ULT_LETTURA_ATTIV_SWINPWR",
    ],
    "creri_train_mstr_volturante": [
        "PK_PROPOSAL_LINE_ITEM",
        "POD_PDR",
        "SDF_CANLE_ATTIVAZ_ATT",
        "CLI_INSOLUTO_LORDO",
        "FLG_CLI_MOROSO",
    ],
}
for table, key_cols in diz.items():
    print(i, table)

    sub_l_df = [pd.DataFrame({"path": [path], "table": [table]})]
    for j, k in enumerate(key_cols):
        query = """
            SELECT COUNT(*) AS n_r{j}ws, count_if({key} IS NOT NULL) AS n_not_null_{key}
            FROM {table_path}
            WHERE flg_lastSnap = 1
            """.format(
            table_path=path + table, key=k, j=j
        )
        df = spark.sql(query)
        df = df.toPandas()
        df[f"perc_not_null_{k}"] = df[f"n_not_null_{k}"].values[0] / df[f"n_r{j}ws"].values[0]

        if df[f"perc_not_null_{k}"].values[0] < perc_threshold / denominatore:
            df[f"TEST_{perc_threshold}%_not_null_{k}"] = True
            mess = f"The {table} has the {k} column with less than {perc_threshold}% of NOT null values, so the join has not worked correctly"
            l_warnings.append(mess)
            logger.warning(mess)
        else:
            df[f"TEST_{perc_threshold}%_not_null_{k}"] = False

        sub_l_df.append(df)
    df = pd.concat(sub_l_df, axis=1)

    df = df[
        ["path", "table", "n_r0ws"]
        + [f"TEST_{perc_threshold}%_not_null_{k}" for k in key_cols]
        + ["n_not_null_" + k for k in key_cols]
        + ["perc_not_null_" + k for k in key_cols]
    ]

    l_df.append(df.set_index("table").T.reset_index().rename(columns={"index": f"columns_{i}"}))
    i += 1

df_joins_train_msr = pd.concat(l_df, axis=1)
df_joins_train_msr

# COMMAND ----------

# DBTITLE 1,TEST join infer master
print(path)
perc_threshold = 1  # %
denominatore = 100
l_df = []
i = 0
diz = {
    "creri_infer_mstr_volturato": [
        "POD_PDR",
        "COD_FISCALE_PIVA",
        "SDF_CANLE_ATTIVAZ_ATT",
        "CLI_AGE",
        "CLI_INSOLUTO_LORDO",
        "PDF_INSOLUTO_LORDO_ALL",
        "DATA_ULT_LETTURA_ATTIV_SWINGAS",
        "DATA_ULT_LETTURA_ATTIV_SWINPWR",
    ],
    "creri_infer_mstr_volturante": ["COD_FISCALE_PIVA_CLIIN", "CLI_INSOLUTO_LORDO_CLIIN"],
}
for table, key_cols in diz.items():
    print(i, table)

    sub_l_df = [pd.DataFrame({"path": [path], "table": [table]})]
    for j, k in enumerate(key_cols):
        query = """
            SELECT COUNT(*) AS n_r{j}ws, count_if({key} IS NOT NULL) AS n_not_null_{key}
            FROM {table_path}
            WHERE flg_lastSnap = 1
            """.format(
            table_path=path + table, key=k, j=j
        )
        df = spark.sql(query)
        df = df.toPandas()
        df[f"perc_not_null_{k}"] = df[f"n_not_null_{k}"].values[0] / df[f"n_r{j}ws"].values[0]

        if df[f"perc_not_null_{k}"].values[0] < perc_threshold / denominatore:
            df[f"TEST_{perc_threshold}%_not_null_{k}"] = True
            mess = f"The {table} has the {k} column with less than {perc_threshold}% of NOT null values, so the join has not worked correctly"
            l_warnings.append(mess)
            logger.warning(mess)
        else:
            df[f"TEST_{perc_threshold}%_not_null_{k}"] = False

        sub_l_df.append(df)
    df = pd.concat(sub_l_df, axis=1)

    df = df[
        ["path", "table", "n_r0ws"]
        + [f"TEST_{perc_threshold}%_not_null_{k}" for k in key_cols]
        + ["n_not_null_" + k for k in key_cols]
        + ["perc_not_null_" + k for k in key_cols]
    ]

    l_df.append(df.set_index("table").T.reset_index().rename(columns={"index": f"columns_{i}"}))
    i += 1

df_joins_infer_msr = pd.concat(l_df, axis=1)
df_joins_infer_msr

# COMMAND ----------

# DBTITLE 1,TEST feature engineering
print(path)
perc_threshold = 10  # %
denominatore = 100
l_df = []
i = 0
diz = {
    "creri_train_mstr_credit_check_volture": [
        "ENCOD_MACROSEGMENTO",
        "ENCOD_MACROSEGMENTO_CLIIN",
        "ENCOD_SDF_CANLE_ATTIVAZ_ATT_CLIIN",
        "FLG_INSOLUTO",
        "FLG_COMTY_GAS",
        "FLG_COMUNE_UGUALE",
        "FLG_PROV_UGUALE",
        "FLG_REGIONE_UGUALE",
        "FLG_NAZ_UGUALE",
        "FLG_COGNOME_UGUALE",
        "FLG_EMAIL_UGUALE",
        "FLG_TELEFONO_UGUALE",
        "AGING_METER_ATTIV_SWIN",
        "UTILITY_SET"
        #   , 'CLI_AGE_RANGE_30-40','CLI_AGE_RANGE_40-50','CLI_AGE_RANGE_50-60'
        #   , 'CLI_AGE_RANGE_60-70','CLI_AGE_RANGE_70-80','CLI_AGE_RANGE_80-200'
    ],
}
for table, key_cols in diz.items():
    print(i, table)

    sub_l_df = [pd.DataFrame({"path": [path], "table": [table]})]
    for j, k in enumerate(key_cols):
        print(j, k)
        query = """
            SELECT COUNT(*) AS n_r{j}ws, count_if(`{key}` IS NOT NULL) AS n_not_null_{key}
            FROM {table_path}
            WHERE flg_lastSnap = 1
            """.format(
            table_path=path + table, key=k, j=j
        )
        df = spark.sql(query)
        df = df.toPandas()
        df[f"perc_not_null_{k}"] = df[f"n_not_null_{k}"].values[0] / df[f"n_r{j}ws"].values[0]

        if df[f"perc_not_null_{k}"].values[0] < perc_threshold / denominatore:
            df[f"TEST_{perc_threshold}%_not_null_{k}"] = True
            mess = f"The {table} has the {k} column with less than {perc_threshold}% of NOT null values, so the join has not worked correctly"
            l_warnings.append(mess)
            logger.warning(mess)
        else:
            df[f"TEST_{perc_threshold}%_not_null_{k}"] = False

        sub_l_df.append(df)
    df = pd.concat(sub_l_df, axis=1)

    df = df[
        ["path", "table", "n_r0ws"]
        + [f"TEST_{perc_threshold}%_not_null_{k}" for k in key_cols]
        + ["n_not_null_" + k for k in key_cols]
        + ["perc_not_null_" + k for k in key_cols]
    ]

    l_df.append(df.set_index("table").T.reset_index().rename(columns={"index": f"columns_{i}"}))
    i += 1

df_feng = pd.concat(l_df, axis=1)

df_feng

# COMMAND ----------

# DBTITLE 1,TEST number of transfers
table = "creri_train_mstr_credit_check_volture"
print(path + table)

# Thresholds parameters in reference to the monthly average
thesholds_num_volture = (10000, 50000)
thesholds_perc_morose = (1, 3)

query = """
    SELECT snapshot_date
    , COUNT(*) as n_volture
    , SUM(CAST(FLG_CLI_MOROSO AS INT)) as n_volture_morose
    , ROUND(100*sum(CAST(FLG_CLI_MOROSO AS INT))/COUNT(*),1) as perc_morose --%
    FROM {table_path}
    WHERE flg_lastSnap = 1
    GROUP BY snapshot_date
    """.format(
    table_path=path + table
)

df_num_trans = spark.sql(query).toPandas()

df_num_trans["TEST_n_volture"] = (df_num_trans["n_volture"] < thesholds_num_volture[0]) | (
    df_num_trans["n_volture"] > thesholds_num_volture[1]
)

df_num_trans["TEST_perc_morose"] = (df_num_trans["perc_morose"] < thesholds_perc_morose[0]) | (
    df_num_trans["perc_morose"] > thesholds_perc_morose[1]
)

if df_num_trans["TEST_n_volture"].sum() > 0:
    months = df_num_trans.loc[df_num_trans["TEST_n_volture"], "snapshot_date"].values
    mess = f"The {table} has less than {thesholds_num_volture[0]} or more than {thesholds_num_volture[1]} contract transfers  in months: {months}."
    l_warnings.append(mess)
    logger.warning(mess)

if df_num_trans["TEST_perc_morose"].sum() > 0:
    months = df_num_trans.loc[df_num_trans["TEST_perc_morose"], "snapshot_date"].values
    mess = f"The {table} has less than {thesholds_perc_morose[0]}% or more than {thesholds_perc_morose[1]}% of fraudulent defaulting of contract transfers in months: {months}."
    l_warnings.append(mess)
    logger.warning(mess)

df_num_trans = df_num_trans.sort_values("snapshot_date", ascending=False)
df_num_trans

# COMMAND ----------

# DBTITLE 1,TEST yellow list
table = "creri_met_morositapregressapodpdr"
print(path + table)

# Thresholds parameters in reference to the monthly average
thesholds_num_morose = (500, 3500)

query = """
    SELECT snapshot_date
    , COUNT(*) AS n_podpdr_clienti
    , SUM(CASE WHEN FLG_MOROSITA_FORN = True THEN 1 ELSE 0 END) AS n_podpdr_morosi
    FROM {table_path}
    WHERE flg_lastSnap = 1
    GROUP BY snapshot_date
    ORDER BY snapshot_date
    """.format(
    table_path=path + table
)

df_yellow = spark.sql(query).toPandas()

df_yellow["TEST_n_yellow_podpdr"] = (df_yellow["n_podpdr_morosi"] < thesholds_num_morose[0]) | (
    df_yellow["n_podpdr_morosi"] > thesholds_num_morose[1]
)

if df_yellow["TEST_n_yellow_podpdr"].sum() > 0:
    months = df_yellow.loc[df_yellow["TEST_n_yellow_podpdr"], "snapshot_date"].values
    mess = f"The {table} has less than {thesholds_num_morose[0]} or more than {thesholds_num_morose[1]} of yellow POD/PDR in months: {months}."
    l_warnings.append(mess)
    logger.warning(mess)

df_yellow = df_yellow.sort_values("snapshot_date", ascending=False)
df_yellow

# COMMAND ----------

# DBTITLE 1,TEST number of met target
table = "creri_met_targetcontofattclientimorosi"
print(path + table)


# Thresholds parameters in reference to the monthly average
thesholds_cli_morosi = (50000, 200000)

query = """
    SELECT snapshot_date
    , COUNT(*) AS n_clienti
    , SUM(CASE WHEN FLG_CLI_MOROSO = True THEN 1 ELSE 0 END) AS n_clienti_morosi
    FROM {table_path}
    WHERE flg_lastSnap = 1
    GROUP BY snapshot_date
    ORDER BY snapshot_date
    """.format(
    table_path=path + table
)

df_target = spark.sql(query).toPandas()

df_target["TEST_n_met_target"] = (df_target["n_clienti_morosi"] < thesholds_cli_morosi[0]) | (
    df_target["n_clienti_morosi"] > thesholds_cli_morosi[1]
)

if df_target["TEST_n_met_target"].sum() > 0:
    months = df_target.loc[df_target["TEST_n_met_target"], "snapshot_date"].values
    mess = f"The {table} has less than {thesholds_cli_morosi[0]} or more than {thesholds_cli_morosi[1]} of True FLG_CLI_MOROSO in months: {months}."
    l_warnings.append(mess)
    logger.warning(mess)

df_target = df_target.sort_values("snapshot_date", ascending=False)
df_target

# COMMAND ----------

# DBTITLE 1,FINAL ASSERT
assert len(l_warnings) == 0, [print(w) for w in l_warnings]
# La metrica creri_met_morositapregressapodpdr NON usa la sua versione aggiornata.
# La metrica creri_met_dataultletturaattivswingas ha dei count molto bassi e strani nei primi mesi.
