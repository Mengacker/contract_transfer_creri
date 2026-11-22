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
import logging
from datetime import datetime
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.utils.input_validation import is_iso_date

from src.data_preparation.model_pre_processing import modeling_pre_processer

from src.evaluation.model_evaluation import evaluate_model_on_test_set

from src.config.config_utils import Config

import logging

logger = logging.getLogger("py4j")
logger.setLevel(logging.ERROR)

# COMMAND ----------

# DBTITLE 1,Notebook arguments
# Reset widgets
dbutils.widgets.removeAll()

# Get job's task arguments
training_run_id = dbutils.jobs.taskValues.get("ModelTraining", "training_run_id", debugValue=0)

# Set notebook widgets
dbutils.widgets.text("env", "dev", "Environment Name")
dbutils.widgets.dropdown("is_cicd", "true", ["true", "false"], "Is run by CICD")

config_setup = Config("../src/config/config_setup.yml")
dev_experiment_path = config_setup.get_config_value("environment.dev.experiment_path")
dbutils.widgets.text("experiment_path", dev_experiment_path, "Experiment path name read by asset workflow file yaml")

dbutils.widgets.text("model_alias", "challenger", "Alias of the model to evaluate")


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
model_alias = dbutils.widgets.get("model_alias")
# macrosegmento = dbutils.widgets.get("macrosegmento")
snapshot_date_str = dbutils.widgets.get("snapshot_date")
snapshot_date_str = snapshot_date_str[:-2] + "01"

# Validate input
if not is_iso_date(snapshot_date_str):
    raise ValueError(f"Invalid snapshot_date: {snapshot_date_str}")

print(env, "env")
print(is_cicd, "is_cicd")
print("Current date and time:", date_and_time.strftime("%Y-%m-%d %H:%M:%S"), "at the {t} timezone".format(t=timezone))
print(snapshot_date_str, model_alias, training_run_id)

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

diz_special_metric = [
    dd for dd in config_setup.get_config_value("metrics") if "Used also in the model evaluation phase." in dd.values()
][0]
metric_flg_volture_lorde_nette = f"{catalog}.{schema}.{diz_special_metric['name']}"
print(f"metric_flg_volture_lorde_nette: '{metric_flg_volture_lorde_nette}'\n")

# COMMAND ----------

# DBTITLE 1,Pre-processing
y_test, X_test, master = modeling_pre_processer(
    table_catalog_path=table_full_name,
    utility_set=["test"],
    oversampling=False,
    # params={"macrosegmento": macrosegmento},
)

master_insolventi = master.loc[master["FLG_INSOLUTO"] == True, :].reset_index(drop=True)
df_test = master_insolventi.loc[master_insolventi["UTILITY_SET"] == "test", :].reset_index(drop=True)
target = y_test.name
predictors = list(X_test.columns)
print(f"'{target}' is the target field\n")
print(f"n°{len(predictors)} predictors: {predictors}\n")

extra_features = [e for e in list(master.columns) if e not in predictors + [target]]
print(f"n°{len(extra_features)} extra_features: {extra_features}\n")

# COMMAND ----------

master_insolventi.sort_values(by="PDF_INSOLUTO_LORDO_ALL", ascending=False)

# COMMAND ----------

master_insolventi["snapshot_date"].unique()

# COMMAND ----------

df_test.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation

# COMMAND ----------

# DBTITLE 1,n volture medie mensili
if True:
    # if macrosegmento not in ("RESIDENZIALI", "MICROBUSINESS"):
    query_sql = """
    WITH FILTRATE AS (
    SELECT *
    , CONCAT(YEAR(DATA_VOLTURA), LPAD('-', 1, ' '), LPAD(MONTH(DATA_VOLTURA), 2, '0')) AS year_month_DATA_VOLTURA
    FROM {tab_path}
    WHERE DATA_VOLTURA >= '2022-06-01'
    AND DATA_VOLTURA < DATE_TRUNC('month', CURRENT_DATE()) -- first day of the current month
    )
    , AGGREGAZIONE AS (
        SELECT year_month_DATA_VOLTURA
        , COUNT(DISTINCT PK_PROPOSAL_LINE_ITEM) AS N_VOLTURE_TOTALI
        FROM FILTRATE
        GROUP BY year_month_DATA_VOLTURA
    )
    , MEDIA_TOTALE AS (
    SELECT 'MEDIA TOTALE PERIODO' AS year_month_DATA_VOLTURA,
        ROUND(AVG(N_VOLTURE_TOTALI) OVER()) AS N_VOLTURE_TOTALI
    FROM AGGREGAZIONE
    )
    , MEDIA_ULTIMI_12MESI AS (
    SELECT 'MEDIA ULTIMI 12 MESI' AS year_month_DATA_VOLTURA,
        AVG(N_VOLTURE_TOTALI) OVER() AS N_VOLTURE_TOTALI
    FROM AGGREGAZIONE
    WHERE year_month_DATA_VOLTURA >= DATE_TRUNC('month', DATE_SUB(CURRENT_DATE(), 365))
    )
    , UNITE AS (
        SELECT *
        FROM AGGREGAZIONE
    UNION
        SELECT *
        FROM MEDIA_TOTALE
    UNION
        SELECT *
        FROM MEDIA_ULTIMI_12MESI
    )
    SELECT *
    FROM UNITE
    SORT BY year_month_DATA_VOLTURA
    """.format(
        tab_path=metric_flg_volture_lorde_nette
    )

# elif macrosegmento in ("RESIDENZIALI", "MICROBUSINESS"):


spark: SparkSession = SparkSession.builder.getOrCreate()
df_sql = spark.sql(query_sql)

if True:
    # if macrosegmento not in ("RESIDENZIALI", "MICROBUSINESS"):
    try:
        volt_mese_att = int(
            df_sql.filter(df_sql["year_month_DATA_VOLTURA"] == "MEDIA ULTIMI 12 MESI").collect()[0]["N_VOLTURE_TOTALI"]
        )
    except:
        logger.warning(
            "'MEDIA ULTIMI 12 MESI' of 'N_VOLTURE_TOTALI' not found in the {tab}".format(
                tab=metric_flg_volture_lorde_nette
            )
        )
        # volt_mese_att = int(
        #     df_sql.filter(df_sql["year_month_DATA_VOLTURA"] == "MEDIA TOTALE PERIODO").collect()[0]["N_VOLTURE_TOTALI"]
        # )
        # volt_mese_att = int(
        #    df_sql.filter(df_sql["year_month_DATA_VOLTURA"] == 'MEDIA PERIODO DI TRAINING').collect()[0]["N_VOLTURE_TOTALI"]
        # )
# # Filtro per MACROSEGMENTO
# elif macrosegmento == "RESIDENZIALI":
# elif macrosegmento == "MICROBUSINESS":

print("volt_mese_att=", volt_mese_att)
display(df_sql)

# COMMAND ----------

# DBTITLE 1,numeriche generali
# volt_mese_att = 19710 # media volture mensili
# 23k circa è la NUMERICA BUSINESS ATTESA
# 22323 numero usato nella review del 17 gennaio 2025
# 19710 numero usato nella review del 20 marzo 2025 FROM p_sbx_data.atum.vw_antifrode_volture_customer_base_volturante
# 18732 residenziali
#   977 microbusiness e assimilate

# NUMERICHE EFFETTIVE
n_mesi = master_insolventi["snapshot_date"].nunique()
perc_i = master_insolventi.shape[0] / master.shape[0]
perc_i_test = sum(master_insolventi["UTILITY_SET"] == "test") / sum(master["UTILITY_SET"] == "test")
perc_test = sum(master_insolventi["UTILITY_SET"] == "test") / master_insolventi.shape[0]

# Coefficiente di proporzione delle numeriche ad un valore medio mensile
coeff_mese = 1 / (perc_test * n_mesi)

# Numero di volture medie mensili effettive
volt_mese_eff = master.shape[0] / n_mesi

# Coefficiente di proporzione delle numeriche al valore business atteso
coeff_att = volt_mese_att / volt_mese_eff

proportion_coeff = coeff_mese * coeff_att

tot_volture = master.shape[0]
tot_volture_i = master_insolventi.shape[0]
volt_mese = tot_volture / n_mesi
volt_i_mese = tot_volture_i / n_mesi

fraud = master["FLG_CLI_MOROSO"].sum()
fraud_i = master_insolventi["FLG_CLI_MOROSO"].sum()
fraud_mese = fraud / n_mesi
fraud_i_mese = fraud_i / n_mesi

tot_volture = int(round(tot_volture * coeff_att, 0))
tot_volture_i = int(round(tot_volture_i * coeff_att, 0))
volt_mese = int(round(volt_mese * coeff_att, 0))
volt_i_mese = int(round(volt_i_mese * coeff_att, 0))
fraud = int(round(fraud * coeff_att, 0))
fraud_i = int(round(fraud_i * coeff_att, 0))
fraud_mese = int(round(fraud_mese * coeff_att, 0))
fraud_i_mese = int(round(fraud_i_mese * coeff_att, 0))

print(n_mesi, "n° mesi considerati\n")
print(f"{round(perc_i * 100,2)}% di volture in casi con insoluto pregresso\n        di cui")
print(f"{round(perc_i_test * 100,2)}% nel test set ({round(perc_test*100,1)}% del dataset totale) \n")

print(tot_volture, "n° tot. volture di cui poi morose fraudolente:", fraud)
print(tot_volture_i, "n° tot. volture con credito insoluto di cui poi morose fraudolente:", fraud_i, "\n")
print(volt_mese, "n° volture al mese di cui poi morose fraudolente:", fraud_mese)
print(
    volt_i_mese, "n° volture al mese con casi d'insoluto pregresso di cui poi morose fraudolente:", fraud_i_mese, "\n"
)

print(proportion_coeff, "proportion_coeff", model_alias)
# proportion_coeff = 0.3445
# print(proportion_coeff, "proportion_coeff modified for the analysis in the review of 20th march 2025")

# COMMAND ----------

# DBTITLE 1,evaluate_model_on_test_set
print(f"'{model_alias}' is the model's alias that is going to be evaluated\n")

# df_test[target]=df_test[target].astype(int) # from boolean to integer

run_id, df_results, df_thresholds, score_model = evaluate_model_on_test_set(
    df_test=df_test,
    target=target,
    predictors=predictors,
    model_alias=model_alias,
    is_cicd=is_cicd,
    experiment_path=experiment_path,
    env=env,
    params={"proportion_coeff": proportion_coeff},
)
print(f"\nThe run_id is {run_id}\n")
df_thresholds

# COMMAND ----------

# DBTITLE 1,execution x 2 macrosegments
# print("is_cicd", is_cicd)

# print(f"'{model_alias}' is the model's alias that is going to be evaluated\n")

# if macrosegmento == "TUTTI":
#     for mac in ["RESIDENZIALI", "MICROBUSINESS"]:
#         if mac == "RESIDENZIALI":
#             ind = df_test["MACROSEGMENTO"] == "RESIDENZIALI"
#             proportion_coeff_segmento = 0.318248287612377
#         else:
#             ind = df_test["MACROSEGMENTO"] != "RESIDENZIALI"
#             proportion_coeff_segmento = 0.461911004501589
#         print(sum(ind), X_test[ind].shape, y_test[ind].shape)
#         print(f"'{mac}' is the macrosegmento analysed\n")

#         run_id_bis, df_results_bis, df_thresholds_bis, same_model_bis = evaluate_model_on_test_set(
#             df_test=df_test[ind].reset_index(drop=True),
#             target=target,
#             predictors=predictors,
#             model_alias=model_alias,
#             is_cicd=is_cicd,
#             experiment_path=experiment_path,
#             env=env,
#             params={"proportion_coeff": proportion_coeff},
#         )
#         print(f"The run_id is {run_id}\n")

# COMMAND ----------

# Set a parameter of the job's task that will be re-used in the next task
dbutils.jobs.taskValues.set("evaluation_run_id", run_id)
run_id

# COMMAND ----------

df_results
