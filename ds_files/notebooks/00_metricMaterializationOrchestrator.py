# Databricks notebook source
# MAGIC %md
# MAGIC 00_metricMaterializationOrchestrator change file type from .ipynb to .py

# COMMAND ----------

# DBTITLE 1, Run another notebook to import and inizialize its functions and classes
# MAGIC %run ./materialization_utils/materialization_utils

# COMMAND ----------

# DBTITLE 1, Local imports and setting
import json
import requests
import databricks.sdk # required version >= 0.205
from datetime import datetime, timedelta
import yaml
from dateutil.relativedelta import relativedelta
import time
from datetime import datetime

start_date = dbutils.widgets.get("start_date")
end_date = dbutils.widgets.get("end_date")

env = dbutils.widgets.get("env")
job_id = dbutils.widgets.get("job_id")
n_months = dbutils.widgets.get("n_months")
ingestion_type = dbutils.widgets.get("ingestion_type")

general_utils = GeneralUtils(spark)
config = general_utils.load_config(general_utils.get_variable()['config_setup_path'])
databricks_host = config['environment'][env]['databricks_host']

if ingestion_type.strip().upper() in ("TRAINING", "INFERENCING"):
    pass
else:
    raise Exception("Ingestion type not supported")
    # pass

# COMMAND ----------

# DBTITLE 1,Function Definitions
def generate_month_starts(start_date: str, end_date: str) -> list[str]:
    """
    Genera una lista di tutte le date comprese tra start_date e end_date,
    corrispondenti al primo giorno di ogni mese.

    Parametri:
        start_date (str): data iniziale nel formato 'YYYY-MM-DD'
        end_date (str): data finale nel formato 'YYYY-MM-DD'

    Ritorna:
        list[str]: elenco di date (stringhe) in formato 'YYYY-MM-DD'
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    start = start.replace(day=1)

    dates = []
    while start <= end:
        dates.append(start.strftime("%Y-%m-%d"))
        start += relativedelta(months=1)

    return dates

def filter_dates_older_than_n_months(dates: list[str], n_months:str) -> list[str]:
    """
    Rimuove tutte le date comprese entro i 15 mesi precedenti a oggi.
    Mantiene solo quelle più vecchie di 15 mesi fa.

    Parametri:
        dates (list[str]): elenco di date in formato 'YYYY-MM-DD'

    Ritorna:
        list[str]: elenco filtrato di date più vecchie di 15 mesi
    """
    today = datetime.today()
    cutoff_date = today - relativedelta(months=int(n_months))

    filtered = [
        d for d in dates
        if datetime.strptime(d, "%Y-%m-%d") < cutoff_date
    ]

    return filtered


def load_yaml(path: str):
    """Load YAML file"""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_materialization_plan(config_path: str):
    """Genera un piano di materializzazione prendendo in considerazione le configurazioni lette aggiungendo le snapshot_date da materializzare classificate per gruppo di metriche"""
    cfg = load_yaml(config_path)

    today = datetime.today().replace(day=1)

    plan = []

    # Metrics group
    for metric in cfg.get("metrics", []):
        inference_date
        for snapshot_date in inference_date:
            plan.append({"group": "metrics", "metric": metric, "snapshot_date": snapshot_date})

    return plan


# COMMAND ----------

# DBTITLE 1,Executions
print("n_months:", n_months)
date_list = generate_month_starts(start_date, end_date)
print("input date_list:", date_list)
date_list_minus_n_months = filter_dates_older_than_n_months(date_list, n_months)
print("date_list_minus_n_months:", date_list_minus_n_months)
inference_date = list(set(date_list) - set(date_list_minus_n_months))
print("inference_date:", inference_date)
data_conf = build_materialization_plan("/Workspace/AI/creri/creri_model/files/src/config/config_setup.yml")
data_conf_materialization = [x for x in data_conf if x['metric']['today_minus_n_months'] == 0]
data_conf_materialization

if ingestion_type == "TRAINING":
    for date in date_list_minus_n_months:
        print(f"Processing snapshot date: {date}")

        w = databricks.sdk.WorkspaceClient()
        run = w.jobs.run_now(
            job_id=job_id, # job id della trasformazione_volture
            job_parameters={"snapshot_date": f"{date}"
                            }
        )
        run_id = run.run_id
        print(f"Run avviato con ID: {run_id}")


if ingestion_type == "INFERENCING":
    for obj in data_conf_materialization:
        print(f"Processing metric: {obj['metric']['name']}")
        print(f"Processing snapshot date: {obj['snapshot_date']}")
        w = databricks.sdk.WorkspaceClient()
        run = w.jobs.run_now(
            job_id=job_id, # job id della metrics_materialization
            job_parameters={"snapshot_date": f"{obj['snapshot_date']}",
                            "input_params": f"{obj['metric']}",
                            "env": f"{env}"
                            }
        )
        run_id = run.run_id
        print(f"Run avviato con ID: {run_id}")


# COMMAND ----------

dbutils.notebook.exit("Success!!")

# COMMAND ----------
