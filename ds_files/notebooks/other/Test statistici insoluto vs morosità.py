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
os.chdir(notebook_path)

# COMMAND ----------

# MAGIC %md
# MAGIC # EDITABLE SECTION
# MAGIC Here you can put your code, starting from imports

# COMMAND ----------

# DBTITLE 1,Import libraries
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, ks_2samp

from src.utils.input_validation import is_iso_date

# from src.features.utils import get_config_input_columns

from src.config.config_utils import Config

import logging

logger = logging.getLogger("py4j")
logger.setLevel(logging.ERROR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load master and data selection

# COMMAND ----------

config = Config("../src/config")

# Retrieve config values
env = "dev"
target = config.get_config_value("targets_to_use.training")
eng_features = config.get_config_value("engineered_features")
not_features = config.get_config_value("features_not_for_modeling")
catalog = config.get_config_value("environment.{env}.catalog".format(env=env))
schema = config.get_config_value("master_training.schema")
master_name = config.get_config_value("master_training.master_name")
table_full_name = f"{catalog}.{schema}.vw_{master_name}"

# Read master
master = spark.sql("""SELECT * FROM IDENTIFIER(:master_table)""", args={"master_table": table_full_name})


# Select only the columns we (need)
config_cols = get_config_input_columns(output="list")
eng_features = list(eng_features.keys())
modeling_cols = ["DATA_SNAPSHOT", "utility_set"] + config_cols + eng_features
modeling_cols = [ff for ff in modeling_cols if ff not in not_features]
bonus_cols = ["POD_PDR", "COD_FISCALE_PIVA", "FLG_MOROSITA_FORN", "CLI_AGE_CLIOUT"]
modeling_cols = bonus_cols + modeling_cols
# print("modeling_cols:", modeling_cols)

master = master.select(*modeling_cols)

# Transform Spark dataframe into a Pandas dataframe
master = cast_pandas_datatype(master.toPandas())
print(master.shape)
print("")

# Filtro solo i casi con del credito insoluto
ind = master["PDF_INSOLUTO_LORDO_ALL"].isnull() & master["CLI_INSOLUTO_LORDO_CLIOUT"].isnull()
master.loc[~ind, "FLAG_INSOLUTO"] = 1
master.loc[ind, "FLAG_INSOLUTO"] = 0

# Sottocasistica dei clienti entranti con insoluto ma senza morosità sul PDF
# ind = (master['CLI_INSOLUTO_LORDO'] > 0) & (master['PDF_INSOLUTO_LORDO_ALL'].isnull()) & (master['CLI_INSOLUTO_LORDO_CLIOUT'].isnull())
# master.loc[~ind, "FLAG_INSOLUTO"] = 0; master.loc[ ind, "FLAG_INSOLUTO"] = 1

ind = master["FLAG_INSOLUTO"] == 1
master_2 = master[ind]
print(master_2.shape)

master_2

# COMMAND ----------

# MAGIC %md
# MAGIC # Explorative Data Analysis

# COMMAND ----------

# DBTITLE 1,coloumns
# Retrieve config values
eng_features = config.get_config_value("engineered_features")
config_cols = get_config_input_columns(output="dict")
config_cols.update(eng_features)

# Select boolean coloumns
bool_cols = [k for k, v in config_cols.items() if v == "bool"]
bool_cols = ["FLAG_INSOLUTO", "FLG_CLI_MOROSO", "FLG_MOROSITA_FORN"] + bool_cols

# Select continuous coloumns
float_cols = [k for k, v in config_cols.items() if v == "float64"]

float_cols

# COMMAND ----------

# DBTITLE 1,plot FLG_CLI_MOROSO
import matplotlib.pyplot as plt

print(master.shape)
print(master_2.shape)

# Plot the data distribution of FLG_CLI_MOROSO
plt.figure(figsize=(10, 6))
value_counts = master["FLG_CLI_MOROSO"].value_counts()
value_counts_2 = master_2["FLG_CLI_MOROSO"].value_counts()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))

value_counts.plot(kind="bar", ax=axes[0])
axes[0].set_title("FLG_CLI_MOROSO all customers dataset")
axes[0].set_xlabel("FLG_CLI_MOROSO")
axes[0].set_ylabel("Count")

value_counts_2.plot(kind="bar", ax=axes[1])
axes[1].set_title("FLG_CLI_MOROSO only dataset of customers with unpaid credit")
axes[1].set_xlabel("FLG_CLI_MOROSO_2")
axes[1].set_ylabel("Count")

# Add percentage on top of bars for the first plot
for index, value in enumerate(value_counts):
    percentage = f"{(value / len(master) * 100):.2f}%"
    axes[0].text(index, value, f"{value}\n({percentage})", ha="center", va="bottom")

# Add percentage on top of bars for the second plot
for index, value in enumerate(value_counts_2):
    percentage = f"{(value / len(master_2) * 100):.2f}%"
    axes[1].text(index, value, f"{value}\n({percentage})", ha="center", va="bottom")

plt.show()

# COMMAND ----------

# DBTITLE 1,Fisher Test on the target
# Variabili di interesse
var_1 = master["FLG_CLI_MOROSO"]
var_2 = master["FLAG_INSOLUTO"]

Ipotesi_nulla = "non c'è nessuna relazione tra le due variabili"
alfa = 0.05

# Contingency table
contingency_table = pd.crosstab(var_1, var_2)
# contingency_table = np.array([[1639,2026],[36089,176227]]) # uguale alla precedente

# Test esatto di Fisher
odds_ratio, p_value = fisher_exact(contingency_table)

# odds_ratio = (176227/2026)/(36089/1639)

print("odds_ratio =", odds_ratio)
assert p_value < alfa
print("p_value =", p_value, "< alfa =", alfa)
print("-> L'ipotesi nulla va rifiutata:", Ipotesi_nulla)
print("")
print(master.shape, "master.shape")
contingency_table

# COMMAND ----------

# DBTITLE 1,old
# Variabili di interesse
var_1 = master["FLG_CLI_MOROSO"]
var_2 = master["FLAG_INSOLUTO"]

Ipotesi_nulla = "non c'è nessuna relazione tra le due variabili"
alfa = 0.05

# Contingency table
contingency_table = pd.crosstab(var_1, var_2)
# contingency_table = np.array([[1639,2026],[36089,176227]]) # uguale alla precedente

# Test esatto di Fisher
odds_ratio, p_value = fisher_exact(contingency_table)

# odds_ratio = (176227/2026)/(36089/1639)

print("odds_ratio =", odds_ratio)
assert p_value < alfa
print("p_value =", p_value, "< alfa =", alfa)
print("-> L'ipotesi nulla va rifiutata:", Ipotesi_nulla)
print("")
print(master.shape, "master.shape")
contingency_table

# COMMAND ----------

# Variabili di interesse
var_1 = master_2["FLG_CLI_MOROSO"]
var_2 = master_2["FLG_COMUNE_UGUALE"]

Ipotesi_nulla = "non c'è nessuna relazione tra le due variabili"
alfa = 0.05

# Contingency table
contingency_table = pd.crosstab(var_1, var_2)
# contingency_table = np.array([[1639,2026],[36089,176227]]) # uguale alla precedente

# Test esatto di Fisher
odds_ratio, p_value = fisher_exact(contingency_table)

# odds_ratio = (176227/2026)/(36089/1639)

print("odds_ratio =", odds_ratio)
assert p_value < alfa
print("p_value =", p_value, "< alfa =", alfa)
print("-> L'ipotesi nulla va rifiutata:", Ipotesi_nulla)
print("")
print(master_2.shape, "master_2.shape")
contingency_table

# COMMAND ----------

1 - (30660 + 1470) / 37487

# COMMAND ----------

200 / (200 + 5157)

# COMMAND ----------

1470 / (30660 + 1470)

# COMMAND ----------

# DBTITLE 1,Fisher Test on boolean predictors
# Variabile di riferimento
var_2 = master["FLAG_INSOLUTO"]

Ipotesi_nulla = "Non c'è nessuna relazione tra le due variabili"
alfa = 0.05

for bb in bool_cols:
    var_1 = master[bb]
    # print(bb, master[bb].unique() )

    # Contingency table
    contingency_table = pd.crosstab(var_1, var_2)
    # print(contingency_table.shape, 'contingency_table')
    if contingency_table.shape == (2, 2):
        # Perform the Fisher test
        odds_ratio, p_value = fisher_exact(contingency_table)

        if p_value < alfa:
            print(
                bb + ": odds_ratio =", odds_ratio, ", p_value =", p_value, "< " + str(alfa)
            )  # -> L'ipotesi nulla va rifiutata!")
        else:
            print(
                bb + ":odds_ratio =",
                odds_ratio,
                ", p_value =",
                p_value,
                ">= "
                + str(alfa)
                + " -> L'ipotesi nulla risulta verificata! "
                + Ipotesi_nulla
                + ": FLAG_INSOLUTO & "
                + bb,
            )
        # print("")

# COMMAND ----------

# DBTITLE 1,Kolmogorov-Smirnov test
# Assunzioni:
# - Variabile continua

Ipotesi_nulla = "i 2 campioni hanno la stessa distribuzione (Ipotesi di conformità)"
alfa = 0.05
c_alfa = 1.358  # critical value

for cc in ["AGING_METER_ATTIV_SWIN", "CLI_AGE_CLIOUT"]:
    # 2 sample for the Kolmogorov-Smirnov test
    sample_0 = master[cc]
    sample_1 = master_2[cc]

    # Perform the Kolmogorov-Smirnov test
    ks_statistic, p_value = ks_2samp(sample_0, sample_1, alternative="two-sided")
    print(cc)
    print("Kolmogorov-Smirnov Statistic =", ks_statistic)  # means the distance between the CDFs
    print("p_value =", p_value, "> " + str(c_alfa) + " critical value (alfa =", alfa, ")")

    # Display the results
    if p_value > c_alfa:
        print(" -> L'ipotesi nulla va rifiutata!")
    else:
        print(" -> L'ipotesi nulla risulta verificata: " + Ipotesi_nulla)
    # If the KS statistic is large, then the p-value will be small, and this may be taken as evidence against the null hypothesis in favor of the alternative.
    print("")
