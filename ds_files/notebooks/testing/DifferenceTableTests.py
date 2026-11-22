# Databricks notebook source
# MAGIC %md
# MAGIC # NOTEBOOK TO CHECK THE DIFFERENCES IN FIELDS IN SEVERAL TABLES
# MAGIC Only SQL queries inside

# COMMAND ----------

# DBTITLE 1,TEST data structure
# ref = 'DEV'
# ref = 'P1'
ref = "P3"

for tab in [
    "creri_met_accounttocodfiscalepiva",
    "creri_met_sdfattivazioneattivi",
    "creri_met_pontesdfproposal",
    "creri_met_metricapdfpodpdrgtwv",
    "creri_met_clienteattributi",
    "creri_met_dataultletturaattivswingas",
    "creri_met_dataultletturaattivswinpwr",
    "creri_met_morositapregressapodpdr",
    "creri_met_morositapregressacliente",
    # "creri_met_targetcontofattclientimorosi",
    # "creri_met_metricaflgvolturelordenette",
    # "creri_train_cb_volturante",
    # "creri_train_cb_volturato",
    # "creri_train_mstr_volturante",
    # "creri_train_mstr_volturato",
    # "creri_train_mstr_credit_check_volture",
    # "creri_infer_mstr_volturante",
    # "creri_infer_mstr_volturato",
    # "creri_out_infer_payload",
    # "creri_out_infer_final"
]:
    print(f"----------------------------------------------------{tab}")

    df_DEV = spark.sql("""SELECT * FROM p_sbx_data.data.""" + tab)
    df_P1 = spark.sql("""SELECT * FROM p_ind_core_ai.creri.""" + tab)
    df_P3 = spark.sql("""SELECT * FROM p_dpp_core_ai.creri.""" + tab)

    s_DEV = df_DEV.schema
    s_P1 = df_P1.schema
    s_P3 = df_P3.schema

    s_DIZ = {"DEV": s_DEV, "P1": s_P1, "P3": s_P3}
    s_REF = {ref: s_DIZ.pop(ref)}

    for env, s_Px in s_DIZ.items():
        print(f"{ref} differences with {env} of {tab}:")

        for s1 in s_REF[ref].fields:
            c_REF = s1.name
            d_REF = s1.dataType
            n_REF = s1.nullable

            check_name = False
            check_type = True
            wrong_type = None
            check_null = True
            for s2 in s_Px.fields:
                c_P = s2.name
                d_P = s2.dataType
                n_P = s2.nullable
                if c_REF == c_P:
                    check_name = True
                    if d_REF != d_P:
                        check_type = False
                        wrong_type = d_P
                    if n_REF != n_P:
                        check_null = False

            if check_name == False:
                print(" - ", c_REF, "not in", env)
            if check_name & (check_type == False):
                print(f" - {d_REF} of {c_REF} mismatch with {wrong_type} in {env}")
            if check_name & (check_null == False):
                print(f" - {n_REF} nullable of {c_REF}  mismatch in {env}")
        print(".")

# COMMAND ----------

# DBTITLE 1,query for snapshot check
query_snapshots = """
with
tab0 as (
  select concat( *{*primary_key1*}* , DATA_SNAPSHOT) as PK_unique0 , DATA_SNAPSHOT as snapshot_date, *
  from *{*t0*}*
),
tab1 as (
  select concat( *{*primary_key2*}* , snapshot_date) as PK_unique1 , *
  from *{*t1*}*
),
tab2 as (
  select concat( *{*primary_key2*}* , snapshot_date) as PK_unique2 , *
  from *{*t2*}*
  WHERE flg_lastSnap = 1
),
tab3 as (
  select concat( *{*primary_key2*}* , snapshot_date) as PK_unique3 , *
  from *{*t3*}*
  WHERE flg_lastSnap = 1
),
group_snap0 as (
  select snapshot_date, count(*) as num0, count(distinct PK_unique0) as PK_unique0, count(*) - count(distinct PK_unique0) as diff0
  from tab0
  group by snapshot_date
),
group_snap1 as (
  select snapshot_date, count(*) as num1, count(distinct PK_unique1) as PK_unique1, count(*) - count(distinct PK_unique1) as diff1
  from tab1
  group by snapshot_date
),
group_snap2 as (
  select snapshot_date, count(*) as num2, count(distinct PK_unique2)  as PK_unique2, count(*) - count(distinct PK_unique2) as diff2
  from tab2
  group by snapshot_date
),
group_snap3 as (
  select snapshot_date, count(*) as num3, count(distinct PK_unique3) as PK_unique3, count(*) - count(distinct PK_unique3) as diff3
  from tab3
  group by snapshot_date
),
join12 as (
  select ifnull(group_snap1.snapshot_date, group_snap2.snapshot_date) as snapshot_date, num1, diff1, num2, diff2
  from group_snap1
  full outer join group_snap2
  on group_snap1.snapshot_date = group_snap2.snapshot_date
),
join123 as (
  select ifnull(join12.snapshot_date, group_snap3.snapshot_date) as snapshot_date, num1, diff1, num2, diff2, num3, diff3
  from join12
  full outer join group_snap3
  on join12.snapshot_date = group_snap3.snapshot_date
),
join0123 as (
  select ifnull(group_snap0.snapshot_date, join123.snapshot_date) as snapshot_date, num0, num1, num2, num3, diff0, diff1, diff2, diff3
  from group_snap0
  full outer join join123
  on group_snap0.snapshot_date = join123.snapshot_date
)
select *
from join0123
"""

# COMMAND ----------

# DBTITLE 1,query for the difference
query_difference = """
    with
    tab0 as (
    select concat(*{*primary_key1*}*, DATA_SNAPSHOT) as PK_unique0 , DATA_SNAPSHOT as snapshot_date, *
    , *{*field1*}* as col_0
    from *{*t0*}*
    where DATA_SNAPSHOT in (*{*snapshot_dates*}*)
    ),
    tab1 as (
    select concat(*{*primary_key2*}*, snapshot_date) as PK_unique1 , *
    , *{*field2*}* as col_1
    from *{*t1*}*
    where snapshot_date in (*{*snapshot_dates*}*)
    ),
    tab2 as (
    select concat(*{*primary_key2*}*, snapshot_date) as PK_unique2 , *
    , *{*field2*}* as col_2
    from *{*t2*}*
    where snapshot_date in (*{*snapshot_dates*}*)
    and flg_lastSnap = 1
    ),
    tab3 as (
    select concat(*{*primary_key2*}*, snapshot_date) as PK_unique3 , *
    , *{*field2*}* as col_3
    from *{*t3*}*
    where snapshot_date in (*{*snapshot_dates*}*)
    and flg_lastSnap = 1
    ),
    join12 as (
    select ifnull(tab1.PK_unique1, tab2.PK_unique2) as PK_unique12, *
    from tab1
    full outer join tab2
    on tab1.PK_unique1 = tab2.PK_unique2
    ),
    join123 as (
    select ifnull(join12.PK_unique12, tab3.PK_unique3) as PK_unique123, *
    from join12
    full outer join tab3
    on join12.PK_unique12 = tab3.PK_unique3
    ),
    join0123 as (
    select ifnull(tab0.PK_unique0, join123.PK_unique123) as PK_unique0123, *
    , col_0 - col_1 as diff_01
    , col_0 - col_2 as diff_02
    , col_0 - col_3 as diff_03
    , col_1 - col_2 as diff_12
    , col_1 - col_3 as diff_13
    , col_2 - col_3 as diff_23
    from tab0
    full outer join join123
    on tab0.PK_unique0 = join123.PK_unique123
    )
    select PK_unique0123
    , col_0, col_1, col_2, col_3
    , diff_01, diff_02, diff_03, diff_12, diff_13, diff_23
    from join0123
    where diff_01 > 0
    or diff_02 > 0
    or diff_03 > 0
    or diff_12 > 0
    or diff_13 > 0
    or diff_23 > 0
    """

# COMMAND ----------

# DBTITLE 1,get_query_dataframe function
def get_query_dataframe(
    query: str,
    args_diz: dict[str, str],
):
    """
    Function to get a query replacing some string with the parameters string and then execute it to extract the dataframe.
    Args:
      query (string): the query to be executed.
      args_diz (dict[string,string]): dictionary of the parameters string to be replaced in the query.
    Returns:
      Spark DataFrame
    """
    for k, v in args_diz.items():
        query = query.replace("*{*" + k + "*}*", v)

    dfs = spark.sql(query)
    return dfs

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training Master Tables

# COMMAND ----------

l_snaps_train = [
    "2024-02-01",
    "2024-01-01",
    "2023-12-01",
    "2023-11-01",
    "2023-10-01",
    "2023-09-01",
    "2023-08-01",
    "2023-07-01",
    "2023-06-01",
    "2023-05-01",
    "2023-04-01",
    "2023-03-01",
    "2023-02-01",
    "2023-01-01",
    "2022-12-01",
    "2022-11-01",
    "2022-10-01",
    "2022-09-01",
    "2022-08-01",
    "2022-07-01",
]

string_snaps_train = "'" + "','".join(l_snaps_train) + "'"


train_mstr_diz = {
    "snapshot_dates": string_snaps_train,
    "primary_key1": "PK_PROPOSAL_LINE_ITEM",
    "primary_key2": "PK_PROPOSAL_LINE_ITEM",
    "field1": "",
    "field2": "",
    "t0": "p_sbx_data.atum.vw_antifrode_volture_master_training",
    "t1": "p_sbx_data.data.creri_train_mstr_credit_check_volture",
    "t2": "p_ind_core_ai.creri.creri_train_mstr_credit_check_volture",
    "t3": "p_dpp_core_ai.creri.creri_train_mstr_credit_check_volture",
}

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM p_sbx_data.atum.vw_antifrode_volture_master_training -- from p_sbx_data.atum.antifrode_volture_master_training
# MAGIC -- from p_sbx_data.data.creri_train_mstr_credit_check_volture
# MAGIC -- from p_ind_core_ai.creri.creri_train_mstr_credit_check_volture
# MAGIC -- from p_dpp_core_ai.creri.creri_train_mstr_credit_check_volture

# COMMAND ----------

display(get_query_dataframe(query=query_snapshots, args_diz=train_mstr_diz))

# COMMAND ----------

train_mstr_diz.update({"field1": "PDF_INSOLUTO_LORDO_ALL", "field2": "PDF_INSOLUTO_LORDO_ALL"})
display(get_query_dataframe(query=query_difference, args_diz=train_mstr_diz))

# COMMAND ----------

train_mstr_diz.update({"field1": "PDF_INSOLUTO_LORDO_LAST", "field2": "PDF_INSOLUTO_LORDO_LAST"})
display(get_query_dataframe(query=query_difference, args_diz=train_mstr_diz))

# COMMAND ----------

train_mstr_diz.update({"field1": "CLI_INSOLUTO_LORDO_CLIOUT", "field2": "CLI_INSOLUTO_LORDO"})
display(get_query_dataframe(query=query_difference, args_diz=train_mstr_diz))

# COMMAND ----------

train_mstr_diz.update({"field1": "CLI_INSOLUTO_LORDO", "field2": "CLI_INSOLUTO_LORDO_CLIIN"})
display(get_query_dataframe(query=query_difference, args_diz=train_mstr_diz))

# COMMAND ----------

train_mstr_diz.update({"field1": "fatturato_rate", "field2": "fatturato_rate"})
display(get_query_dataframe(query=query_difference, args_diz=train_mstr_diz))

# COMMAND ----------

train_mstr_diz.update({"field1": "mancati_incassi_rate", "field2": "mancati_incassi_rate"})
display(get_query_dataframe(query=query_difference, args_diz=train_mstr_diz))

# COMMAND ----------

train_mstr_diz.update({"field1": "perc_mancati_incassi_rate", "field2": "perc_mancati_incassi_rate"})
display(get_query_dataframe(query=query_difference, args_diz=train_mstr_diz))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference Master VOLTURATO Table

# COMMAND ----------

l_snaps_infer = ["2025-11-01"]

string_snaps_infer = "'" + "','".join(l_snaps_infer) + "'"

infer_mstr_cliout_diz = {
    "snapshot_dates": string_snaps_infer,
    "primary_key1": "POD_PDR",
    "primary_key2": "POD_PDR",
    "field1": "",
    "field2": "",
    "t0": "p_sbx_data.data.antifrode_volture_mstr_volturato_copia_TEST",
    "t1": "p_sbx_data.data.creri_infer_mstr_volturato",
    "t2": "p_ind_core_ai.creri.creri_infer_mstr_volturato",
    "t3": "p_dpp_core_ai.creri.creri_infer_mstr_volturato",
}

# COMMAND ----------

# %sql
# CREATE TABLE p_sbx_data.data.antifrode_volture_mstr_volturato_copia_TEST
# select
#   COD_FISCALE_PIVA_CLIOUT,
#   PK_ACCOUNT,
#   COD_PDF,
#   POD_PDR,
#   FLG_MOROSITA_FORN,
#   PDF_MACTO,
#   PDF_STATO_CONTO_CRM_LAST,
#   PDF_DATA_SCAD_RATA_LAST,
#   PDF_AGING_SCAD_RATE_LAST,
#   PDF_NR_FATTURE_INSOLUTE_LAST,
#   PDF_NR_RATE_INSOLUTE_LAST,
#   PDF_INSOLUTO_LORDO_LAST,
#   PDF_INSOLUTO_NETTO_LAST,
#   PDF_INSOLUTO_CANONE_TV_LAST,
#   PDF_INSOLUTO_R_LAST,
#   PDF_NR_FORN_MOROSE_ALL,
#   PDF_DATA_SCAD_RATA_ALL,
#   PDF_AGING_SCAD_RATE_ALL,
#   PDF_NR_FATTURE_INSOLUTE_ALL,
#   PDF_NR_RATE_INSOLUTE_ALL,
#   PDF_INSOLUTO_LORDO_ALL,
#   PDF_INSOLUTO_NETTO_ALL,
#   PDF_INSOLUTO_CANONE_TV_ALL,
#   PDF_INSOLUTO_R_ALL,
#   PDF_INSOLUTO_6_MM_ALL,
#   PDF_INSOLUTO_1_ANNI_ALL,
#   PDF_INSOLUTO_2_ANNI_ALL,
#   PDF_INSOLUTO_3_ANNI_ALL,
#   PDF_INSOLUTO_5_ANNI_ALL,
#   PDF_INSOLUTO_PIU_ANNI_ALL,
#   DATA_AGG_MORPP,
#   COD_POD,
#   DATA_ULT_LETTURA_ATTIV_SWINPWR_CLIOUT,
#   DATA_AGG_SWINPWR,
#   COD_PDR,
#   DATA_ULT_LETTURA_ATTIV_SWINGAS_CLIOUT,
#   DATA_AGG_SWINGAS,
#   CODICE_POD_PDR_CLIOUT,
#   IND_FORN_IND_CLIOUT,
#   IND_FORN_CAP_CLIOUT,
#   IND_FORN_COMUNE_CLIOUT,
#   IND_FORN_COD_PROV_CLIOUT,
#   IND_FORN_NOME_PROV_CLIOUT,
#   IND_FORN_REGIONE_CLIOUT,
#   DATA_AGG_INDGTW,
#   PK_CONTO_FATT,
#   FK_QUOTE_PROPOSAL,
#   DATA_ATT_SITO_ATT,
#   SDF_AGING_GG_ATT,
#   SDF_AGING_MM_ATT,
#   CLI_AGING_GG_ATT,
#   CLI_AGING_MM_ATT,
#   SDF_CANLE_ATTIVAZ_ATT_CLIOUT,
#   DATA_AGG_SDFAA,
#   CLI_SEX_CLIOUT,
#   CLI_NAZTA_CLIOUT,
#   CLI_AGE_CLIOUT,
#   CLI_MACTO_CLIOUT,
#   CLI_CONT_ORIG_CLIOUT,
#   CLI_FLG_REG_PORT_CLIOUT,
#   CLI_PROV_NASC_CLIOUT,
#   CLI_COM_CLIOUT,
#   CLI_PROV_CLIOUT,
#   CLI_REG_CLIOUT,
#   CLI_NAZ_CLIOUT,
#   CLI_FLG_OPZ_VERDE_CLIOUT,
#   SOURCE_CRM_CLIOUT,
#   DATA_AGG_CLIATT,
#   COD_FISCALE_CLIOUT,
#   PIVA_CLIOUT,
#   MACROSEGMENTO_CLIOUT,
#   ENCOD_MACROSEGMENTO_CLIOUT,
#   NOME_CLIOUT,
#   COGNOME_CLIOUT,
#   RAG_SOCIALE_CLIOUT,
#   EMAIL_CLIOUT,
#   PEC_CLIOUT,
#   TELEFONO_CLIOUT,
#   FORMA_GIURIDICA_CLIOUT,
#   SNAPSHOT_DATE_CLIOUT,
#   CLI_NR_CC_FORN_MOROSE_CLIOUT,
#   CLI_DATA_SCAD_RATA_OLDER_CLIOUT,
#   CLI_AGING_SCAD_RATE_OLDER_CLIOUT,
#   CLI_NR_FATTURE_INSOLUTE_CLIOUT,
#   CLI_NR_RATE_INSOLUTE_CLIOUT,
#   CLI_INSOLUTO_LORDO_CLIOUT,
#   CLI_INSOLUTO_NETTO_CLIOUT,
#   CLI_INSOLUTO_CANONE_TV_CLIOUT,
#   CLI_INSOLUTO_NETTO_R_CLIOUT,
#   CLI_INSOLUTO_6_MM_CLIOUT,
#   CLI_INSOLUTO_1_ANNI_CLIOUT,
#   CLI_INSOLUTO_2_ANNI_CLIOUT,
#   CLI_INSOLUTO_3_ANNI_CLIOUT,
#   CLI_INSOLUTO_5_ANNI_CLIOUT,
#   CLI_INSOLUTO_PIU_ANNI_CLIOUT,
#   DATA_AGGIORNAMENTO_CLIOUT,
#   SNAPSHOT_MORPRE_CLIOUT,
#   FLG_COMTY_GAS,
#    `CLI_AGE_RANGE_18-30`,
#    `CLI_AGE_RANGE_30-40`,
#    `CLI_AGE_RANGE_40-50`,
#    `CLI_AGE_RANGE_50-60`,
#    `CLI_AGE_RANGE_60-70`,
#    `CLI_AGE_RANGE_70-80`,
#    `CLI_AGE_RANGE_80-200`,
#    FLG_COMUNE_UGUALE,
#    FLG_PROV_UGUALE,
#    FLG_REGIONE_UGUALE,
#    FLG_NAZ_UGUALE,
#    DATA_ULT_LETTURA_ATTIV_SWIN,
#    '2025-11-01' as DATA_SNAPSHOT,
#    DATA_SNAPSHOT as ACTUAL_DATA_SNAPSHOT
# from p_sbx_data.data.antifrode_volture_mstr_volturato

# COMMAND ----------

display(get_query_dataframe(query=query_snapshots, args_diz=infer_mstr_cliout_diz))

# COMMAND ----------

infer_mstr_cliout_diz.update({"field1": "PDF_INSOLUTO_LORDO_ALL", "field2": "PDF_INSOLUTO_LORDO_ALL"})
display(get_query_dataframe(query=query_difference, args_diz=infer_mstr_cliout_diz))

# COMMAND ----------

infer_mstr_cliout_diz.update({"field1": "PDF_INSOLUTO_LORDO_LAST", "field2": "PDF_INSOLUTO_LORDO_LAST"})
display(get_query_dataframe(query=query_difference, args_diz=infer_mstr_cliout_diz))

# COMMAND ----------

infer_mstr_cliout_diz.update({"field1": "CLI_INSOLUTO_LORDO_CLIOUT", "field2": "CLI_INSOLUTO_LORDO"})
display(get_query_dataframe(query=query_difference, args_diz=infer_mstr_cliout_diz))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference Master VOLTURANTE Table

# COMMAND ----------

l_snaps_infer = ["2025-11-01"]

string_snaps_infer = "'" + "','".join(l_snaps_infer) + "'"

infer_mstr_cliin_diz = {
    "snapshot_dates": string_snaps_infer,
    "primary_key1": "COD_FISCALE_PIVA",
    "primary_key2": "COD_FISCALE_PIVA_CLIIN",
    "field1": "",
    "field2": "",
    "t0": "p_sbx_data.data.antifrode_volture_mstr_volturante_copia_TEST",
    "t1": "p_sbx_data.data.creri_infer_mstr_volturante",
    "t2": "p_ind_core_ai.creri.creri_infer_mstr_volturante",
    "t3": "p_dpp_core_ai.creri.creri_infer_mstr_volturante",
}

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM p_sbx_data.data.antifrode_volture_mstr_volturante_copia_TEST

# COMMAND ----------

# %sql
# CREATE TABLE p_sbx_data.data.antifrode_volture_mstr_volturante_copia_TEST
# select
#   COD_FISCALE_PIVA,
#   CLI_NR_CC_FORN_MOROSE,
#   CLI_DATA_SCAD_RATA_OLDER,
#   CLI_AGING_SCAD_RATE_OLDER,
#   CLI_NR_FATTURE_INSOLUTE,
#   CLI_NR_RATE_INSOLUTE,
#   CLI_INSOLUTO_LORDO,
#   CLI_INSOLUTO_NETTO,
#   CLI_INSOLUTO_CANONE_TV,
#   CLI_INSOLUTO_NETTO_R,
#   CLI_INSOLUTO_6_MM,
#   CLI_INSOLUTO_1_ANNI,
#   CLI_INSOLUTO_2_ANNI,
#   CLI_INSOLUTO_3_ANNI,
#   CLI_INSOLUTO_5_ANNI,
#   CLI_INSOLUTO_PIU_ANNI,
#   DATA_AGGIORNAMENTO,
#    '2025-11-01' as DATA_SNAPSHOT,
#    DATA_SNAPSHOT as ACTUAL_DATA_SNAPSHOT
# from p_sbx_data.data.antifrode_volture_mstr_volturante

# COMMAND ----------

display(get_query_dataframe(query=query_snapshots, args_diz=infer_mstr_cliin_diz))

# COMMAND ----------

infer_mstr_cliin_diz.update({"field1": "CLI_INSOLUTO_LORDO", "field2": "CLI_INSOLUTO_LORDO_CLIIN"})
display(get_query_dataframe(query=query_difference, args_diz=infer_mstr_cliin_diz))

# COMMAND ----------

# MAGIC %md
# MAGIC ## The End.
