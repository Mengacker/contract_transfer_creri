"""
Module with functions to do some data manipulations after the creation of the master dataset,
so to pre-processing the master dataset for modeling by preparing data for tuning, training, test, evaluation and comparison.
"""

import pandas as pd
from pandas import Series as pandasSeries
from pandas import DataFrame as pandasDataFrame
from pyspark.sql import SparkSession

from src.data_preparation.oversampler import oversampler
from src.config.config_utils import Config

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cast_pandas_datatype(df: pandasDataFrame) -> pandasDataFrame:
    """
    Function to cast Pandas dataframe columns to the configured datatypes.
    Args:
        df (Pandas Dataframe): an input dataset.
    Returns:
        the input dataset with the columns casted to the configured datatypes.
    """
    # Retrieve config data values
    config_data = Config("../src/config/config_data.yml")
    target_features = config_data.get_config_value("target_features")
    input_features = config_data.get_config_value("input_features")
    engineered_features = config_data.get_config_value("engineered_features")
    util_columns = config_data.get_config_value("util_columns")
    not_predictors = config_data.get_config_value("not_predictors")

    # Get the Pandas' datatype map
    pandas_dtype_map = {}
    for diz in [target_features, input_features, engineered_features]:
        pandas_dtype_map.update(
            {c: d2["pandas_type"] for group, d1 in diz.items() for c, d2 in d1.items()}
        )
    pandas_dtype_map.update({c: d1["pandas_type"] for c, d1 in util_columns.items()})

    # Manage missing columns
    missing_cols = [c for c in list(pandas_dtype_map.keys()) if c not in df.columns]
    pandas_dtype_map = {
        c: d for c, d in pandas_dtype_map.items() if c not in missing_cols
    }
    missing_cols = [
        c
        for c in missing_cols
        if c
        not in not_predictors
        + [c for m, d in target_features.items() for c in d.keys()]
    ]
    if len(missing_cols) > 0:
        logger.warning(
            f"The dataset does not have the configured 'config_data.yml' columns: {missing_cols}\n"
        )

    # Reshape
    pandas_dtype_map = {
        dd: [c for c, d in pandas_dtype_map.items() if d == dd]
        for dd in set(pandas_dtype_map.values())
    }

    # CAST PANDAS DATATYPES
    for dtype, col_list in pandas_dtype_map.items():
        if dtype == "date":
            for cc in col_list:
                df[cc] = pd.to_datetime(df[cc], format="%Y-%m-%d")
        elif dtype in ["string", "category", "bool", "int64", "float64"]:
            df[col_list] = df[col_list].astype(dtype)
        else:
            logger.warning(
                f"The Pandas datatype '{dtype}' is not expected in the cast_pandas_datatype function, but there are columns with this datatype:{col_list}"
            )
            df[col_list] = df[col_list].astype(dtype)

    return df


def modeling_pre_processer(
    table_catalog_path: str,
    utility_set: list[str],
    oversampling: bool = False,
    params: dict = None,
) -> (pandasSeries, pandasDataFrame, pandasDataFrame):
    """
    Function to pre-process the master data for modeling: read the dataset, cast the Pandas datatype, encode some categorial fields, filter values and select the utility sub-dataset (train, test or all), do the oversampling, split target column by the predictors columns.
    Args:
        table_catalog_path (string): the path of the table to read.
        utility_set (list[string]): the list of the "utility_set"'s categories (train, calib, test) to select a sub-set of the dataset.
        oversampling (bool): True to do the oversampling, False otherwise.
        params (dictionary): a dictionary of other optional parameters.
    Returns:
        target (Pandas Series): a dataset with only the target column and the filtered rows.
        predictors (Pandas DataFrame): a dataset with only the predictors columns and the filtered rows.
        master (Pandas DataFrame): a dataset with all the columns and the filtered rows.
    """
    params = params or {"macrosegmento": "ALL"}
    macrosegmento = params.get("macrosegmento") or "ALL"

    # Retrieve config_data values
    config_model = Config("../src/config/config_model.yml")
    seed_oversampler = config_model.get_config_value("random_state")

    config_data = Config("../src/config/config_data.yml")
    targets = config_data.get_config_value("target_features.training")
    input_features = config_data.get_config_value("input_features")
    engineered_features_ante = config_data.get_config_value(
        "engineered_features.ante_save_master"
    )
    engineered_features_post = config_data.get_config_value(
        "engineered_features.post_save_master"
    )
    util_columns = config_data.get_config_value("util_columns")
    not_predictors = config_data.get_config_value("not_predictors")
    not_predictors_but_utils = config_data.get_config_value("not_predictors_but_utils")

    # Read master
    spark = SparkSession.builder.getOrCreate()
    master_spk = spark.sql("SELECT * FROM {tab}".format(tab=table_catalog_path))
    # cols_master_spk = list(master_spk.columns)

    # Select only the necessary columns
    target = list(targets.keys())[0]  # the only one target "FLG_CLI_MOROSO" to predict
    other_targets = list(targets.keys())[1:]
    input_features = [c for x, d1 in input_features.items() for c, d2 in d1.items()]
    util_columns = list(util_columns.keys())

    cols_to_read = (
        util_columns
        + [target]
        + other_targets
        + not_predictors_but_utils
        + input_features
        + list(engineered_features_ante.keys())
    )
    cols_to_read = [
        cc
        for cc in cols_to_read
        if cc not in not_predictors + list(engineered_features_post.keys())
    ]
    cols_to_read = list(
        dict.fromkeys(cols_to_read)
    )  # remove duplicates without lose order
    print(f"n°{len(cols_to_read)} modeling_cols: {cols_to_read}\n")

    # Filtering columns
    master = master_spk.select(*cols_to_read).toPandas()
    print(f"{master.shape} master shape with the selected columns\n")

    # Target imputation
    ind = master[target].isnull()
    master.loc[ind, target] = 0
    print(f"{ind.sum()} number of nulls in the target column\n")

    # Transform Spark dataframe into a Pandas dataframe
    master = cast_pandas_datatype(df=master)
    nrow_master = master.shape[0]
    print(f"{master.shape} master shape after cast pandas datatype\n")

    # Filtro per MACROSEGMENTO
    ind_macrosegment = master["ENCOD_MACROSEGMENTO"] == "RES"
    if macrosegmento == "RESIDENZIALI":
        master = master.loc[ind_macrosegment, :].reset_index(drop=True)
    elif (
        macrosegmento == "MICROBUSINESS"
    ):  # actually MICROBUSINESS and CONDOMINIO, PUBBLICA AMMINISTRAZIONE, IMPRESE, MULTISITO, null
        master = master.loc[~ind_macrosegment, :].reset_index(drop=True)
    else:
        del ind_macrosegment

    print(
        f"{master.shape} master shape after the '{macrosegmento}' macrosegment selection and filter\n"
    )

    # Filtro solo i casi con del credito insoluto
    print(
        "***************** Selected only customers with bad credit *****************\n"
    )
    master_insolventi = master.loc[master["FLG_INSOLUTO"], :].reset_index(drop=True)
    nrow_master = master_insolventi.shape[0]

    # Select the utility sub-dataset
    ind = master_insolventi["UTILITY_SET"].isin(utility_set)
    sub_master = master_insolventi.loc[ind, :].reset_index(drop=True)
    n_target_1 = sum(sub_master[target] == 1)
    nrow_sub = sub_master.shape[0]
    print(
        f"df {utility_set[0]}: {sub_master.shape} the {100*nrow_sub/nrow_master} % of the total {nrow_master}, {n_target_1} n° of target in the minority class '1'\n"
    )
    if oversampling:
        # Oversample minority class in the training set
        sub_master = oversampler(
            df=sub_master,
            target=target,
            oversampling_ratio="minority",
            random_state=seed_oversampler,
        )
        n_target_1 = sum(sub_master[target] == 1)
        print(
            f"df {utility_set[0]} oversampled: {sub_master.shape} the {100*sub_master.shape[0]/nrow_sub} % of the inital {utility_set[0]} set {nrow_sub}, {n_target_1} n° of target in the minority class '1'\n"
        )

    # Split target and predictors columns
    y = sub_master[target]
    col_to_drop = [
        not_x
        for not_x in list(sub_master.columns)
        if not_x
        in [target]
        + other_targets
        + list(util_columns)
        + [
            "SDF_CANLE_ATTIVAZ_ATT_CLIIN",
            "MACROSEGMENTO_CLIIN",
            "MACROSEGMENTO",
            "FLG_INSOLUTO",
        ]
    ]
    X = sub_master.drop(col_to_drop, axis=1)

    return y, X, master
