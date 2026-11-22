"""Module with functions to do features enginnering on dataframes"""

from pyspark.sql import DataFrame as pyskDataFrame
import pyspark.sql.functions as F
import pyspark.sql.types as T
from typing import Dict

from src.data_preparation.dataset_splitting import dataset_splitting_classifier


import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _commodity_flag(df: pyskDataFrame) -> pyskDataFrame:
    """
    Function to elaborate the feature of the gas commodity's flag:
    if it is true then the commodity is gas, otherwise it is false and the commodity is power.
    Args:
        df (PySpark DataFrame): an input dataset.
    Returns:
        the initial dataset in input with the new column "FLG_COMTY_GAS" added.
    """
    try:
        df = df.withColumn(
            "FLG_COMTY_GAS", F.when(F.col("POD_PDR").rlike("^IT"), 0).otherwise(1)
        )
        df = df.withColumn("FLG_COMTY_GAS", F.col("FLG_COMTY_GAS").cast("boolean"))
    except AttributeError:
        logger.warning("Missing column POD_PDR")
    return df


def _age_ranges(df: pyskDataFrame) -> pyskDataFrame:
    """
    Function to elaborate the flag features of the age ranges of outgoing customers.
    Args:
        df (PySpark DataFrame): an input dataset.
    Returns:
        the initial dataset in input with the new columns "CLI_AGE_RANGE_18-30", "CLI_AGE_RANGE_30-40", "CLI_AGE_RANGE_40-50", "CLI_AGE_RANGE_50-60", "CLI_AGE_RANGE_60-70", "CLI_AGE_RANGE_70-80", "CLI_AGE_RANGE_80-200" added.
    """
    try:
        for a, b in [
            (18, 30),
            (30, 40),
            (40, 50),
            (50, 60),
            (60, 70),
            (70, 80),
            (80, 200),
        ]:
            df = df.withColumn(
                f"CLI_AGE_RANGE_{a}-{b}",
                F.when((F.col("CLI_AGE") > a) & (F.col("CLI_AGE") <= b), 1).otherwise(
                    0
                ),
            )
            df = df.withColumn(
                f"CLI_AGE_RANGE_{a}-{b}",
                F.col(f"CLI_AGE_RANGE_{a}-{b}").cast("boolean"),
            )
    except AttributeError:
        logger.warning("Missing column CLI_AGE")
    return df


def _location_fields(df: pyskDataFrame) -> pyskDataFrame:
    """
    Function to elaborate the flag features for the fields with the same value between outgoing customer's residential location and his energy supply location:
    municipality flag, province flag, region flag and country flag (Italy).
    Args:
        df (PySpark DataFrame): an input dataset.
    Returns:
        the initial dataset in input with the new columns "FLG_COMUNE_UGUALE", "FLG_PROV_UGUALE", "FLG_REGIONE_UGUALE", "FLG_NAZ_UGUALE" added.
    """
    l_cols = df.columns
    cols_to_eng = {
        # "":"IND_FORN_IND", # indirizzo di residenza del volturato non disponibile
        # "":"IND_FORN_CAP", # cap di residenza del volturato non disponibile
        "CLI_COM": "IND_FORN_COMUNE",
        "CLI_PROV": "IND_FORN_COD_PROV",
        # "":"IND_FORN_NOME_PROV", # nome provincia di residenza del volturato non disponibile
        "CLI_REG": "IND_FORN_REGIONE",
        "CLI_NAZ": "ITALIA",  # forniture sempre in Italia
    }
    for cli, forn in cols_to_eng.items():
        if cli in l_cols and forn in l_cols:
            for rex in ["-", "'"]:
                df = df.withColumn(cli, F.upper(F.regexp_replace(df[cli], rex, " ")))
                df = df.withColumn(forn, F.upper(F.regexp_replace(df[forn], rex, " ")))
            campo = forn.split("_")[-1]
            df = df.withColumn(
                f"FLG_{campo}_UGUALE",
                df[cli] == df[forn],
            )
            df = df.withColumn(
                f"FLG_{campo}_UGUALE", F.col(f"FLG_{campo}_UGUALE").cast("boolean")
            )
        elif cli in l_cols and cli == "CLI_NAZ":
            df = df.withColumn(
                "FLG_NAZ_UGUALE",
                F.when(df[cli].like("ITAL%"), True).otherwise(False),
            )
            df = df.withColumn(
                "FLG_NAZ_UGUALE", F.col("FLG_NAZ_UGUALE").cast("boolean")
            )
        else:
            logger.warning(f"Missing column {cli} or {forn}")
    return df


def _identical_fields(
    df: pyskDataFrame,
) -> pyskDataFrame:
    """
    Function to elaborate the flag features for the fields with the same value between volturante and volturato cutomers.
    Args:
        df (PySpark DataFrame): an input dataset.
    Returns:
        the initial dataset in input with the new columns "FLG_COGNOME_UGUALE", "FLG_EMAIL_UGUALE", "FLG_TELEFONO_UGUALE" added.
    """
    l_cols = df.columns
    cols_to_eng = ["COGNOME", "EMAIL", "TELEFONO"]
    for col in cols_to_eng:
        if col in l_cols:
            df = df.withColumn(
                f"FLG_{col}_UGUALE",
                F.upper(df[col]) == F.upper(df[f"{col}_CLIIN"]),
            )
            df = df.withColumn(
                f"FLG_{col}_UGUALE", F.col(f"FLG_{col}_UGUALE").cast("boolean")
            )
        else:
            logger.warning(f"Missing column {col}")
    return df


def _metering_features(
    df: pyskDataFrame,
) -> pyskDataFrame:
    """
    Function to elaborate  the features of aging between meter measurement date of the last switch-in and the date of contract transfer.
    Args:
        df (PySpark DataFrame): an input dataset.
    Returns:
        the initial dataset in input with the new column "AGING_METER_ATTIV_SWIN" added.
    """
    try:
        df = df.withColumn(
            "AGING_METER_ATTIV_SWIN",
            F.when(
                F.col("DATA_ULT_LETTURA_ATTIV_SWINPWR").isNotNull(),
                F.col("DATA_ULT_LETTURA_ATTIV_SWINPWR"),
            ).otherwise(F.col("DATA_ULT_LETTURA_ATTIV_SWINGAS")),
        )
        df = df.withColumn(
            "AGING_METER_ATTIV_SWIN",
            F.datediff(F.col("DATA_VOLTURA"), F.col("AGING_METER_ATTIV_SWIN")),
        )
        df = df.withColumn(
            "AGING_METER_ATTIV_SWIN", F.col("AGING_METER_ATTIV_SWIN").cast("float")
        )
    except AttributeError:
        logger.warning(
            "Missing column DATA_ULT_LETTURA_ATTIV_SWINPWR or DATA_ULT_LETTURA_ATTIV_SWINGAS or DATA_VOLTURA"
        )
    return df


def _dict_map_encoder(
    df: pyskDataFrame, col_to_encode: str, new_col: str, encode_map: dict
) -> pyskDataFrame:
    """
    Encodes the categories of a feature based on the coding map (encode_map) and creates a new column with the encoded values.
    Args:
        df (PySpark DataFrame): an input dataset.
        col_to_encode (str): the name of the column to encode.
        new_col (str): the name of the new column with the encoded values.
        encode_map (dict): an encoding map of the column.
    Returns:
        the initial dataset in PySpark DataFrame with the new encoded column.
    """
    mapping_expr = F.create_map(*[F.lit(x) for kv in encode_map.items() for x in kv])
    df = df.withColumn(
        new_col, mapping_expr.getItem(F.col(col_to_encode).cast(T.StringType()))
    )
    return df


def _unpaid_flag(df: pyskDataFrame) -> pyskDataFrame:
    """
    Function to elaborate the feature of the unpaid credit's flag:
    if the incoming customer or the outgoing customer has an unpaid credit is True, also if the commodity supply has an unpaid credit is True;
    otherwise False.
    Args:
        df (PySpark DataFrame): an input dataset.
    Returns:
        the initial dataset in input with the new column "FLG_INSOLUTO" added.
    """
    df = df.withColumn(
        "FLG_INSOLUTO",
        F.when(
            (F.col("PDF_INSOLUTO_LORDO_ALL") > 0)
            | (F.col("PDF_INSOLUTO_LORDO_LAST") > 0)
            | (F.col("CLI_INSOLUTO_LORDO") > 0)
            | (F.col("CLI_INSOLUTO_LORDO_CLIIN") > 0),
            True,
        ).otherwise(False),
    )
    return df


def features_engineering(
    df: pyskDataFrame,
    process_phase: str = "training_master_creation",
    encode_maps: Dict[str, dict] = None,
) -> pyskDataFrame:
    """
    Function to group as a pipeline the features engineering elaborations depending on the modeling process phase.
    Args:
        df (PySpark DataFrame): an input dataset.
        process_phase (str): a string indicates the modeling process phase.
        encode_maps (dict[str, dict]): a dictionary where each key is the name of a column and its associated value is an encoding map.
    Returns:
        the initial dataset in PySpark DataFrame with the new engineered columns.
    """
    if process_phase == "training_master_creation":
        df = _unpaid_flag(df)
        df = _commodity_flag(df)
        df = _age_ranges(df)
        df = _location_fields(df)
        df = _identical_fields(df)
        df = _metering_features(df)
        for col, emap in encode_maps.items():
            df = _dict_map_encoder(
                df=df, col_to_encode=col, new_col=f"ENCOD_{col}", encode_map=emap
            )
        df = dataset_splitting_classifier(master=df)
    elif process_phase == "inference_master_creation":
        df = _commodity_flag(df)
        df = _age_ranges(df)
        df = _location_fields(df)
        for col, emap in encode_maps.items():
            df = _dict_map_encoder(
                df=df, col_to_encode=col, new_col=f"ENCOD_{col}", encode_map=emap
            )
    # elif process_phase == "inference_preprocess_data":
    #     df = _identical_fields(df)
    #     df = _metering_features(df)
    #     for col, emap in encode_maps.items():
    #         df = _dict_map_encoder(df=df, col_to_encode=col, new_col= f"ENCOD_{col}", encode_map=emap)
    else:
        logger.warning(
            f"Invalid value of process_phase parameter: {process_phase}. It has be one of: 'training_master_creation', 'inference_master_creation', 'inference_preprocess_data'."
        )
    return df


def get_version():
    return "1"
