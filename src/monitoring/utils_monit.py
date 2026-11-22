import json
import pandas as pd
import numpy as np
from typing import Any
from pyspark.sql import types as pyt


def request_json_to_pandas(json_string: str) -> pd.DataFrame:
    """
    Method to trasform a string of a json file into a Pandas dataframe.
    Args:
      json_string (string): a json file as a string containing a dictionary.
    Returns:
      df (Pandas dataframe): a dataset extracted from the json file.
    """

    json_dict = json.loads(json_string)  # <class 'dict'>

    values_stringa = json_dict["dataframe_records"][0]["INPUT_JSON"]
    values_stringa = values_stringa.replace("'", "").replace("\\", "")

    dict_input = json.loads(values_stringa)  # <class 'dict'>

    # Reshape input data for several forniture
    dict_no_forn = {k: v for k, v in dict_input.items() if k != "Forniture"}
    forn_list_dict = dict_input["Forniture"]

    list_of_forn_dict = []
    for ee in forn_list_dict:
        assert isinstance(ee, dict), "A Forniture's element is not a dictionary"
        ee_bis = ee.copy()
        ee_bis.update(dict_no_forn.copy())
        list_of_forn_dict.append(ee_bis)

    df = pd.DataFrame(list_of_forn_dict)
    return df


def boolean_fixer(df: pd.DataFrame, field: str) -> pd.DataFrame:
    """
    Function to replace strings with boolean values.
    Args:
      df (Pandas dataframe): an input dataset.
      field (string): name of the column.
    Returns:
      df (Pandas dataframe): the input dataset with boolean values cleaned.
    """
    df[field] = df[field].replace("true", True)
    df[field] = df[field].replace("True", True)
    df[field] = df[field].replace("false", False)
    df[field] = df[field].replace("False", False)
    df[field] = df[field].replace("None", None)
    df[field] = df[field].replace("nan", None)
    df[field] = df[field].replace(np.nan, None)
    for t in df[field].unique().tolist():
        assert isinstance(t, bool) or isinstance(t, type(None))
    return df


def spark_schema_by_pandas(
    df_pandas: pd.DataFrame, datatypes_tuple_list: list[tuple[str, Any]]
) -> tuple[pd.DataFrame, pyt.StructType]:
    """
    Function to create a PySpark's datatypes schema from a datatypes tuple list.
    At the same time the function fixes the Pandas dataframe to match the PySpark's datatypes schema.
    The fixing operations are imputation of null values and format casting.
    Args:
        df_pandas (Pandas dataframe): an input dataset.
        datatypes_tuple_list (list[tuples[string, any]]): a list of tuples containing the field name and the PySpark's datatype.
    Returns:
        df (Pandas dataframe): the input dataset.
        pyspark_schema (PySpark StructType): the PySpark's datatypes schema of the dataset.
    """
    df = df_pandas
    l_struct = []
    for field, dtype in datatypes_tuple_list:
        ind_null = (df[field].isnull()) | (
            df[field].isin(["", "NoneType", "None", "nan", "NaN", "Nan", "NA", "NaT"])
        )

        try:
            if dtype == "string":
                if ind_null.sum() > 0:
                    df.loc[ind_null, field] = ""
                py_dtype = pyt.StringType()
            elif dtype == "date":
                df[field] = df[field].astype(str)
                if ind_null.sum() > 0:
                    df.loc[ind_null, field] = "2199-01-01"
                df[field] = df[field].apply(lambda x: x.split(" ")[0])
                df[field] = pd.to_datetime(
                    df[field], format="%Y-%m-%d", errors="coerce"
                ).dt.date
                py_dtype = pyt.DateType()
            elif dtype == "timestamp":
                df[field] = df[field].astype(str)
                if ind_null.sum() > 0:
                    df.loc[ind_null, field] = "2199-01-01 01:00:00"
                df[field] = df[field].apply(lambda x: x.split(".")[0])
                df[field] = df[field].apply(
                    lambda x: x + " 00:00:00" if len(x.split(" ")) == 1 else x
                )
                df[field] = pd.to_datetime(df[field], format="%Y-%m-%d %H:%M:%S")
                py_dtype = pyt.TimestampType()
            elif dtype == "boolean":
                df = boolean_fixer(df, field)
                py_dtype = pyt.BooleanType()
            elif dtype in ["float", "int", "double", "bigint"]:
                if ind_null.sum() > 0:
                    df.loc[ind_null, field] = np.nan
                df[field] = df[field].astype(float)
                py_dtype = pyt.FloatType()
            elif dtype == "map<string,string>":
                py_dtype = pyt.MapType(pyt.StringType(), pyt.StringType())
            else:
                raise Exception("dtype not supported: {}".format(dtype))
        except Exception as err:
            print(
                'Error in field: "{f}" with dtype: "{d}" - error: "{e}"'.format(
                    f=field, d=dtype, e=err
                )
            )
        l_struct.append(pyt.StructField(field, py_dtype, True))

        # print(field, dtype)
        # df_spark = spark.createDataFrame(df, schema=pyt.StructType(l_struct)) # prova
    return df, pyt.StructType(l_struct)


def monthly_table_aggregator(
    df: pd.DataFrame,
    date: str = "date",
    nr: str = "NR_REQUEST",
    model: dict = {"USED_MODEL": ["SCORING", "DETERMINISTIC"]},
    colors: dict = {"RESPONSE_CATEGORY": ["RED", "YELLOW", "GREEN", "BAD_REQUEST"]},
) -> pd.DataFrame:
    """
    Function to aggregate a Pandas dataframe to produce a monthly table
    Args:
        df (Pandas dataframe): an input dataset.
        date (string): name of the date column.
        nr (string): name of the aggregated column.
        model (dictionary): dictionary with the used model name as key and the list of categories as value.
        colors (dictionary): dictionary with the color-risk's category name as key and the list of categories as value.
    Returns:
        dfm (Pandas dataframe): the input dataset aggregated at monthly scale.
    """
    df["YEAR_MONTH"] = df[date].dt.strftime("%Y-%m")

    dfm = df.groupby("YEAR_MONTH").agg({nr: "sum"}).reset_index()

    for diz in [model, colors]:
        field = list(diz.keys())[0]
        for category in diz[field]:
            ind = df[field] == category
            dfg = (
                df.loc[ind, :]
                .groupby("YEAR_MONTH")
                .agg({nr: "sum"})
                .reset_index()
                .rename(columns={nr: category})
            )
            dfm = pd.merge(dfm, dfg, how="left", on="YEAR_MONTH")

            if diz == model:
                dfm["%_" + category] = (100 * dfm[category] / dfm[nr]).round(1)

    return dfm
