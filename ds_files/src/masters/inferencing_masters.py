import pyspark.sql.functions as F
from pyspark.sql import DataFrame as pysparkDataFrame
from typing import Any, List, Dict

from src.data_preparation.features_engineering import features_engineering

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class materializationInferenceMasterVolturante:
    """
    Class to build the master dataset of incoming customers for the inference.
    """

    def __init__(self, spark, logger):
        """
        Initialization method to initialize the class.
        Arguments:
            spark (SparkSession): the SparkSession object.
            logger (Logger): the logger object.
        """
        self.spark = spark
        self.logger = logger

    def get_dataframe_as_of(
        self,
        snapshot_date: str,
        catalog_name: str,
        schema_name: str,
        general_utils: Any,
        metric_materializer: Any,
    ) -> pysparkDataFrame:
        """
        Method to get the PySpark dataframe of the dataset as of a given date.
        Args:
            snapshot_date (string): a string representing the snapshot date in the format "YYYY-MM-DD".
            catalog_name (string): the name of the catalog.
            schema_name (string): the name of the schema.
            general_utils (Any): the general_utils object.
            metric_materializer (Any): the metric_materializer object.
        Return:
            df_CLIIN (pysparkDataFrame): The incoming customers dataframe containing the dataset as of the given date.
        """

        metric = f"{catalog_name}.{schema_name}.ccvol_met_morositapregressacliente"

        max_snapshot_date = (
            general_utils.table_reader(
                metric.split(".")[0],
                metric.split(".")[1],
                metric.split(".")[2],
            )
            .select(F.max("snapshot_date").alias("max_date"))
            .first()["max_date"]
        )

        df_CLIIN = (
            general_utils.table_reader(
                metric.split(".")[0],
                metric.split(".")[1],
                metric.split(".")[2],
            ).where(f"snapshot_date = '{max_snapshot_date}' and flg_lastSnap = 1")
            # .withColumnRenamed("COD_FISCALE_PIVA", "COD_FISCALE_PIVA_CLIIN")
        )
        df_CLIIN = metric_materializer.remove_tech_columns(df_CLIIN)
        general_utils.raise_assert_value(df_CLIIN, metric, snapshot_date)

        cols = df_CLIIN.columns

        logger.info(
            f"The shape of the incoming customer's inference master dataset is: ({df_CLIIN.count()}, {len(cols)})"
        )
        print(
            f"The shape of the incoming customer's inference master dataset is: ({df_CLIIN.count()}, {len(cols)})"
        )
        print(f"\nColumn names: {str(cols)}\n")

        for col in cols:
            # if col not in ["COD_FISCALE_PIVA_CLIIN"]:
            df_CLIIN = df_CLIIN.withColumnRenamed(col, f"{col}_CLIIN")

        print(f"\nColumn names: {str(df_CLIIN.columns)}\n")
        return df_CLIIN

    def get_key_columns(self) -> List[str]:
        """
        Method to get the join key columns of the dataset.
        Return:
            A list of strings containing the key columns.
        """
        return ["COD_FISCALE_PIVA_CLIIN"]

    def get_version() -> int:
        """
        Method to get the version of this inference master dataset.
        If significant changes are made to the dataset, the version must be increased.
        Returns:
            version (integer): the version of this inference master dataset.
        """
        return 1


class materializationInferenceMasterVolturato:
    """
    Class to build the master dataset of outgoing customer for the inference. It joins some metrics datasets and then do the feature engineering.
    """

    def __init__(self, spark, logger):
        """
        Initialization method to initialize the class.
        Arguments:
            spark (SparkSession): the SparkSession object.
            logger (Logger): the logger object.
        """
        self.spark = spark
        self.logger = logger

    def get_dataframe_as_of(
        self,
        snapshot_date: str,
        catalog_name: str,
        schema_name: str,
        # df_morosita_cliente: pysparkDataFrame,
        encode_maps: Dict[str, dict],
        general_utils: Any,
        metric_materializer: Any,
    ) -> pysparkDataFrame:
        """
        Method to get the PySpark dataframe of the dataset as of a given date.
        Args:
            snapshot_date (string): a string representing the snapshot date in the format "YYYY-MM-DD".
            catalog_name (string): the name of the catalog.
            schema_name (string): the name of the schema.

            encode_maps (Dict[string, dict]): a dictionary where each key is the name of a column and its associated value is an encoding map.
            general_utils (Any): the general_utils object.
            metric_materializer (Any): the metric_materializer object.
        Return:
            df_CLIOUT (pysparkDataFrame): The outgoing customers dataframe containing the dataset as of the given date.
        """

        # NEW POSSIBLE OPTIMIZED CODE
        metrics = {
            f"{catalog_name}.{schema_name}.ccvol_met_morositapregressapodpdr": {
                "pk": "POD_PDR",
                "rename": [],
                "drop": [],
            },
            f"{catalog_name}.{schema_name}.ccvol_met_dataultletturaattivswinpwr": {
                "pk": "POD_PDR",
                "rename": [("COD_POD", "POD_PDR")],
                "drop": ["tracking_id_ult_swin"],
            },
            f"{catalog_name}.{schema_name}.ccvol_met_dataultletturaattivswingas": {
                "pk": "POD_PDR",
                "rename": [("COD_PDR", "POD_PDR")],
                "drop": ["tracking_id_ult_swin"],
            },
            f"{catalog_name}.{schema_name}.ccvol_met_metricapdfpodpdrgtwv": {
                "pk": "COD_PDF",
                "rename": [],
                "drop": [],
            },
            f"{catalog_name}.{schema_name}.ccvol_met_sdfattivazioneattivi": {
                "pk": "COD_PDF",
                "rename": [("PK_PDF", "COD_PDF")],
                "drop": [],
            },
            f"{catalog_name}.{schema_name}.ccvol_met_clienteattributi": {
                "pk": "PK_ACCOUNT",
                "rename": [],
                "drop": [],
            },
            f"{catalog_name}.{schema_name}.ccvol_met_accounttocodfiscalepiva": {
                "pk": "PK_ACCOUNT",
                "rename": [],
                "drop": [],
            },
            f"{catalog_name}.{schema_name}.ccvol_met_morositapregressacliente": {
                "pk": "COD_FISCALE_PIVA",
                "rename": [],
                "drop": [],
            },
        }
        mets = [k for k in metrics.keys()]
        logger.info(
            f"The inference master dataset of outgoing customers has the following metrics:\n {str(mets)}\n"
        )
        print(
            f"The inference master dataset of outgoing customers has the following metrics:\n {str(mets)}\n"
        )
        df_CLIOUT = self.spark.createDataFrame([], schema="empty_dataframe string")
        for mm, diz in metrics.items():
            # Get the max snapshot date materialized in the metric's table
            max_snapshot_date = (
                general_utils.table_reader(
                    mm.split(".")[0], mm.split(".")[1], mm.split(".")[2]
                )
                .select(F.max("snapshot_date").alias("max_date"))
                .first()["max_date"]
            )
            # Read the chunk of metric's table where the snapshot date is equal to the max snapshot date
            df = general_utils.table_reader(
                mm.split(".")[0], mm.split(".")[1], mm.split(".")[2]
            ).where(f"snapshot_date = '{max_snapshot_date}' and flg_lastSnap = 1")
            # raise_assert_value check
            general_utils.raise_assert_value(df, mm, snapshot_date)
            # Remove technical columns
            df = metric_materializer.remove_tech_columns(df)
            # Rename and drop columns
            for tupl in diz["rename"]:
                df = df.withColumnRenamed(tupl[0], tupl[1])
            for col in diz["drop"]:
                df = df.drop(col)
                logger.info(
                    f'The dataframe of metric "{mm}" has been loaded and joined.'
                )
            # Eventual Join
            if df_CLIOUT.count() > 0:
                df_CLIOUT = df_CLIOUT.join(df, on=diz["pk"], how="left")
            else:
                df_CLIOUT = df

        # -------------------------------------------------------------------------------
        # OLD FUNCTIONAL CODE
        # metrics = [
        #     f"{catalog_name}.{schema_name}.ccvol_met_morositapregressapodpdr",
        #     f"{catalog_name}.{schema_name}.ccvol_met_dataultletturaattivswinpwr",
        #     f"{catalog_name}.{schema_name}.ccvol_met_dataultletturaattivswingas",
        #     f"{catalog_name}.{schema_name}.ccvol_met_metricapdfpodpdrgtwv",
        #     f"{catalog_name}.{schema_name}.ccvol_met_sdfattivazioneattivi",
        #     f"{catalog_name}.{schema_name}.ccvol_met_clienteattributi",
        #     f"{catalog_name}.{schema_name}.ccvol_met_accounttocodfiscalepiva",
        #     f"{catalog_name}.{schema_name}.ccvol_met_morositapregressacliente",
        # ]
        # logger.info(f"The inference master dataset of outgoing customers has the following metrics:\n {str(metrics)}\n")
        # print(f"The inference master dataset of outgoing customers has the following metrics:\n {str(metrics)}\n")

        # mm = f"{catalog_name}.{schema_name}.ccvol_met_morositapregressapodpdr"
        # if mm in metrics:
        #     max_snapshot_date = (
        #         general_utils.table_reader(
        #             mm.split(".")[0], mm.split(".")[1], mm.split(".")[2]
        #         )
        #         .select(F.max("snapshot_date").alias("max_date"))
        #         .first()["max_date"]
        #     )
        #     df_CLIOUT = general_utils.table_reader(
        #         mm.split(".")[0], mm.split(".")[1], mm.split(".")[2]
        #     ).where(f"snapshot_date = '{max_snapshot_date}' and flg_lastSnap = 1")
        #     general_utils.raise_assert_value(df_CLIOUT, mm, snapshot_date)

        #     df_CLIOUT = metric_materializer.remove_tech_columns(df_CLIOUT)
        #     # df_CLIOUT = df_CLIOUT.withColumnRenamed(
        #     #     "DATA_AGGIORNAMENTO", "DATA_AGG_MORPP"
        #     # )
        #     logger.info(f'The dataframe of metric "{mm}" has been loaded correctly.')
        #     print(f'The dataframe of metric "{mm}" has been loaded correctly:\n {df_CLIOUT.columns}\n')
        # else:
        #     logger.warning(f'The dataframe of metric "{mm}" has not been loaded!')
        #     print(f'The dataframe of metric "{mm}" has not been loaded!')

        # mm = f"{catalog_name}.{schema_name}.ccvol_met_dataultletturaattivswinpwr"
        # if mm in metrics:
        #     max_snapshot_date = (
        #         general_utils.table_reader(
        #             mm.split(".")[0], mm.split(".")[1], mm.split(".")[2]
        #         )
        #         .select(F.max("snapshot_date").alias("max_date"))
        #         .first()["max_date"]
        #     )
        #     df_to_join = general_utils.table_reader(
        #         mm.split(".")[0], mm.split(".")[1], mm.split(".")[2]
        #     ).where(
        #         f"snapshot_date = '{max_snapshot_date}' and flg_lastSnap = 1"
        #     )
        #     general_utils.raise_assert_value(df_to_join, mm, snapshot_date)

        #     df_to_join = metric_materializer.remove_tech_columns(df_to_join)
        #     df_to_join = (
        #         df_to_join.withColumn("POD_PDR", F.col("COD_POD"))
        #         #.withColumnRenamed("DATA_AGGIORNAMENTO", "DATA_AGG_SWINPWR")
        #         .drop("tracking_id_ult_swin")
        #     )
        #     df_CLIOUT = df_CLIOUT.join(df_to_join, on="POD_PDR", how="left")
        #     logger.info(f'The dataframe of metric "{mm}" has been loaded and joined correctly.')
        #     print(f'The dataframe of metric "{mm}" has been loaded and joined correctly:\n {df_CLIOUT.columns}\n')
        # else:
        #     logger.warning(f'The dataframe of metric "{mm}" has not been loaded or joined!')
        #     print(f'The dataframe of metric "{mm}" has not been loaded or joined!')

        # mm = f"{catalog_name}.{schema_name}.ccvol_met_dataultletturaattivswingas"
        # if mm in metrics:
        #     max_snapshot_date = (
        #         general_utils.table_reader(
        #             mm.split(".")[0], mm.split(".")[1], mm.split(".")[2]
        #         )
        #         .select(F.max("snapshot_date").alias("max_date"))
        #         .first()["max_date"]
        #     )
        #     df_to_join = general_utils.table_reader(
        #         mm.split(".")[0], mm.split(".")[1], mm.split(".")[2]
        #     ).where(
        #         f"snapshot_date = '{max_snapshot_date}' and flg_lastSnap = 1"
        #     )
        #     general_utils.raise_assert_value(df_to_join, mm, snapshot_date)

        #     df_to_join = metric_materializer.remove_tech_columns(df_to_join)
        #     df_to_join = df_to_join.withColumn("POD_PDR", F.col("COD_PDR"))
        #     df_to_join = (
        #         df_to_join.withColumnRenamed(
        #             "data_ult_lettura_attiv_swingas", "DATA_ULT_LETTURA_ATTIV_SWINGAS"
        #         )
        #         #.withColumnRenamed("DATA_AGGIORNAMENTO", "DATA_AGG_SWINGAS")
        #         .drop("tracking_id_ult_swin")
        #     )
        #     df_CLIOUT = df_CLIOUT.join(df_to_join, on="POD_PDR", how="left")
        #     logger.info(f'The dataframe of metric "{mm}" has been loaded and joined correctly.')
        #     print(f'The dataframe of metric "{mm}" has been loaded and joined correctly:\n {df_CLIOUT.columns}\n')
        # else:
        #     logger.warning(f'The dataframe of metric "{mm}" has not been loaded or joined!')
        #     print(f'The dataframe of metric "{mm}" has not been loaded or joined!')

        # mm = f"{catalog_name}.{schema_name}.ccvol_met_metricapdfpodpdrgtwv"
        # if mm in metrics:
        #     max_snapshot_date = (
        #         general_utils.table_reader(
        #             mm.split(".")[0], mm.split(".")[1], mm.split(".")[2]
        #         )
        #         .select(F.max("snapshot_date").alias("max_date"))
        #         .first()["max_date"]
        #     )
        #     df_to_join = general_utils.table_reader(
        #         mm.split(".")[0], mm.split(".")[1], mm.split(".")[2]
        #     ).where(f"snapshot_date = '{max_snapshot_date}' and flg_lastSnap = 1")
        #     general_utils.raise_assert_value(df_to_join, mm, snapshot_date)

        #     df_to_join = metric_materializer.remove_tech_columns(df_to_join)
        #     # df_to_join = df_to_join.withColumnRenamed(
        #     #     "DATA_AGGIORNAMENTO", "DATA_AGG_INDGTW"
        #     # )
        #     df_CLIOUT = df_CLIOUT.join(df_to_join, on="COD_PDF", how="left")
        #     logger.info(f'The dataframe of metric "{mm}" has been loaded and joined correctly.')
        #     print(f'The dataframe of metric "{mm}" has been loaded and joined correctly:\n {df_CLIOUT.columns}\n')
        # else:
        #     logger.warning(f'The dataframe of metric "{mm}" has not been loaded or joined!')
        #     print(f'The dataframe of metric "{mm}" has not been loaded or joined!')

        # mm = f"{catalog_name}.{schema_name}.ccvol_met_sdfattivazioneattivi"
        # if mm in metrics:
        #     max_snapshot_date = (
        #         general_utils.table_reader(
        #             mm.split(".")[0], mm.split(".")[1], mm.split(".")[2]
        #         )
        #         .select(F.max("snapshot_date").alias("max_date"))
        #         .first()["max_date"]
        #     )
        #     df_to_join = (
        #         general_utils.table_reader(
        #             mm.split(".")[0], mm.split(".")[1], mm.split(".")[2]
        #         )
        #         .where(f"snapshot_date = '{max_snapshot_date}' and flg_lastSnap = 1")
        #         #.withColumn("DATA_AGGIORNAMENTO", F.col("snapshot_date"))
        #     )
        #     general_utils.raise_assert_value(df_to_join, mm, snapshot_date)

        #     df_to_join = metric_materializer.remove_tech_columns(df_to_join)
        #     df_to_join = df_to_join.withColumnRenamed(
        #         "PK_PDF", "COD_PDF"
        #     )#.withColumnRenamed("DATA_AGGIORNAMENTO", "DATA_AGG_SDFAA")
        #     df_CLIOUT = df_CLIOUT.join(df_to_join, on="COD_PDF", how="left")
        #     logger.info(f'The dataframe of metric "{mm}" has been loaded and joined correctly.')
        #     print(f'The dataframe of metric "{mm}" has been loaded and joined correctly:\n {df_CLIOUT.columns}\n')
        # else:
        #     logger.warning(f'The dataframe of metric "{mm}" has not been loaded or joined!')
        #     print(f'The dataframe of metric "{mm}" has not been loaded or joined!')

        # mm = f"{catalog_name}.{schema_name}.ccvol_met_clienteattributi"
        # if mm in metrics:
        #     max_snapshot_date = (
        #         general_utils.table_reader(
        #             mm.split(".")[0], mm.split(".")[1], mm.split(".")[2]
        #         )
        #         .select(F.max("snapshot_date").alias("max_date"))
        #         .first()["max_date"]
        #     )
        #     df_to_join = general_utils.table_reader(
        #         mm.split(".")[0], mm.split(".")[1], mm.split(".")[2]
        #     ).where(f"snapshot_date = '{max_snapshot_date}' and flg_lastSnap = 1")
        #     general_utils.raise_assert_value(df_to_join, mm, snapshot_date)

        #     df_to_join = metric_materializer.remove_tech_columns(df_to_join)
        #     # df_to_join = df_to_join.withColumnRenamed("SOURCE_CRM", "SOURCE_CRM_CLIOUT")
        #     #.withColumnRenamed("DATA_AGGIORNAMENTO", "DATA_AGG_CLIATT")
        #     df_CLIOUT = df_CLIOUT.join(df_to_join, on="PK_ACCOUNT", how="left")
        #     logger.info(f'The dataframe of metric "{mm}" has been loaded and joined correctly.')
        #     print(f'The dataframe of metric "{mm}" has been loaded and joined correctly:\n {df_CLIOUT.columns}\n')
        # else:
        #     logger.warning(f'The dataframe of metric "{mm}" has not been loaded or joined!')
        #     print(f'The dataframe of metric "{mm}" has not been loaded or joined!')

        # mm = f"{catalog_name}.{schema_name}.ccvol_met_accounttocodfiscalepiva"
        # if mm in metrics:
        #     max_snapshot_date = (
        #         general_utils.table_reader(
        #             mm.split(".")[0], mm.split(".")[1], mm.split(".")[2]
        #         )
        #         .select(F.max("snapshot_date").alias("max_date"))
        #         .first()["max_date"]
        #     )
        #     df_to_join = general_utils.table_reader(
        #         mm.split(".")[0], mm.split(".")[1], mm.split(".")[2]
        #     ).where(f"snapshot_date = '{max_snapshot_date}' and flg_lastSnap = 1")
        #     general_utils.raise_assert_value(df_to_join, mm, snapshot_date)

        #     df_to_join = metric_materializer.remove_tech_columns(df_to_join)
        #     # df_to_join = df_to_join.withColumnRenamed(
        #     #     "DATA_ULTIMO_AGGIORNAMENTO", "SNAPSHOT_ACC_CFPIVA"
        #     # )

        #     df_CLIOUT = df_CLIOUT.join(df_to_join, on="PK_ACCOUNT", how="left")
        #     logger.info(f'The dataframe of metric "{mm}" has been loaded and joined correctly.')
        #     print(f'The dataframe of metric "{mm}" has been loaded and joined correctly:\n {df_CLIOUT.columns}\n')
        # else:
        #     logger.warning(f'The dataframe of metric "{mm}" has not been loaded or joined!')
        #     print(f'The dataframe of metric "{mm}" has not been loaded or joined!')

        # mm = f"{catalog_name}.{schema_name}.ccvol_met_morositapregressacliente"
        # if mm in metrics:
        #     max_snapshot_date = (
        #         general_utils.table_reader(
        #             mm.split(".")[0], mm.split(".")[1], mm.split(".")[2]
        #         )
        #         .select(F.max("snapshot_date").alias("max_date"))
        #         .first()["max_date"]
        #     )
        #     df_to_join = general_utils.table_reader(
        #         mm.split(".")[0], mm.split(".")[1], mm.split(".")[2]
        #     ).where(f"snapshot_date = '{max_snapshot_date}' and flg_lastSnap = 1")
        #     general_utils.raise_assert_value(df_to_join, mm, snapshot_date)

        #     df_to_join = metric_materializer.remove_tech_columns(df_to_join)
        #     df_CLIOUT = df_CLIOUT.join(df_to_join, on="COD_FISCALE_PIVA", how="left")

        #     logger.info(f'The dataframe of metric "{mm}" has been loaded and joined correctly.')
        #     print(f'The dataframe of metric "{mm}" has been loaded and joined correctly:\n {df_CLIOUT.columns}\n')
        # else:
        #     logger.warning(f'The dataframe of metric "{mm}" has not been loaded or joined!')
        #     print(f'The dataframe of metric "{mm}" has not been loaded or joined!')

        # ------------------------------------------features_engineering-----------------------------------
        df_CLIOUT = features_engineering(
            df_CLIOUT,
            process_phase="inference_master_creation",
            encode_maps=encode_maps,
        )
        df_CLIOUT = df_CLIOUT.withColumn(
            "DATA_ULT_LETTURA_ATTIV_SWIN",
            F.when(
                F.col("DATA_ULT_LETTURA_ATTIV_SWINPWR").isNotNull(),
                F.col("DATA_ULT_LETTURA_ATTIV_SWINPWR"),
            ).otherwise(F.col("DATA_ULT_LETTURA_ATTIV_SWINGAS")),
        )
        df_CLIOUT = df_CLIOUT.withColumnRenamed(
            "PK_ACCOUNT", "FK_ACCOUNT"
        ).withColumnRenamed("PK_CONTO_FATT", "FK_CONTO_FATT")
        logger.info("The feature engineering has been applied.")

        cols = list(df_CLIOUT.columns)
        logger.info(
            f"The shape of the outgoing customer's inference master dataset is: ({df_CLIOUT.count()}, {len(cols)})"
        )
        print(
            f"The shape of the outgoing customer's inference master dataset is: ({df_CLIOUT.count()}, {len(cols)})"
        )

        df_CLIOUT = df_CLIOUT.dropDuplicates(["POD_PDR"])
        logger.info(
            f"After a drop duplicates operation, the shape of the outgoing customer's inference master dataset is: ({df_CLIOUT.count()}, {len(cols)})"
        )
        print(
            f"After a drop duplicates operation, the shape of the outgoing customer's inference master dataset is: ({df_CLIOUT.count()}, {len(cols)})"
        )
        return df_CLIOUT

    def get_key_columns(self) -> List[str]:
        """
        Method to get the join key columns of the dataset.
        Return:
            A list of strings containing the key columns.
        """
        return ["POD_PDR"]

    def get_version() -> int:
        """
        Method to get the version of this inference master dataset.
        If significant changes are made to the dataset, the version must be increased.
        Returns:
            version (integer): the version of this inference master dataset.
        """
        return 1
