from pyspark.sql import DataFrame as pysparkDataFrame
import pyspark.sql.functions as F
from typing import List, Any


class masterMaterilizer:
    """
    Class to build a master dataset for training a model. It joins the customer base dataset with the target and metrics datasets.
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
        entita_by_name,
        cb_by_name,
        input_params_json: dict,
        catalog_name: str,
        schema_name: str,
        table_name: str,
        snapshot_date: str,
        metric_materializer: Any,
    ) -> pysparkDataFrame:
        """
        Method to get the dataframe as of a given date.
        Args:
            entita_by_name (string): the entity name.
            cb_by_name (string): the customer base name.
            input_params_json (dict): the input parameters json.
            catalog_name (string): the catalog name.
            schema_name (string): the schema name.
            table_name (string): the table name.
            snapshot_date (string): the snapshot date in format YYYY-MM-DD.
            metric_materializer (Any): the metric_materializer object.
        Returns:
            result_df (PySpark dataFrame): the dataframe as of a given date.
        """
        self.logger.info(
            f"STARTING MASTER creation process: {catalog_name}.{schema_name}.{table_name}"
        )
        self.logger.info(f"STARTING Materialization snapshot_date: {snapshot_date}")
        result_df = []
        # c_base = input_params_json.get("src_cb")
        c_base = cb_by_name.get(input_params_json.get("src_cb"))

        if c_base:
            self.logger.info(f"Reading table CUSTOMER_BASE: {c_base}")
            result_df = self.spark.read.table(
                f"{catalog_name}.{schema_name}.{c_base}"
            ).where(f"snapshot_date = '{snapshot_date}' and flg_lastSnap = 1")

            self.logger.info(
                f"SUCCESS read table CUSTOMER_BASE with number of records: {result_df.count()}"
            )
        else:
            self.logger.info(f"CUSTOMER_BASE not found: {c_base}")
            raise Exception(f"CUSTOMER_BASE not found: {c_base}")

        sorgenti = input_params_json.get("src_metrics", [])
        if isinstance(sorgenti, str):
            sorgenti = [sorgenti]

        for sorgente in sorgenti:
            tabella_sorgente = entita_by_name.get(sorgente + "_name")
            key_columns = entita_by_name.get(sorgente + "_keys")
            if tabella_sorgente:
                self.logger.info(f"SOURCE METRIC: {tabella_sorgente}")
                self.logger.info(f"Source metric key columns: {key_columns}")
                df = self.spark.read.table(
                    f"{catalog_name}.{schema_name}.{tabella_sorgente}"
                ).where(f"snapshot_date = '{snapshot_date}' and flg_lastSnap = 1")

                self.logger.info(f"Source metric columns length: {len(df.columns)}")
                self.logger.info(f"Source metric count: {df.count()}")

                assert df.limit(1).count() > 0, (
                    f"La tabella '{catalog_name}.{schema_name}.{tabella_sorgente}' "
                    f"non contiene record per snapshot_date='{snapshot_date}' e flg_lastSnap = 1."
                )

                existing_columns = set(result_df.columns) - set(key_columns)
                self.logger.info(f"Reading source metrics: {tabella_sorgente}")

                df_clean = metric_materializer.remove_tech_columns(df)
                self.logger.info(
                    f"Removed tech columns from source metrics: {tabella_sorgente}"
                )

                metrica1_df_renamed = metric_materializer.rename_conflicting_columns(
                    df_clean, existing_columns, "_SUFFIX"
                )
                self.logger.info(
                    f"Renamed conflicted columns from source metrics: {tabella_sorgente}"
                )

                result_df = result_df.join(metrica1_df_renamed, key_columns, how="left")
                self.logger.info(
                    f"Joining source metrics: {tabella_sorgente} with CUSTOMER BASE"
                )
                self.logger.info(
                    f"Result df after join (no tech columns) - columns length: {len(result_df.columns)}"
                )

                df_filtered = result_df.filter(
                    F.expr(" AND ".join([f"{c} IS NOT NULL" for c in key_columns]))
                )
                self.logger.info(f"Total count not null per key: {df_filtered.count()}")

                df_filtered2 = result_df.join(
                    metrica1_df_renamed, key_columns, how="inner"
                ).select(*key_columns)
                self.logger.info(
                    f"Count not null per inner key between CUSTOMER BASE and {tabella_sorgente}: {df_filtered2.count()}"
                )
            else:
                self.logger.info(f"Source not found: {sorgente}")

        return result_df

    def get_key_columns(self) -> List[str]:
        """
        Method to get the join key columns of the dataset.
        Return:
            A list of strings containing the key columns.
        """
        return ["PK_PROPOSAL_LINE_ITEM"]

    def get_version(self) -> int:
        """
        Method to get the version of this training master dataset.
        If significant changes are made to the dataset, the version must be increased.
        Returns:
            version (integer): the version of this training master dataset.
        """
        return 1
