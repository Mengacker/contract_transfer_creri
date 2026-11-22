from pyspark.sql import SparkSession, DataFrame as PySparkDataFrame, functions as F
from delta.tables import DeltaTable
import pyspark.sql.types as T
from datetime import datetime
from typing import Dict, Tuple


class MetricMaterializer:
    """
    Class to materialize metrics as a table in the Data Platform.
    """

    def __init__(self, spark: SparkSession):
        """
        Initialize the class.
        Attributes:
            spark (SparkSession): SparkSession object.
        """
        self.spark = spark

    def get_max_version(
        self, catalog: str, database: str, table: str, snapshot_date: str
    ) -> int:
        """
        Method to get the max version of a table in the Data Platform.
        Args:
            catalog (string): Catalog name.
            database (string): Database name.
            table (string): Table name.
            snapshot_date (string): Snapshot date.
        Returns:
            max_version (integer): Max version of the table.
        """
        try:
            existing_df = self.spark.read.table(f"{catalog}.{database}.{table}").where(
                f"snapshot_date = '{snapshot_date}'"
            )
            max_version = existing_df.agg({"version": "max"}).collect()[0][0]
            return max_version if max_version is not None else 0
        except Exception as e:
            print(e)
            return 0

    def get_max_iid(self, catalog: str, database: str, table: str) -> int:
        """
        Method to get the max iid of a table in the Data Platform.
        Args:
            catalog (string): Catalog name.
            database (string): Database name.
            table (string): Table name.
        Returns:
            max_iid (integer): Max iid of the table.
        """
        try:
            existing_df = self.spark.read.table(f"{catalog}.{database}.{table}")
            max_iid = existing_df.agg({"iid_processID": "max"}).collect()[0][0]
            return max_iid if max_iid is not None else 0
        except Exception as e:
            print(e)
            return 0

    def is_metrics_updated(
        self,
        catalog: str,
        database: str,
        table: str,
        get_version: int,
        snapshot_date: str,
    ) -> bool:
        """
        Method to check if the metrics are updated.
        Args:
            catalog (string): Catalog name.
            database (string): Database name.
            table (string): Table name.
            get_version (integer): Current version of the table.
            snapshot_date (string): Snapshot date.
        Returns:
            is_updated (boolean): True if the metrics are updated, False otherwise.
        """
        current_version = self.get_max_version(catalog, database, table, snapshot_date)
        if current_version == get_version:
            return True
        else:
            try:
                self.spark.sql(
                    f"DELETE FROM {catalog}.{database}.{table} WHERE snapshot_date = '{snapshot_date}' and flg_lastSnap = 1"
                )
                return False
            except Exception as e:
                print(e)
                return False

    def we_need_a_snapshot_overwrite(
        self,
        catalog: str,
        database: str,
        table: str,
        snapshot_overwrite: bool,
        snapshot_date: str,
    ) -> bool:
        """
        Method to check if we need to overwrite the snapshot.
        Args:
            catalog (string): Catalog name.
            database (string): Database name.
            table (string): Table name.
            snapshot_overwrite (boolean): True if we need to overwrite the snapshot, False otherwise.
            snapshot_date (string): Snapshot date.
        Returns:
            is_snapshot_deleted (boolean): True if the snapshot is deleted, False otherwise.
        """
        if snapshot_overwrite:
            try:
                self.spark.sql(
                    f"DELETE FROM {catalog}.{database}.{table} WHERE snapshot_date = '{snapshot_date}' and flg_lastSnap = 1"
                )
                return True
            except Exception as e:
                print(f"Error deleting snapshot: {e}")
        else:
            return False

    def check_snapshot_overwrite(
        self,
        catalog: str,
        schema: str,
        table: str,
        snapshot_date: str,
        snapshot_overwrite: bool,
    ) -> str:
        """
        Method to check if we need to overwrite the snapshot.
        Args:
            catalog (string): Catalog name.
            database (string): Database name.
            table (string): Table name.
            snapshot_date (string): Snapshot date.
            snapshot_overwrite (boolean): True if we need to overwrite the snapshot, False otherwise.
        Returns:
            snapshot_deletion_assertion (string): Assertion of snapshot deletion status.
        """
        if snapshot_overwrite:
            try:
                self.spark.sql(
                    f"DELETE FROM {catalog}.{schema}.{table} WHERE snapshot_date = '{snapshot_date}' and flg_lastSnap = 1"
                )
                return f"Snapshot {snapshot_date} on table {table} deleted correctly!!"
            except Exception as e:
                print(e)
                return str(e).split(" ")[0]
        else:
            return "No snapshot deletion required!!"

    def add_tech_columns(
        self,
        df: PySparkDataFrame,
        dat_startDate: str,
        runId: str,
        version: int,
        flg_lastSnap: int,
        des_BusinessKey_list: list[str],
        snapshot_date: str,
    ) -> PySparkDataFrame:
        """
        Method to add technical columns to the dataframe.
        Args:
            df (PySpark dataframe): Input dataframe.
            dat_startDate (string): Start date of the snapshot.
            runId (string): Run id of the snapshot.
            version (integer): Version of the snapshot.
            flg_lastSnap (integer): Flag to indicate if the snapshot is the last one.
            des_BusinessKey_list (list[string]): List of business keys.
            snapshot_date (string): Snapshot date.
        Returns:
            df (PySpark dataframe): The input dataframe with the technical columns added.
        """
        from datetime import datetime
        import pyspark.sql.functions as F

        return (
            df.withColumn("snapshot_date", F.lit(snapshot_date))
            .withColumn("dat_startDate", F.lit(dat_startDate))
            .withColumn("dat_endDate", F.lit("null"))
            .withColumn(
                "systemDate", F.lit(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            .withColumn("runId", F.lit(runId))
            .withColumn("version", F.lit(version))
            .withColumn("flg_lastSnap", F.lit(flg_lastSnap))
            .withColumn("des_BusinessKey", F.concat_ws("|", *des_BusinessKey_list))
        )

    def remove_tech_columns(self, df: PySparkDataFrame) -> PySparkDataFrame:
        """
        Method to remove technical columns from the dataframe.
        Args:
            df (PySpark dataframe): Input dataframe.
        Returns:
            df (PySpark dataframe): The input dataframe with the technical columns removed.
        """
        df = df.drop(
            "dat_startDate",
            "dat_endDate",
            "systemDate",
            "runId",
            "version",
            "flg_lastSnap",
            "iid_recordID",
            "iid_processID",
            "iid",
            "des_BusinessKey",
            "snapshot_date",
        )
        ## da rimuovere 'iid' dopo che verranno ripopolate le metriche con il nuovo schema
        return df

    def check_snapshot_data(self, table: str, snapshot_date: str) -> bool:
        """
        Method to check if the snapshot data is present in the table.
        Args:
            table (string): Table name.
            snapshot_date (string): Snapshot date.
        Returns:
            is_snapshot_exist (boolean): True if the snapshot data is present, False otherwise.
        """
        try:
            if (
                self.spark.read.table(table)
                .where(f"snapshot_date = '{snapshot_date}'")
                .limit(1)
                .collect()
            ):
                is_snapshot_exist = True
            else:
                is_snapshot_exist = False
        except Exception as e:
            print(e)
            is_snapshot_exist = False
        return is_snapshot_exist

    def populate_metrics_end_date(self, catalog: str, database: str, table: str):
        """
        Method to populate the metric's end date with the current date.
        Args:
            catalog (string): Catalog name.
            database (string): Database name.
            table (string): Table name.
        """
        import pyspark.sql.functions as F

        delta_table = DeltaTable.forName(self.spark, f"{catalog}.{database}.{table}")

        condition = "dat_endDate == 'null' or dat_endDate IS NULL"

        delta_table.update(
            condition=condition,
            set={"dat_endDate": F.lit(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))},
        )

    def rename_conflicting_columns(
        self, df: PySparkDataFrame, existing_cols: list[str], prefix: str
    ) -> PySparkDataFrame:
        """
        Method to rename conflicting columns in the dataframe.
        Args:
            df (PySpark dataframe): Input dataframe.
            existing_cols (list[string]): List of existing columns.
            prefix (string): Prefix to add to the conflicting columns.
        Returns:
            df (PySpark dataframe): The input dataframe with the conflicting columns renamed.
        """
        existing_cols_set = set(existing_cols)
        new_cols = [
            F.col(c).alias(f"{c}{prefix}") if c in existing_cols_set else F.col(c)
            for c in df.columns
        ]
        return df.select(*new_cols)

    def get_col_list(
        self, cod_productname: str, cod_ObjectName: str, metadata_path: str
    ) -> list[tuple[str, str]]:
        """
        Method to get the list of columns and their data types for a given product and object name.
        Args:
            cod_productname (string): Product name.
            cod_ObjectName (string): Object name.
            metadata_path (string): Path to the metadata table.
        Returns:
            col_list (list[tuple[str, str]]): List of tuples containing the column name and data type.
        """
        return (
            self.spark.read.table(metadata_path)
            .where(
                f"cod_productname = '{cod_productname}' and cod_ObjectName = '{cod_ObjectName}' and (cod_FieldName not in ('iid_RecordID', 'iid_processID', 'iid'))"
            )
            .select("cod_FieldName", "cod_Type")
            .collect()
        )

    def parse_data_type(self, type_str: str) -> str:
        """
        Method to parse the data type string and return the corresponding Spark data type.
        Args:
            type_str (string): Data type string.
        Returns:
            spark_type (Spark data type): Spark data type corresponding to the input data type string.
        """
        if type_str.startswith("DECIMAL"):
            precision, scale = map(
                int, type_str[type_str.find("(") + 1 : type_str.find(")")].split(",")
            )
            spark_type = T.DecimalType(precision, scale)
        elif type_str == "STRING":
            spark_type = "string"
        elif type_str == "INT":
            spark_type == "int"
        elif type_str == "LONG":
            spark_type = "long"
        elif type_str == "BIGINT":
            spark_type = "long"
        elif type_str == "VARCHAR":
            spark_type = "string"
        elif type_str == "FLOAT":
            spark_type = "float"
        elif type_str == "DOUBLE":
            spark_type = "double"
        elif type_str == "BOOLEAN":
            spark_type = "boolean"
        elif type_str == "DATE":
            spark_type = "date"
        else:
            raise ValueError(f"Unsupported type: {type_str}")
        return spark_type

    def get_schema_fields(
        self, schema_map: Dict[str, str], df: PySparkDataFrame
    ) -> Tuple[list[str], list[str]]:
        """
        Method to get the schema fields for the given dataframe.
        Args:
            schema_map (dictionary[string, string]): Dictionary mapping column names to their data types.
            df (PySpark dataframe): Input dataframe.
        Returns:
            common_cols (list[string]): List of common columns between the dataframe and the schema map.
        """
        common_cols = [c for c in df.columns if c in schema_map.keys()]
        df_only_cols = [c for c in df.columns if c not in schema_map.keys()]

        return common_cols, df_only_cols

    def cast_columns(
        self, common_cols: list[str], schema_map: Dict[str, str], df: PySparkDataFrame
    ) -> PySparkDataFrame:
        """
        Method to cast the columns in the dataframe to the corresponding data types in the schema map.
        Args:
            common_cols (list[string]): List of common columns between the dataframe and the schema map.
            schema_map (dictionary[string, string]): Dictionary mapping column names to their data types.
            df (PySpark dataframe): Input dataframe.
        Returns:
            df (PySpark dataframe): The input dataframe with the columns casted to the corresponding data types in the schema map.
        """
        return df.select(*[F.col(c).cast(schema_map[c]).alias(c) for c in common_cols])
