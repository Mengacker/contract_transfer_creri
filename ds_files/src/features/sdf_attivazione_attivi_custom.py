# -------------------------------------------------------------------------------------------- #
#             NUOVA classe di metriche NON CERTIFICATA in Metrics                    #
#             con codice semplificato per usarlo direttamente nel prodotto ccvol.              #
#                                                                                              #
#             La versione attuale di questa metrica è obsoleta e                               #
#             la nuova versione qui presente non verrà ricertificata dal business owner.       #
# -------------------------------------------------------------------------------------------- #

from datetime import datetime
from pyspark.sql import SparkSession, DataFrame as pysparkDataFrame
from typing import Optional, List


class MetricaSdfAttivazioneAttiviCustom:
    """
    Class to define the new (custom) version of the metric 'MetricaSdfActivazioneAttivi'.
    The old version of this metric is being deprecated, and the new version present here will not be recertified by the business owner.
    """

    def __init__(self):
        """
        Initialization method to get the table paths.

        Attributes:
            table_metricaSdfAttivazione (string): the name of the metric table's view.
        """
        self.table_metricaSdfAttivazione = "vw_MetricaSdfAttivazione"

    def _get_query_as_of(self, timestamp_at: str) -> str:
        """
        Method to get the SQL query to extract the data from the tables at the snapshot date.
        Args:
            timestamp_at (string): snapshot date in the format 'YYYY-MM-DD'.
        Returns:
            query (string): the SQL query to extract the data from the tables at the snapshot date.
        """
        if self.table_metricaSdfAttivazione == "vw_MetricaSdfAttivazione":
            from metrics import MetricaSdfAttivazione

            self.query_metricaSdfAttivazione = MetricaSdfAttivazione().get_query()

        query = """
        SELECT *
        FROM {metricaSdfAttivazione}
        WHERE flg_lastSnap = 1 -- logic to filter the metric only in the cases where sdf are active in the snapshot date
        AND DATA_ATT_SITO <= CAST('{snapshot_date}' AS TIMESTAMP)
        AND DATA_CESS_SITO > CAST('{snapshot_date}' AS TIMESTAMP)
        """.format(
            snapshot_date=timestamp_at,
            metricaSdfAttivazione=self.query_metricaSdfAttivazione,
        )
        return query

    def get_dataframe(self, snapshot_date: Optional[str] = None) -> pysparkDataFrame:
        """
        Method to get the PySpark dataframe of the dataset.
        Args:
            snapshot_date: snapshot date in the format 'YYYY-MM-DD'.
                          If None, it is used self.snapshot_date o the current date
        Returns:
            pysparkDataFrame: PySpark dataframe conteining the results
        """

        if snapshot_date is None:
            if self.snapshot_date is None:
                # Usa la data odierna come fallback
                snapshot_date = datetime.now().strftime("%Y-%m-%d")
            else:
                snapshot_date = self.snapshot_date

        # Generates and executes the query
        query = self._get_query_as_of(snapshot_date)
        spark: SparkSession = SparkSession.builder.getOrCreate()
        df_spark = spark.sql(query)

        return df_spark

    def get_key_columns(self) -> List[str]:
        """
        Method to get the join key columns of the dataset.
        Returns:
            key_columns (list[strings]): a list of the name of the key columns.
        """
        return ["PK_PDF", "PK_ACCOUNT", "PK_CONTO_FATT"]

    def get_version(self) -> int:
        """
        Method to get the version of this feature.
        If significant changes are made to the feature's elaboration, the version must be increased.
        Returns:
            version (integer): the version of this feature.
        """
        return 2
