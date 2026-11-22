from datetime import datetime
from pyspark.sql import SparkSession, DataFrame as pysparkDataFrame
from typing import Optional, List


class AccountToCodFiscalePiva:
    """
    Class to define a table that contains the mapping between the 'ACCOUNT' key and the tax code 'COD_FISCALE_PIVA'.
    This elaboration is like a metric but it is not certified.
    """

    def __init__(self):
        """
        Initialization method to get the table paths.
        Attributes:
            table_siebd_cliente_g_p (string): the path of the table 'siebd_cliente_g_p' in the data platform.
            table_cpqom_account (string): the path of the table 'cpqom_account' in the data platform.
            table_dy365_cliente (string): the path of the table 'dy365_cliente' in the data platform.
        """
        self.table_siebd_cliente_g_p = "p_dpp_core.curateddb.siebd_cliente_g_p"
        self.table_cpqom_account = "p_dpp_core.curateddb.cpqom_account"
        self.table_dy365_cliente = "p_dpp_core.curateddb.dy365_cliente"

    def _get_query_as_of(self, timestamp_at: str) -> str:
        """
        Method to get the SQL query to extract the data from the tables at the snapshot date.
        Args:
            timestamp_at (string): snapshot date in the format 'YYYY-MM-DD'.
        Returns:
            query (string): the SQL query to extract the data from the tables at the snapshot date.
        """
        query = """
            WITH tabella AS (
                SELECT
                a.PK_ID_ACCOUNT AS PK_ACCOUNT
                --, a.NR_ACCOUNT AS COD_CLI_NUM
                , IFNULL(c.COD_FISCALE, a.COD_FISCALE) AS COD_FISCALE
                , IFNULL(c.PIVA, a.PIVA) AS PIVA
                , c.MACROSEGMENTO
                , c.NOME, c.COGNOME, c.NOME_COMPLETO_O_RAG_SOCIALE AS RAG_SOCIALE
                , c.EMAIL, c.PEC, c.TELEFONO
                , a.FORMA_LEGALE AS FORMA_GIURIDICA
                -- , a.COD_FISCALE, a.PIVA, a.TIP_CLI AS MACROSEGMENTO, a.NOME_ACCOUNT, a.RAGIONE_SOCIALE_NOME_E_COGNOME
                -- , a.EMAIL_CLI, a.VECCHIA_EMAIL, a.TEL_CLI, a.TEL_ACCOUNT, a.TEL_SECONDARIO, a.VECCHIO_TEL_CELLULARE
                FROM {table_cpqom_account} AS a
                INNER JOIN (
                            SELECT
                            COD_CLI AS COD_CLI_NUM
                            , COD_FISCALE, PIVA
                            , ID_SEGMENTO_CLI_DES AS MACROSEGMENTO
                            , NOME, COGNOME, NOME_COMPLETO_O_RAG_SOCIALE
                            , IFNULL(IND_EMAIL, EMAIL_REGISTRAZIONE_PORTALE) AS EMAIL
                            , PEC
                            , CASE
                                WHEN TEL_PRINCIPALE IS NOT NULL THEN TEL_PRINCIPALE
                                WHEN TEL_SECONDARIO IS NOT NULL THEN TEL_SECONDARIO
                                ELSE NR_CHIAMANTE
                            END AS TELEFONO
                            --, PK_ACCOUNT as ID_ACCOUNT
                            -- FLG_VULN_AUCOMM, FLG_VULN_TOT, FLG_VULN_TOT, FLG_VULN_TOT, FLG_VULN_TOT, FLG_VULN_BONUS_SOCIALE -- Campi utili per FlagVulnerabilityVolturante?
                            FROM {table_dy365_cliente}
                            WHERE dat_startDate <  CAST('{snapshot_date}' AS TIMESTAMP)
                            AND   dat_endDate   >= CAST('{snapshot_date}' AS TIMESTAMP)
                            AND  COD_MIGRAZIONE = 100000001
                ) AS c ON c.COD_CLI_NUM = a.NR_ACCOUNT
                -- ) AS c ON c.ID_ACCOUNT = a.ID_ACCOUNT -- chiave alternativa (peggiore?)
                WHERE
                dat_startDate   <  CAST('{snapshot_date}' AS TIMESTAMP)
                AND dat_endDate >= CAST('{snapshot_date}' AS TIMESTAMP)
            UNION
                SELECT
                PK_COD_CLIENTE AS COD_CLIENTE_ACCOUNT
                --, COD_CLIENTE_NUMERICO
                , COD_FISCALE, PARTITA_IVA, MACROSEGMENTO
                , NOME, COGNOME, DENOMINAZIONE AS RAG_SOCIALE
                , INDIRIZZO_EMAIL AS EMAIL
                , NULL AS PEC
                , CASE
                    WHEN RECAPITO_TELEFONICO_PRIMARIO IS NOT NULL THEN RECAPITO_TELEFONICO_PRIMARIO
                    WHEN RECAPITO_TELEF_SECONDARIO IS NOT NULL THEN RECAPITO_TELEF_SECONDARIO
                    ELSE ALTRO_TELEFONO
                END AS TELEFONO
                , NULL AS FORMA_LEGALE
                -- flg_fornitura_gas, flg_fatt_elettronica, flg_pa, flg_persona_giuridica, natura_soggetto
                FROM {table_siebd_cliente_g_p}
                WHERE dat_startDate < CAST('{snapshot_date}' AS TIMESTAMP)
                AND dat_endDate    >= CAST('{snapshot_date}' AS TIMESTAMP)
                AND STATO_MIGRAZIONE IS NULL
            )
            SELECT  *
            , IFNULL(COD_FISCALE, PIVA) AS COD_FISCALE_PIVA
            , CAST('{snapshot_date}' AS TIMESTAMP) AS SNAPSHOT_DATE
            FROM tabella
            """.format(
            snapshot_date=timestamp_at,
            table_siebd_cliente_g_p=self.table_siebd_cliente_g_p,
            table_cpqom_account=self.table_cpqom_account,
            table_dy365_cliente=self.table_dy365_cliente,
        )
        return query

    def get_dataframe(self, snapshot_date: Optional[str] = None) -> pysparkDataFrame:
        """
        Method to get the PySpark dataframe of this feature's elaboration.
        Args:
            snapshot_date (string): snapshot date in the format 'YYYY-MM-DD'.
                          If None, it is used self.snapshot_date o the current date
        Returns:
            df_spark (PySpark DataFrame): a dataframe conteining the result of the query at the given snapshot date.
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
        return ["PK_ACCOUNT"]

    def get_version(self) -> int:
        """
        Method to get the version of this feature.
        If significant changes are made to the feature's elaboration, the version must be increased.
        Returns:
            version (integer): the version of this feature.
        """
        return 4
