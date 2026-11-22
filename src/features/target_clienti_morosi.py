from datetime import datetime
from pyspark.sql import SparkSession, DataFrame as pysparkDataFrame
from typing import Optional, List


class TargetContoFattClientiMorosi:
    """
    Class to define the target unpaing customers dataset.
    It is important for the subsequent creation of the training dataset, but care must be taken not to include it in the inference datasets.
    A non-paying customer is defined as a customer who has not paid 80% of invoices issued 15 months ago and overdue for at least 3 months.
    The 'FLG_CLI_MOROSO' column contains the flag to indicate whether the customer has not paid (in the future) or not.
    """

    def __init__(self, snapshot_date: Optional[str] = None):
        """
        Initialization method to get the table paths, the snapshot date, the range of months to skip before the snapshot date, and months pending for invoice due dates.

        Arguments:
            snapshot_date (string): snapshot date in the format 'YYYY-MM-DD'.

        Attributes:
            pending_months (integer): number of months to wait for invoices to expire.
            months_interval (integer): number of months to skip before the snapshot date.
            table_netdo_fattura (string): the path of the table 'netdo_fattura' in the data platform.
            table_netdo_rata (string): the path of the table 'netdo_rata' in the data platform.
            table_netdo_incasso (string): the path of the table 'netdo_incasso' in the data platform.
            table_netdo_conti_cliente_bi (string): the path of the table 'netdo_conti_cliente_bi' in the data platform.
            table_cpqom_account_location (string): the path of the table 'cpqom_account_location' in the data platform.
            table_cpqom_account (string): the path of the table 'cpqom_account' in the data platform.
            table_dy365_cliente (string): the path of the table 'dy365_cliente' in the data platform.
            table_siebd_conto_cliente (string): the path of the table 'siebd_conto_cliente' in the data platform.
            table_siebd_cliente_g_p (string): the path of the table 'siebd_cliente_g_p' in the data platform.
        """
        self.snapshot_date = snapshot_date  # Ora può essere None o una data specifica
        self.pending_months = 3
        self.months_interval = 15

        # Tabelle di riferimento
        self.table_netdo_fattura = "p_dpp_core.curateddb.netdo_fattura"
        self.table_netdo_rata = "p_dpp_core.curateddb.netdo_rata"
        self.table_netdo_incasso = "p_dpp_core.curateddb.netdo_incasso"
        self.table_netdo_conti_cliente_bi = (
            "p_dpp_core.curateddb.netdo_conti_cliente_bi"
        )
        self.table_cpqom_account_location = (
            "p_dpp_core.curateddb.cpqom_account_location"
        )
        self.table_cpqom_account = "p_dpp_core.curateddb.cpqom_account"
        self.table_dy365_cliente = "p_dpp_core.curateddb.dy365_cliente"
        self.table_siebd_conto_cliente = "p_dpp_core.curateddb.siebd_conto_cliente"
        self.table_siebd_cliente_g_p = "p_dpp_core.curateddb.siebd_cliente_g_p"

    def _get_query_as_of(self, snapshot_date: str) -> str:
        """
        Method to get the SQL query to extract the data from the tables at the snapshot date.
        Args:
            timestamp_at (string): snapshot date in the format 'YYYY-MM-DD'.
        Returns:
            query (string): the SQL query to extract the data from the tables at the snapshot date.
        """
        query = """
                ------------------------ ESTRAZIONI DA NETA AVANTI NEL TEMPO RISPETTO ALLA DATA SNAPSHOT ------------------------
                WITH fatture AS (
                                SELECT distinct ---!!!!
                                NUMERO_FATTURA
                                , IMPORTO as IMPORTO_fattura
                                , IMPORTO_PAGATO as IMPORTO_PAGATO_fattura
                                , DATA_EMISSIONE as DATA_EMISSIONE_fatt
                                , DATA_SCADENZA as DATA_SCADENZA_fatt
                                , ID_CONTO_CLIENTE
                                , PK_TESTATA_FATTURA
                                FROM {table_netdo_fattura}
                                WHERE
                                dat_startDate      <  DATEADD(month, {months_interval}  , CAST('{timestamp_at}' AS TIMESTAMP))
                                AND dat_endDate    >= DATEADD(month, {months_interval}  , CAST('{timestamp_at}' AS TIMESTAMP))
                                AND DATA_EMISSIONE >= DATEADD(month, 0                  , CAST('{timestamp_at}' AS TIMESTAMP))
                                AND DATA_EMISSIONE <  DATEADD(month, {months_interval}-{pending_months}, CAST('{timestamp_at}' AS TIMESTAMP))
                                AND DATA_SCADENZA  <  DATEADD(month, {months_interval}  , CAST('{timestamp_at}' AS TIMESTAMP))
                )
                , rate_fatture AS (
                SELECT *
                FROM fatture
                INNER JOIN(
                            SELECT
                            ID_RATA
                            , PK_CODICE_FATTURA
                            , IMPORTO_RATA
                            , IMPORTO_PAGATO as IMPORTO_PAGATO_rata
                            , DATA_CREAZIONE
                            , DATA_SCADENZA as DATA_SCADENZA_rata
                            , DATA_PAGAMENTO as DATA_PAGAMENTO_rata
                            , CODICE_INCASSO
                            , DATEDIFF(day, DATA_SCADENZA, DATEADD(month, {months_interval},   CAST('{timestamp_at}' AS TIMESTAMP))) AS aging_days_scadenza_rata
                            , RATA_ATTIVA
                            FROM {table_netdo_rata}
                            WHERE
                            dat_startDate      <  DATEADD(month, {months_interval},   CAST('{timestamp_at}' AS TIMESTAMP))
                            AND dat_endDate    >= DATEADD(month, {months_interval},   CAST('{timestamp_at}' AS TIMESTAMP))
                            AND DATA_CREAZIONE >= DATEADD(month, 0                ,   CAST('{timestamp_at}' AS TIMESTAMP))
                            AND DATA_CREAZIONE <  DATEADD(month, {months_interval}-{pending_months}, CAST('{timestamp_at}' AS TIMESTAMP))
                            AND DATA_SCADENZA  <  DATEADD(month, {months_interval},   CAST('{timestamp_at}' AS TIMESTAMP))
                            AND RATA_ATTIVA = 'S'
                            ) AS rata ON fatture.PK_TESTATA_FATTURA = rata.PK_CODICE_FATTURA
                )
                , incassi_fatture AS (
                        SELECT *
                        , CASE
                            WHEN CODICE_CANALE IN ('-1360','-1361','-1363') THEN 0
                            WHEN aging_days_scadenza_rata >  3*365/12       THEN 1
                            WHEN aging_days_scadenza_rata <= 3*365/12 AND DATA_pagamento_incasso IS NOT NULL THEN 1
                            WHEN aging_days_scadenza_rata <= 3*365/12 AND DATA_pagamento_incasso IS NULL THEN 0
                            ELSE 0
                        END AS rate_considerabili
                        FROM rate_fatture
                        LEFT JOIN(
                                SELECT
                                PK_CODICE_INCASSO
                                -- , COD_OPERAZIONE
                                , IMPORTO as IMPORTO_incasso
                                , DATA_DI_PAGNTO as DATA_pagamento_incasso
                                , COD_CONTO_CLIENTE as COD_CONTO_CLIENTE_incassi
                                , CODICE_CANALE
                                FROM {table_netdo_incasso}
                                WHERE
                                dat_startDate      <  DATEADD(month, {months_interval}, CAST('{timestamp_at}' AS TIMESTAMP))
                                AND dat_endDate    >= DATEADD(month, {months_interval}, CAST('{timestamp_at}' AS TIMESTAMP))
                                ) AS incassi ON incassi.PK_CODICE_INCASSO = rate_fatture.CODICE_INCASSO
                        -- Si devono togliere le rate-fatture su cui si riscontrano gli incassi provenienti sdai seguenti tre canali:
                        --      -1360	riscontro fatture a zero (logica NETA)
                        --      -1361	compensazioni del monte crediti da considerare per ridurre gli importi fatturatio e non per incrementare l'incassato
                        --      -1363	clienti retail per note credito storno IVA
                        WHERE CODICE_CANALE NOT IN ('-1360','-1361','-1363')
                        OR CODICE_CANALE IS NULL
                )
                , group_fatture AS (
                                    SELECT ID_CONTO_CLIENTE, NUMERO_FATTURA
                                    , sum(IMPORTO_RATA)    AS IMPORTO_RATA
                                    , sum(IMPORTO_incasso) AS IMPORTO_incasso
                                    FROM incassi_fatture
                                    WHERE rate_considerabili = 1
                                    GROUP BY ID_CONTO_CLIENTE, NUMERO_FATTURA
                )
                , fatture_in_conto_neta AS (
                                            SELECT * FROM group_fatture AS fatt
                                            INNER JOIN (
                                                        SELECT
                                                            map.CODICE_CONTO AS COD_CONTO_NETA,
                                                            map.PK_ID_CONTO_CLIENTE AS ID_CONTO_CLIENTE_NETA
                                                        FROM {table_netdo_conti_cliente_bi} AS map
                                                        WHERE dat_startDate < DATEADD(month, {months_interval}, CAST('{timestamp_at}' AS TIMESTAMP))
                                                        AND   dat_endDate  >= DATEADD(month, {months_interval}, CAST('{timestamp_at}' AS TIMESTAMP))
                                            ) AS cont ON fatt.ID_CONTO_CLIENTE = cont.ID_CONTO_CLIENTE_NETA
                )
                , group_conto AS (
                                SELECT COD_CONTO_NETA, ID_CONTO_CLIENTE_NETA
                                , sum(IFNULL(IMPORTO_RATA   ,0)) as importo_tot_rate
                                , sum(IFNULL(IMPORTO_incasso,0)) as importo_tot_incassi
                                FROM fatture_in_conto_neta
                                GROUP BY COD_CONTO_NETA, ID_CONTO_CLIENTE_NETA
                )
                --------------------------------------------------------------- JOIN CON LE TABELLE DEI CRM ALLA DATA SNAPSHOT ---------------------------------------------------------------
                , cpqm_dyn AS (
                                SELECT
                                COD_CONTO_FATTURAZIONE,
                                PK_RECORD_ID AS PK_CONTO_FATT -- chiave nella master
                                FROM {table_cpqom_account_location} AS cc
                                INNER JOIN (
                                            SELECT PK_ID_ACCOUNT, NR_ACCOUNT
                                            FROM {table_cpqom_account} AS acc
                                            INNER JOIN (
                                                        SELECT  COD_CLI
                                                        FROM {table_dy365_cliente}
                                                        WHERE
                                                        dat_startDate      <  CAST('{timestamp_at}' AS TIMESTAMP)
                                                        AND   dat_endDate  >= CAST('{timestamp_at}' AS TIMESTAMP)
                                                        AND  COD_MIGRAZIONE = 100000001
                                                        ) AS dyn365 ON dyn365.COD_CLI = acc.NR_ACCOUNT
                                            WHERE
                                            dat_startDate   <  CAST('{timestamp_at}' AS TIMESTAMP)
                                            AND dat_endDate >= CAST('{timestamp_at}' AS TIMESTAMP)
                                ) AS ac ON ac.PK_ID_ACCOUNT = cc.FK_ACCOUNT

                                WHERE dat_startDate < CAST('{timestamp_at}' AS TIMESTAMP)
                                AND dat_endDate    >= CAST('{timestamp_at}' AS TIMESTAMP)
                )
                , siebel AS (
                            SELECT
                            COD_CONTO_CLIENTE_NUMERICO AS COD_CONTO_FATTURAZIONE,
                            PK_COD_CONTO_CLIENTE AS PK_CONTO_FATT -- chiave nella master
                            FROM {table_siebd_conto_cliente} AS cc
                            INNER JOIN (SELECT PK_COD_CLIENTE
                                        FROM {table_siebd_cliente_g_p}
                                        WHERE
                                        dat_startDate   <  CAST('{timestamp_at}' AS TIMESTAMP)
                                        AND dat_endDate >= CAST('{timestamp_at}' AS TIMESTAMP)
                                        AND STATO_MIGRAZIONE IS NULL
                            ) AS migr ON migr.PK_COD_CLIENTE = cc.COD_CLIENTE

                            WHERE dat_startDate < CAST('{timestamp_at}' AS TIMESTAMP)
                            AND dat_endDate    >= CAST('{timestamp_at}' AS TIMESTAMP)

                )
                , join_conto_crm AS (
                                        SELECT *
                                        , importo_tot_rate-importo_tot_incassi as mancati_incassi_rate
                                        , 1 - importo_tot_incassi/importo_tot_rate AS perc_mancati_incassi_rate
                                        FROM group_conto as cn
                                        LEFT JOIN (
                                                    SELECT *
                                                    FROM cpqm_dyn
                                                UNION
                                                    SELECT *
                                                    FROM siebel
                                        ) AS crm ON crm.COD_CONTO_FATTURAZIONE = cn.COD_CONTO_NETA
                )
                , finale AS (
                            SELECT
                            PK_CONTO_FATT, COD_CONTO_NETA, ID_CONTO_CLIENTE_NETA
                            , CASE
                            WHEN DATEDIFF(MONTH, CAST('{timestamp_at}' AS DATE), CURRENT_DATE()) < {months_interval} THEN NULL
                            WHEN perc_mancati_incassi_rate >= 0.8 THEN 1
                            ELSE 0
                            END AS FLG_CLI_MOROSO
                            , ROUND(perc_mancati_incassi_rate, 3) AS perc_mancati_incassi_rate
                            , importo_tot_rate AS fatturato_rate
                            , importo_tot_incassi AS importi_incassati
                            , mancati_incassi_rate
                            FROM join_conto_crm
                )

                SELECT *
                FROM finale
        """.format(
            timestamp_at=snapshot_date,  # Usa il parametro passato
            months_interval=self.months_interval,
            pending_months=self.pending_months,
            table_netdo_fattura=self.table_netdo_fattura,
            table_netdo_rata=self.table_netdo_rata,
            table_netdo_incasso=self.table_netdo_incasso,
            table_netdo_conti_cliente_bi=self.table_netdo_conti_cliente_bi,
            table_cpqom_account_location=self.table_cpqom_account_location,
            table_cpqom_account=self.table_cpqom_account,
            table_dy365_cliente=self.table_dy365_cliente,
            table_siebd_conto_cliente=self.table_siebd_conto_cliente,
            table_siebd_cliente_g_p=self.table_siebd_cliente_g_p,
        )
        return query

    # def _get_dataframe_as_of(self, snapshot_date: str) -> pysparkDataFrame:
    # spark: SparkSession = SparkSession.builder.getOrCreate()

    # # Genera e esegue la query
    # query = self._get_query_as_of(snapshot_date)
    # df_ContoFattClientiMorosi = spark.sql(query)

    # col_output = [
    #     "PK_CONTO_FATT",
    #     "COD_CONTO_NETA",
    #     "ID_CONTO_CLIENTE_NETA",
    #     "FLG_CLI_MOROSO",
    #     "PERC_MANCATI_INCASSI_RATE",
    #     "FATTURATO_RATE",
    #     "MANCATI_INCASSI_RATE",
    #     "IMPORTI_INCASSATI",
    # ]
    # return df_ContoFattClientiMorosi.select(col_output)

    def get_dataframe(self, snapshot_date: Optional[str] = None) -> pysparkDataFrame:
        """
        Method to get the PySpark dataframe of the unpaing customers target dataset.
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
        df_ContoFattClientiMorosi = spark.sql(query)

        col_output = [
            "PK_CONTO_FATT",
            "COD_CONTO_NETA",
            "ID_CONTO_CLIENTE_NETA",
            "FLG_CLI_MOROSO",
            "PERC_MANCATI_INCASSI_RATE",
            "FATTURATO_RATE",
            "MANCATI_INCASSI_RATE",
            "IMPORTI_INCASSATI",
        ]
        return df_ContoFattClientiMorosi.select(col_output)

    def get_key_columns(self) -> List[str]:
        """
        Method to get the join key columns of the dataset.
        Returns:
            key_columns (list[strings]): a list of the name of the key columns.
        """
        return ["PK_CONTO_FATT"]

    def get_version(self) -> int:
        """
        Method to get the version of this feature.
        If significant changes are made to the feature's elaboration, the version must be increased.
        Returns:
            version (integer): the version of this feature.
        """
        return 7
