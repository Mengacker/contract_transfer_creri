from datetime import datetime, timedelta
from pyspark.sql import functions as F, SparkSession, DataFrame as pysparkDataFrame
from typing import List

# --------------------------------------------- CustomBaseVolturante ---------------------------------------------


class cbVolturante:
    """
    Class to define the base dataset of incoming customers.
    It is important for the subsequent creation of the training dataset.
    The dataset is primarily composed of contract transfers to which the "bridge" metric is attached.
    Then, information on the location of gas/electricity supplies and some personal information of the incoming customer, such as tax code and surname, is added.
    """

    def get_dataframe_as_of(
        self,
        spark: SparkSession,
        snapshot_date: str,
        MetricaFlgVoltureLordeNette: str,
        MetricaPonteSdfProposal: str,
        MetricaPdfPodPdrGtwV: str,
    ) -> pysparkDataFrame:
        """
        Method to get the PySpark dataframe of the dataset as of a given date.
        Args:
            spark (SparkSession): a SparkSession object.
            snapshot_date (string): a string representing the snapshot date in the format "YYYY-MM-DD".
            MetricaFlgVoltureLordeNette (string): the name of the table containing the metric FlgVoltureLordeNette.
            MetricaPonteSdfProposal (string): the name of the table containing the metric PonteSdfProposal.
            MetricaPdfPodPdrGtwV (string): the name of the table containing the metric PdfPodPdrGtwV.
        Returns:
            a PySpark DataFrame containing the base dataset of incoming customers.
        """
        self.spark = SparkSession.builder.getOrCreate()

        snapdate = datetime.strptime(snapshot_date, "%Y-%m-%d").date()
        end_date = snapdate - timedelta(days=1)
        start_date = end_date - timedelta(days=end_date.day - 1)

        # Load MetricaFlgVoltureLordeNette
        # volt = MetricaFlgVoltureLordeNette().get_dataframe_as_of(snapshot_date)
        volt = self.spark.read.table(MetricaFlgVoltureLordeNette).where(
            f"snapshot_date = '{snapshot_date}' and flg_lastSnap = 1"
        )  # TODO: check nuovo metodo
        volt = volt.withColumn(
            "DATA_VOLTURA", F.to_date(F.col("DATA_VOLTURA"), "yyyy-MM-dd")
        )
        volt = volt.withColumn(
            "DATA_FIRMA_VOLTURA", F.to_date(F.col("DATA_FIRMA_VOLTURA"), "yyyy-MM-dd")
        )

        filtered_volt = volt.filter(
            (F.col("DATA_VOLTURA") >= start_date)
            & (F.col("DATA_VOLTURA") < end_date)
            & F.col("FLG_VOLTURE_NETTE")
        )

        volture = filtered_volt.select(
            "PK_PROPOSAL_LINE_ITEM",
            F.col("CLIENTE_VOLTURANTE").alias("PK_ACCOUNT"),
            "CLIENTE_VOLTURATO",
            "DATA_FIRMA_VOLTURA",
            "DATA_VOLTURA",
            "FLG_VOLTURE_NETTE",
            "FLG_VOLTURE_LORDE",
            "FLG_VOLTURA_MORTIS_CAUSA",
        )

        # Load MetricaPonteSdfProposal
        # ponte_df = MetricaPonteSdfProposal().get_dataframe_as_of(snapshot_date)
        ponte_df = self.spark.read.table(MetricaPonteSdfProposal).where(
            f"snapshot_date = '{snapshot_date}' and flg_lastSnap = 1"
        )  # TODO: check nuovo metodo
        for c in ponte_df.columns:
            if c.startswith("FK_"):
                ponte_df = ponte_df.withColumnRenamed(c, "PK_" + c[3:])
        ponte_df = ponte_df.drop("PK_ACCOUNT", "PK_COD_FORNITURA")
        base_volture = volture.join(ponte_df, on="PK_PROPOSAL_LINE_ITEM", how="left")

        # Load MetricaPdfPodPdrGtwV
        # pod_pdr_df = MetricaPdfPodPdrGtwV().get_dataframe_as_of(snapshot_date)
        pod_pdr_df = self.spark.read.table(
            MetricaPdfPodPdrGtwV
        )  # .where(f"snapshot_date = '{snapshot_date}' and flg_lastSnap = 1") # TODO: check nuovo metodo
        pod_pdr_df = pod_pdr_df.withColumnRenamed(
            "CODICE_POD_PDR", "POD_PDR"
        ).withColumnRenamed("COD_PDF", "PK_PDF")
        ind = F.col("POD_PDR").like("IT%")
        pod_pdr_df = pod_pdr_df.withColumn("COD_POD", F.when(ind, F.col("POD_PDR")))
        pod_pdr_df = pod_pdr_df.withColumn("COD_PDR", F.when(~ind, F.col("POD_PDR")))
        pod_pdr_df = pod_pdr_df.select(
            "PK_PDF",
            "POD_PDR",
            "COD_POD",
            "COD_PDR",
            "IND_FORN_IND",
            "IND_FORN_CAP",
            "IND_FORN_COMUNE",
            "IND_FORN_COD_PROV",
            "IND_FORN_NOME_PROV",
            "IND_FORN_REGIONE",
        )

        # left join MetricaFlgVoltureLordeNette with MetricaPdfPodPdrGtwV
        base_volture = base_volture.join(pod_pdr_df, on="PK_PDF", how="left")

        # Codice fiscale / PIVA query dinamica
        table_siebd_cliente_g_p = "p_dpp_core.curateddb.siebd_cliente_g_p"
        table_cpqom_account = "p_dpp_core.curateddb.cpqom_account"
        table_dy365_cliente = "p_dpp_core.curateddb.dy365_cliente"

        query_cod_fiscale = f"""
        WITH tabella AS (
            SELECT a.PK_ID_ACCOUNT AS PK_ACCOUNT,
                   IFNULL(c.COD_FISCALE, a.COD_FISCALE) AS COD_FISCALE,
                   IFNULL(c.PIVA, a.PIVA) AS PIVA,
                   c.MACROSEGMENTO, c.NOME, c.COGNOME, c.NOME_COMPLETO_O_RAG_SOCIALE AS RAG_SOCIALE,
                   c.EMAIL, c.PEC, c.TELEFONO, a.FORMA_LEGALE AS FORMA_GIURIDICA
            FROM {table_cpqom_account} AS a
            INNER JOIN (
                SELECT COD_CLI AS COD_CLI_NUM, COD_FISCALE, PIVA,
                       ID_SEGMENTO_CLI_DES AS MACROSEGMENTO,
                       NOME, COGNOME, NOME_COMPLETO_O_RAG_SOCIALE,
                       IFNULL(IND_EMAIL, EMAIL_REGISTRAZIONE_PORTALE) AS EMAIL,
                       PEC,
                       COALESCE(TEL_PRINCIPALE, TEL_SECONDARIO, NR_CHIAMANTE) AS TELEFONO
                FROM {table_dy365_cliente}
                WHERE dat_startDate < CAST('{snapshot_date}' AS TIMESTAMP)
                  AND dat_endDate >= CAST('{snapshot_date}' AS TIMESTAMP)
                  AND COD_MIGRAZIONE = 100000001
            ) AS c ON c.COD_CLI_NUM = a.NR_ACCOUNT
            WHERE dat_startDate < CAST('{snapshot_date}' AS TIMESTAMP)
              AND dat_endDate >= CAST('{snapshot_date}' AS TIMESTAMP)
            UNION
            SELECT PK_COD_CLIENTE AS PK_ACCOUNT, COD_FISCALE, PARTITA_IVA AS PIVA,
                   MACROSEGMENTO, NOME, COGNOME, DENOMINAZIONE AS RAG_SOCIALE,
                   INDIRIZZO_EMAIL AS EMAIL, NULL AS PEC,
                   COALESCE(RECAPITO_TELEFONICO_PRIMARIO, RECAPITO_TELEF_SECONDARIO, ALTRO_TELEFONO) AS TELEFONO,
                   NULL AS FORMA_GIURIDICA
            FROM {table_siebd_cliente_g_p}
            WHERE dat_startDate < CAST('{snapshot_date}' AS TIMESTAMP)
              AND dat_endDate >= CAST('{snapshot_date}' AS TIMESTAMP)
              AND STATO_MIGRAZIONE IS NULL
        )
        SELECT *, IFNULL(COD_FISCALE, PIVA) AS COD_FISCALE_PIVA FROM tabella
        """

        df_cod_fiscale = self.spark.sql(query_cod_fiscale)

        # left join MetricaFlgVoltureLordeNette with cod_fiscale
        base_volture = base_volture.join(df_cod_fiscale, on="PK_ACCOUNT", how="left")
        base_volture = base_volture.dropDuplicates(["PK_PROPOSAL_LINE_ITEM"])

        return base_volture

    def get_key_columns(self) -> List[str]:
        """
        Method to get the join key columns of the dataset.
        Return:
            A list of strings containing the key columns.
        """
        return ["PK_PROPOSAL_LINE_ITEM"]

    def get_version(self) -> int:
        """
        Method to get the version of the dataset.
        If significant changes are made to the customer base dataset, the version must be increased.
        Return:
            An integer representing the version of the dataset.
        """
        return 14


# --------------------------------------------- CustomBaseVolturato ---------------------------------------------
class cbVolturato:
    """
    Class to define the base dataset of outgoing customers.
    It is important for the subsequent creation of the training dataset.
    The dataset is primarily composed of contract transfers to which the "bridge" metric is attached.
    Then, information on the location of gas/electricity supplies and some personal information of the outgoing customer, such as tax code and surname, is added.
    """

    def get_dataframe_as_of(
        self,
        spark: SparkSession,
        snapshot_date: str,
        MetricaFlgVoltureLordeNette: str,
        MetricaPonteSdfProposal: str,
        MetricaPdfPodPdrGtwV: str,
    ) -> pysparkDataFrame:
        """
        Method to get the PySpark dataframe of the dataset as of a given date.
        Args:
            spark (SparkSession): a SparkSession object.
            snapshot_date (string): a string representing the snapshot date in the format "YYYY-MM-DD".
            MetricaFlgVoltureLordeNette (string): the name of the table containing the metric FlgVoltureLordeNette.
            MetricaPonteSdfProposal (string): the name of the table containing the metric PonteSdfProposal.
            MetricaPdfPodPdrGtwV (string): the name of the table containing the metric PdfPodPdrGtwV.
        Returns:
            a PySpark DataFrame containing the base dataset of outgoing customers.
        """
        self.spark = SparkSession.builder.getOrCreate()

        snapdate = datetime.strptime(snapshot_date, "%Y-%m-%d").date()
        end_date = snapdate - timedelta(days=1)
        start_date = end_date.replace(day=1)

        # Catalog references
        table_cpqom_account = "p_dpp_core.curateddb.cpqom_account"
        table_dy365_cliente = "p_dpp_core.curateddb.dy365_cliente"
        table_siebd_cliente_g_p = "p_dpp_core.curateddb.siebd_cliente_g_p"

        # Load MetricaFlgVoltureLordeNette
        # volt = MetricaFlgVoltureLordeNette().get_dataframe_as_of(snapshot_date)
        volt = self.spark.read.table(MetricaFlgVoltureLordeNette).where(
            f"snapshot_date = '{snapshot_date}' and flg_lastSnap = 1"
        )
        volt = volt.withColumn(
            "DATA_VOLTURA", F.to_date("DATA_VOLTURA", "yyyy-MM-dd")
        ).withColumn(
            "DATA_FIRMA_VOLTURA", F.to_date("DATA_FIRMA_VOLTURA", "yyyy-MM-dd")
        )

        filtered_volt = volt.filter(
            (F.col("DATA_VOLTURA") >= F.lit(start_date))
            & (F.col("DATA_VOLTURA") < F.lit(end_date))
            & F.col("FLG_VOLTURE_NETTE")
        )

        volture = filtered_volt.select(
            "PK_PROPOSAL_LINE_ITEM",
            "CLIENTE_VOLTURATO",
            "CLIENTE_VOLTURANTE",
            "DATA_FIRMA_VOLTURA",
            "DATA_VOLTURA",
        ).withColumnRenamed("CLIENTE_VOLTURATO", "PK_ACCOUNT")

        # Load MetricaPonteSdfProposal
        # ponte_df = MetricaPonteSdfProposal().get_dataframe_as_of(snapshot_date)
        ponte_df = self.spark.read.table(MetricaPonteSdfProposal).where(
            f"snapshot_date = '{snapshot_date}' and flg_lastSnap = 1"
        )
        for c in ponte_df.columns:
            if c.startswith("FK_"):
                ponte_df = ponte_df.withColumnRenamed(c, "PK_" + c[3:])
        ponte_df = ponte_df.drop("PK_ACCOUNT", "PK_COD_FORNITURA")

        base_volture = volture.join(ponte_df, on="PK_PROPOSAL_LINE_ITEM", how="left")

        # Load MetricaPdfPodPdrGtwV
        # pod_pdr_df = MetricaPdfPodPdrGtwV().get_dataframe_as_of(snapshot_date)
        pod_pdr_df = self.spark.read.table(
            MetricaPdfPodPdrGtwV
        )  # .where(f"snapshot_date = '{snapshot_date}' and flg_lastSnap = 1") # TODO: check nuovo metodo
        pod_pdr_df = pod_pdr_df.withColumnRenamed(
            "CODICE_POD_PDR", "POD_PDR"
        ).withColumnRenamed("COD_PDF", "PK_PDF")
        is_pod = F.col("POD_PDR").like("IT%")
        pod_pdr_df = pod_pdr_df.withColumn(
            "COD_POD", F.when(is_pod, F.col("POD_PDR"))
        ).withColumn("COD_PDR", F.when(~is_pod, F.col("POD_PDR")))
        pod_pdr_df = pod_pdr_df.select(
            "PK_PDF",
            "POD_PDR",
            "COD_POD",
            "COD_PDR",
            "IND_FORN_IND",
            "IND_FORN_CAP",
            "IND_FORN_COMUNE",
            "IND_FORN_COD_PROV",
            "IND_FORN_NOME_PROV",
            "IND_FORN_REGIONE",
        )

        # left join MetricaFlgVoltureLordeNette with MetricaPdfPodPdrGtwV
        base_volture = base_volture.join(pod_pdr_df, on="PK_PDF", how="left")

        # Codice fiscale / PIVA query dinamica
        query_cod_fiscale = f"""
        WITH tabella AS (
            SELECT
            a.PK_ID_ACCOUNT AS PK_ACCOUNT,
            IFNULL(c.COD_FISCALE, a.COD_FISCALE) AS COD_FISCALE,
            IFNULL(c.PIVA, a.PIVA) AS PIVA,
            c.MACROSEGMENTO, c.NOME, c.COGNOME, c.NOME_COMPLETO_O_RAG_SOCIALE AS RAG_SOCIALE,
            c.EMAIL, c.PEC, c.TELEFONO,
            a.FORMA_LEGALE AS FORMA_GIURIDICA
            FROM {table_cpqom_account} AS a
            INNER JOIN (
                SELECT
                COD_CLI AS COD_CLI_NUM, COD_FISCALE, PIVA,
                ID_SEGMENTO_CLI_DES AS MACROSEGMENTO,
                NOME, COGNOME, NOME_COMPLETO_O_RAG_SOCIALE,
                IFNULL(IND_EMAIL, EMAIL_REGISTRAZIONE_PORTALE) AS EMAIL,
                PEC,
                CASE
                    WHEN TEL_PRINCIPALE IS NOT NULL THEN TEL_PRINCIPALE
                    WHEN TEL_SECONDARIO IS NOT NULL THEN TEL_SECONDARIO
                    ELSE NR_CHIAMANTE
                END AS TELEFONO
                FROM {table_dy365_cliente}
                WHERE dat_startDate < TIMESTAMP('{snapshot_date}')
                AND   dat_endDate   >= TIMESTAMP('{snapshot_date}')
                AND COD_MIGRAZIONE = 100000001
            ) AS c ON c.COD_CLI_NUM = a.NR_ACCOUNT
            WHERE dat_startDate < TIMESTAMP('{snapshot_date}')
            AND   dat_endDate   >= TIMESTAMP('{snapshot_date}')
            UNION
            SELECT
            PK_COD_CLIENTE AS PK_ACCOUNT, COD_FISCALE, PARTITA_IVA AS PIVA, MACROSEGMENTO,
            NOME, COGNOME, DENOMINAZIONE AS RAG_SOCIALE,
            INDIRIZZO_EMAIL AS EMAIL, NULL AS PEC,
            CASE
                WHEN RECAPITO_TELEFONICO_PRIMARIO IS NOT NULL THEN RECAPITO_TELEFONICO_PRIMARIO
                WHEN RECAPITO_TELEF_SECONDARIO IS NOT NULL THEN RECAPITO_TELEF_SECONDARIO
                ELSE ALTRO_TELEFONO
            END AS TELEFONO,
            NULL AS FORMA_GIURIDICA
            FROM {table_siebd_cliente_g_p}
            WHERE dat_startDate < TIMESTAMP('{snapshot_date}')
            AND   dat_endDate   >= TIMESTAMP('{snapshot_date}')
            AND STATO_MIGRAZIONE IS NULL
        )
        SELECT *, IFNULL(COD_FISCALE, PIVA) AS COD_FISCALE_PIVA FROM tabella
        """
        df_cod_fiscale = self.spark.sql(query_cod_fiscale)

        # left join MetricaFlgVoltureLordeNette with cod_fiscale
        base_volture = base_volture.join(df_cod_fiscale, on="PK_ACCOUNT", how="left")

        base_volture = base_volture.dropDuplicates(["PK_PROPOSAL_LINE_ITEM"])

        return base_volture

    def get_key_columns(self) -> List[str]:
        """
        Method to get the join key columns of the dataset.
        Return:
            A list of strings containing the key columns.
        """
        return ["PK_PROPOSAL_LINE_ITEM"]

    def get_version(self) -> int:
        """
        Method to get the version of the dataset.
        If significant changes are made to the customer base dataset, the version must be increased.
        Return:
            An integer representing the version of the dataset.
        """
        return 14
