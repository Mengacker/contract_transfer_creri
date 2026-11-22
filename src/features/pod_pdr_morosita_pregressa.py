# -------------------------------------------------------------------------------------------- #
#          NUOVA classe di metriche NON ANCORA certificata in Metrics                #
#          con codice semplificato per usarlo direttamente nel prodotto ccvol.                 #
# -------------------------------------------------------------------------------------------- #
from datetime import datetime
from pyspark.sql import SparkSession, DataFrame as pysparkDataFrame
from typing import Optional, List


class MetricheMorositaPregressaPodPdr:
    """
    Class to define metrics of previous unpaid credit arrears in reference to the gas/electricity supply point POD/PDR.
    There are 26 metrics, some of which express the amount of outstanding debt in various ways, while others express the time of pending payments.
    """

    def __init__(self):
        """
        Initialization method to get the table paths.
        Attributes:
          table_metricaContoClienteCreditoNeta (string): the name of the metric table's view.
          table_metricaPonteSdfProposal (string): the name of the metric table's view.
          table_metricaSdfAttivazione (string): the name of the metric table's view.
          table_punto_di_fornitura (function): the add_catalog's function of Metrics Package with the name of the metric's table and its schema name.
          table_punto_di_fornitura_gas (function): the add_catalog's function of Metrics Package with the name of the metric's table and its schema name.
          table_punto_di_fornitura_pwr (function): the add_catalog's function of Metrics Package with the name of the metric's table and its schema name.
        """
        self.table_metricaContoClienteCreditoNeta = "vw_MetricheContoClienteCreditoNeta"
        self.table_metricaPonteSdfProposal = "vw_MetricaPonteSdfProposal"
        self.table_metricaSdfAttivazione = "vw_MetricaSdfAttivazione"
        self.table_punto_di_fornitura = "p_dpp_ita.curato.tdw01_punto_di_fornitura"
        self.table_punto_di_fornitura_gas = (
            "p_dpp_ita.curato.tdw01_punto_di_fornitura_gas"
        )
        self.table_punto_di_fornitura_pwr = (
            "p_dpp_ita.curato.tdw01_punto_di_fornitura_pwr"
        )

    def _get_query_as_of(self, timestamp_at: str) -> str:
        """
        Method to get the SQL query to extract the data from the tables at the snapshot date.
        Args:
            timestamp_at (string): snapshot date in the format 'YYYY-MM-DD'.
        Returns:
            query (string): the SQL query to extract the data from the tables at the snapshot date.
        """
        from metrics import MetricheContoClienteCreditoNeta
        from metrics import MetricaPonteSdfProposal
        from metrics import MetricaSdfAttivazione

        if (
            self.table_metricaContoClienteCreditoNeta
            == "vw_MetricheContoClienteCreditoNeta"
        ):
            metricaContoClienteCreditoNeta = MetricheContoClienteCreditoNeta()
            dfMetricaContoClienteCreditoNeta = (
                metricaContoClienteCreditoNeta.get_dataframe_as_of(timestamp_at)
            )
            dfMetricaContoClienteCreditoNeta.createOrReplaceTempView(
                self.table_metricaContoClienteCreditoNeta
            )

        if self.table_metricaPonteSdfProposal == "vw_MetricaPonteSdfProposal":
            metricaPonteSdfProposal = MetricaPonteSdfProposal()
            dfMetricaPonteSdfProposal = metricaPonteSdfProposal.get_dataframe_as_of(
                timestamp_at
            )
            dfMetricaPonteSdfProposal.createOrReplaceTempView(
                self.table_metricaPonteSdfProposal
            )

        if self.table_metricaSdfAttivazione == "vw_MetricaSdfAttivazione":
            metricaSdfAttivazione = MetricaSdfAttivazione()
            dfMetricaSdfAttivazione = metricaSdfAttivazione.get_dataframe_as_of(
                timestamp_at
            )
            dfMetricaSdfAttivazione.createOrReplaceTempView(
                self.table_metricaSdfAttivazione
            )

        query = """
        WITH CONTO_CLIENTE_CREDITO_NETA AS (
          SELECT
            PK_CONTO_FATT,
            CC_MACTO,
            CRED_CC_STATO_CONTO_CRM,
            DATA_SCAD_RATA_OLDER,
            CRED_CC_AGING_SCAD_RATE,
            CRED_CC_NR_FATTURE_INSOLUTE,
            CRED_CC_NR_RATE_INSOLUTE,
            CRED_CC_INSOLUTO_NETA,
            CRED_CC_INSOLUTO_CANONE_TV,
            CRED_CC_INSOLUTO_NETA_R,
            CRED_CC_INSOLUTO_NETA_6_MM,
            CRED_CC_INSOLUTO_NETA_1_ANNI,
            CRED_CC_INSOLUTO_NETA_2_ANNI,
            CRED_CC_INSOLUTO_NETA_3_ANNI,
            CRED_CC_INSOLUTO_NETA_5_ANNI,
            CRED_CC_INSOLUTO_NETA_PIU_ANNI,
            (
              CRED_CC_INSOLUTO_NETA - CRED_CC_INSOLUTO_CANONE_TV
            ) AS CC_INSOLUTO_NETTO_NETA
          FROM
            {table_metricaContoClienteCreditoNeta}
        ),
        PONTE AS (
          SELECT
            PK_ACCOUNT,
            PK_PDF,
            PK_CONTO_FATT
          FROM
            {table_metricaPonteSdfProposal}
        ),
 ----inizio logica per i podpdr ---
--logica per andare piu in dietro di luglio 2022 uguale a quella della metrica MetricaPdfPodPdrGtwV() con fix per 2022, perchè la startDate si inizia a popolare per ogni mese solo dal 2022-07-01 in avanti.---
        PDF AS (
          SELECT
            COD_PDF,
            PK_ID_PDF
          FROM
            {table_punto_di_fornitura} pdf
          WHERE
            pdf.flg_lastSnap = 1
            AND COD_PDF <> '8000000001'

          GROUP BY
            COD_PDF,
            PK_ID_PDF
        ),
        PDR AS (
          SELECT
            pdf.COD_PDF,
            pdr.CODICE_POD_PDR
          FROM
            PDF pdf
            INNER JOIN (
              SELECT
                ID_PDF,
                COD_PDR AS CODICE_POD_PDR,
                row_number() OVER (
                  PARTITION BY ID_PDF
                  ORDER BY
                    valido_dal DESC,
                    ID_TIPOLOGIA_UTILIZZO DESC NULLS LAST
                ) AS N_ROW
              FROM
                {table_punto_di_fornitura_gas}
              WHERE
                flg_lastSnap = 1
                AND COD_PDR IS NOT NULL
            ) pdr ON pdf.PK_ID_PDF = pdr.ID_PDF
          WHERE
            N_ROW = 1
          GROUP BY
            pdf.COD_PDF,
            pdr.CODICE_POD_PDR
        ),
        POD AS (
          SELECT
            pdf.COD_PDF,
            pod.CODICE_POD_PDR
          FROM
            PDF pdf
            INNER JOIN (
              SELECT
                ID_PDF,
                COD_POD AS CODICE_POD_PDR,
                row_number() OVER (
                  PARTITION BY ID_PDF
                  ORDER BY
                    valido_dal DESC,
                    ID_TIPOLOGIA_UTILIZZO DESC NULLS LAST
                ) AS N_ROW
              FROM
                {table_punto_di_fornitura_pwr}
              WHERE
                flg_lastSnap = 1
                AND new_pod IS NOT NULL
                AND COD_POD IS NOT NULL
                AND INSTR(NVL(new_pod, 'x'), 'ERRORE') = 0
            ) pod ON pdf.PK_ID_PDF = pod.ID_PDF
          WHERE
            N_ROW = 1
          GROUP BY
            pdf.COD_PDF,
            pod.CODICE_POD_PDR
        ),
        UNION_POD_PDR AS (
          SELECT
            COD_PDF,
            CODICE_POD_PDR
          FROM
            PDR
          UNION ALL
          SELECT
            COD_PDF,
            CODICE_POD_PDR
          FROM
            POD
        ),
        PODPDRGTW AS (
          SELECT
            COD_PDF,
            CODICE_POD_PDR
          FROM
            UNION_POD_PDR
          -- WHERE CODICE_POD_PDR NOT IN ('IT001E65793804') -- caso duplicato?
        ),

-----------fine logica per i podpdr ---
        SDF_ATTIVAZIONI AS (

          SELECT
            PK_PDF,
            PK_CONTO_FATT,
            PK_ACCOUNT,
            CODICE_POD_PDR
          FROM
            (
            SELECT
                PK_PDF,
                PK_CONTO_FATT,
                PK_ACCOUNT,
                CODICE_POD_PDR,
                ROW_NUMBER() OVER (PARTITION BY PK_PDF ORDER BY DATA_ATT_SITO DESC) AS ROW_N
            FROM (
                    SELECT *
                    FROM {table_metricaSdfAttivazione}
                    WHERE
                    flg_lastSnap = 1 -- logic to filter the metric only in the cases where sdf are active in the snapshot date
                    AND
                    DATA_ATT_SITO <= CAST('{snapshot_date}' AS TIMESTAMP)
                    AND
                    DATA_CESS_SITO > CAST('{snapshot_date}' AS TIMESTAMP)
                ) AS f
            INNER JOIN PODPDRGTW p ON f.PK_PDF = p.COD_PDF
            )
          WHERE
            ROW_N = 1
        ),

        CCCN_PONTE_JOINED AS (

          SELECT
            cccn.CC_MACTO,
            cccn.CRED_CC_STATO_CONTO_CRM,
            cccn.DATA_SCAD_RATA_OLDER,
            cccn.CRED_CC_AGING_SCAD_RATE,
            cccn.CRED_CC_NR_FATTURE_INSOLUTE,
            cccn.CRED_CC_NR_RATE_INSOLUTE,
            cccn.CRED_CC_INSOLUTO_NETA,
            cccn.CC_INSOLUTO_NETTO_NETA,
            cccn.CRED_CC_INSOLUTO_CANONE_TV,
            cccn.CRED_CC_INSOLUTO_NETA_R,
            cccn.CRED_CC_INSOLUTO_NETA_6_MM,
            cccn.CRED_CC_INSOLUTO_NETA_1_ANNI,
            cccn.CRED_CC_INSOLUTO_NETA_2_ANNI,
            cccn.CRED_CC_INSOLUTO_NETA_3_ANNI,
            cccn.CRED_CC_INSOLUTO_NETA_5_ANNI,
            cccn.CRED_CC_INSOLUTO_NETA_PIU_ANNI,
            p.PK_ACCOUNT,
            p.PK_PDF,
            p.PK_CONTO_FATT,
            pp.CODICE_POD_PDR as POD_PDR
          FROM
            CONTO_CLIENTE_CREDITO_NETA cccn
          LEFT JOIN PONTE p ON cccn.PK_CONTO_FATT = p.PK_CONTO_FATT
          INNER JOIN PODPDRGTW pp ON p.PK_PDF = pp.COD_PDF
        ),
        LAST_FORN AS (

          SELECT
            sdf_att.PK_PDF,
            sdf_att.PK_CONTO_FATT as FK_CONTO_FATT,
            sdf_att.PK_ACCOUNT AS FK_ACCOUNT,
            sdf_att.CODICE_POD_PDR,
            jn.CC_MACTO,
            jn.CRED_CC_STATO_CONTO_CRM,
            jn.DATA_SCAD_RATA_OLDER,
            jn.CRED_CC_AGING_SCAD_RATE,
            jn.CRED_CC_NR_FATTURE_INSOLUTE,
            jn.CRED_CC_NR_RATE_INSOLUTE,
            jn.CRED_CC_INSOLUTO_NETA,
            jn.CC_INSOLUTO_NETTO_NETA,
            jn.CRED_CC_INSOLUTO_CANONE_TV,
            jn.CRED_CC_INSOLUTO_NETA_R,
            jn.CRED_CC_INSOLUTO_NETA_6_MM,
            jn.CRED_CC_INSOLUTO_NETA_1_ANNI,
            jn.CRED_CC_INSOLUTO_NETA_2_ANNI,
            jn.CRED_CC_INSOLUTO_NETA_3_ANNI,
            jn.CRED_CC_INSOLUTO_NETA_5_ANNI,
            jn.CRED_CC_INSOLUTO_NETA_PIU_ANNI,
            CASE
              WHEN jn.CC_MACTO like 'RESIDENZ%'
              AND jn.CC_INSOLUTO_NETTO_NETA >= 5000
              AND jn.CRED_CC_AGING_SCAD_RATE >= 90 THEN true
              WHEN jn.CC_MACTO not like 'RESIDENZ%'
              AND jn.CC_INSOLUTO_NETTO_NETA >= 2500
              AND jn.CRED_CC_AGING_SCAD_RATE >= 30 THEN true
              WHEN jn.CC_MACTO not like 'RESIDENZ%'
              AND jn.CC_INSOLUTO_NETTO_NETA >= 500
              AND jn.CRED_CC_AGING_SCAD_RATE >= 90 THEN true
              ELSE false
            END AS FLG_MOROSITA_FORN
          FROM
            SDF_ATTIVAZIONI sdf_att
            LEFT JOIN CCCN_PONTE_JOINED jn
            ON sdf_att.PK_PDF = jn.PK_PDF
            AND sdf_att.PK_CONTO_FATT = jn.PK_CONTO_FATT
            AND sdf_att.PK_ACCOUNT=jn.PK_ACCOUNT
        ),
        ALL_FORN AS (

          SELECT
            POD_PDR,
            COUNT(DISTINCT PK_CONTO_FATT) AS PDF_NR_FORN_MOROSE_ALL,
            MIN(DATA_SCAD_RATA_OLDER) AS PDF_DATA_SCAD_RATA_ALL,
            MAX(CRED_CC_AGING_SCAD_RATE) AS PDF_AGING_SCAD_RATE_ALL,
            SUM(CRED_CC_NR_FATTURE_INSOLUTE) AS PDF_NR_FATTURE_INSOLUTE_ALL,
            SUM(CRED_CC_NR_RATE_INSOLUTE) AS PDF_NR_RATE_INSOLUTE_ALL,
            SUM(CRED_CC_INSOLUTO_NETA) AS PDF_INSOLUTO_LORDO_ALL,
            SUM(CC_INSOLUTO_NETTO_NETA) AS PDF_INSOLUTO_NETTO_ALL,
            SUM(CRED_CC_INSOLUTO_CANONE_TV) AS PDF_INSOLUTO_CANONE_TV_ALL,
            SUM(CRED_CC_INSOLUTO_NETA_R) AS PDF_INSOLUTO_R_ALL,
            SUM(CRED_CC_INSOLUTO_NETA_6_MM) AS PDF_INSOLUTO_6_MM_ALL,
            SUM(CRED_CC_INSOLUTO_NETA_1_ANNI) AS PDF_INSOLUTO_1_ANNI_ALL,
            SUM(CRED_CC_INSOLUTO_NETA_2_ANNI) AS PDF_INSOLUTO_2_ANNI_ALL,
            SUM(CRED_CC_INSOLUTO_NETA_3_ANNI) AS PDF_INSOLUTO_3_ANNI_ALL,
            SUM(CRED_CC_INSOLUTO_NETA_5_ANNI) AS PDF_INSOLUTO_5_ANNI_ALL,
            SUM(CRED_CC_INSOLUTO_NETA_PIU_ANNI) AS PDF_INSOLUTO_PIU_ANNI_ALL
          FROM
            CCCN_PONTE_JOINED
          GROUP BY
            POD_PDR
        )
        SELECT
          f.CODICE_POD_PDR as POD_PDR,
          f.PK_PDF as COD_PDF ,
          f.FLG_MOROSITA_FORN,
          f.CC_MACTO AS PDF_MACTO,
          f.CRED_CC_STATO_CONTO_CRM AS PDF_STATO_CONTO_CRM_LAST,
          f.DATA_SCAD_RATA_OLDER AS PDF_DATA_SCAD_RATA_LAST,
          f.CRED_CC_AGING_SCAD_RATE AS PDF_AGING_SCAD_RATE_LAST,
          f.CRED_CC_NR_FATTURE_INSOLUTE AS PDF_NR_FATTURE_INSOLUTE_LAST,
          f.CRED_CC_NR_RATE_INSOLUTE AS PDF_NR_RATE_INSOLUTE_LAST,
          CAST(f.CRED_CC_INSOLUTO_NETA AS DOUBLE) AS PDF_INSOLUTO_LORDO_LAST,
          CAST(f.CC_INSOLUTO_NETTO_NETA AS DOUBLE) AS PDF_INSOLUTO_NETTO_LAST,
          CAST(f.CRED_CC_INSOLUTO_CANONE_TV AS DOUBLE) AS PDF_INSOLUTO_CANONE_TV_LAST,
          CAST(f.CRED_CC_INSOLUTO_NETA_R AS DOUBLE) AS PDF_INSOLUTO_R_LAST,
          f2.PDF_NR_FORN_MOROSE_ALL AS PDF_NR_FORN_MOROSE_ALL,
          f2.PDF_DATA_SCAD_RATA_ALL,
          f2.PDF_AGING_SCAD_RATE_ALL AS PDF_AGING_SCAD_RATE_ALL,
          f2.PDF_NR_FATTURE_INSOLUTE_ALL AS PDF_NR_FATTURE_INSOLUTE_ALL,
          f2.PDF_NR_RATE_INSOLUTE_ALL AS PDF_NR_RATE_INSOLUTE_ALL,
          CAST(f2.PDF_INSOLUTO_LORDO_ALL AS DOUBLE) AS PDF_INSOLUTO_LORDO_ALL,
          CAST(f2.PDF_INSOLUTO_NETTO_ALL AS DOUBLE) AS PDF_INSOLUTO_NETTO_ALL,
          CAST(f2.PDF_INSOLUTO_CANONE_TV_ALL AS DOUBLE) AS PDF_INSOLUTO_CANONE_TV_ALL,
          CAST(f2.PDF_INSOLUTO_R_ALL AS DOUBLE) AS PDF_INSOLUTO_R_ALL,
          CAST(f2.PDF_INSOLUTO_6_MM_ALL AS DOUBLE) AS PDF_INSOLUTO_6_MM_ALL,
          CAST(f2.PDF_INSOLUTO_1_ANNI_ALL AS DOUBLE) AS PDF_INSOLUTO_1_ANNI_ALL,
          CAST(f2.PDF_INSOLUTO_2_ANNI_ALL AS DOUBLE) AS PDF_INSOLUTO_2_ANNI_ALL,
          CAST(f2.PDF_INSOLUTO_3_ANNI_ALL AS DOUBLE) AS PDF_INSOLUTO_3_ANNI_ALL,
          CAST(f2.PDF_INSOLUTO_5_ANNI_ALL AS DOUBLE) AS PDF_INSOLUTO_5_ANNI_ALL ,
          CAST(f2.PDF_INSOLUTO_PIU_ANNI_ALL AS DOUBLE) AS PDF_INSOLUTO_PIU_ANNI_ALL
        FROM
          LAST_FORN f
        LEFT JOIN ALL_FORN f2 ON f.CODICE_POD_PDR = f2.POD_PDR

      """.format(
            snapshot_date=timestamp_at,
            table_metricaContoClienteCreditoNeta=self.table_metricaContoClienteCreditoNeta,
            table_metricaPonteSdfProposal=self.table_metricaPonteSdfProposal,
            table_metricaSdfAttivazione=self.table_metricaSdfAttivazione,
            table_punto_di_fornitura=self.table_punto_di_fornitura,
            table_punto_di_fornitura_gas=self.table_punto_di_fornitura_gas,
            table_punto_di_fornitura_pwr=self.table_punto_di_fornitura_pwr,
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
        return ["POD_PDR"]

    def get_version(self) -> int:
        """
        Method to get the version of this feature.
        If significant changes are made to the feature's elaboration, the version must be increased.
        Returns:
            version (integer): the version of this feature.
        """
        return 3
