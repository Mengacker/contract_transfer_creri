# -------------------------------------------------------------------------------------------- #
#       Copia di sola lettura della classe di metriche certificata in Metrics        #
# -------------------------------------------------------------------------------------------- #
from metrics.metrics import SimpleMetric
from metrics.utils import add_catalog


class MetricaSdfAttivazioneAttivi(SimpleMetric):
    # Logica: per ogni tripletta Account, Conto Fattura e Codice Pdf (ex sito fornitura) ricavo la data di attivazione e di cancellazione
    # Poichè per ogni tripletta si possono susseguire più forniture, la logica è la seguente: prendo la data di attivazione della prima fornitura attivata, e la data di cessazione dell'ultima fornitura attivata.
    # Se una tripletta presenta una fornitura attiva, allora la data di cancellazione sarà di default pari a '2099-12-31'

    def __init__(self):
        self.table_sito_fornitura = add_catalog("curato", "siebd_sito_di_fornitura")
        self.table_fornitura = add_catalog("curato", "siebd_fornitura")
        self.elemento_ordine = add_catalog("curato", "siebd_elemento_dell_ordine ")
        self.elemento_offerta = add_catalog("curato", "siebd_elemento_dell_offerta")

        self.table_asset_line_item = add_catalog("curato", "cpqom_asset_line_item")
        self.table_order_line_item = add_catalog("curato", "cpqom_order_line_item")
        self.table_quote_proposal = add_catalog("curato", "cpqom_quote_proposal")

        self.table_slfin_canale_di_attivazione = add_catalog(
            "curato", "slfin_canale_di_attivazione"
        )
        self.table_metrica_cod_agenzia_esteso = "vs_metrica_cod_agenzia_esteso"
        self.table_metrica_flg_forn_attiv = "vw_flg_forn_attiv"

    def _get_query_as_of(self, timestamp_at: str):
        if self.table_metrica_cod_agenzia_esteso == "vs_metrica_cod_agenzia_esteso":
            from metrics.quote_proposal import MetricaCodAgenziaEsteso

            metrica_cod_agenzia_esteso = MetricaCodAgenziaEsteso()
            df_metrica_cod_agenzia_esteso = (
                metrica_cod_agenzia_esteso.get_dataframe_as_of(timestamp_at)
            )
            df_metrica_cod_agenzia_esteso.createOrReplaceTempView(
                self.table_metrica_cod_agenzia_esteso
            )

        if self.table_metrica_flg_forn_attiv == "vw_flg_forn_attiv":
            from metrics.asset_line_item import MetricaFlgFornAttiv

            flg_forn_attiv = MetricaFlgFornAttiv()
            df_metrica_flg_forn_attiv = flg_forn_attiv.get_dataframe_as_of(timestamp_at)
            df_metrica_flg_forn_attiv.createOrReplaceTempView(
                self.table_metrica_flg_forn_attiv
            )

        query = """
                with ATTRIBUTI_SITO_FORNITURE_SIEBD as (
                  -- pk_cod_fornitura + attributi tripletta pdf/cliente/conto alla prima attivazione (SIEBD).
                  -- CTE necessaria per ricondursi al cod_offerta_sito per i migrati (vedi ATTRIBUTI_SITO_DYN)
                  with TABELLA_FORNITURE_SIEBD as (
                    with TABELLA_SITO_SIEBD as (
                      select
                        pk_cod_sito_di_fornitura,
                        cod_pdf
                      from(
                          select
                            pk_cod_sito_di_fornitura,
                            cod_pdf,
                            row_number() over(
                              partition by pk_cod_sito_di_fornitura
                              order by
                                pk_data_inizio_validita desc
                            ) as rn
                          from
                            {table_sito_fornitura}
                          where
                            cod_pdf is not null
                        )
                      where
                        rn = 1
                    )
                    select
                      *
                    except(rn)
                    from(
                        select
                          pk_cod_fornitura,
                          b.cod_pdf,
                          cod_cliente,
                          cod_conto_cliente,
                          data_attivazione,
                          case
                            when data_cessazione is null then cast('2099-12-31' as date)
                          end as data_cessazione,
                          cod_fornitura_numerico,
                          cod_offerta,
                          data_creazione,
                          row_number() over(
                            partition by pk_cod_fornitura
                            order by
                              pk_data_inizio_validita desc,
                              stato_fornitura desc,
                              data_creazione desc
                          ) as rn
                        from
                          {table_fornitura} as a
                          inner join TABELLA_SITO_SIEBD as b on a.cod_sito_di_fornitura = b.pk_cod_sito_di_fornitura
                        where
                          stato_fornitura in ("ATTIVO", "CESSATO")
                          and cod_cliente is not null
                          and cod_conto_cliente is not null
                          and cod_sito_di_fornitura is not null -- Non posso storicizzare perché nei periodi di
                            -- migrazione possono esserci delle forniture del sito ancora nei vecchi sistemi e potrei
                            -- non prendermi gli attributi corretti -> metto i vincoli sotto per storicizzare,
                            -- questo è importante per le cessazioni non commerciali per cui il sito può cessare
                            -- per un certo intervallo di tempo per poi riattivarsi.
                          and data_attivazione is not null
                          and data_attivazione <= CAST('{snapshot_date}'  AS DATE)
                          and (
                            data_cessazione <= CAST('{snapshot_date}'  AS DATE)
                            or data_cessazione = '2099-12-31'
                            or data_cessazione is null
                          )
                      )
                    where
                      rn = 1
                  ),
                  OFFERTA as (
                    select
                      cod_fornitura_numerico,
                      cod_offerta
                    from(
                        select
                          cod_fornitura_numerico,
                          cod_offerta,
                          row_number() over(
                            partition by cod_fornitura_numerico
                            order by
                              cod_offerta nulls last
                          ) as rn
                        from
                          {elemento_ordine} as a
                          inner join {elemento_offerta} as b on pk_cod_elemento_offerta = cod_elemento_offerta
                        where
                          a.flg_lastSnap = 1
                          and b.flg_lastSnap = 1
                          and cod_fornitura_numerico is not null
                      )
                    where
                      rn = 1
                  )
                  select
                    pk_cod_fornitura,
                    cod_pdf,
                    cod_cliente,
                    cod_conto_cliente,
                    data_attivazione,
                    data_cessazione,
                    first_value(a.data_attivazione) over w as data_attivazione_sito,
                    first_value(b.cod_offerta) over w as cod_offerta_sito
                  from
                    TABELLA_FORNITURE_SIEBD as a
                    left join OFFERTA as b on a.cod_fornitura_numerico = b.cod_fornitura_numerico window w as (
                      partition by cod_pdf,
                      cod_cliente,
                      cod_conto_cliente
                      order by
                        data_attivazione asc,
                        data_cessazione asc,
                        data_creazione asc
                    )
                ),
                ATTRIBUTI_SITO_SIEBD as (
                  -- Attributi della tripletta pdf/cliente/conto alla prima attivazione (SIEBD)
                  select
                    cod_pdf as pk_cod_pdf,
                    cod_cliente as pk_cod_cliente,
                    cod_conto_cliente as pk_cod_conto_cliente,
                    cod_offerta_sito as fk_cod_offerta_sito,
                    data_attivazione_sito --,
                    --data_cessazione     as data_cessazione_sito
                  from(
                      select
                        cod_pdf,
                        cod_cliente,
                        cod_conto_cliente,
                        cod_offerta_sito,
                        data_attivazione_sito,
                        --data_cessazione,
                        row_number() over(
                          partition by cod_pdf,
                          cod_cliente,
                          cod_conto_cliente
                          order by
                            data_cessazione asc
                        ) as rn
                      from
                        ATTRIBUTI_SITO_FORNITURE_SIEBD
                    )
                  where
                    rn = 1
                ),
                ATTRIBUTI_SITO_DYN as (
                  -- Attributi della tripletta pdf/cliente/conto alla prima attivazione (DYN)
                  with OLI AS (
                    SELECT
                      FK_ASSET_LINE_ITEM AS PK_ASSET_LINE_ITEM,
                      FK_QUOTE_PROPOSAL
                    FROM
                      (
                        select
                          A.FK_ASSET_LINE_ITEM,
                          A.FK_QUOTE_PROPOSAL,
                          ROW_NUMBER() OVER (
                            PARTITION BY A.FK_ASSET_LINE_ITEM
                            ORDER BY
                              A.DATA_CREAZIONE
                          ) AS RNA
                        from
                          {table_order_line_item} A
                        WHERE
                          A.flg_lastSnap = 1
                          AND A.stato NOT IN ("Cancelled", "Superseded")
                          AND A.PROCESSO_DI_VEND <> "CESSAZIONE SU VOLTURATO"
                          AND A.PROCESSO_DI_VEND NOT LIKE "%SW%OUT%"
                          AND A.PROCESSO_DI_VEND NOT LIKE "%CESSAZ%"
                          AND A.PROCESSO_DI_VEND NOT LIKE "CAE%"
                          AND A.PROCESSO_DI_VEND NOT LIKE "CA1%"
                          AND A.PROCESSO_DI_VEND NOT LIKE "CA2%"
                          AND A.PROCESSO_DI_VEND NOT LIKE "CA4%"
                          AND A.PROCESSO_DI_VEND NOT LIKE "CA5%"
                          AND A.PROCESSO_DI_VEND NOT LIKE "CA7%"
                          AND A.DATA_FIRMA IS NOT NULL
                      )
                    WHERE
                      RNA = 1
                  ),
                  ALI_PACCHETTO as (
                    select
                      pk_record_id,
                      cod_punto_di_forn,
                      fk_venduto_a,
                      fk_location,
                      fk_quote_proposal,
                      rowid_siebel_forn,
                      data_inizio,
                      case
                        when data_fine is null then cast('2099-12-31' as date)
                        else data_fine
                      end as data_fine,
                      data_creazione
                    from
                      {table_asset_line_item}
                    where
                      asset_status in ("Cancelled", "Activated")
                      and tip_prod like 'Prodotto Commodity%'
                      and data_inizio is not null
                      and cod_punto_di_forn is not null
                      and fk_venduto_a is not null
                      and fk_location is not null -- Non posso storicizzare perché nei periodi di
                      --migrazione possono esserci delle forniture del sito ancora nei vecchi sistemi e
                      --potrei non prendermi gli attributi corretti -> metto i vincoli
                      --sotto per storicizzare, questo è importante per le cessazioni non commerciali
                      --per cui il sito può cessare per un certo intervallo di tempo per poi riattivarsi.
                      and data_inizio <= CAST('{snapshot_date}'  AS DATE)
                      and (
                        data_fine <= CAST('{snapshot_date}'  AS DATE)
                        or data_fine = '2099-12-31'
                        or data_fine is null
                      )
                  )
                  select
                    cod_punto_di_forn as pk_cod_pdf,
                    fk_venduto_a as pk_account,
                    fk_location as pk_conto_fatt,
                    case
                      when rowid_siebel_forn is not null then cod_offerta_sito
                      else fk_quote_proposal
                    end as fk_quote_proposal_sito,
                    data_inizio as data_attivazione_sito
                  from(
                      select
                        cod_punto_di_forn,
                        fk_venduto_a,
                        fk_location,
                        data_fine,
                        rowid_siebel_forn,
                        c.cod_offerta_sito,
                        data_inizio,
                        b.fk_quote_proposal,
                        row_number() over (
                          partition by cod_punto_di_forn,
                          fk_venduto_a,
                          fk_location
                          order by
                            data_inizio asc,case
                              when rowid_siebel_forn is not null then 1
                              else 0
                            end desc,
                            data_fine asc,
                            data_creazione asc
                        ) as rn -- per prendere la data fine dell'ultima fornitura
                      from
                        ALI_PACCHETTO as a
                        left join OLI as b on a.pk_record_id = b.pk_asset_line_item
                        left join ATTRIBUTI_SITO_FORNITURE_SIEBD as c on a.rowid_siebel_forn = c.pk_cod_fornitura
                    )
                  where
                    rn = 1
                ),
                UNION_ATTRIBUTI_SITO as (
                  -- Attributi della tripletta pdf/cliente/conto alla prima attivazione
                  select
                    pk_cod_pdf,
                    pk_account,
                    pk_conto_fatt,
                    fk_quote_proposal_sito,
                    data_attivazione_sito
                  from
                    ATTRIBUTI_SITO_DYN
                  union all
                  select
                    pk_cod_pdf,
                    pk_cod_cliente,
                    pk_cod_conto_cliente,
                    fk_cod_offerta_sito,
                    data_attivazione_sito
                  from
                    ATTRIBUTI_SITO_SIEBD
                ),
                SITI_ATTIVI as (
                  with OLI_PROPOSAL AS (
                    select
                      A.pk_record_Id as pk_order_line_item,
                      A.FK_QUOTE_PROPOSAL
                    from
                      {table_order_line_item} A
                    WHERE
                      A.flg_lastSnap = 1
                      AND A.stato NOT IN ("Cancelled", "Superseded")
                      AND A.PROCESSO_DI_VEND <> "CESSAZIONE SU VOLTURATO"
                      AND A.PROCESSO_DI_VEND NOT LIKE "%SW%OUT%"
                      AND A.PROCESSO_DI_VEND NOT LIKE "%CESSAZ%"
                      AND A.PROCESSO_DI_VEND NOT LIKE "CAE%"
                      AND A.PROCESSO_DI_VEND NOT LIKE "CA1%"
                      AND A.PROCESSO_DI_VEND NOT LIKE "CA2%"
                      AND A.PROCESSO_DI_VEND NOT LIKE "CA4%"
                      AND A.PROCESSO_DI_VEND NOT LIKE "CA5%"
                      AND A.PROCESSO_DI_VEND NOT LIKE "CA7%"
                      AND A.DATA_FIRMA IS NOT NULL
                  )
                  select
                    cod_pdf,
                    fk_conto_fatt,
                    fk_account,
                    fk_quote_proposal_asset_fittizio,
                    data_attivazione
                  from
                    (
                      select
                        cod_pdf,
                        fk_conto_fatt,
                        fk_account,
                        source_crm,
                        case
                          when source_crm = 'D365 - NO ASSET' then fk_quote_proposal
                          else null
                        end as fk_quote_proposal_asset_fittizio,
                        data_attivazione,
                        row_number() over(
                          partition by cod_pdf,
                          fk_conto_fatt,
                          fk_account
                          order by
                            data_attivazione asc
                        ) as rn
                      from
                        {table_metrica_flg_forn_attiv} as a
                        left join OLI_PROPOSAL as b on a.fk_order_line_item = b.pk_order_line_item
                    )
                  where
                    rn = 1
                ),
                ATTRIBUTI_SITO_AGING as (
                  -- UNION_ATTRIBUTI_SITO + metriche aging
                  with AGING_FORNITURA as (
                    select
                       coalesce(pk_cod_pdf,cod_pdf ) as pk_cod_pdf,
                       coalesce(pk_account,  fk_account) as pk_account,
                       coalesce(pk_conto_fatt, fk_conto_fatt) as pk_conto_fatt,
                      coalesce(
                        fk_quote_proposal_sito,
                        fk_quote_proposal_asset_fittizio
                      ) as fk_quote_proposal_sito,
                      coalesce(data_attivazione_sito, data_attivazione) as data_attivazione_sito,
                      datediff(
                        '{snapshot_date}' ,
                        coalesce(data_attivazione_sito, data_attivazione)
                      ) as aging_fornitura_gg,
                      months_between(
                        '{snapshot_date}' ,
                        coalesce(data_attivazione_sito, data_attivazione)
                      ) as aging_fornitura_mm
                    from
                      SITI_ATTIVI as a
                      left join UNION_ATTRIBUTI_SITO as b on a.cod_pdf = b.pk_cod_pdf
                      and a.fk_account = b.pk_account
                      and a.fk_conto_fatt = b.pk_conto_fatt
                  ),
                  AGING_CLIENTE as (
                    select
                      pk_account,
                      max(aging_fornitura_gg) as aging_cliente_gg,
                      max(aging_fornitura_mm) as aging_cliente_mm
                    from
                      AGING_FORNITURA
                    group by
                      pk_account
                  )
                  select
                    pk_cod_pdf,
                    a.pk_account,
                    pk_conto_fatt,
                    fk_quote_proposal_sito,
                    data_attivazione_sito,
                    aging_fornitura_gg,
                    aging_fornitura_mm,
                    aging_cliente_gg,
                    aging_cliente_mm
                  from
                    AGING_FORNITURA as a
                    left join AGING_CLIENTE as b on a.pk_account = b.pk_account
                                ),
                CANALE_ATTIVAZIONE as (
                  -- Canale attivazione in chiave offerta
                  select
                    pk_quote_proposal,
                    canale_attivaz,
                    canale_sales
                  from
                    {table_metrica_cod_agenzia_esteso} as a
                    inner join {table_slfin_canale_di_attivazione} as b on a.cod_agenzia_esteso = b.PK_COD_AGENZIA
                    and b.flg_lastSnap = 1
                )
                  select
                  pk_cod_pdf as PK_PDF,
                  pk_account as PK_ACCOUNT,
                  pk_conto_fatt as PK_CONTO_FATT,
                  fk_quote_proposal_sito as FK_QUOTE_PROPOSAL,
                  data_attivazione_sito as DATA_ATT_SITO_ATT,
                  aging_fornitura_gg as SDF_AGING_GG_ATT,
                  aging_fornitura_mm as SDF_AGING_MM_ATT,
                  aging_cliente_gg as CLI_AGING_GG_ATT,
                  aging_cliente_mm as CLI_AGING_MM_ATT,
                  case
                    when canale_attivaz is null then 'Z_NOPLICO'
                    else canale_attivaz
                  --end as canale_attivazione_sito
                  end as SDF_CANLE_ATTIVAZ_ATT
                from
                  ATTRIBUTI_SITO_AGING as a
                  left join CANALE_ATTIVAZIONE as b on a.fk_quote_proposal_sito = b.pk_quote_proposal

                """.format(
            table_sito_fornitura=self.table_sito_fornitura,
            table_fornitura=self.table_fornitura,
            elemento_ordine=self.elemento_ordine,
            elemento_offerta=self.elemento_offerta,
            table_asset_line_item=self.table_asset_line_item,
            table_order_line_item=self.table_order_line_item,
            table_metrica_cod_agenzia_esteso=self.table_metrica_cod_agenzia_esteso,
            table_metrica_flg_forn_attiv=self.table_metrica_flg_forn_attiv,
            table_slfin_canale_di_attivazione=self.table_slfin_canale_di_attivazione,
            snapshot_date=timestamp_at,
        )

        return query

    def get_key_columns(self):
        return ["PK_ACCOUNT", "PK_CONTO_FATT", "PK_PDF"]
