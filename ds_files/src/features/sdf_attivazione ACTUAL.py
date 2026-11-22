# -------------------------------------------------------------------------------------------- #
#       Copia di sola lettura della classe di metriche certificata in Metrics        #
# -------------------------------------------------------------------------------------------- #

from metrics.metrics import SimpleMetric
from metrics.utils import add_catalog


class MetricaSdfAttivazione(SimpleMetric):
    def __init__(self):
        self.table_tdw01_punto_di_fornitura = add_catalog(
            "curato", "tdw01_punto_di_fornitura"
        )
        self.table_tdw01_punto_di_fornitura_gas = add_catalog(
            "curato", "tdw01_punto_di_fornitura_gas"
        )
        self.table_tdw01_punto_di_fornitura_pwr = add_catalog(
            "curato", "tdw01_punto_di_fornitura_pwr"
        )

        self.table_tdw01_stato_pratica = add_catalog("curato", "tdw01_stato_pratica")
        self.table_tdw01_pratiche_dati_tecnici_pwr = add_catalog(
            "curato", "tdw01_pratiche_dati_tecnici_pwr"
        )
        self.table_tdw01_pratiche_dati_tecnici_gas = add_catalog(
            "curato", "tdw01_pratiche_dati_tecnici_gas"
        )
        self.table_tdw01_pratiche_dati_pwr = add_catalog(
            "curato", "tdw01_pratiche_dati_pwr"
        )
        self.table_tdw01_pratiche_dati_gas = add_catalog(
            "curato", "tdw01_pratiche_dati_gas"
        )
        self.table_tdw01_pratiche_gas = add_catalog("curato", "tdw01_pratiche_gas")
        self.table_tdw01_pratiche_pwr = add_catalog("curato", "tdw01_pratiche_pwr")
        self.table_cpqom_order_line_item = add_catalog(
            "curato", "cpqom_order_line_item"
        )
        self.table_cpqom_order = add_catalog("curato", "cpqom_order")
        self.table_cpqom_asset_line_item = add_catalog(
            "curato", "cpqom_asset_line_item"
        )

        self.table_siebd_sito_di_fornitura = add_catalog(
            "curato", "siebd_sito_di_fornitura"
        )
        self.table_siebd_fornitura = add_catalog("curato", "siebd_fornitura")
        self.table_siebd_elemento_dell_ordine = add_catalog(
            "curato", "siebd_elemento_dell_ordine"
        )
        self.table_siebd_elemento_dell_offerta = add_catalog(
            "curato", "siebd_elemento_dell_offerta"
        )
        self.table_siebd_cliente_g_p = add_catalog("curato", "siebd_cliente_g_p")
        self.table_siebd_offerta = add_catalog("curato", "siebd_offerta")
        self.table_slfin_canale_di_attivazione = add_catalog(
            "curato", "slfin_canale_di_attivazione"
        )

        self.table_metrica_asset_migrato_d365 = "vw_asset_migrato_d365"
        self.table_metrica_flg_migrato_d365 = "vw_flg_migrato_d365"
        self.table_metrica_asset_data_attivaz_cessaz_pdf = (
            "vw_asset_data_attivaz_cessaz_pdf"
        )
        self.table_metrica_cod_agenzia_esteso = "vw_cod_agenzia_esteso"

    def _get_query_as_of(self, timestamp_at: str):
        if self.table_metrica_asset_migrato_d365 == "vw_asset_migrato_d365":
            from metrics.asset_line_item import MetricaAssetMigratoD365

            self.table_metrica_asset_migrato_d365 = (
                MetricaAssetMigratoD365().get_query()
            )

        if self.table_metrica_flg_migrato_d365 == "vw_flg_migrato_d365":
            from metrics.account import MetricaFlgMigratoD365

            self.table_metrica_flg_migrato_d365 = MetricaFlgMigratoD365().get_query()

        if (
            self.table_metrica_asset_data_attivaz_cessaz_pdf
            == "vw_asset_data_attivaz_cessaz_pdf"
        ):
            from metrics.asset_line_item import (
                MetricaAssetDataAttivazCessazPdf,
            )

            self.table_metrica_asset_data_attivaz_cessaz_pdf = (
                MetricaAssetDataAttivazCessazPdf().get_query()
            )

        if self.table_metrica_cod_agenzia_esteso == "vw_cod_agenzia_esteso":
            from metrics.quote_proposal import MetricaCodAgenziaEsteso

            self.table_metrica_cod_agenzia_esteso = (
                MetricaCodAgenziaEsteso().get_query_as_of(timestamp_at)
            )

        query = """
        with
        /*
        --- STEP 0 (CTE propedeutiche GTW) ---

        In generale servono come REFINEMENT di altre tabelle dove gtw è considerato più affidabile e con tempistiche più ridotte.
            1. PUNTO_DI_FORN_GTW -> Contiene pdf e il rispettivo pod/pdr che, a meno di sporcature dovrebbero essere 1 a 1
            2. PRATICHE_GTW -> Contiene le pratiche di in (swi,attivazioni) e OUT (swo,cessazioni non commerciali)
            3. SWO_GTW -> Contiene gli swo partendo da PRATICHE_GTW -> serve per riclassificare la data_fine degli asset che non risultano ancora cessati per tempistiche crm.
        */
        PDF as (
            select
                cod_pdf,
                PK_ID_PDF
            from
                {table_tdw01_punto_di_fornitura}
            where
                flg_lastSnap = 1
            group by
                cod_pdf,
                PK_ID_PDF
        ),
        PDR as (
            select
                pdf.cod_pdf,
                pdr.codice_pod_pdr,
                pdr.ID_TIPOLOGIA_UTILIZZO,
                pdr.valido_dal as DATA_VALIDO_DAL_GTW,
                pdr.valido_fino_al as DATA_VALIDO_FINO_AL_GTW
            from
                PDF pdf
                inner join (
                select
                    ID_PDF,
                    COD_PDR as codice_pod_pdr,
                    ID_TIPOLOGIA_UTILIZZO,
                    valido_dal,
                    coalesce(valido_fino_al, '2099-12-31') as valido_fino_al,
                    row_number() over (
                    partition by ID_PDF
                    order by
                        valido_dal desc
                    ) as N_ROW
                from
                    {table_tdw01_punto_di_fornitura_gas}
                where
                    flg_lastSnap = 1
                    and valido_dal <= cast('{snapshot_date}' as date)
                    and coalesce(valido_fino_al, '2099-12-31') >= cast('{snapshot_date}' as date)
                ) pdr on pdf.PK_ID_PDF = pdr.ID_PDF
            where
                N_ROW = 1
            group by
                pdf.cod_pdf,
                pdr.codice_pod_pdr,
                pdr.ID_TIPOLOGIA_UTILIZZO,
                pdr.valido_dal,
                pdr.valido_fino_al
        ),
        POD as (
            select
                pdf.cod_pdf,
                pod.codice_pod_pdr,
                pod.ID_TIPOLOGIA_UTILIZZO,
                pod.valido_dal as DATA_VALIDO_DAL_GTW,
                pod.valido_fino_al as DATA_VALIDO_FINO_AL_GTW
            from
                PDF pdf
                inner join (
                select
                    ID_PDF,
                    COD_POD as codice_pod_pdr,
                    ID_TIPOLOGIA_UTILIZZO,
                    valido_dal,
                    coalesce(valido_fino_al, '2099-12-31') as valido_fino_al,
                    row_number() over (
                    partition by ID_PDF
                    order by
                        valido_dal desc
                    ) as N_ROW
                from
                    {table_tdw01_punto_di_fornitura_pwr}
                where
                    flg_lastSnap = 1
                    and valido_dal <= cast('{snapshot_date}' as date)
                    and coalesce(valido_fino_al, '2099-12-31') >= cast('{snapshot_date}' as date)
                    and new_pod is not null
                    and instr(nvl(new_pod, 'x'), 'ERRORE') = 0
                ) pod on pdf.PK_ID_PDF = pod.ID_PDF
            where
                N_ROW = 1
            group by
                pdf.cod_pdf,
                pod.codice_pod_pdr,
                pod.ID_TIPOLOGIA_UTILIZZO,
                pod.valido_dal,
                pod.valido_fino_al
        ),
        PUNTO_DI_FORN_GTW as(
            select
                distinct *
            from
                PDR
            union all
            select
                distinct *
            from
            POD
        ),
        STATO_PRATICA as (
            select
                PK_ID_STATO_PRATICA
                ,STATO_GTW
            from {table_tdw01_stato_pratica} where flg_lastSnap = 1
        ),
        DATI_TECNICI as (
            select
            PK_COD_PRAT_CORR
            ,COD_PDR
            ,COD_POD
            from {table_tdw01_pratiche_dati_tecnici_pwr} where flg_lastSnap = 1 and isLasT="Y"
            union all
            select
            PK_COD_PRAT_CORR
            ,COD_PDR
            ,COD_POD
            from {table_tdw01_pratiche_dati_tecnici_gas} where flg_lastSnap = 1 and isLasT="Y"
        ),
        DATI as (
            select
                data_decorrenza
                ,PK_COD_PRAT_CORR
                ,MOTIVAZIonE
                ,CODICE_ConTRATTO
                ,codice_odl
                ,CODICE_PDF
                ,DATA_ESECUZIonE as data_attivazione
                ,data_disattivazione
            from {table_tdw01_pratiche_dati_pwr} where flg_lastSnap = 1 and isLasT="Y"
            union all
            select
                data_decorrenza
                ,PK_COD_PRAT_CORR
                ,MOTIVAZIonE
                ,CODICE_ConTRATTO
                ,codice_odl
                ,CODICE_PDF
                ,data_attivazione
                ,data_disattivazione
            from {table_tdw01_pratiche_dati_gas} where flg_lastSnap = 1  and isLasT="Y"
        ),
        PRATICHE as(
            select
                PK_ID_PRATICA
                ,ID_STATO_PRATICA
                ,SERV_NAME
                ,SERVICE_TYPE
                ,COD_SERVIZIO_RETE
                ,COD_PRAT_CORR
                ,PROVENIENZA
                ,SORGENTE
            from
                {table_tdw01_pratiche_gas}
            where
                flg_lastSnap = 1
            union all
            select
                PK_ID_PRATICA
                ,ID_STATO_PRATICA
                ,SERV_NAME
                ,SERVICE_TYPE
                ,COD_SERVIZIO_RETE
                ,COD_PRAT_CORR
                ,PROVENIENZA
                ,SORGENTE
            from
                {table_tdw01_pratiche_pwr}
            where
                flg_lastSnap = 1
        ),
        PRATICHE_GTW as (
            select distinct PRAT.PK_ID_PRATICA
                ,PRAT.SERV_NAME
                ,PRAT.SERVICE_TYPE
                ,STATO.STATO_GTW
                ,DATI.data_decorrenza
                ,DATI.data_attivazione
                ,DATI.data_disattivazione
                ,DATI.MOTIVAZIonE
                ,DATI.CODICE_ConTRATTO
                ,DATI.codice_odl
                ,DATI.CODICE_PDF
                ,SORGENTE
                ,case when DETT.COD_PDR is null then DETT.COD_POD else DETT.COD_PDR end as cod_pod_pdr

            from PRATICHE PRAT

            left join STATO_PRATICA STATO on PRAT.ID_STATO_PRATICA = STATO.PK_ID_STATO_PRATICA
            left join DATI on PRAT.COD_PRAT_CORR = DATI.PK_COD_PRAT_CORR
            left join DATI_TECNICI DETT on PRAT.COD_PRAT_CORR = DETT.PK_COD_PRAT_CORR
        ),
        SWO_GTW as (
            select
                C.PK_ID_PRATICA,
                C.data_decorrenza,
                c.cod_pod_pdr,
                -- +cod_pdf come chiave ponte perchè più affidabile del pod_pdr nell'ALI e OLI. Prima D perchè in C può avere alcuni problemi
                coalesce(D.cod_pdf,C.CODICE_PDF)  as cod_pdf
            from PRATICHE_GTW C
            left join PUNTO_DI_FORN_GTW D on C.cod_pod_pdr = D.codice_pod_pdr
            where
                C.SERVICE_TYPE in ("SE3", "SWG3", "VT4")
                -- Anche stato ERRORE prima o poi dovrebbero andare in COMPLETATA
                AND C.STATO_GTW in ("COMPLETATA","PRATICA SOSPESA","ERRORE","AMMISSIBILE")
                AND coalesce(D.cod_pdf,C.CODICE_PDF) is not null
        ),
        /*
        --- STEP 1 (SITI ASSET)---

        Siti composti da asset che sono presenti nell'asset_line_item (ALI)
        Si vuole:
        1. Creare una tabella con tutti gli asset (ALI_ADJ_ESTESA) con:
            - Le rispettive offerte/processi di vendita con cui sono stati generati (ASSET_OFFERTA_NATIVI/MIGRATI)
            - Sicomme a vecchio il sito era un entità e la tripletta di pk di questa metrica non riproduce il sito perfettamente per alcuni processi, si crea fk_sito_adj che è la rowid_siebel_sito (corrispondente alla chiave del sito a vecchio) riclassificata.
            - rank di processi di vendita per prendere correttamente offerta che ha generato il sito in caso di asset iniziato lo stesso giorno.
        2. Partendo da ALI_ADJ_ESTESA si crea il sito:
            1. Si porta la tabella in chiave tripletta e in seguito si riclassificano gli attributi della tripletta con fk_sito_adj che avrà a questo punto granularità maggiore.
            2. Si trattano a parte i siti "particolari" cioè quelli che possono cambiare ancora tripletta dopo questi aggiustamenti.
            3. Si riclassificano i siti particolari all'interno della tabella con tutti i siti.
        */

        -- asset nativi con processo_di_vend di attivazione, swi, voltura o altri  che possono innescare un cambio della tripletta.
        ASSET_OFFERTA_NATIVI as (
            select
                fk_asset_line_item as pk_asset_line_item,
                processo_di_vend as processo_di_vend_attivaz,
                fk_quote_proposal
            from
                (
                select
                    a.pk_record_id as fk_order_line_item,
                    a.processo_di_vend,
                    a.fk_asset_line_item,
                    a.fk_quote_proposal,
                    row_number() over (
                    partition by a.fk_asset_line_item
                    order by
                        migration_external_id is null desc,
                        a.stato not in ("Cancelled", "Superseded") desc,
                        a.data_creazione asc,
                        a.data_firma is not null desc,
                        a.pk_record_id asc
                    ) as RNA
                from
                    {table_cpqom_order_line_item} as A
                    inner join (
                    select
                        pk_id_record,
                        migration_external_id
                    from
                        {table_cpqom_order}
                    where
                        flg_lastsnap = 1
                    ) O on a.fk_order = O.pk_id_record
                    inner join ({table_metrica_asset_migrato_d365}) B on a.fk_asset_line_item = B.pk_asset_line_item
                    and a.flg_lastsnap = 1
                    and a.processo_di_vend in (
                    -- Processi di attivazione, swi, voltura che a meno di sporcature, sono quelli che possono creare un sito
                        "ALLACCIO CON POSA E ATTIVAZIONE",
                        "ALLACCIO CON POSA E ATTIVAZIONE AMM.",
                        "ATT. SEMPLICE AMM. STD",
                        'ATT. SEMPLICE AMM. RECUPERO',
                        "ATTIVAZIONE",
                        "ATTIVAZIONE CON ACCERTAMENTO",
                        "ATTIVAZIONE CON ACCERTAMENTO AMM",
                        "ATTIVAZIONE CON ISTANZA",
                        "ATTIVAZIONE CON ISTANZA AMM",
                        "ATTIVAZIONE CON ISTANZA AMM. VIP",
                        "ATTIVAZIONE SEMPLICE",
                        "ATTIVAZIONE SEMPLICE AMM. - VIP",
                        "PREVENTIVO E ATTIVAZIONE GAS",
                        "PREVENTIVO ATTIVAZIONE GAS AMM",
                        "SUBENTRO",
                        "SUBENTRO AMM. - VIP",
                        "SUBENTRO AMM. STD",
                        "SUBENTRO AMM. RECUPERO",
                        "SWIN AMM. - RECUPERO",
                        "SWITCH IN",
                        "SWITCH-IN AMM. - VIP",
                        "VOLTURA",
                        "VOLTURA CON SWITCH",
                        "VOLTURA CON SWITCH AMM. - RECUPERO",
                        "VOLTURA CON SWITCH AMM. - VIP",
                        "VOLTURA PARZIALE",
                        "VOLTURA AMM. - VIP",
                        "VOLTURA AMM. - NO PROVISIONING",
                        "VOLTURA AMM. - SI PROVISIONING",
                    -- Processi che innescano in alcuni casi un cambio di tripletta e che non dovrebbero creare un sito
                        "AGGREGA DISAGGREGA FORNITURA",
                        "CAMBIO ACCISA FORNITURA SPEZZATA",
                        "CAMBIO IVA FORNITURA SPEZZATA",
                        "CAMBIO ACCISA FORNITURA MODIFICATA",
                        "VARIAZIONE RESIDENZA",
                        "VARIAZIONE TECNICA LAVORI PREVENTIVO AUMENTO POTENZA"
                    )
                    and (
                    b.flg_asset_d365 = 1
                    or B.FLG_asset_D365_anomalo = 1
                    )
                )
            where
                RNA = 1
        ),
        -- Passaggi per ottenere forniture migrate in chiave asset con la stessa logica di sopra.
        FORNITURA as (
            select
                pk_cod_fornitura,
                cod_sito_di_fornitura,
                cod_cliente,
                cod_conto_cliente,
                cod_fornitura_numerico,
                data_attivazione,
                coalesce(data_cessazione,cast("2099-12-31" as date)) as data_cessazione
            from {table_siebd_fornitura}
            where
                COD_ConTO_CLIENTE is not null
                AND cod_cliente is not null
                AND cod_sito_di_fornitura is not null
                AND data_attivazione is not null
                AND stato_fornitura in ("ATTIVO","CESSATO")
            qualify row_number() over(partition by pk_cod_fornitura order by pk_data_inizio_validita desc) = 1
        ),
        BRIDGE_FORNITURA_NUMERICO_OFFERTA as (
            select
                EOr.cod_fornitura_numerico as pk_cod_fornitura_numerico,
                EOF.cod_offerta as fk_cod_offerta,
                case
                    when EOF.flg_switch_in = "SI" AND B.modalita_operativa = "ATTIVAZIONE" then "SWITCH IN"
                    else B.modalita_operativa
                end as modalita_operativa

            from  {table_siebd_elemento_dell_offerta} EOF
            inner join {table_siebd_elemento_dell_ordine} EOR
                on EOF.pk_cod_elemento_offerta = EOR.cod_elemento_offerta
                AND EOR.flg_lastsnap = 1
            inner join {table_siebd_offerta}   B
                on EOF.cod_offerta=B.PK_COD_OFFERTA
                AND B.flg_lastsnap = 1
            where
                eof.flg_lastsnap = 1
                and (B.modalita_operativa in ("ATTIVAZIONE","VOLTURA","VOLTURA PARZIALE") OR B.modalita_operativa LIKE "CAMBIO%")
            qualify row_number() over (partition by eor.cod_fornitura_numerico order by EOR.stato_elemento_ordine="COMPLETATO" desc, EOR.data_apertura desc,EOF.pk_cod_elemento_offerta desc) = 1
        ),
        ASSET_OFFERTA_MIGRATI as (
            select
                c.pk_asset_line_item,
                b.fk_cod_offerta as fk_quote_proposal,
                b.modalita_operativa as processo_di_vend_attivaz
            from FORNITURA as a
                left join BRIDGE_FORNITURA_NUMERICO_OFFERTA as b
                    on a.cod_fornitura_numerico = b.pk_cod_fornitura_numerico
                inner join ({table_metrica_asset_migrato_d365}) as c
                    on a.pk_cod_fornitura = c.cod_forn_commerciale
            qualify row_number() over (partition by c.pk_asset_line_item order by fk_cod_offerta is not null desc,c.pk_asset_line_item desc) = 1
        ),
        -- ALI con alcuni workaround
        ALI_ADJ as (
            select
                pk_asset_line_item,
                cod_pdf,
                fk_account,
                fk_conto_fatt,
                data_inizio,
                flg_inactive,
                data_inizio,
                flg_inactive,
                rowid_siebel_sito,
                data_creazione,
                flg_stesso_inizio_fine
            from(
                -- Il record più recente è considerato in generale il più affidabile, ma può comunque contenere account o/e conto_fatt null per problemi di quality -> cerco di valorizzarlI con il valore più recente prima di filtrare in flg_lastsnap = 1.
                select
                    pk_record_id as pk_asset_line_item,
                    cod_punto_di_forn as cod_pdf,
                    first_value(fk_venduto_a) over (
                        partition by pk_record_id
                        order by fk_venduto_a is null asc, dat_startdate desc
                        ) as fk_account,
                    first_value(fk_location) over (
                        partition by pk_record_id
                        order by fk_location is null asc, dat_startdate desc
                        ) as fk_conto_fatt,
                    data_inizio,
                    flg_inactive,
                    rowid_siebel_sito,
                    data_creazione,
                    data_inizio == data_fine as flg_stesso_inizio_fine,
                    flg_lastsnap
                from
                    {table_cpqom_asset_line_item} as a
                where
                    tip_prod like 'Prodotto Commodity%'
            )
            where
                flg_lastsnap = 1
                and fk_conto_fatt is not null
                and fk_account is not null
        ),
        -- ALI_ADJ con riclassificazioni e campi aggiuntivi
        ALI_ADJ_ESTESA as (
            select
                a1.pk_asset_line_item,
                coalesce(a2.cod_pdf_fix,a1.cod_pdf) as cod_pdf,
                a1.fk_account,
                a1.fk_conto_fatt,
                a1.data_inizio,
                a1.flg_inactive,
                a1.flg_stesso_inizio_fine,
                a2.data_attivazione_fix,
                a2.data_cessazione_fix,
                coalesce(
                    a1.rowid_siebel_sito,
                    first_value(a1.rowid_siebel_sito) over (
                        partition by coalesce(a2.cod_pdf_fix,a1.cod_pdf),a1.fk_account,a1.fk_conto_fatt
                        order by a1.rowid_siebel_sito is not null desc, a1.data_inizio desc, a1.data_creazione desc, a1.pk_asset_line_item desc
                    ),
                    concat(coalesce(a2.cod_pdf_fix,a1.cod_pdf), '_', a1.fk_account, '_', a1.fk_conto_fatt)
                ) as  fk_sito_adj,
                min(a2.data_attivazione_fix) over (partition by coalesce(a2.cod_pdf_fix,a1.cod_pdf),a1.fk_account,a1.fk_conto_fatt) as data_att_tripletta,
                max(a2.data_cessazione_fix) over (partition by coalesce(a2.cod_pdf_fix,a1.cod_pdf),a1.fk_account,a1.fk_conto_fatt) as data_cess_tripletta,
                coalesce(b.fk_quote_proposal,c.fk_quote_proposal) as fk_quote_proposal,
                coalesce(b.processo_di_vend_attivaz, c.processo_di_vend_attivaz) as processo_di_vend_attivaz,
                /*
                Siccome in un sito ci possono essere più asset iniziati lo stesso giorno, per capire qual'è effettivamente l'asset che ha generato il sito con la rispettiva offerta, si vuole vedere se il processo_di_vend appartiene tra quelli che possono generare un sito. Per problemi di quality principalmente sulla data_attivazione possono capitare dei siti generati da processi non previsti, ma i casi sono contenuti.
                */
                case
                    when coalesce(b.processo_di_vend_attivaz, c.processo_di_vend_attivaz) in (
                            "ALLACCIO CON POSA E ATTIVAZIONE"
                            ,"ALLACCIO CON POSA E ATTIVAZIONE AMM."
                            ,"ATT. SEMPLICE AMM. STD"
                            ,'ATT. SEMPLICE AMM. RECUPERO'
                            ,"ATTIVAZIONE"
                            ,"ATTIVAZIONE CON ACCERTAMENTO"
                            ,"ATTIVAZIONE CON ACCERTAMENTO AMM"
                            ,"ATTIVAZIONE CON ISTANZA"
                            ,"ATTIVAZIONE CON ISTANZA AMM"
                            ,"ATTIVAZIONE CON ISTANZA AMM. VIP"
                            ,"ATTIVAZIONE SEMPLICE"
                            ,"ATTIVAZIONE SEMPLICE AMM. - VIP"
                            ,"PREVENTIVO E ATTIVAZIONE GAS"
                            ,"PREVENTIVO ATTIVAZIONE GAS AMM"
                            ,"SUBENTRO"
                            ,"SUBENTRO AMM. - VIP"
                            ,"SUBENTRO AMM. STD"
                            ,"SUBENTRO AMM. RECUPERO"
                            ,"SWIN AMM. - RECUPERO"
                            ,"SWITCH IN"
                            ,"SWITCH-IN AMM. - VIP"
                            ,"VOLTURA"
                            ,"VOLTURA CON SWITCH"
                            ,"VOLTURA CON SWITCH AMM. - RECUPERO"
                            ,"VOLTURA CON SWITCH AMM. - VIP"
                            ,"VOLTURA PARZIALE"
                            ,"VOLTURA AMM. - VIP"
                            ,"VOLTURA AMM. - NO PROVISIONING"
                            ,"VOLTURA AMM. - SI PROVISIONING"
                        ) then  3 -- Processi che possono generare un sito
                    when coalesce(b.processo_di_vend_attivaz, c.processo_di_vend_attivaz) is null
                        then    2 -- Per i siti molto vecchi è normale che non ci sia un'offerta associata
                    when coalesce(b.processo_di_vend_attivaz, c.processo_di_vend_attivaz) in (
                            "AGGREGA DISAGGREGA FORNITURA"
                            ,"CAMBIO ACCISA FORNITURA SPEZZATA"
                            ,"CAMBIO IVA FORNITURA SPEZZATA"
                            ,"CAMBIO ACCISA FORNITURA MODIFICATA"
                            ,"VARIAZIONE RESIDENZA"
                            ,"VARIAZIONE TECNICA LAVORI PREVENTIVO AUMENTO POTENZA"
                        ) then  1 -- Processi  che non dovrebbero generare un sito, ma possono generare una nuova tripletta e quindi dovranno essere riclassificati
                    else        0 -- Processi  che non dovrebbero generare un sito
                end as rank_processo_di_vend
            from
                ALI_ADJ as a1
                /*
                MetricaassetDataAttivazCessazPdf è un' "ali_bonificata": corregge le date inizio/fine dell' ali,cod_pdf... tenendo conto dei problemi di quality e di delay dei processi. Siccome esclude gli asset con data_inizio = data_fine e potrebbe capitare che questi asset anomali abbiano associata l'offerta di inizio sito corretta, la metrica è in left e non in inner join.
                */
                left join  ({table_metrica_asset_data_attivaz_cessaz_pdf}) as a2
                    on a1.pk_asset_line_item = a2.pk_asset_line_item
                left join asSET_OFFERTA_MIGRATI as b
                    on a1.pk_asset_line_item = b.pk_asset_line_item
                left join asSET_OFFERTA_NATIVI as c
                    on a1.pk_asset_line_item = c.pk_asset_line_item
            where coalesce(a2.cod_pdf_fix,a1.cod_pdf) is not null
        ),
        -- Siti in chiave tripletta riclassificati con fk_sito_adj
        SITI_ASSET_STAGE_1 as (
            select
                cod_pdf as pk_pdf,
                fk_account as pk_account,
                fk_conto_fatt as pk_conto_fatt,
                min(data_att_tripletta) over w_sito_1 as data_att_sito_grezza,
                max(data_cess_tripletta) over w_sito_1 as data_cess_sito_grezza,
                first_value(fk_quote_proposal) over w_sito_2 as fk_quote_proposal,
                first_value(processo_di_vend_attivaz) over w_sito_2 as sdf_mod_ope_grezza,
                rank_processo_di_vend,
                case when rank_processo_di_vend = 1 then true else false end as flg_sito_mult_triplette
            from ALI_ADJ_ESTESA
            where
                data_att_tripletta is not null
                and data_cess_tripletta is not null
                -- Per prendere solo le forniture attivate la stessa data in cui si è attivata la tripletta, nel caso in cui l'attivazione della tripletta non corrisponda all'attivazione del sito la fornitura (o le forniture eventualmente) associata avrà un rank_processo_di_vend inferiore e verrà scartata
                and (
                    data_attivazione_fix = data_att_tripletta
                    or (data_attivazione_fix is null and data_inizio <= data_att_tripletta)
                )
            window
            w_sito_1 as (partition by fk_sito_adj),
            w_sito_2 as(
                    partition by
                        fk_sito_adj
                    order by
                        rank_processo_di_vend desc,
                        data_att_tripletta asc nulls last,
                        data_inizio asc nulls last,
                        flg_stesso_inizio_fine asc,
                        flg_inactive asc,
                        pk_asset_line_item asc
            )
            qualify row_number() over(
                partition by
                    pk_pdf,
                    pk_account,
                    pk_conto_fatt
                order by
                data_att_sito_grezza asc,
                rank_processo_di_vend desc,
                pk_asset_line_item asc
            ) = 1
        ),
        /*
        Per i siti che cambiano tripletta:
            1. Si rilassa il conto e si creano dei gruppi(numero progressivo partendo da 0) con intervalli contigui pdf,account. Intervallo contiguo: coppia attivata al max 1 giorno dopo la cessazione della precedente
            2. Si prendono solo le triplette (pdf,cliente,conto) appartenti a gruppi di pdf/account che sono in contiguità con siti che cambiano tripletta e si ricalcolano gli attributi del sito.
        */
        SITI_ASSET_PARTICOLARI_STAGE_1 as(
            select
                a.*,
                SUM(
                    case
                    --Condizione di non contiguità -> +1 al gruppo
                        when date_add(data_att_sito_grezza,-1) > LAG(MAX(data_cess_sito_grezza) over w) over w then 1
                        else 0
                    end
                ) over w as gruppo_contiguo
            from SITI_ASSET_STAGE_1 as a
                inner join (select distinct pk_account, pk_pdf from SITI_ASSET_STAGE_1 where flg_sito_mult_triplette = true) as b
                on a.pk_account = b.pk_account
                and a.pk_pdf = b.pk_pdf

            window w as (partition by a.pk_account,a.pk_pdf order by data_att_sito_grezza asc)
        ),
        SITI_ASSET_PARTICOLARI_STAGE_2 as (
            select
                a.pk_pdf,
                a.pk_account,
                pk_conto_fatt,
                MIN(data_att_sito_grezza) over w_contiguita_1  as data_att_sito,
                MAX(data_cess_sito_grezza) over w_contiguita_1  as data_cess_sito,
                first_value(a.fk_quote_proposal) over w_contiguita_2 as fk_quote_proposal,
                first_value(a.sdf_mod_ope_grezza) over w_contiguita_2 as sdf_mod_ope,
                true as flg_sito_particolare
            from SITI_ASSET_PARTICOLARI_STAGE_1 as a
                inner join (
                    select pk_account,pk_pdf,gruppo_contiguo
                    from SITI_ASSET_PARTICOLARI_STAGE_1
                    group by pk_account,pk_pdf,gruppo_contiguo
                    having max (flg_sito_mult_triplette) = true
                ) as b
                    on a.pk_account = b.pk_account
                    and a.pk_pdf = b.pk_pdf
                    and a.gruppo_contiguo = b.gruppo_contiguo
            window
                w_contiguita_1 as (partition by a.pk_account,a.pk_pdf,a.gruppo_contiguo),
                w_contiguita_2 as (
                    partition by a.pk_account,a.pk_pdf,a.gruppo_contiguo
                    order by data_att_sito_grezza asc, rank_processo_di_vend desc, pk_conto_fatt desc
                )

        ),
        -- Partendo da SITI_ASSET_STAGE_1, si sostituiscono i vecchi valori con i nuovi per i siti particolari
        SITI_ASSET_STAGE_2 as (
            select
                a.pk_pdf,
                a.pk_account,
                a.pk_conto_fatt,
                coalesce(b.data_att_sito,data_att_sito_grezza) as data_att_sito,
                coalesce(b.data_cess_sito,data_cess_sito_grezza) as data_cess_sito,
                -- E' normale che quote/sdf_mod_ope possa essere null
                case when flg_sito_particolare == true then b.fk_quote_proposal else a.fk_quote_proposal end as fk_quote_proposal,
                case when flg_sito_particolare == true  then b.sdf_mod_ope else a.sdf_mod_ope_grezza end as sdf_mod_ope
            from SITI_ASSET_STAGE_1 as a
            left join SITI_ASSET_PARTICOLARI_STAGE_2 as b
                on a.pk_pdf = b.pk_pdf
                and a.pk_account = b.pk_account
                and a.pk_conto_fatt = b.pk_conto_fatt
        ),
        /*
        --- STEP 2 (SITI NO ASSET) ---
        Siti composti da asset che non sono ancora presenti nell'asset_line_item (ALI)
            1. Si prendono gli ordini derivati da processi di attivazione/swi/ voltura/cessazione che possono avere un ritardo significativo nella creazione dell'asset
            2. Si crea una variabile data_decorrenza che rappresenta una data_attivazione o data_cessazione (per distinguere le tipologie viene creata nel punto 3. una variabile apposita).
            Ogni processo ha un suo modo di calcolare la data_attivazione/cessazione:
            - data_decorrenza: swo,swi,voltura
            - coalesce(data_disattivazione,data_decorrenza): cessazioni non commerciali. La data_disattivazione in questo caso è la "data ufficiale" di cessazione, data_decorrenza è quella stimata
            - coalesce(data_attivazione,data_decorrenza): attivazioni non commerciali. Stessa logica di sopra
            3. Si crea una variabile transazione al fine di distinguere se l'ordine è derivato da un processo di IN 1 o di OUT -1
        */
        ORDINI_ASSET_FITTIZI as (
            select
                O.fk_venduto_a as fk_account,
                a.fk_location as fk_conto_fatt,
                coalesce(a.cod_punto_di_forn,d.cod_pdf) as cod_pdf,
                a.fk_quote_proposal,
                a.processo_di_vend,
                case when processo_di_vend LIKE "%CESSAZ%"
                                    or processo_di_vend LIKE "CAE%"
                                    or processo_di_vend LIKE "CA1%"
                                    or processo_di_vend LIKE "CA2%"
                                    or processo_di_vend LIKE "CA3%"
                                    or processo_di_vend LIKE "CA4%"
                                    or processo_di_vend LIKE "CA5%"
                                    or processo_di_vend LIKE "CA7%"
                                                then coalesce(c.data_disattivazione,c.data_decorrenza)
                        when processo_di_vend in (
                                                "ALLACCIO CON POSA E ATTIVAZIONE"
                                                ,"ATT. SEMPLICE AMM. STD"
                                                ,"ATTIVAZIONE"
                                                ,"ATTIVAZIONE CON ACCERTAMENTO"
                                                ,"ATTIVAZIONE CON ACCERTAMENTO AMM"
                                                ,"ATTIVAZIONE CON ISTANZA"
                                                ,"ATTIVAZIONE CON ISTANZA AMM"
                                                ,"ATTIVAZIONE SEMPLICE"
                                                ,"PREVENTIVO E ATTIVAZIONE GAS"
                                                ,"SUBENTRO"
                                                ,"SUBENTRO AMM. - VIP"
                                                ,"SUBENTRO AMM. STD")
                                                then coalesce(c.data_attivazione,c.data_decorrenza)
                        else  c.data_decorrenza
                        end as data_decorrenza,
                case when processo_di_vend LIKE "%CESSAZ%"
                                    or processo_di_vend LIKE "CAE%"
                                    or processo_di_vend LIKE "CA1%"
                                    or processo_di_vend LIKE "CA2%"
                                    or processo_di_vend LIKE "CA3%"
                                    or processo_di_vend LIKE "CA4%"
                                    or processo_di_vend LIKE "CA5%"
                                    or processo_di_vend LIKE "CA7%"
                                    or processo_di_vend LIKE "%SW%OUT%"  then -1
                        when processo_di_vend in (
                                                "ALLACCIO CON POSA E ATTIVAZIONE"
                                                ,"ATT. SEMPLICE AMM. STD"
                                                ,"ATTIVAZIONE"
                                                ,"ATTIVAZIONE CON ACCERTAMENTO"
                                                ,"ATTIVAZIONE CON ACCERTAMENTO AMM"
                                                ,"ATTIVAZIONE CON ISTANZA"
                                                ,"ATTIVAZIONE CON ISTANZA AMM"
                                                ,"ATTIVAZIONE SEMPLICE"
                                                ,"PREVENTIVO E ATTIVAZIONE GAS"
                                                ,"SUBENTRO"
                                                ,"SUBENTRO AMM. - VIP"
                                                ,"SUBENTRO AMM. STD"
                                                ,"SWIN AMM. - RECUPERO"
                                                ,"SWITCH IN"
                                                ,"SWITCH-IN AMM. - VIP"
                                                ,"VOLTURA CON SWITCH"
                                                ,"VOLTURA CON SWITCH AMM. - RECUPERO"
                                                ,"VOLTURA CON SWITCH AMM. - VIP") then 1
                        end as transazione

            from {table_cpqom_order_line_item} A

            inner join (
                        select pk_id_record,
                                fk_venduto_a
                        from {table_cpqom_order} A
                        inner join ({table_metrica_flg_migrato_d365}) B on a.fk_venduto_a = B.pk_account_cpq
                        where flg_lastSnap = 1
                                AND migration_external_id is null
                                AND a.stato <> 'Cancelled'
                        ) O on a.fk_order = O.pk_id_record

            inner join PRATICHE_GTW as c on a.pk_record_id = c.codice_odl
            left join PUNTO_DI_FORN_GTW as d on c.cod_pod_pdr = D.codice_pod_pdr

            where
            a.flg_lastSnap = 1
            and o.fk_venduto_a is not null and fk_location is not null and coalesce(a.cod_punto_di_forn,d.cod_pdf) is not null
            and a.TIP_PROD LIKE "Prodotto Commodity%"
            AND a.fk_asset_line_item is null
            AND a.stato <> "Cancelled"
            AND ((( processo_di_vend LIKE "%CESSAZ%"
                            or processo_di_vend LIKE "CAE%"
                            or processo_di_vend LIKE "CA1%"
                            or processo_di_vend LIKE "CA2%"
                            or processo_di_vend LIKE "CA3%"
                            or processo_di_vend LIKE "CA4%"
                            or processo_di_vend LIKE "CA5%"
                            or processo_di_vend LIKE "CA7%") AND c.stato_gtw in ("COMPLETATA"))
                OR (processo_di_vend in ("SWITCH OUT"
                                ,"VOLTURA CON SWITCH OUT") AND c.stato_gtw in ("COMPLETATA","PRATICA SOSPESA","ERRORE","AMMISSIBILE"))

                OR (processo_di_vend in ("ALLACCIO CON POSA E ATTIVAZIONE"
                                ,"ATT. SEMPLICE AMM. STD"
                                ,"ATTIVAZIONE"
                                ,"ATTIVAZIONE CON ACCERTAMENTO"
                                ,"ATTIVAZIONE CON ACCERTAMENTO AMM"
                                ,"ATTIVAZIONE CON ISTANZA"
                                ,"ATTIVAZIONE CON ISTANZA AMM"
                                ,"ATTIVAZIONE SEMPLICE"
                                ,"PREVENTIVO E ATTIVAZIONE GAS"
                                ,"SUBENTRO"
                                ,"SUBENTRO AMM. - VIP"
                                ,"SUBENTRO AMM. STD")
                            AND c.stato_gtw in ("COMPLETATA","ESEGUITO TECNICO","IN ATTESA VS2")
                            AND service_type != 'E01')
                OR (processo_di_vend in ("SWIN AMM. - RECUPERO"
                                ,"SWITCH IN"
                                ,"SWITCH-IN AMM. - VIP"
                                ,"VOLTURA CON SWITCH"
                                ,"VOLTURA CON SWITCH AMM. - RECUPERO"
                                ,"VOLTURA CON SWITCH AMM. - VIP") AND c.stato_gtw in ("COMPLETATA","AMMISSIBILE"))
                )
        ),
        /*
        1. Si valorizzano fk_quote_proposal, processo_di_vend_attivaz solo per gli IN
        2. Si contano il numero di ordini per tripletta e gli si da un numero progressivo.
        */
        ORDINI_ASSET_FITTIZI_TRANSAZIONI as (
            select
                fk_account
                ,fk_conto_fatt
                ,a.cod_pdf
                ,case when a.transazione = 1 then  a.fk_quote_proposal else null end as fk_quote_proposal
                ,data_decorrenza
                ,transazione
                ,case when a.transazione = 1 then  a.processo_di_vend else null end as processo_di_vend_attivaz
                ,COUNT(*) over (partition by a.fk_account, a.fk_conto_fatt, a.cod_pdf) as nr_transazioni
                ,row_number() over (partition by a.fk_account, a.fk_conto_fatt, a.cod_pdf order by a.data_decorrenza) as rank_transazioni
            from ORDINI_ASSET_FITTIZI A
        ),
        -- Passaggi per ottenere asset_fittizi con i suoi attributi, che sono per definizione gli stessi del sito.
        SITI_NO_ASSET_STAGE_1 as (
            select
                fk_account
                ,fk_conto_fatt
                ,a.cod_pdf
                -- Uso max per fk_quote_proposal e processo_di_vend_attivaz perché sono valorizzate solo per la transazione in ingresso
                ,MAX(fk_quote_proposal) as fk_quote_proposal
                ,MAX(processo_di_vend_attivaz) as processo_di_vend_attivaz
                -- Il primo ordine di IN della tripletta è l'asset (to be) che ha generato il sito, l'ultimo è quello che ha l'informazione della data di cessazione del sito
                ,MIN(case when transazione = 1 AND rank_transazioni = 1 then data_decorrenza else cast("2099-12-31" as date) end) as data_att_sito
                ,MIN(case when nr_transazioni = rank_transazioni AND transazione = -1 then data_decorrenza else cast("2099-12-31" as date) end) as data_cess_sito_grezza
            from ORDINI_ASSET_FITTIZI_TRANSAZIONI A
            -- primo e ultimo asset fittizio della tripletta
            where (transazione = 1 AND rank_transazioni = 1)
                    or (nr_transazioni = rank_transazioni AND transazione = -1)
            group by ALL
        ),
        SITI_NO_ASSET_STAGE_2 as (
            -- Per gli swo non si forma neanche l'oli in tempo a differenza degli altri processi, quindi devo escluderli inferenzialmente tramite pod_pdr e data_decorrenza.
            select
                fk_account as pk_account
                ,fk_conto_fatt as pk_conto_fatt
                ,a.cod_pdf as pk_pdf
                ,fk_quote_proposal
                ,data_att_sito
                ,coalesce(swo_gtw.data_decorrenza,a.data_cess_sito_grezza) as data_cess_sito
                ,processo_di_vend_attivaz as sdf_mod_ope
            from
                SITI_NO_ASSET_STAGE_1 as a
                left join SWO_GTW as swo_gtw on a.cod_pdf = swo_gtw.cod_pdf
                                                    AND a.data_att_sito < swo_gtw.data_decorrenza
                                                    AND a.data_cess_sito_grezza = "2099-12-31"
            where data_att_sito < coalesce(swo_gtw.data_decorrenza,a.data_cess_sito_grezza)
            qualify row_number() over(
                    partition by fk_account
                    ,fk_conto_fatt
                    ,a.cod_pdf
                    order by coalesce(swo_gtw.data_decorrenza,a.data_cess_sito_grezza) asC) = 1
        ),
        /*
        --- STEP 3 (SITI SIEBD) ---
        Si considerano tutte le forniture siebd non migrate( flg_lastsnap =1)
        1. Si considerano tutte le forniture di clienti resi e micro
        2. Si porta il sito a tripletta e si riclassifica per cod_sito_di_fornitura con logica simile allo step 1, ma più semplice.
        */
        SITO_DI_FORNITURA as (
            select pk_cod_sito_di_fornitura,cod_pdf
            from  {table_siebd_sito_di_fornitura}
            where cod_pdf is not null
            -- qualify invece che flg_lastsnap per evitare problemi di quality e scartare forniture poi nel join
            qualify row_number() over(partition by pk_cod_sito_di_fornitura order by pk_data_inizio_validita desc) = 1
        ),
        CLIENTE_MICRO_RESI as (
            select distinct pk_cod_cliente
            from {table_siebd_cliente_g_p}                    -- -- no flg_lastsnap per evitare di quality e scartare forniture poi nel join
            where (macrosegmento in ("RESIDENZIALI","MICROBUSINESS","RESIDENZIALE") or macrosegmento is null)
        ),
        FORNITURA_LASTSNAP as (
            select
            pk_cod_fornitura,
            cod_sito_di_fornitura,
            cod_cliente,
            cod_conto_cliente,
            cod_fornitura_numerico,
            data_attivazione,
            coalesce(data_cessazione,cast("2099-12-31" as date)) as data_cessazione
            from {table_siebd_fornitura}
            where  cod_conto_cliente is not null
                AND cod_cliente is not null
                AND cod_sito_di_fornitura is not null
                AND data_attivazione is not null
                AND stato_fornitura in ("ATTIVO","CESSATO")
                and flg_lastsnap=1
            -- Ci possono essere duplicati se uso solo il flg_lastsnap, una mia supposizione è che l'univocità è garantita solo per le chiavi e siccome a vecchio sono pk_cod_fornitura e pk_data_inizio_validita se ci sono problemi su dat_startdate/enddate che si riflettono sul flg_lastsnap non se ne accorgono. Il qualify sotto garantisce univocità.
            qualify row_number() over(partition by pk_cod_fornitura order by pk_data_inizio_validita desc) =1
        ),
        SITI_FORNITURA_SIEBD as (
            select
                pk_cod_cliente,
                pk_pdf,
                pk_cod_conto_cliente,
                min(data_attivazione_tripletta) over(partition by cod_sito_di_fornitura) as data_att_sito,
                max(data_cess_tripletta) over (partition by cod_sito_di_fornitura) as data_cess_sito,
                first_value(fk_cod_offerta) over w as fk_cod_offerta,
                first_value(modalita_operativa) over w as sdf_mod_ope
            from (
                select
                    cod_cliente  as pk_cod_cliente
                    ,cod_conto_cliente as pk_cod_conto_cliente
                    ,SF.cod_pdf as pk_pdf
                    ,f.data_attivazione as data_attivazione_tripletta
                    ,fk_cod_offerta
                    ,modalita_operativa
                    ,max(coalesce(cess.data_decorrenza,data_cessazione))  over(partition by sf.cod_pdf,cod_cliente,cod_conto_cliente )  as data_cess_tripletta
                    ,f.cod_sito_di_fornitura
                    ,case
                        when modalita_operativa in ("ATTIVAZIONE","VOLTURA","VOLTURA PARZIALE", "SWITCH IN") then 2
                        when modalita_operativa is null then 1
                        else 0
                    end as rank_modalita_operativa
                from FORNITURA_LASTSNAP F
                inner join CLIENTE_MICRO_RESI CL on F.cod_cliente = CL.PK_cod_cliente
                inner join SITO_DI_FORNITURA SF on F.cod_sito_di_fornitura=SF.PK_cod_sito_di_fornitura
                left join BRIDGE_FORNITURA_NUMERICO_OFFERTA on F.COD_FORNITURA_NUMERICO = BRIDGE_FORNITURA_NUMERICO_OFFERTa.pk_cod_fornitura_numerico
                left join SWO_GTW CESS on SF.cod_pdf = CESS.cod_pdf
                                                AND F.data_attivazione < CESS.data_decorrenza
                                                AND data_cessazione = "2099-12-31"
                qualify row_number() over(
                    partition by pk_pdf, pk_cod_cliente, pk_cod_conto_cliente
                    order by data_attivazione asc,
                        rank_modalita_operativa desc
                ) = 1
            )
            where data_attivazione_tripletta < data_cess_tripletta
            window w as (partition by cod_sito_di_fornitura order by data_attivazione_tripletta asc, rank_modalita_operativa desc)
        ),
        -- Union all di tutti i siti e calcolo dell'aging
        UNION_ALL_SITI as(
            select *,
            case
                    when data_cess_sito > cast('{snapshot_date}' as date)
                        then datediff(cast('{snapshot_date}' as date), data_att_sito)
                    else datediff(data_cess_sito, data_att_sito)
                end as sdf_aging_gg,
                case
                    when data_cess_sito > cast('{snapshot_date}' as date)
                        then months_between(cast('{snapshot_date}' as date), data_att_sito)
                    else months_between(data_cess_sito, data_att_sito)
                end as sdf_aging_mm
            from (
                select
                    pk_pdf,
                    pk_account,
                    pk_conto_fatt,
                    fk_quote_proposal,
                    data_att_sito,
                    data_cess_sito,
                    sdf_mod_ope,
                    'D365 - ASSET' as source_crm
                from SITI_ASSET_STAGE_2
                union all
                select
                    pk_pdf,
                    pk_account,
                    pk_conto_fatt,
                    fk_quote_proposal,
                    data_att_sito,
                    data_cess_sito,
                    sdf_mod_ope,
                    'D365 - NO ASSET' as source_crm
                from SITI_NO_ASSET_STAGE_2 as a
                left anti join SITI_ASSET_STAGE_1 as b
                    on a.pk_pdf = b.pk_pdf
                    and a.pk_account = b.pk_account
                    and a.pk_conto_fatt = b.pk_conto_fatt
                union all
                select
                    pk_pdf,
                    pk_cod_cliente,
                    pk_cod_conto_cliente,
                    fk_cod_offerta,
                    data_att_sito,
                    data_cess_sito,
                    sdf_mod_ope,
                    'SBL' as source_crm
                from SITI_FORNITURA_SIEBD as a

            )
            where data_att_sito <= cast('{snapshot_date}' as date)
        ),
        -- Calcolo dell'aging cliente che è definita come l'aging massimo del sito che ha ancora attivo il cliente alla data di snapshot, altrimenti null.
        AGING_CLIENTE as (
            select
                pk_account,
                max(sdf_aging_gg) as cli_aging_gg,
                max(sdf_aging_mm) as cli_aging_mm
            from
                UNION_ALL_SITI
            where
                data_cess_sito > cast('{snapshot_date}' as date)
            group by
                pk_account
        ),
        CANALE_ATTIVAZIONE as (
                select
                    pk_quote_proposal,
                    canale_attivaz
                from
                    ({table_metrica_cod_agenzia_esteso})  as a
                    inner join {table_slfin_canale_di_attivazione} as b
                        on a.cod_agenzia_esteso = b.PK_COD_AGENZIA
                        and b.flg_lastSnap = 1
        )
        -- Ci sono milioni di asset e quindi siti che non hanno un'offerta associata, se non erro uno dei problemi principali è che in una vecchia migrazione (sistemi precedenti a siebd -> siebd) si sono perse le offerte. Per convenzione si mette 'Z_NOPLICO' in questi casi.
        select
            a.PK_PDF,
            a.PK_ACCOUNT,
            a.PK_CONTO_FATT,
            a.FK_QUOTE_PROPOSAL,
            a.DATA_ATT_SITO,
            a.DATA_CESS_SITO,
            a.SDF_AGING_GG,
            a.SDF_AGING_MM,
            b.CLI_AGING_GG,
            b.CLI_AGING_MM,
            coalesce(a.sdf_mod_ope,'Z_NOPLICO') as SDF_MOD_OPE,
            coalesce(c.canale_attivaz,'Z_NOPLICO')as SDF_CANLE_ATTIVAZ,
            a.SOURCE_CRM
        from
        UNION_ALL_SITI as a
        left join AGING_CLIENTE as b
            on a.pk_account = b.pk_account
        left join CANALE_ATTIVAZIONE as c
            on a.fk_quote_proposal = c.pk_quote_proposal
        """.format(
            snapshot_date=timestamp_at,
            table_tdw01_punto_di_fornitura=self.table_tdw01_punto_di_fornitura,
            table_tdw01_punto_di_fornitura_gas=self.table_tdw01_punto_di_fornitura_gas,
            table_tdw01_punto_di_fornitura_pwr=self.table_tdw01_punto_di_fornitura_pwr,
            table_tdw01_stato_pratica=self.table_tdw01_stato_pratica,
            table_tdw01_pratiche_dati_tecnici_pwr=self.table_tdw01_pratiche_dati_tecnici_pwr,
            table_tdw01_pratiche_dati_tecnici_gas=self.table_tdw01_pratiche_dati_tecnici_gas,
            table_tdw01_pratiche_dati_pwr=self.table_tdw01_pratiche_dati_pwr,
            table_tdw01_pratiche_dati_gas=self.table_tdw01_pratiche_dati_gas,
            table_tdw01_pratiche_gas=self.table_tdw01_pratiche_gas,
            table_tdw01_pratiche_pwr=self.table_tdw01_pratiche_pwr,
            table_cpqom_order_line_item=self.table_cpqom_order_line_item,
            table_cpqom_order=self.table_cpqom_order,
            table_cpqom_asset_line_item=self.table_cpqom_asset_line_item,
            table_siebd_sito_di_fornitura=self.table_siebd_sito_di_fornitura,
            table_siebd_fornitura=self.table_siebd_fornitura,
            table_siebd_elemento_dell_ordine=self.table_siebd_elemento_dell_ordine,
            table_siebd_elemento_dell_offerta=self.table_siebd_elemento_dell_offerta,
            table_siebd_cliente_g_p=self.table_siebd_cliente_g_p,
            table_siebd_offerta=self.table_siebd_offerta,
            table_slfin_canale_di_attivazione=self.table_slfin_canale_di_attivazione,
            table_metrica_asset_migrato_d365=self.table_metrica_asset_migrato_d365,
            table_metrica_flg_migrato_d365=self.table_metrica_flg_migrato_d365,
            table_metrica_asset_data_attivaz_cessaz_pdf=self.table_metrica_asset_data_attivaz_cessaz_pdf,
            table_metrica_cod_agenzia_esteso=self.table_metrica_cod_agenzia_esteso,
        )
        return query

    def get_key_columns(self):
        return ["PK_PDF", "PK_ACCOUNT", "PK_CONTO_FATT"]
