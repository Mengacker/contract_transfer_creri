# from pyspark.sql import SparkSession
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import (
    OnlineTable,
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# spark: SparkSession = SparkSession.builder.getOrCreate()


def check_online_table_status(online_table_name: str) -> None:
    """
    Function to manually test the status of an online table. It is not used in the notebooks.
    Args:
        online_table_name (string): the name of the online table.
    Returns:
        some print statement with the status of the online table.
    """
    w = WorkspaceClient()
    print(f"\n🔍 Controllo della ONLINE TABLE: {online_table_name}")

    try:
        # Recupera la online table
        otab = w.online_tables.get(online_table_name)
        print(f"✅ Online Table trovata: {online_table_name}")
        print(f"📌 Tabella sorgente: {otab.spec.source_table_full_name}")
        print(f"🔑 Chiave primaria: {otab.spec.primary_key_columns}")

        # Recupera info pipeline
        pipeline_id = otab.spec.pipeline_id
        print(f"⚙️ Pipeline ID: {pipeline_id}")

        # Stato pipeline
        pipeline = w.pipelines.get(pipeline_id=pipeline_id)
        print(f"📊 Stato pipeline: {pipeline.state}")

        if pipeline.state == "FAILED":
            print(
                "❌ La pipeline è in stato FAILED. Potrebbe esserci un errore bloccante."
            )
            updates = w.pipelines.list_updates(pipeline_id=pipeline_id)
            for update in updates:
                print(
                    f"🧾 Update: ID={update.update_id}, state={update.state}, started={update.start_time}"
                )
        elif pipeline.state == "IDLE":
            print("🟢 La pipeline è idle. Provo a forzare l’aggiornamento...")

            # Forza update con full refresh
            try:
                w.pipelines.start_update(pipeline_id=pipeline_id, full_refresh=True)
                print("🔁 Update forzato lanciato con successo!")
            except Exception as e:
                print(f"⚠️ Errore durante l’update: {e}")
        else:
            print(
                f"ℹ️ Pipeline in stato: {pipeline.state}. Nessuna azione necessaria per ora."
            )
    except Exception as e:
        print(f"❌ Errore nel recuperare la online table: {e}")


def online_table_creator(
    pk_col: list[str], table_path: str, online_table_path: str
) -> list[str]:
    """
    Function to create an online table from a table in the catalog.
    Args:
        pk_col (list[string]): the list of name of the primary key columns.
        table_path (string): the path-name of the table in the unity catalog.
        online_table_path (string): the path of the on-line table in the unity catalog.
    Returns:
        l_messages (list): a list of messages with the status of the online table creation.
    Reference:
        https://databricks-sdk-py.readthedocs.io/en/stable/workspace/catalog/online_tables.html
    """
    assert isinstance(pk_col, list) and (len(pk_col) > 0), logger.warning(
        f"""Primary key {str(pk_col)} is not a list or is empty, please check your input"""
    )

    l_messages = []
    w = WorkspaceClient()

    # Create an online table
    spec = OnlineTableSpec(
        primary_key_columns=pk_col,
        source_table_full_name=table_path,
        run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict(
            {"triggered": "true"}
        ),  # la sincronizzazione può essere in tre modalità: istantanea, attivata cioè su trigger, oppure continua
    )
    otab = OnlineTable(name=online_table_path, spec=spec)
    try:
        w.online_tables.get(name=online_table_path)
        online_table_exists = True
    except NotFound:
        online_table_exists = False

    if not online_table_exists:
        # Create an online table
        w.online_tables.create(table=otab)
        message = f"Online table '{online_table_path}' is created"
        logger.info(message)
        l_messages.append(message)
    else:
        message = f"Online table '{online_table_path}' already exists"
        logger.info(message)
        l_messages.append(message)

        # Refresh the on-line table
        pipeline_id = w.online_tables.get(online_table_path).spec.pipeline_id
        # la sincronizzazione può essere in tre modalità: istantanea, attivata cioè su trigger, oppure continua

        # Check if there is an active update
        active_updates = w.pipelines.list_updates(pipeline_id=pipeline_id)
        try:
            if active_updates:
                w.pipelines.start_update(
                    # pipeline_id=pipeline_id, full_refresh=False
                    pipeline_id=pipeline_id,
                    full_refresh=True,  # Forza un full_refresh=True della pipeline--questo è quello che ha funzionato per la online table che non si aggiorna, cosi si aggiorna
                )  # CDF:delta.enableChang...e true, se non funziona si va a riga 154 per forzare il full refresh
                message = f"Online table '{online_table_path}' updated incrementally with pipeline '{pipeline_id}'"

            else:
                message = f"There are not active updates of online table '{online_table_path}' in the pipeline '{pipeline_id}'"
        except Exception as e:
            logger.warning(f"Incremental update failed: {e}. Trying full refresh...")
            try:
                w.pipelines.start_update(
                    pipeline_id=pipeline_id,
                    full_refresh=True,  # Fallback se l'incrementale fallisce
                )
                message = f"Online table '{online_table_path}' forced full refresh with pipeline '{pipeline_id}'"
            except Exception as e2:
                message = f"Full refresh also failed: {e2}"

        logger.info(message)
        l_messages.append(message)

    return l_messages


def feature_spec_creator(pk_col: str, table_path: str, spec_path: str) -> None:
    """
    Function to create a feature spec to serve the feature in the serving endpoint.
    The feature spec automatically uses the online table associated with the Delta table.
    Args:
        pk_col (string): the name of the primary key column.
        table_path (string): the path-name of the table in the unity catalog.
        spec_path (string): the path of the feature spec function in the unity catalog.
    """

    # The feature spec function can be created only with the cluster "IT..."
    w = WorkspaceClient()
    fe = FeatureEngineeringClient()
    try:
        w.functions.get(name=spec_path)
        feature_spec_exists = True
    except NotFound:
        feature_spec_exists = False

    if not feature_spec_exists:
        fe.create_feature_spec(
            name=spec_path,
            features=[FeatureLookup(table_name=table_path, lookup_key=pk_col)],
        )
        print(
            f"UDF feature spec function '{spec_path}' created to lookup the online table '{table_path}'"
        )
    else:
        # The feature spec function no needs to be refreshed, it is a function, not data
        print(
            f"UDF feature spec function '{spec_path}' already exists and it lookups the online table '{table_path}'"
        )
