import mlflow
from pyspark.sql import SparkSession
from datetime import datetime

from src.model.model_utils import get_latest_model_version

from src.serving.signature_serving import ServingSignature
from src.serving.custom_serving_model_deterministic import CustomServingModel

from src.config.config_utils import Config

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def mlflow_publish_model(
    is_cicd: bool = False, env: str = None, experiment_path: str = None
) -> str:
    """
    Function to deploy the Deterministic model in the MLFlow registry. It extracts and wraps the yellow list of "POD_PDR" with previous unpaid credits into the Deterministic model, then it is registered in the registry as a new version of Deterministic model.
    Args:
        is_cicd (boolean): a flag to enable the CICD deployment.
        env (string): the name of environment where deploy the model.
        experiment_path (string): MLflow experiment path-name.
    Returns:
        run_id (string): the run id of the MLflow run.
    """
    # Retrive config params
    logger.info("Retrieving config params")

    config_setup = Config("../src/config/config_setup.yml")
    config_model = Config("../src/config/config_model.yml")

    # MLflow informations
    tracking_uri = config_model.get_config_value("mlflow_config.tracking_uri")
    run_name = config_model.get_config_value("mlflow_config.run_name.rollback")
    tags = config_model.get_config_value("mlflow_config.tags.rollback")

    # Tables informations
    catalog = config_setup.get_config_value("environment.{env}.catalog".format(env=env))
    schema = config_setup.get_config_value("environment.{env}.schema".format(env=env))

    # Yellow list table informations
    # path_to_the_table = config_setup.get_config_value("deterministic_model_pod_pdr_tab") # OLD
    diz_special_metric = [
        dd
        for dd in config_setup.get_config_value("metrics")
        if "Used also to extract the yellow-list of POD/PDR." in dd.values()
    ][0]
    path_to_the_table = f"{catalog}.{schema}.{diz_special_metric['name']}"

    # Model informations
    model_name = config_setup.get_config_value("modeling.model_deterministic")["name"]
    catalog_model_deterministic_path = f"{catalog}.{schema}.{model_name}"

    # MlFlow settings
    logger.info("Setting mlflow configurations")
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(experiment_path)
    experiment_id = mlflow.get_experiment_by_name(experiment_path).experiment_id
    current_date = datetime.today().date()
    client = mlflow.MlflowClient()

    # Get the list of POD_PDR with previous morosity
    query = """
    SELECT *
    FROM {table_path}
    WHERE FLG_MOROSITA_FORN IS TRUE
    AND snapshot_date = (SELECT MAX(snapshot_date) FROM {table_path})
    """.format(table_path=path_to_the_table)
    spark: SparkSession = SparkSession.builder.getOrCreate()
    df_morosita = spark.sql(query)
    pod_pdr_list = df_morosita.select("POD_PDR").collect()
    pod_pdr_list = [row["POD_PDR"] for row in pod_pdr_list]
    snapshot_date = df_morosita.select("snapshot_date").collect()[0][0]

    # Wrap the yellow list in the serving model
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
        tags=tags,  # {"mlflow.note.content": desc_str},
    ) as run:
        run_id = run.info.run_id
        # Wrap model
        wrapped_model = CustomServingModel(
            model_name=catalog_model_deterministic_path,
            registry_uri=tracking_uri,
            yellow_list=pod_pdr_list,
        )

        # Define signature
        serving_sign = ServingSignature().get_model_signature()

        # Log params
        mlflow.log_param("run date", str(current_date))
        mlflow.log_param("snapshot date of the yellow list", str(snapshot_date))
        mlflow.log_param("n° of POD/PDR in the yellow list", len(pod_pdr_list))

        # Log model
        logger.info("CICD: registering deterministic model on Databricks catalog")
        mlflow.pyfunc.log_model(
            python_model=wrapped_model,
            registered_model_name=catalog_model_deterministic_path,
            artifact_path="model",
            signature=serving_sign,
            code_paths=["../src"],
        )

        model_version = get_latest_model_version(
            model_name=catalog_model_deterministic_path, registry_uri=tracking_uri
        )

        logger.info(f"Last available version of a model is: {model_version}")
        client.set_registered_model_alias(
            catalog_model_deterministic_path, "deterministic", model_version
        )
        mlflow.end_run
    return run_id
