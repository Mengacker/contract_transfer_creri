import mlflow
from pyspark.sql import SparkSession
from datetime import datetime
from mlflow.models.model import get_model_info
from typing import Any

from src.model.model_utils import (
    load_deployed_model,
    get_latest_model_version,
    get_version_mapped_to_alias,
)

from src.serving.signature_serving import ServingSignature
from src.serving.custom_serving_model_scoring import CustomServingModel
from src.serving.custom_serving_model_scoring_testing import TestServingModel

from src.config.config_utils import Config

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def mlflow_publish_model(
    is_cicd: bool = False,
    env: str = None,
    experiment_path: str = None,
    token: str = None,
    databricks_host: str = None,
) -> (str, Any, Any):
    """
    Function to deploy the Serving model in the MLFlow registry. It loads the Champion Scoring model from the registry and wraps it in the Serving model, then the Serving model is deployed in the registry as a new version of the Serving model.
    Moreover the function extracts and wraps the yellow list of "POD_PDR" with previous unpaid credits into the Deterministic model, then it is registered in the registry as a new version of Deterministic model.
    Args:
        is_cicd (boolean): a flag to enable the CICD deployment.
        env (string): the name of environment where deploy the model.
        experiment_path (string): MLflow experiment path-name.
        token (string): the authentication token.
        databricks_host (string): the url of Databricks workspace.
    Returns:
        run_id (string): the run id of the MLflow run.
        serving_model_info (dictionary): MLflow info of the Serving model.
        wrapped_serving_model (any): the Serving model wrapped in the custom entry script.
    """

    # Retrive config params
    logger.info("Retrieving config params")

    config_setup = Config("../src/config/config_setup.yml")
    config_model = Config("../src/config/config_model.yml")
    config_data = Config("../src/config/config_data.yml")

    # MLflow informations
    tracking_uri = config_model.get_config_value("mlflow_config.tracking_uri")
    run_name = config_model.get_config_value("mlflow_config.run_name.serving")
    tags = config_model.get_config_value("mlflow_config.tags.serving")

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
    model_scoring_name = config_setup.get_config_value("modeling.model_scoring")["name"]
    model_serving_name = config_setup.get_config_value("modeling.model_serving")["name"]
    catalog_model_scoring_path = f"{catalog}.{schema}.{model_scoring_name}"
    catalog_model_serving_path = f"{catalog}.{schema}.{model_serving_name}"

    # Endpoint references
    ep_mstr_cli_out = config_setup.get_config_value(
        "serving_endpoint.ep_mstr_volturato"
    )["name"]
    ep_mstr_cli_in = config_setup.get_config_value(
        "serving_endpoint.ep_mstr_volturante"
    )["name"]
    pk_ep_cli_out = config_setup.get_config_value("serving_endpoint.ep_mstr_volturato")[
        "primary_key"
    ]
    pk_ep_cli_in = config_setup.get_config_value("serving_endpoint.ep_mstr_volturante")[
        "primary_key"
    ]

    # Retrieve encoding maps
    input_data = config_data.get_config_value("input_data")
    encode_maps = {}
    encode_maps["MACROSEGMENTO_CLIIN"] = input_data["SegmentoVolturante"]["encoding"]
    emap = input_data["CanaleAcquisizione"]["decoding"]
    encode_maps["SDF_CANLE_ATTIVAZ_ATT_CLIIN"] = {v: k for k, v in emap.items()}

    # MlFlow settings
    logger.info("Setting mlflow configurations")
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(experiment_path)
    experiment_id = mlflow.get_experiment_by_name(experiment_path).experiment_id
    current_date = datetime.today().date()
    client = mlflow.MlflowClient()

    # Load the trained model as a Champion
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
        tags=tags,
    ) as run:
        run_id = run.info.run_id

        # Load model
        alias = "champion"
        model_champion_version = get_version_mapped_to_alias(
            model_name=catalog_model_scoring_path,
            alias=alias,
            registry_uri=tracking_uri,
        )
        model_uri = "models:/{model_name}@{alias}".format(
            model_name=catalog_model_scoring_path, alias=alias
        )
        mlflow_wrap = load_deployed_model(
            model_uri=model_uri, as_spark_udf=False, result_type="long"
        )
        wrapped_scoring_model = mlflow_wrap.unwrap_python_model()
        raw_lgb_model = wrapped_scoring_model.model
        dtypes_map = wrapped_scoring_model.get_pandas_dtypes_map()
        predictors = [pp for sublist in dtypes_map.values() for pp in sublist]
        score_thresholds = wrapped_scoring_model.get_score_thresholds()
        logger.info(f"Loaded model correctly from uri: {model_uri}")

        # Get the list of POD_PDR with previous morosity
        query = """
        SELECT *
        FROM {table_path}
        WHERE FLG_MOROSITA_FORN IS TRUE
        AND snapshot_date = (SELECT MAX(snapshot_date) FROM {table_path})
        """.format(table_path=path_to_the_table)
        spark: SparkSession = SparkSession.builder.getOrCreate()
        df_morosita = spark.sql(query)
        pod_pdr_yellow_list = df_morosita.select("POD_PDR").collect()
        pod_pdr_yellow_list = [row["POD_PDR"] for row in pod_pdr_yellow_list]

        # Wrap the model with endpoint details
        online_table_dict = {
            pk_ep_cli_out: ep_mstr_cli_out,
            pk_ep_cli_in: ep_mstr_cli_in,
        }

        if env == "prod-sbx":
            wrapped_serving_model = TestServingModel(
                model=raw_lgb_model,
                dtypes_map=dtypes_map,
                encode_maps=encode_maps,
                yellow_red_threshold=max(score_thresholds),
                yellow_green_threshold=min(score_thresholds),
                yellow_list=pod_pdr_yellow_list,
                databricks_host=databricks_host,
                token_ep=token,
                tables_endpoints=online_table_dict,
                use_requests=True,  #  (requests.post) -> funziona anche in un serving endpoint
                # , use_requests= False # (client.predict) -> non funziona in un serving endpoint
            )
        else:
            wrapped_serving_model = CustomServingModel(
                model=raw_lgb_model,
                dtypes_map=dtypes_map,
                encode_maps=encode_maps,
                yellow_red_threshold=max(score_thresholds),
                yellow_green_threshold=min(score_thresholds),
                yellow_list=pod_pdr_yellow_list,
                databricks_host=databricks_host,
                token_ep=token,
                tables_endpoints=online_table_dict,
                use_requests=True,  #  (requests.post) -> funziona anche in un serving endpoint
                # , use_requests= False # (client.predict) -> non funziona in un serving endpoint
            )

        # Define serving model signature
        serving_sign = ServingSignature(env=env).get_model_signature()
        serving_model_info = "None because is_cicd is False"
        if is_cicd:
            logger.info("CICD: registering serving model on Databricks catalog")
            # Log params
            mlflow.log_param(
                "scoring model uri by {alias} alias".format(alias=alias),
                str(model_uri),
            )
            mlflow.log_param("environment", str(env))
            mlflow.log_param("databricks_host", str(databricks_host))
            mlflow.log_param("run date", str(current_date))
            mlflow.log_param("snapshot date", str(current_date.replace(day=1)))
            mlflow.log_param("version of scoring model", str(model_champion_version))
            mlflow.log_param("yellow-red threshold", str(max(score_thresholds)))
            mlflow.log_param("yellow-green threshold", str(min(score_thresholds)))
            mlflow.log_param("list of predictors", predictors)
            # Log model
            mlflow.pyfunc.log_model(
                artifact_path="model/wrapper",
                # name="serving_model",  # model_serving_name,
                python_model=wrapped_serving_model,
                registered_model_name=catalog_model_serving_path,
                signature=serving_sign,
                code_paths=["../src"],
            )
            model_serving_version = get_latest_model_version(
                model_name=catalog_model_serving_path, registry_uri=tracking_uri
            )
            logger.info(
                f"Last available version of the serving model is: {model_serving_version}"
            )
            alias = "serving_champion"
            client.set_registered_model_alias(
                catalog_model_serving_path, alias, model_serving_version
            )
            serving_model_uri = "models:/{model_name}@{alias}".format(
                model_name=catalog_model_serving_path, alias=alias
            )
            serving_model_info = get_model_info(serving_model_uri)

        mlflow.end_run()
    return run_id, serving_model_info, wrapped_serving_model
