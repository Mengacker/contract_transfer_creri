import mlflow
from pandas import DataFrame as pandasDataFrame
from datetime import datetime
from typing import List

from src.model.model_utils import load_deployed_model, get_version_mapped_to_alias
from src.config.config_utils import Config

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def batch_inference(
    data: pandasDataFrame = None,
    model_use_mode: str = None,
    is_cicd: bool = False,
    env: str = None,
) -> (str, pandasDataFrame, List[str], List[float]):
    """
    Function to do a prediction on a dataset with the Serving model in a batch inference mode.
    Args:
        data (Pandas dataframe): dataset for inference.
        model_use_mode (string): mode of the model to be used for inference.
        is_cicd (boolean): a flag to enable the CICD deployment.
        env (string): the name of environment where deploy the model.
    Returns:
        run_id (string): the run_id of the mlflow run.
        results (Pandas dataframe): the output of the inference.
        predictors (list[string]): the list of predictors.
        score_thresholds (list[float]): the list of score thresholds.
    """
    # Retrive config params
    logger.info("Retrieving config params")

    config_setup = Config("../src/config/config_setup.yml")
    tracking_uri = config_setup.get_config_value("mlflow_config.tracking_uri")
    experiment_path = config_setup.get_config_value("mlflow_config.experiment_path")

    catalog = config_setup.get_config_value("environment.{env}.catalog".format(env=env))
    schema = config_setup.get_config_value("environment.{env}.schema".format(env=env))

    # model_scoring_name = config_setup.get_config_value("modeling.model_scoring")['name']
    model_serving_name = config_setup.get_config_value("modeling.model_serving")["name"]
    # catalog_model_scoring_path = f"{catalog}.{schema}.{model_scoring_name}"
    catalog_model_serving_path = f"{catalog}.{schema}.{model_serving_name}"

    # ep_mstr_volturato = config_setup.get_config_value("serving_endpoint.ep_mstr_volturato")["name"]
    # ep_mstr_volturante = config_setup.get_config_value("serving_endpoint.ep_mstr_volturante")["name"]

    # databricks_host = config_setup.get_config_value("environment.dev.databricks_host")
    # host = "https://{databricks_host}".format(databricks_host=databricks_host)

    config_model = Config("../src/config/config_model.yml")
    run_name = config_model.get_config_value("mlflow_config.run_name.batch_inference")
    tags = config_model.get_config_value("mlflow_config.tags.batch_inference")

    # MlFlow settings
    logger.info("Setting mlflow configurations")
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(experiment_path)
    experiment_id = mlflow.get_experiment_by_name(experiment_path).experiment_id
    current_date = datetime.today().date()
    # client = mlflow.MlflowClient()

    # Load the trained Champion model
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
        tags=tags,
    ) as run:
        run_id = run.info.run_id

        # Load model
        alias = "serving_champion"
        model_version = get_version_mapped_to_alias(
            model_name=catalog_model_serving_path,
            alias=alias,
            registry_uri=tracking_uri,
        )
        model_uri = "models:/{model_name}@{alias}".format(
            model_name=catalog_model_serving_path, alias=alias
        )  # = model_uri_by_runid = model_uri_by_alias

        mlflow_wrap = load_deployed_model(
            model_uri=model_uri, as_spark_udf=False, result_type="long"
        )
        wrapped_model = mlflow_wrap.unwrap_python_model()
        # raw_lgb_model = wrapped_model.model
        dtypes_map = wrapped_model.get_pandas_dtypes_map()
        predictors = [pp for sublist in dtypes_map.values() for pp in sublist]
        score_thresholds = wrapped_model.get_score_thresholds()
        logger.info(f"Loaded model correctly from uri: {model_uri}")

        if is_cicd:
            logger.info(
                "Do inference in batch mode to compare with the real-time inference"
            )

            # semaforo,  score
            results = wrapped_model.predict(
                user_input=data,
                params={
                    "real_time_input": False,  # "True" is to use real-time inference
                    "use_mode": model_use_mode,  # "ab-test" or "deterministic" or "scoring"
                },
            )

            # Log params
            mlflow.log_param(
                "served model uri by {alias} alias".format(alias=alias),
                str(model_uri),
            )
            mlflow.log_param("run date", str(current_date))
            mlflow.log_param("snapshot date", str(current_date.replace(day=1)))
            mlflow.log_param("version of served model", str(model_version))
            mlflow.log_param("list of predictors", predictors)
            mlflow.log_param("yellow-red threshold", str(max(score_thresholds)))
            mlflow.log_param("yellow-green threshold", str(min(score_thresholds)))

            mlflow.log_param("real-time input", str(False))
            mlflow.log_param("use mode", str(model_use_mode))
            mlflow.log_param("dataframe shape", str(results.shape))

        mlflow.end_run()
    return run_id, results, predictors, score_thresholds
