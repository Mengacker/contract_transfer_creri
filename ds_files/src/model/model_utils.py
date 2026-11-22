"""Utilities functions related to model."""

import logging
from typing import Any, Dict
from pyspark.sql import SparkSession

import mlflow
from mlflow.pyfunc import PythonModel
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

spark = SparkSession.builder.getOrCreate()


def check_if_alias_exist(model_name: str, alias: str, registry_uri: str) -> bool:
    """
    Function to check if a specific alias exists for the model.
    Args:
        model_name (string): name of the model.
        alias (string): alias of the model.
        registry_uri (string): registry uri that indicates where is the model.
    Returns:
        check_bool (boolean): a flag to indicate if the alias exists or not.
    """
    client = MlflowClient(registry_uri=registry_uri)
    # Get a dictionary mapping aliases to version
    aliases_dict = client.get_registered_model(model_name).aliases
    check_bool = alias in aliases_dict.keys()
    return check_bool


def get_version_mapped_to_alias(model_name: str, alias: str, registry_uri: str) -> int:
    """
    Function to return the model's version associated with a specific alias.
    Args:
        model_name (string): name of the model.
        alias (string): alias of the model.
        registry_uri (string): registry uri that indicates where is the model.
    Returns:
        model_version (integer): model version associated with the alias.
    """
    client = MlflowClient(registry_uri=registry_uri)
    # Get a dictionary mapping aliases to version
    aliases_dict = client.get_registered_model(model_name).aliases
    model_version = int(aliases_dict[alias])
    logger.info("Model version associated to alias %s is %s", alias, model_version)
    return model_version


def get_latest_model_version(model_name: str, registry_uri: str) -> int:
    """
    Function to retrieve the last available version of the specified model.
    Args:
        model_name (string): name of the model.
        registry_uri (string): registry uri that indicates where is the model.
    Returns:
        latest_version (integer): latest model version.
    """
    latest_version = 1
    mlflow_client = MlflowClient(registry_uri=registry_uri)
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version


def retrieve_tuned_params_from_run_id(
    run_id: str, params_list: list[str]
) -> Dict[str, Any]:
    """
    Function to get the tuned hyperparameters from a given tuning run id.
    Args:
        run_id (string): the id that identifies the MLflow run.
        params_list (list): list of parameters to filter.
    Returns:
        filt_params_dict (dict): dictionary containing the filtered parameters.
    """
    client = MlflowClient()
    # Get a dict with all registered params
    all_params_dict = client.get_run(run_id).data.params
    filt_params_dict = {
        key: value for key, value in all_params_dict.items() if key in params_list
    }
    return filt_params_dict


def load_deployed_model(
    model_uri: str, as_spark_udf: bool = True, result_type: str = "long"
) -> PythonModel:
    """
    Function to load the last model registered in Databricks and return it as Spark function.
    The model usi is like "models:/{model_name}@alias".
    Args:
        model_uri (string): the Uniform Resource Identifier of the model to load.
        as_spark_udf (boolean): flag to indicate if the model should be loaded as Spark UDF or not. "True" is the default value.
        result_type (string): datatype of the result to return. "Long" is the default datatype.
    Returns:
        model_func (PythonModel): the loaded model.
    """
    if as_spark_udf:
        logger.info("Loading model as spark udf from uri: {model_uri}")
        model_func = mlflow.pyfunc.spark_udf(
            spark, model_uri=model_uri, result_type=result_type
        )
    else:
        logger.info("Loading model as pyfunc from uri: {model_uri}")
        model_func = mlflow.pyfunc.load_model(model_uri)
    logger.info("Loaded model correctly")
    return model_func


def model_version_eraser(
    model_name_to_delete: str = "catalog.schema.model_name",
    versions_to_delete: list = [1, 2, 3],
    tracking_uri: str = "databricks-uc",
    experiment_path: str = "experiment_path_name",
) -> str:
    """
    Function to delete model versions deployed in the catalog with MLflow.
    Args:
        model_name_to_delete (string): name of the model.
        versions_to_delete (list): list of model's versions.
        tracking_uri (string): the Uniform Resource Identifier of the traking in Unity Catalog.
        experiment_path (string): MLflow experiment path-name.
    Returns:
        run_id (string): the id that identifies the MLflow run.
    """
    run_name = "deleting_model_versions"
    desc_str = "delete some model versions"

    # MlFlow settings
    logger.info("Setting mlflow configurations")
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(experiment_path)
    experiment_id = mlflow.get_experiment_by_name(experiment_path).experiment_id
    client = mlflow.MlflowClient()

    # Load the trained model as a Champion
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
        tags={"mlflow.note.content": desc_str},
    ) as run:
        run_id = run.info.run_id
        for version in versions_to_delete:
            print(f"model version: {version}", end="->")
            try:
                client.delete_model_version(name=model_name_to_delete, version=version)
                print("deleted")
            except Exception:
                print("version not found")
    return run_id
