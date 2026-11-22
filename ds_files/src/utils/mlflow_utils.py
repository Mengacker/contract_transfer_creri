"""Utility functions for MLflow model management and tracking."""

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.pyfunc import PythonModel
from typing import Any, Dict, List, Optional, Union
from pyspark.sql import SparkSession

from src.utils.logger import get_logger

logger = get_logger()
spark = SparkSession.builder.getOrCreate()

# Constants
DATABRICKS_REGISTRY_URI = "databricks-uc"
DEFAULT_ARTIFACT_PATH = "model/wrapper"
DEFAULT_CODE_PATHS = ["../src"]


def register_mlflow_model(
    model: Any,
    model_name: str,
    signature: Any,
    registry_uri: str = DATABRICKS_REGISTRY_URI,
    artifact_path: str = DEFAULT_ARTIFACT_PATH,
    code_paths: List[str] = DEFAULT_CODE_PATHS,
) -> None:
    """
    Register the model in MLflow and Databricks Unity Catalog.

    Args:
        model: The model to register.
        model_name: Name of the model.
        signature: Model signature.
        registry_uri: MLflow registry URI.
        artifact_path: Path for model artifact.
        code_paths: List of code paths to include.
    """
    # Setup Mlflow
    mlflow.set_registry_uri(registry_uri)
    client = MlflowClient()

    # Log mlflow model
    mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=model,
        registered_model_name=model_name,
        signature=signature,
        code_paths=code_paths,
    )

    # Promote to challenger and if not exist also to champion
    model_version = get_latest_model_version(model_name, registry_uri)
    logger.info("Last available version is: %s", model_version)
    client.set_registered_model_alias(model_name, "challenger", model_version)

    if not check_if_alias_exist(model_name, "champion", registry_uri):
        logger.info("There is no champion model -> promoting model to champion")
        client.set_registered_model_alias(model_name, "champion", model_version)


def load_deployed_model(
    model_uri: str,
    # spark: Optional[SparkSession] = None,
    as_spark_udf: bool = True,
    result_type: str = "long",
    params: Optional[Dict[str, Any]] = None,
) -> Union[Any, PythonModel]:
    """
    Load the last model registered in Databricks and return it as Spark UDF or PyFunc.

    Args:
        model_uri: URI of the model.
        spark: SparkSession instance (required if as_spark_udf is True).
        as_spark_udf: Whether to load as Spark UDF.
        result_type: Result type for Spark UDF.
        params: Parameters for Spark UDF.

    Returns:
        The loaded model as Spark UDF or PythonModel.
    """
    if as_spark_udf:
        if spark is None:
            raise ValueError("SparkSession must be provided when loading as Spark UDF.")
        logger.info("Loading model as Spark UDF from uri: {model_uri}")
        model_func = mlflow.pyfunc.spark_udf(
            spark, model_uri=model_uri, result_type=result_type, params=params
        )
    else:
        logger.info("Loading model as PyFunc from uri: {model_uri}")
        model_func = mlflow.pyfunc.load_model(model_uri)
    logger.info("Loaded model correctly")
    return model_func


def check_if_alias_exist(
    model_name: str, alias: str, registry_uri: str = DATABRICKS_REGISTRY_URI
) -> bool:
    """
    Check if a specific alias exists for the model.

    Args:
        model_name: Name of the model.
        alias: Alias to check.
        registry_uri: MLflow registry URI.

    Returns:
        True if alias exists, False otherwise.
    """
    client = MlflowClient(registry_uri=registry_uri)
    aliases_dict = client.get_registered_model(model_name).aliases
    check_bool = alias in aliases_dict.keys()
    return check_bool


def get_version_mapped_to_alias(
    model_name: str, alias: str, registry_uri: str = DATABRICKS_REGISTRY_URI
) -> int:
    """
    Return the model version associated with a specific alias.

    Args:
        model_name: Name of the model.
        alias: Alias name.
        registry_uri: MLflow registry URI.

    Returns:
        Model version as integer.
    """
    client = MlflowClient(registry_uri=registry_uri)
    aliases_dict = client.get_registered_model(model_name).aliases
    model_version = int(aliases_dict[alias])
    logger.info("Model version associated to alias %s: %s", alias, model_version)
    return model_version


def retrieve_tuned_params_from_run_id(
    run_id: str, params_list: List[str]
) -> Dict[str, Any]:
    """
    Return tuned hyperparameters given tuning run_id.

    Args:
        run_id: MLflow run ID.
        params_list: List of parameter names to retrieve.

    Returns:
        Dictionary of filtered parameters.
    """
    client = MlflowClient()
    all_params_dct = client.get_run(run_id).data.params
    filt_params_dct = {
        key: value for key, value in all_params_dct.items() if key in params_list
    }
    return filt_params_dct


def retrieve_n_iterations_from_run_id(run_id: str, param_name: str) -> int:
    """
    Return a specific param value (as integer) from a run given its run_id.

    Args:
        run_id: MLflow run ID.
        param_name: Parameter name.

    Returns:
        Parameter value as integer.
    """
    client = MlflowClient()
    all_params_dct = client.get_run(run_id).data.params
    int_param = int(all_params_dct[param_name])
    return int_param


def get_latest_model_version(
    model_name: str, registry_uri: str = DATABRICKS_REGISTRY_URI
) -> int:
    """
    Retrieve the last available version for the specified model.

    Args:
        model_name: Name of the model.
        registry_uri: MLflow registry URI.

    Returns:
        Latest model version as integer.
    """
    latest_version = 1
    mlflow_client = MlflowClient(registry_uri=registry_uri)
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version


def setup_mlflow(tracking_uri: str, experiment_name: str) -> str:
    """
    Set up MLflow tracking and return experiment_id.

    Args:
        tracking_uri: MLflow tracking URI.
        experiment_name: Name of the experiment.

    Returns:
        Experiment ID as string.
    """
    logger.info("Setting MLflow configurations: setting registry URI and experiment")
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(experiment_name=experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    return experiment.experiment_id
