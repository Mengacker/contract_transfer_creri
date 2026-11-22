import mlflow
from mlflow import MlflowClient
from pandas import DataFrame as pandasDataFrame
from sklearn import metrics as sk_metrics

from src.deployment.model_validator import ModelValidator
from src.utils.mlflow_utils import get_version_mapped_to_alias

from src.config.config_utils import Config

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compare_models(
    eval_data: pandasDataFrame = None,
    target: str = "FLG_CLI_MOROSO",
    predictors: list[str] = [],
    experiment_path: str = None,
    env: str = None,
) -> (str, dict):
    """
    Function to evaluate the performance of two models, compares them and eventually promote the Challenger model to Champion modifing their alias.
    Args:
        eval_data (Pandas DataFrame): the evaluation dataset.
        target (string): the name of the target column.
        predictors (list[string]): the list of predictor names.
        experiment_path (string): MLflow experiment path-name.
    Returns:
        run_id (string): the MLflow run ID of the comparison run.
        validator_report (dictionary): a dictionary containing the comparison criteria and their outcome.
    """
    # Sort predictors
    sorted_predictors = sorted(predictors)
    eval_data = eval_data.reset_index(drop=True).copy()

    # Retrive config params
    logger.info("Retrieving configuration parameters and informations")

    config_setup = Config("../src/config/config_setup.yml")
    config_model = Config("../src/config/config_model.yml")

    catalog = config_setup.get_config_value("environment.{env}.catalog".format(env=env))
    schema = config_setup.get_config_value("environment.{env}.schema".format(env=env))
    model_name = config_setup.get_config_value("modeling.model_scoring")["name"]
    catalog_model_path = f"{catalog}.{schema}.{model_name}"

    tracking_uri = config_model.get_config_value("mlflow_config.tracking_uri")
    run_name = config_model.get_config_value("mlflow_config.run_name.comparison")
    tags = config_model.get_config_value("mlflow_config.tags.comparison")
    metric_name = config_model.get_config_value("comparison_params.metric_name")
    greater_is_better = config_model.get_config_value(
        "comparison_params.greater_is_better"
    )
    comparison_params = config_model.get_config_value("comparison_params.metric_values")

    # Import the metric corresponding to metric name
    metric_fn = getattr(sk_metrics, metric_name, None)
    if metric_fn is None:
        raise ValueError(f"Metric '{metric_name}' not found in sklearn.metrics")

    # MlFlow settings
    logger.info("Setting MLflow configurations")
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(experiment_name=experiment_path)

    experiment_id = mlflow.get_experiment_by_name(experiment_path).experiment_id

    # Evaluate models
    with mlflow.start_run(
        experiment_id=experiment_id, run_name=run_name, tags=tags
    ) as run:
        run_id = run.info.run_id

        # Instantiate validator and compare models
        logger.info("Instantiate validator and compare models")
        validator = ModelValidator(
            metric_name=metric_name,
            metric_fn=metric_fn,
            greater_is_better=greater_is_better,
            comparison_params=comparison_params,
        )
        comparison_passed = validator.compare_models(
            champion_model_uri=f"models:/{catalog_model_path}@champion",
            challenger_model_uri=f"models:/{catalog_model_path}@challenger",
            eval_data=eval_data,
            target=target,
            predictors=sorted_predictors,
        )
        validator_report = validator.get_report()

        challenger_version = get_version_mapped_to_alias(
            catalog_model_path, "challenger", tracking_uri
        )
        champion_version = get_version_mapped_to_alias(
            catalog_model_path, "champion", tracking_uri
        )

        # Log metrics and results
        logger.info("Logging metrics and parameters on MLFlow experiment")
        mlflow.log_metrics(validator_report["metrics"])
        mlflow.log_params(validator_report["params"])
        mlflow.log_params(validator_report["validation_outputs"])
        mlflow.log_params({"challenger_version": str(challenger_version)})
        mlflow.log_params({"champion_version": str(champion_version)})

        # Promote Challenger to Champion if passed
        if comparison_passed:
            client = MlflowClient()

            client.set_registered_model_alias(
                catalog_model_path, "champion", challenger_version
            )
            logger.info(
                "Comparison passed -> The Challenger model has be promoted to Champion."
            )
            for k, v in validator_report["validation_outputs"].items():
                if (v) & (k != "is_all_comparison_criteria_passed"):
                    logger.info(f"The reason of promotion: {k} = {v}")
        else:
            logger.info(
                "Comparison not passed -> The Challenger model will not be promoted. The Champion model remains the same old one."
            )
            for k, v in validator_report["validation_outputs"].items():
                if (not v) & (k != "is_all_comparison_criteria_passed"):
                    logger.info(f"The reason of not promotion: {k} = {v}")

    return run_id, validator_report
