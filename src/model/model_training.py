from pandas import DataFrame as pandasDataFrame, Series as pandasSeries
from datetime import datetime
import mlflow
import lightgbm as lgb
from typing import Any, Dict

from src.model.signature import DataModelSignature
from src.model.model_wrapper import ModelWrapper
from src.model.model_utils import get_latest_model_version, check_if_alias_exist

from src.config.config_utils import Config

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fit_model(
    train_set: tuple[pandasDataFrame, pandasSeries],
    eval_set: tuple[pandasDataFrame, pandasSeries] = None,
    best_params: Dict[str, Any] = None,
    is_cicd: bool = False,
    experiment_path: str = None,
    env: str = None,
) -> tuple[str, ModelWrapper]:
    """
    Function to train a LightGBM model on the given training dataset with the best parameters found during tuning.
    The raw model is wrapped in a custom PyModel and registered as a model in MLFlow.
    Args:
        train_set (tuple[Pandas DataFrame, Pandas Series]): tuple containing the training dataset and the target variable.
        eval_set (tuple[Pandas DataFrame, Pandas Series]): tuple containing the evaluation dataset and the target variable.
        best_params (dictionary[string, any]): dictionary containing the best parameters found during tuning.
        is_cicd (boolean): flag to indicate if the method is called from a CI/CD pipeline or not.
        experiment_path (str): MLflow experiment path-name.
        env (str): the name of the environment.
    Returns:
        run_id (string): the id that identifies the MLflow run.
        model_wrapper (ModelWrapper): the raw LightGBM model wrapped in a custom PyModel.
    """
    # Sort predictors
    X_train = train_set[0][sorted(train_set[0].columns.to_list())]
    y_train = train_set[1]

    # Get the datatypes-map of predictors
    dfx = X_train.dtypes.reset_index()
    dfx[0] = dfx[0].astype(str)
    dtypes_map = dfx.groupby(0).agg(list).to_dict()["index"]

    # Retrive config parameters and data
    logger.info("Retrieving config params")

    config_setup = Config("../src/config/config_setup.yml")
    catalog = config_setup.get_config_value("environment.{env}.catalog".format(env=env))
    schema = config_setup.get_config_value("environment.{env}.schema".format(env=env))
    scoring_model_name = config_setup.get_config_value("modeling.model_scoring")["name"]
    catalog_scoring_model_path = f"{catalog}.{schema}.{scoring_model_name}"

    config_thresholds = Config("../src/config/config_thresholds.yml")
    threshold_red = config_thresholds.get_config_value("soglia_giallorossa")
    threshold_green = config_thresholds.get_config_value("soglia_verdegialla")

    config_model = Config("../src/config/config_model.yml")
    tracking_uri = config_model.get_config_value("mlflow_config.tracking_uri")
    training_eval_metrics = config_model.get_config_value(
        "training_params.eval_metrics"
    )
    lgb_params = config_model.get_config_value("training_params.lgb_fixed_params")

    run_name = config_model.get_config_value("mlflow_config.run_name.training")
    tags = config_model.get_config_value("mlflow_config.tags.training")

    config_data = Config("../src/config/config_data.yml")
    target_features = config_data.get_config_value("target_features")
    input_features = config_data.get_config_value("input_features")
    engineered_features = config_data.get_config_value("engineered_features")
    util_columns = config_data.get_config_value("util_columns")
    # not_predictors = config_data.get_config_value("not_predictors")

    # MlFlow settings
    logger.info("Setting mlflow configurations")
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(experiment_name=experiment_path)
    mlflow.autolog(
        # mlflow.lightgbm.autolog( # WARNING mlflow.utils.autologging_utils: You are using an unsupported version of lightgbm.
        # Autologging is known to be compatible with the package versions: 3.3.5 <= lightgbm <= 4.5.0. Autologging may not succeed when used with package versions outside of this range.
        # registered_model_name=catalog_scoring_model_path,
        log_models=False,  # log only the model wrapper
    )
    experiment_id = mlflow.get_experiment_by_name(experiment_path).experiment_id

    # Handle tunable params
    if best_params is not None:
        lgb_params.update(best_params)

    # Handle eval set
    if eval_set is None:
        actual_eval_set = train_set
    else:
        actual_eval_set = eval_set

    # Start mlflow logging
    logger.info("START MLFLOW TRAINING RUN")
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
        tags=tags,  # {"mlflow.note.content": desc_str},
    ) as run:
        run_id = run.info.run_id
        # Log params
        current_date = datetime.today().date()
        mlflow.log_param("run_date", str(current_date))
        mlflow.log_param("snapshot_date", str(current_date.replace(day=1)))
        mlflow.log_param("is_cicd", str(is_cicd))
        mlflow.log_param(
            "cliin_macrosegments",
            X_train["ENCOD_MACROSEGMENTO_CLIIN"].unique().tolist(),
        )
        mlflow.log_param("score_thresholds", (threshold_red, threshold_green))

        # Fit model
        logger.info("Training model with following parameters: {}".format(lgb_params))
        clf = lgb.LGBMClassifier(**lgb_params)
        clf.fit(
            X=X_train,
            y=y_train,
            eval_set=actual_eval_set,
            eval_metric=training_eval_metrics,
            categorical_feature=dtypes_map["category"],  #'auto',
        )

        # Get the Pyspark's datatype map
        pyspark_dtype_map = {}
        for diz in [target_features, input_features, engineered_features]:
            pyspark_dtype_map.update(
                {
                    c: d2["pyspark_type"]
                    for group, d1 in diz.items()
                    for c, d2 in d1.items()
                }
            )
        pyspark_dtype_map.update(
            {c: d1["pyspark_type"] for c, d1 in util_columns.items()}
        )

        # Manage missing columns
        missing_cols = [
            c for c in list(pyspark_dtype_map.keys()) if c not in X_train.columns
        ]
        pyspark_dtype_map = {
            c: d for c, d in pyspark_dtype_map.items() if c not in missing_cols
        }
        if len(missing_cols) > 0:
            logger.warning(
                f"The dataset in the 'fit_model' function does not have the configured 'config_data.yml' columns: {missing_cols}\n"
            )

        # Define model signature
        scoring_sign = DataModelSignature(
            predictors_dtype_map=pyspark_dtype_map
        ).get_model_signature()

        # Encapsulate the model in a custom wrapper
        clf_wrapped = ModelWrapper(
            model=clf,
            dtypes_map=dtypes_map,
            yellow_red_threshold=threshold_red,
            yellow_green_threshold=threshold_green,
        )

        # Register model in UC after cicd
        if is_cicd:
            logger.info("CICD: registering model on Databricks")
            mlflow.set_registry_uri(tracking_uri)
            client = mlflow.MlflowClient()  # call it after setting UC as registry URI

            # SPECIFIC lightgbm log code
            # mlflow.lightgbm.log_model(
            #     artifact_path="model",
            #     lgb_model=clf,
            #     registered_model_name=catalog_scoring_model_path,
            #     signature=scoring_sign,
            # )
            # GENERAL pyfunc log code
            mlflow.pyfunc.log_model(
                artifact_path="model/wrapper",
                python_model=clf_wrapped,
                registered_model_name=catalog_scoring_model_path,
                signature=scoring_sign,
                code_paths=["../src"],
                # infer_code_paths = True,
                # input_example={"POD_PDR": "12345678901234", "COD_FISCALE_PIVA": ...},
                # UserWarning: An input example was not provided when logging the model. To ensure the model signature functions correctly, specify the `input_example` parameter. See https://mlflow.org/docs/latest/model/signatures.html#model-input-example for more details about the benefits of using input_example.
            )
            model_version = get_latest_model_version(
                catalog_scoring_model_path, tracking_uri
            )
            logger.info(
                f"Lastest available version of challenger model is: {model_version}"
            )
            client.set_registered_model_alias(
                catalog_scoring_model_path, "challenger", model_version
            )

            # If champion does not exist promote the model
            champion_exist = check_if_alias_exist(
                catalog_scoring_model_path, "champion", tracking_uri
            )
            if not champion_exist:
                logger.info(
                    "There is no champion model -> promoting challenger model to champion"
                )
                client.set_registered_model_alias(
                    catalog_scoring_model_path, "champion", model_version
                )

        mlflow.end_run()
    return run_id, clf_wrapped
