from typing import Any, Dict
from datetime import datetime
import pandas as pd
from pandas import DataFrame as pandasDataFrame
from typing import Optional
import mlflow
from sklearn.metrics import roc_auc_score, accuracy_score  # , precision_score

from src.utils.evaluation import create_metrics_df, get_confusion_matrix_plot
from src.model.model_utils import load_deployed_model  # , load_model_from_run_id
from src.evaluation.model_evaluation import _risk_classification_barplot
from src.config.config_utils import Config

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def batch_inference(
    df_input: pandasDataFrame = None,
    target: str = "FLG_CLI_MOROSO",
    model_alias: str = "champion",
    is_cicd: bool = False,
    experiment_path: str = None,
    env: str = None,
    params: Optional[Dict[str, Any]] = None,
) -> (str, pandasDataFrame, pandasDataFrame, Any):
    """
    Function to do a prediction on a dataset with the Scoring model in a batch inference mode.
    Args:
        df_input (Pandas dataframe): the input dataset to predict.
        target (string): the name of the target column.
        model_alias (string): the alias of the model to use for prediction.
        is_cicd (boolean): a flag to enable the CICD deployment.
        experiment_path (string): MLflow experiment path-name.
        env (string): the name of environment where deploy the model.
        params (dictionary[string, any]): optional parameters on a dictionary with the following keys:
            - config_path (string): the folder's path of the configuration files.
    Returns:
        run_id (string): the run_id of the mlflow run.
        df_results (Pandas dataframe): the output of the inference.
        wrapped_scoring_model (Any): the wrapping of Scoring Model.
    """

    params = params or {"config_path": "../src/config"}
    config_path = params.get("config_path")

    # First retrive configuration parameters and data
    logger.info("Retrieving config params")

    config_setup = Config(f"{config_path}/config_setup.yml")
    catalog = config_setup.get_config_value("environment.{env}.catalog".format(env=env))
    schema = config_setup.get_config_value("environment.{env}.schema".format(env=env))
    scoring_model_name = config_setup.get_config_value("modeling.model_scoring")["name"]
    catalog_scoring_model_path = f"{catalog}.{schema}.{scoring_model_name}"
    model_uri = f"models:/{catalog_scoring_model_path}@{model_alias}"

    config_model = Config(f"{config_path}/config_model.yml")
    tracking_uri = config_model.get_config_value("mlflow_config.tracking_uri")

    run_name = "batch_inference_run"
    # descr = "Run to do a batch inference."

    # MlFlow settings
    logger.info("Setting mlflow configurations")
    current_date = datetime.today().date()
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(experiment_name=experiment_path)
    experiment_id = mlflow.get_experiment_by_name(experiment_path).experiment_id

    # Load model
    logger.info("Loading model and predicting")
    print(f"model_uri: {model_uri}\n")
    mlflow_wrap = load_deployed_model(
        model_uri=model_uri, as_spark_udf=False, result_type="long"
    )
    wrapped_scoring_model = mlflow_wrap.unwrap_python_model()
    raw_lgb_model = wrapped_scoring_model.model

    # Get model attributes
    dtypes_map = wrapped_scoring_model.dtypes_map
    predictors = [pp for sublist in dtypes_map.values() for pp in sublist]
    print(f"\nPredictors: n° {len(predictors)}\n", predictors, "\n")

    score_threshold_red = wrapped_scoring_model.yellow_red_threshold
    score_threshold_green = wrapped_scoring_model.yellow_green_threshold
    print("\nYellow-red threshold:", score_threshold_red)
    print("Yellow-green threshold:", score_threshold_green, "\n")

    # Dataset breakdown
    df_input = df_input.reset_index(drop=True).copy()
    sorted_predictors = sorted(predictors)
    X_input = df_input[sorted_predictors]
    y_input = df_input[target]
    Z_input = df_input[[z for z in df_input.columns if z not in predictors + [target]]]

    # Cast Pandas datatypes
    for dtype, cols in dtypes_map.items():
        if dtype in ["category", "bool", "float64"]:
            X_input[cols] = X_input[cols].astype(dtype)

    # Probabilistic prediction
    predict_probas = raw_lgb_model.predict_proba(X_input)
    df_proba = (
        pd.DataFrame(predict_probas)
        .rename(
            columns={
                0: "PROBA_0",
                1: "score",  # i.e. "PROBA_1"
            }
        )
        .reset_index(drop=True)
    )

    # Binary Prediction
    ind_red = df_proba["score"] >= score_threshold_red
    preds = ind_red.astype(int)
    df_proba["BINARY_PREDICTION"] = preds

    # Three-level risk Classification
    df_proba.loc[ind_red, "esitoSemaforico"] = "1 Semaforo Rosso"
    ind_green = df_proba["score"] < score_threshold_green
    df_proba.loc[ind_green, "esitoSemaforico"] = "3 Semaforo Verde"
    ind_yellow = df_proba["esitoSemaforico"].isnull()
    df_proba.loc[ind_yellow, "esitoSemaforico"] = "2 Semaforo Giallo"

    # Data Concatenation
    df_results = pd.concat(
        [y_input.reset_index(drop=True), df_proba, X_input, Z_input],
        axis=1,
    )

    # FILTERING OF CASES WITH PREVIOUS UNPAID CREDIT
    ind_insoluto = (
        (df_results["PDF_INSOLUTO_LORDO_ALL"] > 0)
        | (df_results["PDF_INSOLUTO_LORDO_LAST"] > 0)
        | (df_results["CLI_INSOLUTO_LORDO"] > 0)
        | (df_results["CLI_INSOLUTO_LORDO_CLIIN"] > 0)
    )

    # The case without unpaid credit have not to be predicted but they can free pass as "Semaforo Verde"
    df_results.loc[~ind_insoluto, "score"] = -1
    df_results.loc[~ind_insoluto, "esitoSemaforico"] = "3 Semaforo Verde"

    # Add a column of which model is used
    df_results["USED_MODEL"] = "SCORING"
    df_results.loc[~ind_insoluto, "USED_MODEL"] = "NO ONE"

    # Aggregation on Classification ----------------------------------------------------------------------------------------------------
    df_grouped = df_results.groupby(["esitoSemaforico", "FLG_CLI_MOROSO"]).agg(
        {
            "BINARY_PREDICTION": "count",
            "FLG_CLI_MOROSO": "sum",
            "MANCATI_INCASSI_RATE": "sum",
            "IMPORTI_INCASSATI": "sum",
        }
    )
    df_grouped = (
        df_grouped.rename(columns={"FLG_CLI_MOROSO": "OBSERVED"})
        .reset_index()
        .sort_values(by="FLG_CLI_MOROSO", ascending=False)
        .sort_values(by="esitoSemaforico")
    )
    df_grouped["SAVINGS"] = (
        df_grouped["MANCATI_INCASSI_RATE"] - df_grouped["IMPORTI_INCASSATI"]
    )
    df_grouped.loc[df_grouped["esitoSemaforico"] == "3 Semaforo Verde", "SAVINGS"] = 0
    # hypothesis that operators block all and only yellow customers actually observed as fraudulent defaulters
    df_grouped.loc[
        (df_grouped["esitoSemaforico"] == "2 Semaforo Giallo")
        & (df_grouped["FLG_CLI_MOROSO"] == 0),
        "SAVINGS",
    ] = 0
    df_grouped = df_grouped.groupby(["esitoSemaforico"]).agg(
        {"BINARY_PREDICTION": "sum", "OBSERVED": "sum", "SAVINGS": "sum"}
    )
    df_grouped = (df_grouped.astype(float)).round(0).astype(int)

    # grouped_data_map = {
    #     "SCORE THRESHOLD": {"RED": score_threshold_red, "GREEN": score_threshold_green},
    #     "PREDICTION": df_grouped["BINARY_PREDICTION"].to_dict(),
    #     "OBSERVED": df_grouped["OBSERVED"].to_dict(),
    #     "SAVINGS": df_grouped["SAVINGS"].to_dict(),
    # }
    df_grouped = df_grouped.rename(
        columns={"BINARY_PREDICTION": "PREDICTION"}
    ).reset_index()
    grouped_data_map = {
        "SCORE THRESHOLD": {"RED": score_threshold_red, "GREEN": score_threshold_green},
        "PREDICTION": {
            "1 Semaforo Rosso": 0,
            "2 Semaforo Giallo": 0,
            "3 Semaforo Verde": 0,
        },
        "OBSERVED": {
            "1 Semaforo Rosso": 0,
            "2 Semaforo Giallo": 0,
            "3 Semaforo Verde": 0,
        },
        "SAVINGS": {
            "1 Semaforo Rosso": 0,
            "2 Semaforo Giallo": 0,
            "3 Semaforo Verde": 0,
        },
    }
    for x in ["PREDICTION", "OBSERVED", "SAVINGS"]:
        grouped_data_map[x].update(df_grouped[x].to_dict())

    # Evaluate and plot performancers ----------------------------------------------------------------------------------------------------
    logger.info("Creating evaluation metrics")
    metrics_df = create_metrics_df(y_true=y_input, y_pred=preds)
    cm_plot = get_confusion_matrix_plot(y_input, preds)
    accuracy = accuracy_score(y_input, preds)
    auc_score = roc_auc_score(y_input, preds)

    try:
        risk_barplot = _risk_classification_barplot(grouped_data_map)
    except Exception as e:
        logger.error(f"Error in risk_barplot: {e}")
        risk_barplot = None

    # Logging in MLFlow experiment
    run_id = "no run"
    if is_cicd:
        logger.info("Logging params, metrics figures on MLFlow experiment")
        mlflow.set_registry_uri(tracking_uri)

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
            # tags={mlflow.note.content: descr},
        ) as run:
            run_id = run.info.run_id

            # Log params
            mlflow.log_param("snapshot_date", str(current_date.replace(day=1)))
            mlflow.log_param(
                "macrosegments_cliin",
                df_input["ENCOD_MACROSEGMENTO_CLIIN"].unique().tolist(),
            )
            mlflow.log_param("score_threshold_red", score_threshold_red)
            mlflow.log_param("score_threshold_green", score_threshold_green)

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("auc", auc_score)

            # loop over cells in metrics_df
            for index, row in metrics_df.iterrows():
                for column in metrics_df.columns:
                    cell_value = row[column]
                    mlflow.log_metric(f"{index}_{column}", cell_value)

            # Log figures
            mlflow.log_figure(cm_plot.figure_, "confusion_matrix.png")
            if risk_barplot is not None:
                mlflow.log_figure(risk_barplot, "risk_classification.png")

    # Clean the field 'esitoSemaforico' to remove the initial ordering number
    df_results["esitoSemaforico"] = df_results["esitoSemaforico"].str.replace(
        r"^\d+\s*", "", regex=True
    )
    return run_id, df_results, wrapped_scoring_model
