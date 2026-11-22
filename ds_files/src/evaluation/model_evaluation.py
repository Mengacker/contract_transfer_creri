"""Functions for getting evaluation metrics"""

from typing import Any, Dict, Optional
from datetime import datetime
import pandas as pd
from pandas import DataFrame as pandasDataFrame
import matplotlib.pyplot as plt
import mlflow
import tempfile
from sklearn.metrics import roc_auc_score, accuracy_score  # , precision_score
from lightgbm import plot_metric, plot_importance

from src.utils.evaluation import create_metrics_df, get_confusion_matrix_plot

from src.model.model_utils import load_deployed_model  # , load_model_from_run_id
from src.evaluation.model_thresholds import performance_table_by_proba_thresholds

from src.config.config_utils import Config

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _risk_classification_barplot(data_map: Dict[str, Any] = None) -> Any:
    """
    Function to plot a barplot that summarizes the risk classification of the model.
    Args:
        data_map (dictionary[string, any]): dictionary containing the data to plot.
    Returns:
        fig (Matplotlib figure): figure object containing the plot.
    """
    # Coordinates
    # labels = list(data_map["PREDICTION"].keys())
    labels = ["high", "medium", "low"]
    prediction = list(data_map["PREDICTION"].values())
    observation = list(data_map["OBSERVED"].values())
    economic = list(data_map["SAVINGS"].values())

    # Figure
    fig, axs = plt.subplots(1, figsize=(10, 5))
    bars = axs.barh(
        labels[::-1], prediction[::-1], color=["#00FF00", "#FFFF00", "#FF0000"]
    )
    axs.set_title("Risk Classification")
    axs.set_xlabel("n° of customers contract transfer re-proportioned to the month")
    axs.set_xscale("log")
    l_xtext = [b.get_width() for b in bars]
    min_width = min([x for x in l_xtext if x > 0])
    l_xtext = [min_width, l_xtext[1], l_xtext[0] * 0.1]
    n = len(bars) - 1
    for j, bar in enumerate(bars):
        i = n - j
        if labels[i] in ["high", "medium"]:
            s = f"{bar.get_width()} n°predicted ({observation[i]} n°observed, {economic[i]}€ hypothetical savings) "
        elif labels[i] in ["low"]:
            s = f"{bar.get_width()} n°predicted ({observation[i]} n°observed) "
        axs.text(
            x=l_xtext[i],
            y=bar.get_y() + bar.get_height() / 2,
            s=s,
            va="center",
        )
    for i, c in enumerate(["GREEN", "RED"]):
        st = data_map["SCORE THRESHOLD"][c]
        y = (i + 1) / 3
        axs.text(
            0.5,
            y,
            f"---------------------------------------- Threshold Score = {st} ---------------------------------------",
            horizontalalignment="center",
            transform=plt.gca().transAxes,
        )
    return fig


def evaluate_model_on_test_set(
    df_test: pandasDataFrame = None,
    target: str = "FLG_CLI_MOROSO",
    predictors: list[str] = [],
    model_alias: str = "champion",  # training_run_id,
    is_cicd: bool = False,
    experiment_path: str = None,
    env: str = None,
    params: Optional[Dict[str, Any]] = None,
) -> (str, pandasDataFrame, pandasDataFrame, Any):
    """
    Method to evaluate the performance of a model on a test dataset.
    Args:
        df_test (Pandas DataFrame): dataset to test.
        target (string): the name of the target column.
        predictors (list[string]): the list of predictor names.
        model_alias (string): alias of the model to use.
        is_cicd (boolean): flag to indicate if the method is called from a CI/CD pipeline or not.
        experiment_path (string): MLflow experiment path-name.
        env (string): the name of the environment.
        params (dictionary[string, any]): optional parameters on a dictionary with the following keys:
            - proportion_coeff (float): proportionality coefficient.

    Returns:
        run_id (string): MLflow run ID of the evaluation.
        df_results (Pandas DataFrame): a dataframe containing the input data and the evaluation results.
        df_thresholds (Pandas DataFrame): a table reporting some performance metrics (precision, lift, true positive, false negative, percentile, etc.), as the classification threshold varies.
        wrapped_scoring_model (Spark UDF): the scoring model wrapped as pyfunc.
    """
    params = params or {"proportion_coeff": 1}
    proportion_coeff = params.get("proportion_coeff") or 1

    # First retrive configuration parameters and data
    logger.info("Retrieving config params")

    config_setup = Config("../src/config/config_setup.yml")
    catalog = config_setup.get_config_value("environment.{env}.catalog".format(env=env))
    schema = config_setup.get_config_value("environment.{env}.schema".format(env=env))
    scoring_model_name = config_setup.get_config_value("modeling.model_scoring")["name"]
    catalog_scoring_model_path = f"{catalog}.{schema}.{scoring_model_name}"
    model_uri = f"models:/{catalog_scoring_model_path}@{model_alias}"

    config_thresholds = Config("../src/config/config_thresholds.yml")
    score_threshold_red = config_thresholds.get_config_value("soglia_giallorossa")
    score_threshold_green = config_thresholds.get_config_value("soglia_verdegialla")

    config_model = Config("../src/config/config_model.yml")
    tracking_uri = config_model.get_config_value("mlflow_config.tracking_uri")
    run_name = config_model.get_config_value("mlflow_config.run_name.evaluation")
    tags = config_model.get_config_value("mlflow_config.tags.evaluation")

    # MlFlow settings
    logger.info("Setting mlflow configurations")
    current_date = datetime.today().date()
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(experiment_name=experiment_path)
    experiment_id = mlflow.get_experiment_by_name(experiment_path).experiment_id

    # Load model
    logger.info("Loading model and predicting on test set")
    mlflow_wrap = load_deployed_model(
        model_uri=model_uri, as_spark_udf=False, result_type="long"
    )
    wrapped_scoring_model = mlflow_wrap.unwrap_python_model()
    raw_lgb_model = wrapped_scoring_model.model
    # dtypes_map = wrapped_scoring_model.dtypes_map

    # Dataset breakdown
    df_test = df_test.reset_index(drop=True).copy()
    sorted_predictors = sorted(predictors)
    X_test = df_test[sorted_predictors]
    y_test = df_test[target]
    Z_test = df_test[[z for z in df_test.columns if z not in predictors + [target]]]

    # Probabilistic prediction
    predict_probas = raw_lgb_model.predict_proba(X_test)
    df_proba = (
        pd.DataFrame(predict_probas)
        .rename(
            columns={
                0: "PROBA_0",
                1: "PROBA_1",  # i.e. the score
            }
        )
        .reset_index(drop=True)
    )

    y_proba = pd.concat([y_test, df_proba["PROBA_1"]], axis=1)

    # Calculate performance metrics and score/precision thresholds
    df_thresholds, thresholds_map = performance_table_by_proba_thresholds(
        data=y_proba,
        target_col="FLG_CLI_MOROSO",
        proba_col="PROBA_1",
        params={
            "proportion_coeff": proportion_coeff,
            "score_threshold_red": score_threshold_red,  # enabled
            "score_threshold_green": score_threshold_green,  # enabled
        },
    )
    score_threshold_red = thresholds_map["SCORE THRESHOLD"]["RED"]
    score_threshold_green = thresholds_map["SCORE THRESHOLD"]["GREEN"]

    # Binary Prediction
    ind_red = df_proba["PROBA_1"] >= score_threshold_red
    preds = ind_red.astype(int)
    df_proba["PREDICTION"] = preds

    # Three-level risk Classification, i.e. 'esitoSemaforico'
    df_proba.loc[ind_red, "CLASSIFICATION"] = "1 RED"
    ind_green = df_proba["PROBA_1"] < score_threshold_green
    df_proba.loc[ind_green, "CLASSIFICATION"] = "3 GREEN"
    ind_yellow = df_proba["CLASSIFICATION"].isnull()
    df_proba.loc[ind_yellow, "CLASSIFICATION"] = "2 YELLOW"

    # Data Concatenation
    df_results = pd.concat(
        [y_test.reset_index(drop=True), df_proba, X_test, Z_test],
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
    df_results.loc[~ind_insoluto, "PROBA_1"] = -1
    df_results.loc[~ind_insoluto, "CLASSIFICATION"] = "3 GREEN"

    # Add a column of which model is used
    df_results["USED_MODEL"] = "SCORING"
    df_results.loc[~ind_insoluto, "USED_MODEL"] = "NO ONE"

    # Aggregation on Classification ----------------------------------------------------------------------------------------------------
    df_grouped = df_results.groupby(["CLASSIFICATION", "FLG_CLI_MOROSO"]).agg(
        {
            "PREDICTION": "count",
            "FLG_CLI_MOROSO": "sum",
            "MANCATI_INCASSI_RATE": "sum",
            "IMPORTI_INCASSATI": "sum",
        }
    )
    df_grouped = (
        df_grouped.rename(columns={"FLG_CLI_MOROSO": "OBSERVED"})
        .reset_index()
        .sort_values(by="FLG_CLI_MOROSO", ascending=False)
        .sort_values(by="CLASSIFICATION")
    )
    df_grouped["SAVINGS"] = (
        df_grouped["MANCATI_INCASSI_RATE"] - df_grouped["IMPORTI_INCASSATI"]
    )
    df_grouped.loc[df_grouped["CLASSIFICATION"] == "3 GREEN", "SAVINGS"] = 0
    # hypothesis that operators block all and only yellow customers actually observed as fraudulent defaulters
    df_grouped.loc[
        (df_grouped["CLASSIFICATION"] == "2 YELLOW")
        & (df_grouped["FLG_CLI_MOROSO"] == 0),
        "SAVINGS",
    ] = 0
    df_grouped = df_grouped.groupby(["CLASSIFICATION"]).agg(
        {"PREDICTION": "sum", "OBSERVED": "sum", "SAVINGS": "sum"}
    )
    df_grouped = (df_grouped.astype(float) * proportion_coeff).round(0).astype(int)

    grouped_data_map = {
        "SCORE THRESHOLD": {"RED": score_threshold_red, "GREEN": score_threshold_green},
        "PREDICTION": df_grouped["PREDICTION"].to_dict(),
        "OBSERVED": df_grouped["OBSERVED"].to_dict(),
        "SAVINGS": df_grouped["SAVINGS"].to_dict(),
    }

    # Evaluate and plot performancers ----------------------------------------------------------------------------------------------------
    logger.info("Creating evaluation metrics on test set")
    metrics_df = create_metrics_df(y_true=y_test, y_pred=preds)
    cm_plot = get_confusion_matrix_plot(y_test, preds)
    accuracy = accuracy_score(y_test, preds)
    auc_score = roc_auc_score(y_test, preds)

    # Performance plots
    fitting_metric = plot_metric(raw_lgb_model)  # booster must be dict or LGBMModel
    fig_height = X_test.shape[1] / 6
    importance_gain = plot_importance(
        raw_lgb_model,  # must be Booster or LGBMModel
        importance_type="gain",
        title="features importance by gain",
        figsize=(25, fig_height),
    )
    importance_split = plot_importance(
        raw_lgb_model,  # must be Booster or LGBMModel
        importance_type="split",
        title="features importance by split",
        figsize=(25, fig_height),
    )
    try:
        risk_barplot = _risk_classification_barplot(grouped_data_map)
    except Exception as e:
        logger.error(f"Error in plotting risk barplot: {e}")
        risk_barplot = None

    # Logging in MLFlow experiment
    run_id = "no run"
    if is_cicd:
        logger.info(
            "Logging model, params, metrics, figures and artifacts on MLFlow experiment"
        )
        mlflow.set_registry_uri(tracking_uri)

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
            tags=tags,  # {"mlflow.note.content": desc_str},
        ) as run:
            run_id = run.info.run_id

            # Log params
            mlflow.log_param("snapshot_date", str(current_date.replace(day=1)))
            mlflow.log_param(
                "macrosegments_cliin",
                df_test["ENCOD_MACROSEGMENTO_CLIIN"].unique().tolist(),
            )
            mlflow.log_param("score_threshold_red", score_threshold_red)
            mlflow.log_param("score_threshold_green", score_threshold_green)
            mlflow.log_param("proportion_coeff", proportion_coeff)
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("auc", auc_score)

            # loop over cells in metrics_df
            for index, row in metrics_df.iterrows():
                for column in metrics_df.columns:
                    cell_value = row[column]
                    mlflow.log_metric(f"{index}_{column}", cell_value)

            # Log figures
            mlflow.log_figure(fitting_metric.get_figure(), "fitting_metrics.png")
            mlflow.log_figure(
                importance_gain.get_figure(), "features_importance_by_gain.png"
            )
            mlflow.log_figure(
                importance_split.get_figure(), "features_importance_by_split.png"
            )
            mlflow.log_figure(cm_plot.figure_, "confusion_matrix.png")
            if risk_barplot is not None:
                mlflow.log_figure(risk_barplot, "risk_classification.png")

            # Log artifacts
            with tempfile.TemporaryDirectory() as tmpdir:
                fname_csv = tmpdir + "/thresholds_performance_table.csv"
                df_thresholds.to_csv(fname_csv, index=False)
                mlflow.log_artifact(fname_csv)

    # Clean the field 'CLASSIFICATION' to remove the initial ordering number
    df_results["CLASSIFICATION"] = df_results["CLASSIFICATION"].str.replace(
        r"^\d+\s*", "", regex=True
    )
    return run_id, df_results, df_thresholds, wrapped_scoring_model
