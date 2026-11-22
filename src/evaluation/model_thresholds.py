"""Functions for getting evaluation metrics"""

import logging
import pandas as pd
from pandas import Series as pandasSeries
from pandas import DataFrame as pandasDataFrame
import numpy as np
from typing import Any, Dict, Optional

from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
)  # , accuracy_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _classification_metrics_creator(
    y_true: pandasSeries, y_pred: pandasSeries
) -> pandasDataFrame:
    """
    Function to create a Pandas dataframe with the classification metrics.
    Args:
        y_true (Pandas Series): a dataset of the True labels of a target categorical variable
        y_pred (Pandas Series): a dataset of the Predicted labels of a target categorical variable
    Returns:
        metrics_df (Pandas DataFrame): a dataframe with the classification metrics
    """
    metrics = precision_recall_fscore_support(y_true, y_pred)
    metrics_df = pd.DataFrame(
        [m for m in metrics],
        index=["precision", "recall", "fscore", "support"],
        columns=["target_0", "target_1"],
    )
    metrics_df["avg"] = metrics_df.mean(axis=1)
    return metrics_df


def performance_table_by_proba_thresholds(
    data: pandasDataFrame,
    target_col: str,
    proba_col: str,
    params: Optional[Dict[str, Any]],
) -> (pandasDataFrame, Dict[str, Dict[str, float]]):
    """
    Function to create a dataframe with classification metrics values ​​based on the variation of the probability threshold.
    Args:
        data (Pandas DataFrame): a dataset to evaluate.
        target_col (string): the name of target column with the True labels.
        proba_col (string): the name of column with the prediction probability.
        params (dictionary[string, any]): optional parameters on a dictionary with the following keys:
            - proportion_coeff (float): proportionality coefficient.
            - score_threshold_red (float): probability threshold to define the RED risk class
            - score_threshold_green (float): probability threshold to define the GREEN risk class
    Returns:
        df_results (Pandas DataFrame): a dataframe with classification metrics values ​​based on the variation of the probability threshold.
        results_map (dictionary[string, dictionary[string, float]]): a dictionary with the following keys:
            - score_threshold_red (float): probability threshold to define the RED risk class
            - score_threshold_green (float): probability threshold to define the GREEN risk class
            - precision_threshold_red (float): precision threshold to define the RED risk class
            - precision_threshold_green (float): precision threshold to define the GREEN risk class
    """
    params = params or {
        "proportion_coeff": 1,
        "score_threshold_red": 0.8,
        "score_threshold_green": 0.1,
        # "precision_threshold_red": None,
        # "precision_threshold_green": None,
    }
    proportion_coeff = params.get("proportion_coeff") or 1
    score_threshold_red = params.get("score_threshold_red") or 0.8
    score_threshold_green = params.get("score_threshold_green") or 0.1
    # precision_threshold_red = params.get("precision_threshold_red") or None
    # precision_threshold_green = params.get("precision_threshold_green") or None

    y_true = data[target_col]
    n_total = data.shape[0]
    l_results = []
    l_threshold = np.arange(0, 1.001, 0.001)
    for threshold in l_threshold:
        # Classify predictions by threshold
        ind = data[proba_col] > threshold
        y_pred = ind.astype(int)
        percentile = 100 * sum(ind) / n_total

        # Create metrics
        df_results = _classification_metrics_creator(y_true=y_true, y_pred=y_pred)
        # accuracy = accuracy_score(y_true, y_pred)

        # Select focus metric
        metric = "precision"
        df_reshaped = df_results.T.rename(
            index={
                "target_0": metric + "_t0",
                "target_1": metric + "_t1",
                "avg": metric + "_avg",
            }
        )
        # Confusion matrix
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
        TN = cm[0][0]
        FN = cm[1][0]
        FP = cm[0][1]
        TP = cm[1][1]
        cm = {
            "VOLT.INS.TOT.": int(round(n_total * proportion_coeff, 0)),
            "P True": int(round((TP + FN) * proportion_coeff, 0)),
            "TP": int(round((TP) * proportion_coeff, 0)),
            "FN": int(round((FN) * proportion_coeff, 0)),
            "P Pred": int(round((TP + FP) * proportion_coeff, 0)),
            "%P Pred/True": round(100 * ((TP + FP) / (TP + FN)), 2),
        }
        # Lift elaboration
        lift_t1 = round((TP / (TP + FP)) / ((TP + FN) / (TP + FP + FN + TN)), 2)

        # Append to results
        dict_results = {
            "score threshold": threshold,
            "% percentile": percentile,
        }  # , "accuracy": accuracy}
        dict_results.update(
            {metric + "_t1": df_reshaped.loc[:, metric][metric + "_t1"]}
        )  # Select focus metric # df_reshaped.loc[:, metric].to_dict())
        dict_results.update({"lift_t1": lift_t1})
        dict_results.update(cm)
        l_results.append(dict_results)

    df_results = pd.DataFrame(l_results)
    df_results.sort_values(by="score threshold", ascending=False, inplace=True)

    # Score/Precision Thresholds selection and Risk Classification

    # priori_threshold
    l_score_trh = [score_threshold_red, score_threshold_green]
    # l_precision_trh = [precision_threshold_red, precision_threshold_green]

    # if (score_threshold_red is None) & (score_threshold_green is None):
    #     metric_a_priori = "precision"
    #     metric_to_find = "score"
    #     source_column = "precision_t1"
    #     objective_column = "score threshold"
    #     l_priori_trh = l_precision_trh
    #     l_found_trh = []
    # else:
    metric_a_priori = "score"
    metric_to_find = "precision"
    source_column = "score threshold"
    objective_column = "precision_t1"
    l_priori_trh = l_score_trh
    l_found_trh = []

    for i, color in enumerate(["RED", "GREEN"]):
        ind = df_results[source_column] >= l_priori_trh[i]
        if sum(ind) > 0:
            found_trh = df_results.loc[ind, objective_column].min()
            l_found_trh.append(found_trh)
            logger.info(
                f"The {metric_a_priori} {color} threshold {l_priori_trh[i]} corresponds to the {metric_to_find} {color} threshold {found_trh}"
            )
        else:
            l_found_trh.append(None)
            logger.warning(
                f"There isn't any {metric_to_find} {color} threshold because the {metric_a_priori} {color} threshold {l_priori_trh[i]} is not reached"
            )

    # if (score_threshold_red is None) & (score_threshold_green is None):
    #     score_threshold_red = l_found_trh[0]
    #     score_threshold_green = l_found_trh[1]
    # else:
    precision_threshold_red = l_found_trh[0]
    precision_threshold_green = l_found_trh[1]

    results_map = {
        "SCORE THRESHOLD": {"RED": score_threshold_red, "GREEN": score_threshold_green},
        "PRECISION THRESHOLD": {
            "RED": precision_threshold_red,
            "GREEN": precision_threshold_green,
        },
    }

    return df_results, results_map
