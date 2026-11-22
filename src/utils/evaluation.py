"""Utilities functioncs for evaluating a model"""

from typing import Any, List, Optional
import pandas as pd
from pandas import DataFrame as pandasDataFrame, Series as pandasSeries
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay,
)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_metrics_df(y_true: pandasSeries, y_pred: pandasSeries) -> pandasDataFrame:
    """
    Function to create a dataframe with classification metrics like precision, recall, fscore and support for each class.
    Args:
        y_true (Pandas series): the dataset of observed labels.
        y_pred (Pandas series): the dataset of predicted labels.
    Returns:
        metrics_df (Pandas dataframe): a dataframe with classification metrics like precision, recall, fscore and support of each class.
    """
    metrics = precision_recall_fscore_support(y_true, y_pred)
    metrics_df = pd.DataFrame(
        [m for m in metrics],
        index=["precision", "recall", "fscore", "support"],
        columns=["target_0", "target_1"],
    )
    metrics_df["avg"] = metrics_df.mean(axis=1)
    return metrics_df


def get_confusion_matrix_plot(
    y_true: pandasSeries,
    y_pred: pandasSeries,
    display_labels: Optional[List[str]] = ["0", "1"],
) -> Any:
    """
    Function to elaborate the confusion matrix plot.
    Args:
        y_true (Pandas series): the dataset of observed labels.
        y_pred (Pandas series): the dataset of predicted labels.
        display_labels (list[string]): the list of labels to display in the plot. Default is ["0", "1"].
    Returns:
        cm_plot (any): the confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_plot = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=display_labels
    ).plot()
    return cm_plot
