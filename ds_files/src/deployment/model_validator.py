from pandas import DataFrame as pandasDataFrame
import operator
from typing import Any, Dict

# from src.utils.mlflow_utils import load_deployed_model
from src.model.model_utils import load_deployed_model

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelValidator:
    """
    Class to validate the Challenger model's performance, and then compare it against the Champion model's performance.
    """

    def __init__(
        self,
        metric_name: str,
        metric_fn: Any,
        greater_is_better: bool,
        comparison_params: Dict[str, Any],
    ):
        """
        Initialization method to load metrics and evaluation criteria in the class.

        Arguments:
            metric_name (string): The name of the metric used for comparison.
            metric_fn (any): The function that calculates the metric.
            greater_is_better (boolean): Indicates if a higher metric value is the best (or the opposite).
            comparison_params (dictionary[string, any]): Parameters for comparison including min_absolute_metric, min_relative_change, and min_absolute_change.

        Attributes:
            min_absolute_metric (float): Minimum acceptable value (threshold) of the metric for the Challenger model.
            min_relative_change (float): The minimum relative change in the metric value required for the Challenger model.
            min_absolute_change (float): The minimum absolute change in the metric value required for the Challenger model.
            comparator_fn (function): Operator function for comparison.
            validator_report (dictionary[string, any]): Report of the comparison results including parameters, metrics, and validation outputs.
        """
        self.metric_name = metric_name
        self.metric_fn = metric_fn
        self.greater_is_better = greater_is_better

        self.min_absolute_metric = comparison_params["min_absolute_metric"]
        self.min_relative_change = comparison_params["min_relative_change"]
        self.min_absolute_change = comparison_params["min_absolute_change"]
        self.comparator_fn = (
            operator.__ge__ if self.greater_is_better else operator.__le__
        )
        self.validator_report = {
            "params": {},
            "metrics": {
                "metric_champion": 0,
                "metric_challenger": 0,
            },
            "validation_outputs": {
                "is_predictor_list_changed": False,
                "is_score_thresholds_changed": False,
                "is_minimum_absolute_passed": False,
                "is_min_absolute_change_passed": False,
                "is_min_relative_change_passed": False,
                "is_all_comparison_criteria_passed": False,
            },
        }
        self.validator_report["params"] = {
            "metric_name": self.metric_name,
            "min_absolute_metric": self.min_absolute_metric,
            "min_relative_change": self.min_relative_change,
            "min_absolute_change": self.min_absolute_change,
            "greater_is_better": self.greater_is_better,
            "predictors_added": [None],
            "predictors_removed": [None],
            "score_thresholds_of_champion": (),
            "score_thresholds_of_challenger": (),
        }

    def _predict_comparison(
        self,
        eval_data: pandasDataFrame,
        target: str,
        predictors: list[str] = [],
        predict_arg: dict = None,
    ) -> bool:
        """
        Method to do prediction on evaluation data of the two models, Champion and Challenger, then to evaluate and to compare the performance of both.
        Args:
            eval_data (pandasDataFrame): the evaluation dataset.
            target (string): the name of the target column.
            predictors (list[string]): the list of predictor names.
            predict_arg (dictionary): optional arguments for prediction.
        Returns:
            flag_promote (boolean): True if the Challenger model is better than the Champion model, False otherwise.
        """

        # Dataset breakdown
        eval_data = eval_data.reset_index(drop=True).copy()
        X_eval = eval_data[predictors]
        y_eval = eval_data[target]

        if predict_arg is None:
            eval_data["pred_champion"] = self.champion_model.predict(X_eval)
            eval_data["pred_challenger"] = self.challenger_model.predict(X_eval)
        else:
            eval_data["pred_champion"] = self.champion_model.predict(
                X_eval, params=predict_arg
            )
            eval_data["pred_challenger"] = self.challenger_model.predict(
                X_eval, params=predict_arg
            )

        # Get metric
        metric_champion = self.metric_fn(y_eval, eval_data["pred_champion"])
        metric_challenger = self.metric_fn(y_eval, eval_data["pred_challenger"])
        self.validator_report["metrics"]["metric_champion"] = metric_champion
        self.validator_report["metrics"]["metric_challenger"] = metric_challenger

        # Get validation outputs
        validation_outputs = {}
        if isinstance(self.min_absolute_metric, float):
            validation_outputs["is_minimum_absolute_passed"] = self.comparator_fn(
                metric_challenger, self.min_absolute_metric
            )

        if isinstance(self.min_absolute_change, float):
            abs_change = metric_challenger - metric_champion
            validation_outputs["is_min_absolute_change_passed"] = self.comparator_fn(
                abs_change, self.min_absolute_change
            )

        if isinstance(self.min_relative_change, float):
            relative_change = (metric_challenger - metric_champion) / metric_champion
            validation_outputs["is_min_relative_change_passed"] = self.comparator_fn(
                relative_change, self.min_relative_change
            )

        validation_outputs["is_all_comparison_criteria_passed"] = (
            True if all(validation_outputs.values()) else False
        )

        self.validator_report["validation_outputs"].update(validation_outputs)
        flag_promote = self.validator_report["validation_outputs"][
            "is_all_comparison_criteria_passed"
        ]
        return flag_promote

    def compare_models(
        self,
        champion_model_uri: str,
        challenger_model_uri: str,
        eval_data: pandasDataFrame,
        target: str,
        predictors: list[str] = [],
        predict_arg: dict = None,
    ) -> bool:
        """
        Method to evaluate the performance of two models, a Champion and a Challenger, and then determine if the Challenger model should be promoted as a Champion, or not.

        Args:
            champion_model_uri (string): URI of the Champion model.
            challenger_model_uri (string): URI of the Challenger model.
            eval_data (Pandas DataFrame): the evaluation dataset.
            target (string): the name of the target column.
            predictors (list[string]): the list of predictor names.
            predict_arg (dictionary): optional arguments for prediction.
        Returns:
            flag_promote (boolean): True if the Challenger model is better than the Champion model, False otherwise.
        """

        # Inizialization
        flag_promote = False
        logger.info(
            f"ModelValidator is comparing models by the metric '{self.metric_fn}'."
        )

        # Load champion model
        champion_mlflow_wrap = load_deployed_model(
            champion_model_uri, as_spark_udf=False
        )
        champion_wrapped_scoring_model = champion_mlflow_wrap.unwrap_python_model()
        self.champion_model = champion_wrapped_scoring_model.model
        self.champion_dtypes_map = (
            champion_wrapped_scoring_model.get_pandas_dtypes_map()
        )
        self.validator_report["params"]["score_thresholds_of_champion"] = (
            champion_wrapped_scoring_model.get_score_thresholds()
        )

        # Load challenger model
        challenger_mlflow_wrap = load_deployed_model(
            challenger_model_uri, as_spark_udf=False
        )
        challenger_wrapped_scoring_model = challenger_mlflow_wrap.unwrap_python_model()
        self.challenger_model = challenger_wrapped_scoring_model.model
        self.challenger_dtypes_map = (
            challenger_wrapped_scoring_model.get_pandas_dtypes_map()
        )
        self.validator_report["params"]["score_thresholds_of_challenger"] = (
            challenger_wrapped_scoring_model.get_score_thresholds()
        )

        # Comparison of predictors list
        set_champ = {c for d, ll in self.champion_dtypes_map.items() for c in ll}
        set_chall = {c for d, ll in self.challenger_dtypes_map.items() for c in ll}
        if set_champ != set_chall:
            self.validator_report["validation_outputs"]["is_predictor_list_changed"] = (
                True
            )
            self.validator_report["params"]["predictors_added"] = list(
                set_chall - set_champ
            )
            self.validator_report["params"]["predictors_removed"] = list(
                set_champ - set_chall
            )
            flag_promote = True

        # Comparison of score thresholds values
        if (
            self.validator_report["params"]["score_thresholds_of_champion"]
            != self.validator_report["params"]["score_thresholds_of_challenger"]
        ):
            self.validator_report["validation_outputs"][
                "is_score_thresholds_changed"
            ] = True
            flag_promote = True

        # Comparison of prediction performances
        if not flag_promote:
            flag_promote = self._predict_comparison(
                eval_data, target, predictors, predict_arg
            )

        return flag_promote

    def get_report(self) -> Dict[str, Any]:
        """
        Method to get the report of model's validation.

        Returns:
            validator_report (dictionary[string, Any]): the validator report containing comparison results.
        """
        return self.validator_report

    # def get_models_datatype_maps(self) -> (Dict[str, list[string]], Dict[str, list[string]]):
    #     """
    #     Method to get the Pandas datatypes map of both, Champion and Challenger, models.

    #     Returns:
    #         champion_dtypes_map (dictionary[string, list[string]]): the Pandas datatypes map of the Champion model's predictors.
    #         challenger_dtypes_map (dictionary[string, list[string]]): the Pandas datatypes map of the Challenger model's predictors.
    #     """
    #     return self.champion_dtypes_map, self.challenger_dtypes_map

    # def get_models_score_thresholds(self) -> (tuple[float, float], tuple[float, float]):
    #     """
    #     Method to get the thresholds to categorize the score into a color-level risk of both, Champion and Challenger, models.

    #     Returns:
    #         champion_score_thresholds (tuple[float, float]): the Champion model's thresholds to categorize the score into a color-level risk.
    #         challenger_score_thresholds (tuple[float, float]): the Challenger model's thresholds to categorize the score into a color-level risk.
    #     """
    #     champion_score_thresholds = self.validator_report["params"]["score_thresholds_of_champion"]
    #     challenger_score_thresholds = self.validator_report["params"]["score_thresholds_of_challenger"]
    #     return champion_score_thresholds, challenger_score_thresholds
