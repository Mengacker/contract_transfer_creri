from mlflow.pyfunc import PythonModel
import pandas as pd
from typing import Any, Dict, List


class ModelWrapper(PythonModel):
    """
    Class to wrap any model's flavor in to a PyFunc-PythonModel, so that MLflow can deploy it to the registry.
    The raw binary classification model is then wrapped to become the Scoring model that outputs the classification prediction probability.
    """

    def __init__(
        self,
        model: PythonModel,
        dtypes_map: Dict[str, list[str]],
        yellow_red_threshold: float,
        yellow_green_threshold: float,
    ):
        """
        Initialization method to load into the serving script the trained scoring model and its data-types map. Moreover, a score-tresholds are also loaded.
        Attributes:
            model (PythonModel): the Scoring model.
            dtypes_map (dictionary[string, list[string]]): a dictionary with the datatype as keys and the list of the predictors as values.
            yellow_red_threshold (float): the score threshold for classifying between red and yellow traffic lights.
            yellow_green_threshold (float): the score threshold for classifying between yellow and green traffic lights.
        """
        self.model = model
        self.dtypes_map = dtypes_map
        self.yellow_red_threshold = yellow_red_threshold
        self.yellow_green_threshold = yellow_green_threshold

    def _pre_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method to change the Pandas datatype of categorial variables from object to category.
        It is necessary to respect the datatypes of the trained model so this can work.
        Args:
            df (Pandas dataframe): a dataset to pre-process.
        Returns:
            df (Pandas dataframe): the pre-processed dataframe.
        """
        # categories_cols = self.dtypes_map["category"]
        # df[categories_cols] = df[categories_cols].astype("category")

        for dtype, cols in self.dtypes_map.items():
            if dtype in ["category", "bool", "float64"]:
                df[cols] = df[cols].astype(dtype)
        return df

    def predict(self, context: Any, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Method to execute the prediction of classification probability on the input data.
        Args:
            context (any): context of the model (optional argument not used in this case).
            model_input (Pandas dataframe): input dataset for the model.
        Returns:
            predict_proba (Pandas dataframe): the model's classification probability.
        """
        # Data preparation
        model_input = self._pre_processing(model_input)

        # Execute prediction
        return self.model.predict_proba(model_input)

    def get_pandas_dtypes_map(self) -> Dict[str, List[str]]:
        """
        Method to get the Pandas datatypes map of the predictors.
        Returns:
            dtypes_map (dictionary): Pandas datatypes map of the predictors.
        """
        return self.dtypes_map

    def get_score_thresholds(self) -> (float, float):
        """
        Method to get the thresholds to categorize the score into a color-level risk.
        Returns:
            yellow_red_threshold (float): the threshold to categorize the score into a red or yellow color-level risk.
            yellow_green_threshold (float): the threshold to categorize the score into a yellow or green color-level risk.
        """
        return self.yellow_red_threshold, self.yellow_green_threshold
