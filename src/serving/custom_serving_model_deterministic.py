import mlflow
import json
import pandas as pd
from typing import Optional, Any, Dict, List

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class CustomServingModel(mlflow.pyfunc.PythonModel):
    """
    Class to wrap any model's flavor in to a PyFunc-PythonModel, so that MLflow can deploy it to the registry.
    A logic applied to the yellow list of POD/PDR is then wrapped to become a binary-classification model called the Deterministic model.
    The Deterministic model outputs two case of a traffic light classification (green or yellow) and its associated binary score.
    """

    def __init__(self, model_name: str, registry_uri: str, yellow_list: List[str]):
        """
        Initialization method to load the yellow list of POD/PDR into the serving script.
        Args:
            model_name (string): name of the model.
            registry_uri (string): registry_uri of the model.
            yellow_list (list[string]): list of POD/PDR tagged as defaulters.
        """
        self.model_name = model_name
        self.registry_uri = registry_uri
        self.yellow_list = yellow_list

    def load_context(self, context: Optional[Any]) -> None:
        """
        Load context method has to be in the class.
        Args:
            context (any): context of the model (optional argument not used in this case).
        """
        logger.info("load_context")

    def _pre_processing(self, df_pandas: pd.DataFrame):
        """
        Method to pre-process the user's input, hence, the input json file is transformed and reshaped into a Pandas dataframe.
        Afterward, some fields are renamed and some other fields are created.
        Args:
            df_pandas (Pandas dataframe): user's input.
        Returns:
            df (Pandas dataframe): a dataset with the "POD_PDR" column.
        """
        values_stringa = df_pandas["INPUT_JSON"].values[0]  # <class 'str'>
        values_stringa = values_stringa.replace("\\'", "'")  # clean string
        dict_input = json.loads(values_stringa)  # <class 'dict'>

        assert isinstance(
            dict_input, dict
        ), "The json input not contain a string of a dictionary"

        # Reshape input data for several forniture
        dict_no_forn = {k: v for k, v in dict_input.items() if k != "Forniture"}
        forn_list_dict = dict_input["Forniture"]
        assert isinstance(
            forn_list_dict, list
        ), "The json input not contain a Forniture's list"
        list_of_forn_dict = []
        for ee in forn_list_dict:
            assert isinstance(ee, dict), "A Forniture's element is not a dictionary"
            ee_bis = ee.copy()
            ee_bis.update(dict_no_forn.copy())
            list_of_forn_dict.append(ee_bis)
        assert len(forn_list_dict) > 0, logger.warning("Forniture list is empty")

        df = pd.DataFrame(list_of_forn_dict)
        df.rename(columns={"POD/PDR": "POD_PDR"}, inplace=True)

        return df

    def _deterministic_model(self, df_pandas: pd.DataFrame) -> dict:
        """
        Method to apply the Deterministic model, i.e. the deterministic logic, which checks if the POD/PDR are present in the "yellow list", if so the contract transfer has a yellow traffic light, otherwise the traffic light is green.
        Args:
            df_pandas (Pandas dataframe): dataset with the "POD_PDR" column.
        Returns:
            dict_esito (dictionary[string, any]): dictionary with the output of the model's prediction:
             - risk_class (string): the traffic light risk-classification of the Deterministic model.
             - score (float): the score of the Deterministic model.
        """
        lista_pod_pdr = df_pandas["POD_PDR"].tolist()
        try:
            for pod_pdr in lista_pod_pdr:
                if pod_pdr in self.yellow_list:
                    risk_class = "Semaforo Giallo"
                    score = 1
                    break
                else:
                    risk_class = "Semaforo Verde"
                    score = 0

        except Exception as e:
            risk_class = "Non Disponibile"
            score = None
            logger.error(f"ERROR IN THE DETERMINISTIC MODEL: {e}")
        dict_esito = {"esitoSemaforico": risk_class, "score": score}
        return dict_esito

    def predict(
        self,
        context: Optional[Any] = None,
        user_input: pd.DataFrame = None,
        params=None,
    ) -> Dict[str, Any]:
        """
        Method to prepare data and then do the prediction of the Deterministic model, i.e. apply the deterministic logic with the yellow list of POD/PDR.
        Args:
            context (any): context of the model (optional argument not used in this case).
            user_input (Pandas dataframe): user's input.
            params (Optional[any]): optional parameters (not used in this case)
        Returns:
            dict_esito (dictionary[string, any]): dictionary with the output of the model's prediction:
             - risk_class (string): the traffic light risk-classification prediction.
             - score (float): the score of prediction.
        """
        # DATA PREPARATION
        df_user_input = self._pre_processing(df_pandas=user_input)

        # DETERMINISTIC MODEL
        dict_esito = self._deterministic_model(df_pandas=df_user_input)

        return dict_esito
