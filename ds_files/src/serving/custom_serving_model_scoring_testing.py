import json
from datetime import date
import re
import pandas as pd
from mlflow.pyfunc import PythonModel
from mlflow.deployments import get_deploy_client
import requests
from typing import Optional, Any, Dict, List

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestServingModel(PythonModel):
    """
    Class to wrap any model's flavor in to a PyFunc-PythonModel, so that MLflow can deploy it to the registry.
    The Scoring model is then wrapped to become a three-class classification model. The latter will be the Serving model to outputs the traffic light classification and its associated score.
    """

    def __init__(
        self,
        model: PythonModel,
        dtypes_map: Dict[str, List[str]],
        encode_maps: Dict[str, Dict[str, str]],
        yellow_red_threshold: float,
        yellow_green_threshold: float,
        yellow_list: List[str],
        databricks_host: str,
        token_ep: str,
        tables_endpoints: Dict[str, str] = {
            "POD_PDR": None,
            "COD_FISCALE_PIVA_CLIIN": None,
        },
        use_requests: bool = False,
    ):
        """
        Initialization method to load into the serving script the trained champion scoring model, its thresholds, predictors informations, the yellow list of POD/PDR and the on-line tables endpoint name.
        Arguments:
            model (PythonModel): the Scoring model.
            dtypes_map (dictionary[string, list[string]]): a dictionary with the datatype as keys and the list of the predictors as values.
            encode_maps (dictionary[string, dictionary[string, string]]): a dictionary with the predictors as keys and the encoding map as values.
            yellow_red_threshold (float): the score threshold for classifying between red and yellow traffic lights.
            yellow_green_threshold (float): the score threshold for classifying between yellow and green traffic lights.
            yellow_list (list[string]): list of POD/PDR tagged as defaulters.
            databricks_host (string): the Databricks host (not the endpoint URL).
            token_ep (string): the authentication token.
            tables_endpoints (dictionary[string, string]): a dictionary with the primary key as keys and the endpoint's name as values.
            use_requests (boolean): if True, a request post is used to query the on-line tables, otherwise the Databricks SQL connector is used.

        Attributes:
            predictors (list[string]): the list of predictors.
        """
        self.model = model
        self.dtypes_map = dtypes_map
        self.encode_maps = encode_maps
        self.predictors = sorted(
            [pp for sublist in dtypes_map.values() for pp in sublist]
        )
        self.yellow_red_threshold = yellow_red_threshold
        self.yellow_green_threshold = yellow_green_threshold
        self.yellow_list = yellow_list
        self.use_requests = use_requests
        self.databricks_host = databricks_host
        self.token_ep = token_ep
        self.tables_endpoints = tables_endpoints
        self._init_url()
        self.bugfix_scoring_model = {}

    def _init_url(self):
        """
        Method to obtain the url of serving endpoints that serves the on-line tables by their name.
        """
        ep_dict = {}
        for key, ep in self.tables_endpoints.items():
            ep_dict[key] = f"{self.databricks_host}/serving-endpoints/{ep}/invocations"
        self.tables_endpoints_url = ep_dict

    def load_context(self, context: Optional[Any]) -> None:
        """
        Load context method has to be in the class.
        Args:
            context (any): context of the model (optional argument not used in this case).
        """
        logger.info("load_context")

    def _pre_processing(self, df_pandas: pd.DataFrame) -> pd.DataFrame:
        """
        Method to pre-process the user's input, hence, the input json file is transformed and reshaped into a Pandas dataframe.
        Afterward, some fields are renamed and some other fields are created.
        Args:
            df_pandas (Pandas dataframe): a dataset to pre-process.
        Returns:
            df_pandas (Pandas dataframe): the pre-processed dataframe.
        """
        values_stringa = df_pandas["INPUT_JSON"].values[0]  # <class 'str'>
        values_stringa = values_stringa.replace("\\'", "'")  # clean string
        dict_input = json.loads(values_stringa)  # <class 'dict'>
        assert isinstance(dict_input, dict), logger.warning(
            "The json input not contain a string of a dictionary"
        )

        # Reshape input data for several forniture
        dict_no_forn = {k: v for k, v in dict_input.items() if k != "Forniture"}
        forn_list_dict = dict_input["Forniture"]
        assert isinstance(forn_list_dict, list), logger.warning(
            "The json input not contain a Forniture's list"
        )
        list_of_forn_dict = []
        for ee in forn_list_dict:
            assert isinstance(ee, dict), "A Forniture's element is not a dictionary"
            ee_bis = ee.copy()
            ee_bis.update(dict_no_forn.copy())
            list_of_forn_dict.append(ee_bis)
        assert len(forn_list_dict) > 0, logger.warning("Forniture list is empty")
        df = pd.DataFrame(list_of_forn_dict)

        # Fix some specific fields
        df.rename(
            columns={
                "POD/PDR": "POD_PDR",
                "SegmentoVolturante": "MACROSEGMENTO_CLIIN",
                "CanaleAcquisizione": "ENCOD_SDF_CANLE_ATTIVAZ_ATT_CLIIN",
                "CellulareVolturante": "TELEFONO_CLIIN",
                "MailVolturante": "EMAIL_CLIIN",
            },
            inplace=True,
        )
        if "TELEFONO" in df.columns:
            df["TELEFONO_CLIIN"] = df["TELEFONO_CLIIN"].str.lower()
        else:
            df["TELEFONO_CLIIN"] = ""

        if "EMAIL" in df.columns:
            df["EMAIL_CLIIN"] = df["EMAIL_CLIIN"].str.lower()
        elif "PECVolturante" in df.columns:
            df["EMAIL_CLIIN"] = df["PECVolturante"].str.lower()
        else:
            df["EMAIL_CLIIN"] = ""

        df["MACROSEGMENTO_CLIIN"] = df["MACROSEGMENTO_CLIIN"].astype(str)

        if "NomeCognome" in df.columns:
            df["COGNOME_CLIIN"] = df["NomeCognome"].astype(str)
            df["COGNOME_CLIIN"] = df["COGNOME_CLIIN"].apply(lambda x: x.split("|")[-1])
            if "RagioneSociale" in df.columns:
                ind = df["COGNOME_CLIIN"].isin(["", "nan"])
                df.loc[ind, "COGNOME_CLIIN"] = df.loc[ind, "RagioneSociale"]
        elif "RagioneSociale" in df.columns:
            df["COGNOME_CLIIN"] = df["RagioneSociale"]
        else:
            df["COGNOME_CLIIN"] = ""

        for cc in [
            "COGNOME_CLIIN",
            "MACROSEGMENTO_CLIIN",
            "ENCOD_SDF_CANLE_ATTIVAZ_ATT_CLIIN",
        ]:
            df[cc] = df[cc].str.upper()

        # The date of contract transfer is today, i.e. the date of the inference (It is important to elaborate some aging features)
        df["DATA_VOLTURA"] = date.today()
        df["DATA_VOLTURA"] = pd.to_datetime(df["DATA_VOLTURA"], format="%Y-%m-%d")

        return df

    def _dict_map_encoder(
        self,
        df: pd.DataFrame,
        col_to_encode: str,
        new_col: str,
        encode_map: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Encodes the categories of a feature based on the coding map (encode_map) and creates a new column with the encoded values.
        Args:
            df (Pandas DataFrame): an input dataset.
            col_to_encode (string): the name of the column to encode.
            new_col (string): the name of the new column with the encoded values.
            encode_map (dict[string, Any]): the encoding map.
        Returns:
            the initial dataset in Pandas DataFrame with the new encoded column.
        """
        col_position = df.columns.get_loc(col_to_encode)
        df.insert(
            col_position + 1,
            new_col,
            df[col_to_encode].astype(str).replace(encode_map).astype("category"),
        )
        return df

    def _identical_fields(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Function to elaborate the flag features for the fields with the same value between volturante and volturato cutomers.
        Args:
            df (Pandas DataFrame): an input dataset.
        Returns:
            the initial dataset in input with the new columns "FLG_COGNOME_UGUALE", "FLG_EMAIL_UGUALE", "FLG_TELEFONO_UGUALE" added.
        """
        l_cols = df.columns
        cols_to_eng = ["COGNOME", "EMAIL", "TELEFONO"]
        for col in cols_to_eng:
            if col in l_cols:
                df[f"FLG_{col}_UGUALE"] = (
                    df[col].str.upper() == df[f"{col}_CLIIN"].str.upper()
                )
            else:
                logger.warning(f"Missing column {col}")
        return df

    def _metering_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Function to elaborate  the features of aging between meter measurement date of the last switch-in and the date of contract transfer.
        Args:
            df (Pandas DataFrame): an input dataset.
        Returns:
            the initial dataset in input with the new column "AGING_METER_ATTIV_SWIN" added.
        """
        try:
            # Merge the two columns of DATA_ULT_LETTURA_ATTIV_SWINPWR and DATA_ULT_LETTURA_ATTIV_SWINGAS
            df["DATA_ULT_LETTURA_ATTIV_SWINPWR"] = pd.to_datetime(
                df["DATA_ULT_LETTURA_ATTIV_SWINPWR"], format="%Y-%m-%d"
            )
            df["DATA_ULT_LETTURA_ATTIV_SWINGAS"] = pd.to_datetime(
                df["DATA_ULT_LETTURA_ATTIV_SWINGAS"], format="%Y-%m-%d"
            )
            df["DATA_ULT_LETTURA_ATTIV_SWIN"] = df[
                "DATA_ULT_LETTURA_ATTIV_SWINPWR"
            ].fillna(df["DATA_ULT_LETTURA_ATTIV_SWINGAS"])
            # Convert the date columns to datetime
            df["DATA_ULT_LETTURA_ATTIV_SWIN"] = pd.to_datetime(
                df["DATA_ULT_LETTURA_ATTIV_SWIN"], format="%Y-%m-%d"
            )
            df["DATA_VOLTURA"] = pd.to_datetime(df["DATA_VOLTURA"], format="%Y-%m-%d")
            # Calculate the difference in days
            df["AGING_METER_ATTIV_SWIN"] = (
                df["DATA_VOLTURA"] - df["DATA_ULT_LETTURA_ATTIV_SWIN"]
            ).apply(lambda x: x.days)
            df["AGING_METER_ATTIV_SWIN"] = df["AGING_METER_ATTIV_SWIN"].astype(
                "float64"
            )
        except AttributeError:
            logger.warning(
                "Missing column DATA_ULT_LETTURA_ATTIV_SWINPWR or DATA_ULT_LETTURA_ATTIV_SWINGAS or DATA_VOLTURA"
            )
        return df

    def _deterministic_model(self, df: pd.DataFrame) -> tuple:
        """
        Method to apply the Deterministic model, i.e. the deterministic logic, which checks if the POD/PDR are present in the "yellow list", if so the contract transfer has a yellow traffic light, otherwise the traffic light is green.
        Args:
            df (Pandas dataframe): dataset with the "POD_PDR" column.
        Returns:
            risk_class (string): the traffic light risk-classification of the Deterministic model.
            score (float): the score of the Deterministic model.
        """
        lista_pod_pdr = df["POD_PDR"].tolist()
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
            logger.error(
                f"ERROR IN THE DETERMINISTIC MODEL WITH THE YELLOW LIST OF POD/PDR{str(lista_pod_pdr)}. Error: {e}"
            )

        return risk_class, score

    def _request_ep_inference_table(
        self, keys: List[Dict[str, List[Dict[str, str]]]], endpoint_url: str
    ) -> pd.DataFrame:
        """
        Method to query the online tables with a request post method or with the Databricks SQL connector.
        Args:
            keys (list[dictionary[string, list[dictionary[string, string]]): a list of dataframe records as a dicionary of lists of dictionaries, where each of the latter has a pramary key of the table.
            endpoint_url (string): the url of endpoint that serves an online table.
        Returns:
            df_output (Pandas dataframe): the output dataset of the online table's query.
        """
        df_output = None
        bugfix_request = None
        try:
            if self.use_requests:
                # Using requests -> questo funziona fra endpoint !
                response = requests.post(
                    endpoint_url,
                    headers={"Authorization": f"Bearer {self.token_ep}"},
                    json=keys,
                    timeout=60,  # TIMEOUT, parameter expressed in seconds
                )
                if response.status_code == 200:
                    tab_output = response.json()
                else:
                    raise Exception(
                        f"Error code: {response.status_code}, {response.text}"
                    )
            else:
                # Using MLFlow deployments -> questo non funziona fra endpoint, anche se funziona nei notebook.
                client = get_deploy_client(
                    "databricks"
                )  # non è il tracking_uri = "databricks-uc" ??
                tab_output = client.predict(inputs=keys, endpoint=self.endpoint_name)

            try:
                df_output = pd.DataFrame(tab_output["outputs"])
                bugfix_request = f"Request to endpoint {endpoint_url} completed"
            except Exception as e:
                bugfix_request = f'ERROR TO CONVERT THE RESPONSE OF THE ENDPOINT "{endpoint_url}" TO A DATAFRAME. Error: {e}'
                logger.error(bugfix_request)

        except Exception as e:
            bugfix_request = f'ERROR IN "_request_ep_inference_table" METHOD FOR THE ENDPOINT "{endpoint_url}" WITH KEYS "{str(keys)}". Error: {e}'
            logger.error(bugfix_request)

        return df_output, bugfix_request

    def _scoring_model(self, df: pd.DataFrame) -> tuple:
        """
        Method to apply the Scoring model, which performs the prediction, but first it must perform some preliminary processing:
         - extract the predictors from the on-line tables with "query-request-post" to the relative endpoints,
         - join the different dataframes obtained,
         - filter the cases that have an unpaid credit,
         - perform the features engineering and transform the datatypes.

        Args:
            df (Pandas dataframe): the pre-processed dataframe.
        Returns:
            risk_class (string): the traffic light risk-classification of the Scoring model.
            score (float): the score of the Scoring model.
        """
        self.bugfix_scoring_model = {"scoring_model": "method start"}
        # INFERENCE MASTER EXTRACTION
        try:
            # Primary keys
            # POD e/o PDR
            keys_pod_pdr = [
                {"POD_PDR": kk} for kk in df["POD_PDR"].tolist() if kk != ""
            ]
            # self.bugfix_scoring_model.update({"POD_PDR": keys_pod_pdr})

            # CODICE FISCALE o PARTITA IVA
            col_cf_piva = [
                cc for cc in ["CodiceFiscale", "PartitaIva"] if cc in df.columns
            ]
            keys_cf_piva = [
                {"COD_FISCALE_PIVA_CLIIN": kk}
                for kk in df[col_cf_piva].values.tolist()[0]
                if kk != ""
            ]
            if len(keys_cf_piva) == 2:
                df["CF_PIVA_INPUT"] = df["CodiceFiscale"].fillna(df["PartitaIva"])
            else:
                df["CF_PIVA_INPUT"] = df[col_cf_piva[0]]
            ind = df["CF_PIVA_INPUT"] == ""
            if (sum(ind) > 0) & (len(keys_cf_piva) > 1):
                df.loc[ind, "CF_PIVA_INPUT"] = df.loc[ind, col_cf_piva[-1]]

            # self.bugfix_scoring_model.update({"CF_PIVA_INPUT": keys_cf_piva})
        except Exception as e:
            error = f"ERROR IN THE EXTRACTION OF PRIMARY KEYS: {e}"
            logger.error(error)
            self.bugfix_scoring_model.update({"PK_error": error})
            return "Non Disponibile", None, {"bugfix_scoring_model": error}

        try:
            # Request to inference table endpoints
            dfmi_volturato_pod_pdr, bugfix_request = self._request_ep_inference_table(
                keys={"dataframe_records": keys_pod_pdr},
                endpoint_url=self.tables_endpoints_url["POD_PDR"],
            )
            if dfmi_volturato_pod_pdr is not None:
                message = f"The request to inference table endpoint ccvol_sep_mstr_volturato is completed. Its output is: {str(dfmi_volturato_pod_pdr.shape)}."
                logger.info(message)
            else:
                message = bugfix_request
                logger.warning(message)
            self.bugfix_scoring_model.update({"request_ep1_volturato": message})
        except Exception as e:
            error = f'ERROR IN THE REQUEST TO INFERENCE TABLE ENDPOINT "{str(self.tables_endpoints_url["POD_PDR"])}". Error: {e}'
            logger.error(error)
            self.bugfix_scoring_model.update({"REQUEST_EP1_error": error})
            return "Non Disponibile", None, {"bugfix_scoring_model": error}

        try:
            dfmi_volturante_cf_piva, bugfix_request = self._request_ep_inference_table(
                keys={"dataframe_records": keys_cf_piva},
                endpoint_url=self.tables_endpoints_url["COD_FISCALE_PIVA_CLIIN"],
            )
            if dfmi_volturante_cf_piva is not None:
                message = f"The request to inference table endpoint ccvol_sep_mstr_volturante is completed. Its output is: {str(dfmi_volturante_cf_piva.shape)}."
                logger.info(message)
            else:
                message = bugfix_request
                logger.warning(message)
            self.bugfix_scoring_model.update({"request_ep2_volturante": message})
        except Exception as e:
            error = f'ERROR IN THE REQUEST TO INFERENCE TABLE ENDPOINT "{str(self.tables_endpoints_url["POD_PDR"])}". Error: {e}'
            logger.error(error)
            self.bugfix_scoring_model.update({"REQUEST_EP2_error": error})
            return "Non Disponibile", None, {"bugfix_scoring_model": error}

        # Joins
        try:
            df_master_inference = df.merge(
                dfmi_volturato_pod_pdr,
                left_on="POD_PDR",
                right_on="POD_PDR",
                how="left",
            )
            df_master_inference = df_master_inference.merge(
                dfmi_volturante_cf_piva,
                left_on="CF_PIVA_INPUT",
                right_on="COD_FISCALE_PIVA_CLIIN",
                how="left",
            )
            message = f"Joins between the dataframe is completed. The shape of result dataframe is: {str(df_master_inference.shape)}."
            logger.info(message)
            self.bugfix_scoring_model.update({"joins_result": message})
        except Exception as e:
            error = f"ERROR IN THE JOINS OF DATAFRAMES:: {e}"
            logger.error(error)
            self.bugfix_scoring_model.update({"JOINS_error": error})
            return "Non Disponibile", None, {"bugfix_scoring_model": error}

        # FILTERING OF CASES WITH PREVIOUS UNPAID CREDIT
        ind_insoluto = (
            (df_master_inference["PDF_INSOLUTO_LORDO_ALL"] > 0)
            | (df_master_inference["PDF_INSOLUTO_LORDO_LAST"] > 0)
            | (df_master_inference["CLI_INSOLUTO_LORDO"] > 0)
            | (df_master_inference["CLI_INSOLUTO_LORDO_CLIIN"] > 0)
        )

        # if sum(ind_insoluto) > 0:
        #     df_master_inference = df_master_inference.loc[ind_insoluto, :].copy()
        if True:
            try:
                # FEATURES ENGINEERING
                # Encoding categorical column
                # "SDF_CANLE_ATTIVAZ_ATT_CLIIN" is already encoded
                col = "MACROSEGMENTO_CLIIN"  # i.e. SegmentoVolturante
                emap = self.encode_maps[col]
                df_master_inference = self._dict_map_encoder(
                    df=df_master_inference,
                    col_to_encode=col,
                    new_col=f"ENCOD_{col}",
                    encode_map=emap,
                )
                df_master_inference = self._identical_fields(df_master_inference)
                df_master_inference = self._metering_features(df_master_inference)

                # CAST PANDAS DATATYPES
                for dtype, col_list in self.dtypes_map.items():
                    df_master_inference[col_list] = df_master_inference[
                        col_list
                    ].astype(dtype)

                self.bugfix_scoring_model.update({"feature_engineering": "done"})
            except Exception as e:
                error = f"ERROR IN THE FEATURES ENGINEERING WITH THE DATAFRAME: \n {str(df_master_inference.info())}.\n Error: {e}"
                logger.error(error)
                self.bugfix_scoring_model.update({"FEATURE_ENGINEERING_error": error})
                return "Non Disponibile", None, {"bugfix_scoring_model": error}

        inference_masters_fields = {
            "campi_master_volturato": dfmi_volturato_pod_pdr.to_dict(),
            "campi_master_volturante": dfmi_volturante_cf_piva.to_dict(),
            "campi_master_finale": df_master_inference.to_dict(),
        }

        if sum(ind_insoluto) > 0:
            df_filtered = df_master_inference.loc[ind_insoluto, :].copy()
            # MODEL'S PREDICTIONS
            try:
                predict_probas = self.model.predict_proba(df_filtered[self.predictors])
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
                score = round(df_proba["PROBA_1"].max(), 2)
                self.bugfix_scoring_model.update({"predict_proba": "done"})
            except Exception as e:
                error = f"ERROR IN THE PREDICTION OF SCORING MODEL WITH THE DATAFRAME {str(df_filtered[self.predictors].shape)}. Error: {e}"
                logger.error(error)
                self.bugfix_scoring_model.update({"PREDICTION_error": error})
                return "Non Disponibile", None, {"bugfix_scoring_model": error}

            # Categorization of the score into a color-level risk
            if score == 1:
                score = 0.99  # per poter distinguere tale output dal quello del modello deterministico
                risk_class = "Semaforo Rosso"
            elif score > self.yellow_red_threshold:
                risk_class = "Semaforo Rosso"
            elif score > self.yellow_green_threshold:
                risk_class = "Semaforo Giallo"
            elif score > 0:
                risk_class = "Semaforo Verde"
            elif score == 0:
                score = 0.01  # per poter distinguere tale output dal quello del modello deterministico
                risk_class = "Semaforo Verde"
            else:
                risk_class = "Non Disponibile"
                score = None

        else:
            message = "There is no previous unpaid credit, then the transfers are not considered at risk of fraudulent default."
            risk_class = "Semaforo Verde"
            score = -1
            self.bugfix_scoring_model.update({"filter_result": message})
            return risk_class, score, inference_masters_fields  # {"info": message}

        return risk_class, score, inference_masters_fields

    def predict(
        self,
        context: Optional[Any] = None,
        user_input: pd.DataFrame = None,
    ) -> Dict[str, Any]:
        """Method to prepare data, do the inference predition and its post-processing.
        It applies a try-except logic because if the model fails, then the Deterministic model will be applied.
        In addition, an A/B test logic is applied instead of using the Scoring model directly. This is a temporary solution that will be removed in the future.
        The A/B test logic is explained as follows:
        - if the number contained in the "POD_PDR" code is even, then the scoring model will be applied;
        - if the number contained in the "POD_PDR" code is odd, then the deterministic model will be applied;
        - otherwise a service unavailability result is returned.
        Tips
        In order to be able to recognize which model has actually been applied in each use of the service, in the output results, which are tracked and stored in a table, the score value also has a hidden meaning:
        - the Deterministic model always returns one of the two integer values ​​"0" or "1" as the result score;
        - the Scoring model returns any decimal numeric value between "0" and "1", but is forced to never return the extreme integer values;
        - in the special case where the Scoring model should be applied, but in reality the unpaid credit flag filter avoids its application, then the score is returned with the negative value "-1".

        Args:
            context (any): context of the model (optional argument not used in this case).
            user_input (Pandas dataframe): user's input.
        Returns:
            dict_esito (dictionary[string, any]): dictionary with the output of the model's prediction:
             - risk_class (string): the traffic light risk-classification prediction.
             - score (float): the score of prediction.
        """
        bugfix = {
            "model": str(type(self.model)),
            # "dtypes_map": self.dtypes_map,
            # "encode_maps": self.encode_maps,
            "yellow_red_threshold": self.yellow_red_threshold,
            "yellow_green_threshold": self.yellow_green_threshold,
            # "databricks_host": self.databricks_host,
            # "token_ep": self.token_ep,
            # "tables_endpoints": self.tables_endpoints,
        }
        input_data = {
            # "predictors": self.predictors,
            "user_input_pre_processed": None,
            "sum_podpdr_num": None,
            "campi_master_volturato": None,
            "campi_master_volturante": None,
            "campi_master_finale": None,
        }

        # DATA PREPARATION
        try:
            df_user_input = self._pre_processing(df_pandas=user_input)
            input_data.update(
                {"user_input_pre_processed": str(df_user_input.to_dict())}
            )
        except Exception as e:
            bugfix.update({"user_input": "data preparation failed"})
            logger.error(
                f"ERROR IN THE DATA PREPARATION WITH THE INPUT: \n {str(user_input)} .\n Error: {e}"
            )

        # TRY SCORING MODEL OTHERWISE USE DETERMINISTIC MODEL AS ROLLBACK
        try:
            # CASE SEPARATION FOR A/B TEST
            sum_podpdr_num = 0
            for pod_pdr in df_user_input["POD_PDR"].tolist():
                cleaned_number = re.sub(r"\D", "", str(pod_pdr))
                int_number = int(cleaned_number) if cleaned_number != "" else None
                sum_podpdr_num = sum_podpdr_num + int_number
            input_data.update({"sum_podpdr_num": str(sum_podpdr_num)})

            if sum_podpdr_num % 2 == 0:  # even
                bugfix.update(
                    {
                        "metadata": "scoring_model used because the number of POD_PDR is even"
                    }
                )
                # SCORING MODEL
                risk_class, score, inference_masters_fields = self._scoring_model(
                    df=df_user_input
                )

            elif sum_podpdr_num % 2 == 1:  # odd
                bugfix.update(
                    {
                        "metadata": "deterministic_model used because the number of POD_PDR is odd"
                    }
                )
                # DETERMINISTIC MODEL
                risk_class, score = self._deterministic_model(df=df_user_input)
            else:
                bugfix.update({"metadata": "no valid POD_PDR number"})
                risk_class = "Non Disponibile"
                score = None
                logger.error("The number of POD_PDR is not even or odd")

        except Exception as e:
            bugfix.update(
                {
                    "metadata": "deterministic_model is used because something in the scoring model failed"
                }
            )
            # DETERMINISTIC MODEL
            risk_class, score = self._deterministic_model(df=df_user_input)
            logger.error(
                f"ERROR IN THE SCORING MODEL, SO THE DETERMINISTIC MODEL IS USED. Error: {e}"
            )
        dict_esito = {
            "esitoSemaforico": risk_class,
            "score": score,
            "bugfix": bugfix,
            "input_data": input_data,
            "bugfix_scoring_model": self.bugfix_scoring_model,
            "final_inference_master": inference_masters_fields["campi_master_finale"],
        }
        return dict_esito
