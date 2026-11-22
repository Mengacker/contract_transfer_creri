import json
from datetime import date
import re
import pandas as pd
import numpy as np
from mlflow.pyfunc import PythonModel

from mlflow.deployments import get_deploy_client
import requests
from typing import Optional, Any, Dict, List

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class CustomServingModel(PythonModel):
    """
    Class to wrap any model's flavor in to a PyFunc-PythonModel, so that MLflow can deploy it to the registry.
    The scoring model is then wrapped to become a three-class classification model that will be the serving model to outputs the traffic light classification and its associated score.
    """

    def __init__(
        self,
        model: PythonModel,
        dtypes_map: dict[str, list[str]],
        encode_maps: dict[str, dict[str, str]],
        yellow_red_threshold: float,
        yellow_green_threshold: float,
        yellow_list: list,
        host: str,
        token_ep: str,
        tables_endpoints: dict = {"POD_PDR": None, "COD_FISCALE_PIVA_CLIIN": None},
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
            host (string): the url of Databricks workspace.
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
        self.host = host
        self.token_ep = token_ep
        self.tables_endpoints = tables_endpoints
        self._init_url()

    def _init_url(self):
        """
        Method to obtain the url of serving endpoints that serves the on-line tables by their name.
        """
        ep_dict = {}
        for key, ep in self.tables_endpoints.items():
            ep_dict[key] = f"{self.host}/serving-endpoints/{ep}/invocations"
            # ep_dict.update({key: f"{self.host}/serving-endpoints/{ep}/invocations"})
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

    def _query_online_table(
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
                raise Exception(f"Error code: {response.status_code}, {response.text}")
        else:
            # Using MLFlow deployments -> questo non funziona fra endpoint anche se lo è nei notebook
            # client = mlflow.deployments.get_deploy_client("databricks")
            client = get_deploy_client(
                "databricks"
            )  # non è il tracking_uri = "databricks-uc" ??
            tab_output = client.predict(inputs=keys, endpoint=self.endpoint_name)
        df_output = pd.DataFrame(tab_output["outputs"])
        return df_output

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

    def _deterministic_model(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method to apply the Deterministic model, i.e. the deterministic logic, which checks if the POD/PDR are present in the "yellow list", if so the contract transfer has a yellow traffic light, otherwise the traffic light is green.
        Args:
            df (Pandas dataframe): dataset with the "POD_PDR" column.
        Returns:
            the initial dataset in input with the new column "score", "esitoSemaforico" and "USED_MODEL" added.
        """
        df["score"] = df["POD_PDR"].isin(self.yellow_list).astype("int64")
        df["esitoSemaforico"] = df["score"].map(
            {1: "Semaforo Giallo", 0: "Semaforo Verde"}
        )
        df["USED_MODEL"] = "DETERMINISTIC"
        return df

    def _scoring_model(
        self,
        df: pd.DataFrame,
        real_time_input: bool = True,  # otherwise "False" to use batch input
    ) -> pd.DataFrame:
        """
        Method to apply the Scoring model, which performs the prediction, but first it must perform some preliminary processing:
         - extract the predictors from the on-line tables with "query-request-post" to the relative endpoints,
         - join the different dataframes obtained,
         - filter the cases that have an unpaid credit,
         - perform the features engineering and transform the datatypes.

        Args:
            df (Pandas dataframe): the pre-processed dataframe.
            real_time_input (boolean): a parameter to change the data input type that change also the way of operations like pre-processing, queries and joins online tables. If True, real-time input is used, otherwise "False" uses batch input.
        Returns:
            the initial dataset joined with the on-line tables, the feature engineering and the addition of the new columns "score", "esitoSemaforico" and "USED_MODEL" added.
        """

        if real_time_input:
            # INFERENCE MASTER EXTRACTION FOR REAL-TIME PREDICTION

            # Primary keys
            # POD e/o PDR
            keys_pod_pdr = [{"POD_PDR": kk} for kk in df["POD_PDR"].tolist()]
            # CODICE FISCALE o PARTITA IVA
            col_cf_piva = [
                cc for cc in ["CodiceFiscale", "PartitaIva"] if cc in df.columns
            ]
            keys_cf_piva = [
                {"COD_FISCALE_PIVA_CLIIN": kk}
                for kk in df[col_cf_piva].values.tolist()[0]
            ]
            if len(col_cf_piva) == 2:
                df["CF_PIVA_INPUT"] = df["CodiceFiscale"].fillna(df["PartitaIva"])
            else:
                df["CF_PIVA_INPUT"] = df[col_cf_piva[0]]
            ind = df["CF_PIVA_INPUT"] == ""
            if (sum(ind) > 0) & (len(col_cf_piva) > 1):
                df.loc[ind, "CF_PIVA_INPUT"] = df.loc[ind, col_cf_piva[-1]]

            # Query on-line tables
            dfmi_volturato_pod_pdr = self._query_online_table(
                keys={"dataframe_records": keys_pod_pdr},
                endpoint_url=self.tables_endpoints_url["POD_PDR"],
            )
            dfmi_volturante_cf_piva = self._query_online_table(
                keys={"dataframe_records": keys_cf_piva},
                endpoint_url=self.tables_endpoints_url["COD_FISCALE_PIVA_CLIIN"],
            )

            # Joins
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
        else:
            # USE RAW-BATCH INPUT FOR BATCH INFERENCE
            df_master_inference = df.copy()

        # FILTERING OF CASES WITH PREVIOUS UNPAID CREDIT
        ind_insoluto = (
            (df_master_inference["PDF_INSOLUTO_LORDO_ALL"] > 0)
            | (df_master_inference["PDF_INSOLUTO_LORDO_LAST"] > 0)
            | (df_master_inference["CLI_INSOLUTO_LORDO"] > 0)
            | (df_master_inference["CLI_INSOLUTO_LORDO_CLIIN"] > 0)
        )
        if sum(ind_insoluto) == 0:
            # if there is no previous unpaid credit, then the transfers are not considered at risk of client's fraudulent default
            df_master_inference.loc[~ind_insoluto, "score"] = -1
            df_master_inference.loc[~ind_insoluto, "esitoSemaforico"] = "Semaforo Verde"
            df_master_inference.loc[~ind_insoluto, "USED_MODEL"] = "NO ONE"

        elif sum(ind_insoluto) > 0:
            # df_master_inference = df_master_inference.loc[ind_insoluto, :].copy()

            # FEATURES ENGINEERING
            # Encoding categorical column
            # "SDF_CANLE_ATTIVAZ_ATT_CLIIN" is already encoded
            cols_to_encode = ["MACROSEGMENTO_CLIIN"]  # i.e. SegmentoVolturante
            for col in cols_to_encode:
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
                df_master_inference[col_list] = df_master_inference[col_list].astype(
                    dtype
                )

            # MODEL'S PREDICTIONS
            predict_probas = self.model.predict_proba(
                df_master_inference[self.predictors]
            )
            # scores = round(predict_probas[:, 1].max(), 2) # OLD
            df_master_inference["score"] = predict_probas[:, 1]
            df_master_inference["score"] = df_master_inference["score"].round(2)

            # The case without unpaid credit have not to be predicted but they can free pass as "Semaforo Verde"
            df_master_inference.loc[~ind_insoluto, "score"] = -1

            # Modifica dei valori di score estremi 1->0.99 e 0->0.01 per poter distinguere tale output del modello di scoring dal quello del modello deterministico
            df_master_inference["score"] = df_master_inference["score"].apply(
                lambda x: 0.99 if x == 1 else (0.01 if x == 0 else x)
            )

            # Categorization of the score into a color-level risk
            ind_red = df_master_inference["score"] > self.yellow_red_threshold
            ind_green = df_master_inference["score"] <= self.yellow_green_threshold

            df_master_inference["esitoSemaforico"] = "Semaforo Giallo"
            df_master_inference.loc[ind_red, "esitoSemaforico"] = "Semaforo Rosso"
            df_master_inference.loc[ind_green, "esitoSemaforico"] = "Semaforo Verde"

            # Add a column of which model is used
            df_master_inference["USED_MODEL"] = "SCORING"
            df_master_inference.loc[~ind_insoluto, "USED_MODEL"] = "NO ONE"

        else:
            raise Exception("ERROR: cases with previous unpaid credit have some errors")
        return df_master_inference

    def predict(
        self,
        context: Optional[Any] = None,
        user_input: pd.DataFrame = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any] | pd.DataFrame:
        """Method to prepare data, do the inference predition and its post-processing.
        It applies a try-except logic because if the model fails, then the deterministic model will be applied.
        In addition, an A/B test logic is applied instead of using the scoring model directly. This is a temporary solution that will be removed in the future.
        The A/B test logic is explained as follows:
        - if the number contained in the POD_PDR code is even, then the scoring model will be applied;
        - if the number contained in the POD_PDR code is odd, then the deterministic model will be applied;
        - otherwise a service unavailability result is returned.
        Tips
        In order to be able to recognize which model has actually been applied in each use of the service, in the output results, which are tracked and stored in a table, the score value also has a hidden meaning:
        - the deterministic model always returns one of the two integer values "0" or "1" as the result score;
        - the scoring model returns any decimal numeric value between "0" and "1", but is forced to never return the extreme integer values;
        - in the special case where the scoring model should be applied, but in reality the bad debt filter avoids its application, then the score is returned with the negative value "-1".

        Args:
            context (any): context of the model (optional argument not used in this case).
            user_input (Pandas dataframe): user's input.
            params (dictionary[string, any]): optional parameters on a dictionary with the following keys:
             - real_time_input (boolean): a parameter to change the data input type that change also the way of operations like pre-processing, queries and joins online tables. If True, real-time input is used, otherwise "False" uses batch input.
             - use_mode (string): "ab-test" to use the A/B test logic, "deterministic" to use the deterministic model, "scoring" to use the scoring model.
        Returns:

            If real_time_input is True:
                dict_esito (dictionary[string, any]): dictionary with the output of the model's prediction:
                - risk_class (string): the traffic light risk-classification prediction.
                - score (float): the score of prediction.

            If real_time_input is False:
                df_results (Pandas dataframe): the result dataset, i.e. the pre-processed input dataset joined with the on-line tables, the feature engineering and the addition of the new columns "score", "esitoSemaforico" and "USED_MODEL" added.
        """
        # params = params or {
        params = {
            "real_time_input": True,  # "True" to use real-time input, otherwise "False" to use batch input.
            "use_mode": "ab-test",  # or "deterministic" or "scoring"
        }
        real_time_input = params.get("real_time_input")
        use_mode = params.get("use_mode")

        if user_input.shape[0] == 1:
            real_time_input = True  # "True" to use real-time input
        else:
            real_time_input = False  # otherwise "False" to use batch input

        # DATA PREPARATION
        if real_time_input:
            df = self._pre_processing(df_pandas=user_input)
            df_null_podpdr = pd.DataFrame()
            df["databricks_request_id"] = "real-time-request"
        else:
            df = user_input
            # Remove the rows with null POD_PDR by the dataframe
            ind = df["databricks_request_id"].isin(
                df.loc[df["POD_PDR"].isnull(), "databricks_request_id"].tolist()
            )
            df_null_podpdr = df.loc[ind, :].reset_index(drop=True)
            df = df.loc[~ind, :].reset_index(drop=True)

        # SPLIT DATAFRAME?
        diz_df = {0: pd.DataFrame(), 1: pd.DataFrame()}
        if use_mode == "scoring":
            diz_df[0] = df  # even

        elif use_mode == "deterministic":
            diz_df[1] = df  # odd

        elif use_mode == "ab-test":
            # Extract the numbers on the strings of "POD_PDR"
            df["PODPDR_num"] = df["POD_PDR"].apply(lambda x: re.sub(r"\D", "", str(x)))
            df["PODPDR_num"] = df["PODPDR_num"].apply(
                lambda x: int(x) if x != "" else None
            )
            # Sum the "POD_PDR" numbers at "databricks_request_id" aggregation scale
            df_group = (
                df.groupby("databricks_request_id")
                .agg(
                    PODPDR_num=(
                        "PODPDR_num",
                        lambda x: np.nan if x.isnull().any() else x.sum(),
                    )
                )
                .reset_index()
            )
            # Split the dataset with odd and even POD+PDR numbers at "databricks_request_id" scale
            df_group["flag_odd"] = df_group["PODPDR_num"].apply(lambda x: x % 2)
            for c in [0, 1]:
                id_list = df_group.loc[
                    df_group["flag_odd"] == c, "databricks_request_id"
                ].tolist()
                diz_df[c] = df.loc[
                    df["databricks_request_id"].isin(id_list), :
                ].reset_index(drop=True)
        else:
            logger.error(
                "The 'use_mode' parameter is bad-defined; use one of the following strings instead: 'ab-test', 'deterministic' or 'scoring'"
            )

        # TRY SCORING MODEL ANR/OR DETERMINISTIC MODEL, OTHERWISE USE DETERMINISTIC MODEL
        try:
            # Execute the model

            if diz_df[0].shape[0] > 0:  # even
                # SCORING MODEL
                diz_df[0] = self._scoring_model(
                    df=diz_df[0], real_time_input=real_time_input
                )

            if diz_df[1].shape[0] > 0:  # odd
                # DETERMINISTIC MODEL
                diz_df[1] = self._deterministic_model(df=diz_df[1])

            df_results = pd.concat([diz_df[0], diz_df[1]], axis=0)

            if df_null_podpdr.shape[0] > 0:  # null POD_PDR
                df_null_podpdr["score"] = None
                df_null_podpdr["esitoSemaforico"] = "Non Disponibile"
                df_null_podpdr["USED_MODEL"] = None
                df_results = pd.concat([df_results, df_null_podpdr], axis=0)
                logger.error("Some POD_PDR are missing")

        except Exception as e:
            # DETERMINISTIC MODEL
            df_results = self._deterministic_model(df=df)
            logger.error(
                f"ERROR IN THE SCORING MODEL, SO THE DETERMINISTIC MODEL IS USED. Error: {e}"
            )

        # RETURN THE OUTPUT
        if real_time_input:
            # In the Real-Time output, it is expected a single value of "score" and "esitoSemaforico"
            databricks_request_id = (
                df_results.groupby("databricks_request_id")
                .agg({"score": "max"})
                .reset_index()["databricks_request_id"]
                .tolist()
            )
            ind = df_results["databricks_request_id"].isin(databricks_request_id)
            if sum(ind) > 0:
                dict_result = {
                    "esitoSemaforico": df_results.loc[ind, "esitoSemaforico"].values[0],
                    "score": df_results[ind, "score"].values[0],
                }
            else:
                dict_result = {
                    "esitoSemaforico": f"#############{str(df_results)}####################",
                    "score": len(databricks_request_id),
                }
            return dict_result
        else:
            return df_results
