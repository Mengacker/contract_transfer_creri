# from mlflow.models import infer_signature
from mlflow.models.signature import ModelSignature
from mlflow.types import ColSpec, Schema


class ServingSignature:
    """
    Class to manage signatures to register models with MLflow.
    A model has to be deployed in the Model-Registry with a predefined signature, which is a representative object of the datatypes of inputs (predictors) and outputs (targets).
    This class define the signature of Serving model.
    """

    def __init__(
        self,
        env: str = None,
    ):
        """
        Method to initialize the class.
        Args:
            env (string): the name of environment where deploy the model.
        Attributes:
            input_schema (mlflow.types.Schema): the schema of the input data, i.e. a single column "INPUT_JSON" contains a json string with some fields.
            output_schema (mlflow.types.Schema): the schema of the output data, i.e. the target variable "score" and its final classification as "esitoSemaforico".
        """
        self.input_schema = self._get_data_schema(
            pyspark_dtype_map={"INPUT_JSON": "string"}
        )
        if env == "prod-sbx":
            self.output_schema = self._get_data_schema(
                pyspark_dtype_map={
                    "esitoSemaforico": "string",
                    "score": "long",
                    "input_data": "string",
                    "bugfix": "string",
                }
            )
        else:
            self.output_schema = self._get_data_schema(
                pyspark_dtype_map={"esitoSemaforico": "string", "score": "long"}
            )

    def _get_data_schema(self, pyspark_dtype_map: dict[str, str]):
        """
        Method to define the data schema of predictors or targets for a model.
        Args:
            pyspark_dtype_map (dictionary[string, string]): a dictionary with the mapping between the column names and their PySpark datatype.
        Returns:
            data_schema (mlflow.types.Schema): the schema of the data.
        """
        schema_list = []
        for field, pyspark_type in pyspark_dtype_map.items():
            schema_list.append(ColSpec(pyspark_type, field))
        data_schema = Schema(schema_list)
        return data_schema

    def get_model_signature(self):
        """
        Method to return the PySpark-MLflow's signature of the Serving model.
        Returns:
            model_signature (mlflow.models.signature): the signature of the Serving model.
        """
        return ModelSignature(self.input_schema, self.output_schema)
