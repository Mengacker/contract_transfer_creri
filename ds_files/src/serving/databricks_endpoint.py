import requests
import time
from tenacity import retry, stop_after_attempt, wait_fixed
from typing import Any

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logging.getLogger("py4j").setLevel(logging.ERROR)

TIMEOUT = 30


class DatabricksEndpoint:
    """
    Class to create, to eventually update and to interact with a Databricks Serving Endpoint that it can serve a model or an on-line table.
    """

    def __init__(
        self,
        auth_token: str,
        databricks_host: str,
        endpoint_name: str,
        entity_type: str,
    ):
        """
        Initialization method to load in the class the reference to the Databricks instance host, the authentication token, the endpoint's name and (optionally) the param for feature spec name (function to query on-line tables).

        Arguments:
            auth_token (string): the authentication token.
            databricks_host (string): the Databricks instance host.
            endpoint_name (string): the serving endpoint's name.
            entity_type (string): the type of entity esposed in the endpoint (model or udf).

        Attributes:
            request_endpoint_url (string): the url to use in the requests package to modity the serving endpoint.
            invocation_endpoint_url (string): the url for the invocations of the serving endpoint.
            headers (string): the headers for the request.
        """
        self.entity_type = entity_type
        self.databricks_host = databricks_host
        self.endpoint_name = endpoint_name
        self.request_endpoint_url = f"{databricks_host}/api/2.0/serving-endpoints"
        self.invocation_endpoint_url = (
            f"{databricks_host}/serving-endpoints/{endpoint_name}/invocations"
        )
        self.headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
        }
        # OLD CODE USED AT DEVELOPING PHASE IN SBX_DATA, NOT YET USED AT INDUSTRIALIZATION PHASE
        # # SDK client per gestione permessi, ecc.
        # self.client = WorkspaceClient(
        #     host=databricks_host, token=auth_token
        # )

    @retry(wait=wait_fixed(10), stop=stop_after_attempt(5))
    def exists(self) -> bool:
        """
        Method to check if the endpoint exists by getting the status of the endpoint.
        Returns:
            flag_exist (boolean): a flag to indicate if the serving endpoint already exists or not.
        """
        url = f"{self.request_endpoint_url}/{self.endpoint_name}"
        # url = self.invocation_endpoint_url # IS WRONG!
        response = requests.get(url=url, headers=self.headers, timeout=TIMEOUT)
        r_json = response.json()
        flag_exist = "error_code" not in r_json.keys()
        return flag_exist

    @retry(wait=wait_fixed(10), stop=stop_after_attempt(5))
    def create_update_endpoint(self, json_config: dict[str, Any]) -> None:
        """
        Method to create or eventually to update the serving endpoint.
        If the endpoint serves a model,  it will update the endpoint to a new model.
        If the endpoint serves an online table, it will do nothing (no refresh will be needed).
        Args:
            json_config (dictionary): configuration of the endpoint.
        """
        response = None  # Inizializzato per evitare UnboundLocalError
        if not self.exists():
            url = self.request_endpoint_url
            # url = self.invocation_endpoint_url # IS WRONG!
            logger.info(f"✅ Creating endpoint at: {url}")
            response = requests.post(
                url=url,
                headers=self.headers,
                json=json_config,
                timeout=TIMEOUT,
            )

        else:
            logger.info(
                f"✅ The endpoint '{self.endpoint_name}' already exists, so its type is checking..."
            )
            if self.entity_type == "model":
                model_version = json_config["config"]["served_entities"][0][
                    "entity_version"
                ]
                logger.info(
                    f"🔁 The entity esposed in the endpoint is a model, so updating it to the version: {model_version}"
                )
                url = f"{self.request_endpoint_url}/{self.endpoint_name}/config"
                # url = self.invocation_endpoint_url # is WRONG!

                # Do the replacement of the model with its new version
                response = requests.put(
                    url=url,
                    headers=self.headers,
                    json=json_config["config"],
                    timeout=TIMEOUT,
                )
                self.wait_for_config_applied()
                logger.info(
                    f"⚠️ Test response of the requests.put in the create_update_endpoint method:{response}"
                )
            else:
                logger.info(
                    "⚠️ The endpoint does not serve a model. No update provided fot it."
                )
                return

        if response.status_code != 200:
            error = f"❌ Expected HTTP 200, but got {response.status_code}"
            logger.warning(error)
            logger.warning(response.text)
            raise ValueError(response)

    def wait_for_config_applied(self) -> None:
        """
        Method to wait for the new configuration to be applied and then the endpoint is ready.
        """
        retry = True
        total_wait = 0
        while retry:
            url = f"{self.request_endpoint_url}/{self.endpoint_name}"
            # url = self.invocation_endpoint_url # is WRONG!
            response = requests.get(url=url, headers=self.headers, timeout=TIMEOUT)

            if response.status_code != 200:
                logger.warning(f"❌ Expected HTTP 200, got {response.status_code}")

            endpoint = response.json()
            if "pending_config" in endpoint:
                seconds = 10
                logger.info("⏳ Config pending. Waiting...")
                if total_wait < 600:
                    logger.info(
                        f"⏱️ Waiting {seconds} seconds (total so far: {total_wait})"
                    )
                    time.sleep(seconds)
                    total_wait += seconds
                else:
                    logger.warning(f"⚠️ Gave up after {total_wait} seconds")
                    retry = False
            else:
                logger.info("✅ New config applied!")
                retry = False

    # @retry(wait=wait_fixed(10), stop=stop_after_attempt(5))
    # def update_permission(self, user_name: str, permission_level: str) -> None: # OLD CODE USED AT DEVELOPING PHASE IN SBX_DATA
    #     """
    #     Method to update the permission of the endpoint.
    #     Args:
    #         user_name (string): the user name to update the permission.
    #         permission_level (string): the permission level to set.
    #     """
    #     assert permission_level in ["CAN_MANAGE", "CAN_QUERY", "CAN_VIEW"]
    #     serv_ep_id = self.client.serving_endpoints.get(self.endpoint_name).id
    #     if permission_level == "CAN_MANAGE":
    #         lvl = ServingEndpointPermissionLevel.CAN_MANAGE
    #     elif permission_level == "CAN_QUERY":
    #         lvl = ServingEndpointPermissionLevel.CAN_QUERY
    #     elif permission_level == "CAN_VIEW":
    #         lvl = ServingEndpointPermissionLevel.CAN_VIEW
    #     self.client.serving_endpoints.update_permissions(
    #         serving_endpoint_id=serv_ep_id,
    #         access_control_list=[
    #             ServingEndpointAccessControlRequest(
    #                 user_name=user_name, permission_level=lvl
    #             )
    #         ],
    #     )
    #     logger.info(
    #         f" Permission {permission_level} set for user '{user_name}' on endpoint '{self.endpoint_name}'"
    #     )
