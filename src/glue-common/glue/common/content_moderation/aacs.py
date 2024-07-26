from typing import Dict
import requests
import json

from glue.common.base_classes import AACS, SetupConfig, OperationMode
from glue.common.constants.str_literals import InstallLibs, URLs
from glue.common.exceptions import GlueAuthenticationException
from glue.common.utils.runtime_tasks import install_lib_if_missing
from glue.common.content_moderation.base_class import ContentModeration
from glue.common.utils.logging import get_glue_logger

logger = get_glue_logger(__name__)
install_lib_if_missing(InstallLibs.AZ_IDENTITY)
install_lib_if_missing(InstallLibs.AZ_CONTENT_SAFETY)
install_lib_if_missing(InstallLibs.AZ_MGMT_COG_SERVICE)

from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import  AnalyzeTextOptions
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient
from azure.mgmt.cognitiveservices.models import Account, Sku, AccountProperties


def sliding_window(text, max_chars=1024, overlap_words=2):
    words = text.split()
    windows = []
    start = 0
    while start < len(words):
        window = []
        current_length = 0
        for i in range(start, len(words)):
            word_length = len(words[i]) + 1 
            if current_length + word_length > max_chars:
                break
            window.append(words[i])
            current_length += word_length
        windows.append(' '.join(window))
        start += len(window) - overlap_words
    return windows

def merge_dicts(dict_a, dict_b):
    if not isinstance(dict_a, dict) or not isinstance(dict_b, dict):
        return [dict_a, dict_b] if dict_a != dict_b else dict_a

    merged = dict_a.copy()
    for key, value in dict_b.items():
        if key in dict_a:
            if isinstance(dict_a[key], dict) and isinstance(value, dict):
                merged[key] = merge_dicts(dict_a[key], value)
            elif isinstance(dict_a[key], list):
                merged[key] += value if isinstance(value, list) else [value]
            elif isinstance(value, list):
                merged[key] = [dict_a[key]] + value
            elif isinstance(dict_a[key], (int, float, str)) and isinstance(value, (int, float, str)):
                merged[key] = [dict_a[key], value] if dict_a[key] != value else dict_a[key]
            else:
                merged[key] = value
        else:
            merged[key] = value
    return merged

class AACSContentModeration(ContentModeration):
    def __init__(self, setup_config: SetupConfig):
        self.setup_config = setup_config
        aacs_config = setup_config.content_moderation.aacs
        self.include_metaprompt_guidelines = setup_config.content_moderation.include_metaprompt_guidelines
        try:
            credential = DefaultAzureCredential()
            self.auth_token = credential.get_token(URLs.AZ_COGNITIVE_SERVICES_URL).token
        except Exception as e:
            if setup_config.mode == OperationMode.OFFLINE.value:
                credential = InteractiveBrowserCredential()
                self.auth_token = credential.get_token(URLs.AZ_COGNITIVE_SERVICES_URL).token
            else:
                raise GlueAuthenticationException(f"For using DefaultAzureCredential to authenticate config.json needs to "
                                                  f"be present. Refer: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment?view=azureml-api-2"
                                                  f"If running in offline mode InteractiveBrowserCredential() can be used to authentication using pop-up window in web browser.\n", e)

        aacs_client = CognitiveServicesManagementClient(credential, aacs_config.subscription_id)
        self.aacs = self.find_or_create_aacs(aacs_config, aacs_client)
        if aacs_config.use_azure_ad:
            self.aacs_access_key = None
        else:
            self.aacs_access_key = aacs_client.accounts.list_keys(
                resource_group_name=aacs_config.resource_group, account_name=self.aacs.name).key1
            self.auth_token = None
            
    def check_attack_detected(self, result):
        if isinstance(result, dict):
            for value in result.values():
                if isinstance(value, bool) and value:
                    return True
                elif isinstance(value, dict):
                    if self.check_attack_detected(value):
                        return True
        elif isinstance(result, list):
            for item in result:
                if self.check_attack_detected(item):
                    return True
        return False
    
    def shield_prompt(self,
            user_prompt: str,
            documents: list
        ) -> dict:
        """
        Detects unsafe content using the Content Safety API.

        Args:
        - user_prompt (str): The user prompt to analyze.
        - documents (list): The documents to analyze.

        Returns:
        - dict: The response from the Content Safety API.
        """
        
        api_version = "2024-02-15-preview"
        url = f"{self.aacs.properties.endpoint}/contentsafety/text:shieldPrompt?api-version={api_version}"
        headers = self.build_headers()
        if len(user_prompt) <= 110:
            user_prompt += " " + "_"*(110-len(user_prompt))
        if isinstance(documents, list) and len(documents) > 0:
            for i in range(len(documents)):
                if len(documents[i]) <= 110:
                    documents[i] += " " + "_"*(110-len(documents[i]))
        else:
            if len(documents) <= 110:
                documents += " " + "_"*(110-len(documents))
                
        data = {
            "userPrompt": user_prompt,
            "documents": documents
        }
        response =  requests.post(url, headers=headers, json=data)
        return self.check_attack_detected(response.json())
    
    def text_analyze(self, content, blocklists=[]):
        api_version = "2023-10-01"
        url = f"{self.aacs.properties.endpoint}/contentsafety/text:analyze?api-version={api_version}"
        headers = self.build_headers()
        results = None
        for windows in sliding_window(content, 9990, 0):
            if len(windows)<=110:
                windows += " " + "_"*(110-len(windows))
            request_body = {
                "text": windows,
                "blocklistNames": blocklists,
            }
            payload = json.dumps(request_body)

            response = requests.post(url, headers=headers, data=payload)
            res_content = response.json()
            if response.status_code != 200:
                raise Exception(
                    res_content
                )
            
            if results is None:
                results = res_content
            else:
                results = merge_dicts(results, res_content)
        return results
        
        
    def build_headers(self) -> Dict[str, str]:
        """
        Builds the headers for the Content Safety API request.

        Returns:
        - Dict[str, str]: The headers for the Content Safety API request.
        """
        if not self.aacs_access_key:
            return {
                "Authorization": "Bearer "+ self.auth_token,
                "Content-Type": "application/json",
            }
        else:
            return {
                "Ocp-Apim-Subscription-Key": self.aacs_access_key,
                "Content-Type": "application/json",
            }
        
    def is_text_safe(self, text) -> bool:
        harm_response = self.text_analyze(text)
        safe_bool = self.is_below_threshold(harm_response["categoriesAnalysis"])
        if not safe_bool:
            return safe_bool
        jailbreak_bool = self.shield_prompt(text, [])
        safe_bool = safe_bool and (not jailbreak_bool)
        return safe_bool
        
    def is_below_threshold(self, category_list) -> bool:
        """
        Based on response received from AACS, check if the threshold of content_severity, set by user is crossed.

        :param response: Response from AACS service for a given data.
        :return: True if severity for the given data is below the allowed threshold, else return False.
        """

        severity = 0
        for category in category_list:
            severity = max(severity, category["severity"])

        if severity < self.setup_config.content_moderation.content_severity_threshold:
            return True

        return False

    @staticmethod
    def find_or_create_aacs(aacs_config: AACS, aacs_client):
        """
        Check if AACS account with the given name exists in Azure portal. If yes use it.
        If AACS account with given name is not present in Azure portal, then check is any
        AACs account is present in given subscription. If yes use it.
        If there are no AACS account present in Azure portal, create a new AACs account & return.

        :param aacs_config: Object having AACS related configs
        :param aacs_client:

        :return:
        """

        def find_acs(accounts):
            return next(
                x
                for x in accounts
                if x.kind == "ContentSafety"
                and x.location == aacs_config.location
                and x.sku.name == aacs_config.sku_name
            )
        parameters = Account(
                sku=Sku(name=aacs_config.sku_name),
                kind="ContentSafety",
                location=aacs_config.location,
                properties=AccountProperties(
                    custom_sub_domain_name=aacs_config.name, public_network_access="Enabled"
                ),
            )

        try:
            # check if AACS exists
            aacs = aacs_client.accounts.get(aacs_config.resource_group, aacs_config.name)
            logger.info(f"Found existing AACS Account {aacs.name}.")
        except:
            try:
                # check if there is an existing AACS resource within same resource group
                aacs = find_acs(aacs_client.accounts.list_by_resource_group(aacs_config.resource_group))
                logger.info(
                    f"Found existing AACS Account {aacs.name} in resource group {aacs_config.resource_group}."
                )
            except:
                logger.info(f"Creating AACS Account {aacs_config.name}.")
                aacs_client.accounts.begin_create(aacs_config.resource_group, aacs_config.name, parameters).wait()
                logger.info("Resource created.")
                aacs = aacs_client.accounts.get(aacs_config.resource_group, aacs_config.name)

        return aacs
