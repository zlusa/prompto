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


class AACSContentModeration(ContentModeration):
    def __init__(self, setup_config: SetupConfig):
        self.setup_config = setup_config
        aacs_config = setup_config.content_moderation.aacs
        try:
            credential = DefaultAzureCredential()
            credential.get_token(URLs.AZ_CREDENTIAL_URL)
        except Exception as e:
            if setup_config.mode == OperationMode.OFFLINE.value:
                credential = InteractiveBrowserCredential()
            else:
                raise GlueAuthenticationException(f"For using DefaultAzureCredential to authenticate config.json needs to "
                                                  f"be present. Refer: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment?view=azureml-api-2"
                                                  f"If running in offline mode InteractiveBrowserCredential() can be used to authentication using pop-up window in web browser.\n", e)

        aacs_client = CognitiveServicesManagementClient(credential, aacs_config.subscription_id)
        aacs = self.find_or_create_aacs(aacs_config, aacs_client)
        aacs_access_key = aacs_client.accounts.list_keys(
            resource_group_name=aacs_config.resource_group, account_name=aacs.name).key1
        self.aacs_client = ContentSafetyClient(aacs.properties.endpoint, AzureKeyCredential(aacs_access_key))

    def is_text_safe(self, text) -> bool:
        response = self.aacs_client.analyze_text(AnalyzeTextOptions(text=text))
        return self.is_below_threshold(response["categoriesAnalysis"])

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
