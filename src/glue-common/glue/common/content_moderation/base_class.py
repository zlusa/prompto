from glue.common.base_classes import AACS, SetupConfig
from glue.common.exceptions import GlueValidaionException


class ContentModeration:
    """
    Parent class for all Content Moderation services
    """
    def __init__(self, setup_config: SetupConfig):
        self.setup_config = setup_config
        self.include_metaprompt_guidelines = setup_config.content_moderation.include_metaprompt_guidelines
        pass

    def is_text_safe(self, text) -> bool:
        """
        Analyze the text and return True if text is safe for work. Else return False

        :param text: Text that we need to analyze if it's safe for work environment.
        :return: True if text is safe for work else return False
        """
        pass

    def is_below_threshold(self, response):
        """
        Based on response received from Content Moderator, check if the threshold of content_severity, set by user is crossed.

        :param response: Response from AACS service for a given data.
        :return: True if severity for the given data is below the allowed threshold, else return False.
        """
        pass

    def validate_list_of_txt(self, list_txt):
        for txt in list_txt:
            if not self.is_text_safe(txt):
                raise GlueValidaionException(f"{txt} is detected as un-safe for work by the content moderator.", None)



