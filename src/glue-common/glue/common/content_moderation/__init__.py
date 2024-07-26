from glue.common.base_classes import SetupConfig
from glue.common.content_moderation.aacs import AACSContentModeration
from glue.common.content_moderation.base_class import ContentModeration

class ByPassContentModeration(ContentModeration):
    def __init__(self, setup_config: SetupConfig):
        self.setup_config = setup_config
        self.include_metaprompt_guidelines = setup_config.content_moderation.include_metaprompt_guidelines

    def is_text_safe(self, text) -> bool:
        return True

def get_content_moderator_handle(setup_config: SetupConfig) -> ContentModeration:
    content_moderator_handle = None
    if setup_config.content_moderation.enable_moderation:
        if setup_config.content_moderation.aacs:
            content_moderator_handle = AACSContentModeration(setup_config)
    if content_moderator_handle is None:
        content_moderator_handle = ByPassContentModeration(setup_config)

    return content_moderator_handle
