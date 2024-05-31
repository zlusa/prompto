from glue.common.base_classes import SetupConfig
from glue.common.content_moderation.aacs import AACSContentModeration


def get_content_moderator_handle(setup_config: SetupConfig):
    content_moderator_handle = {}
    if setup_config.content_moderation.aacs:
        content_moderator_handle["aacs"] = AACSContentModeration(setup_config)

    return content_moderator_handle
