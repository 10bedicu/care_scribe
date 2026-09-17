import sys

from django.apps import AppConfig
from care.utils.registries.feature_flag import FlagRegistry, FlagType

PLUGIN_NAME = "care_scribe"

BUILD_TIME_COMMANDS = {
    "collectstatic",
    "makemigrations",
    "migrate",
    "compilemessages",
    "makemessages",
    "spectacular",
    "test",
}


class CareScribeConfig(AppConfig):
    name = PLUGIN_NAME

    def ready(self):
        if len(sys.argv) > 1 and sys.argv[1] in BUILD_TIME_COMMANDS:
            return

        from care_scribe.settings import plugin_settings

        plugin_settings.validate()
