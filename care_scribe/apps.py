from django.apps import AppConfig
from care.utils.registries.feature_flag import FlagRegistry, FlagType

PLUGIN_NAME = "care_scribe"


class CareScribeConfig(AppConfig):
    name = PLUGIN_NAME
