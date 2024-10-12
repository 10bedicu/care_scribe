from django.apps import AppConfig
from care.utils.registries.feature_flag import FlagRegistry, FlagType

PLUGIN_NAME = "care_scribe"
class CareScribeConfig(AppConfig):
    name = PLUGIN_NAME

    def ready(self):
        FlagRegistry.register(FlagType.FACILITY, "SCRIBE_ENABLED")
        FlagRegistry.register(FlagType.USER, "SCRIBE_ENABLED")

