from typing import Any

import environ
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.core.signals import setting_changed
from rest_framework.settings import perform_import
from django.dispatch import receiver

from care_scribe.apps import PLUGIN_NAME

env = environ.Env()


class PluginSettings:  # pragma: no cover
    """
    A settings object that allows plugin settings to be accessed as
    properties. For example:

        from plugin.settings import plugin_settings
        print(plugin_settings.API_KEY)

    Any setting with string import paths will be automatically resolved
    and return the class, rather than the string literal.

    """

    def __init__(
        self,
        plugin_name: str = None,
        defaults: dict | None = None,
        import_strings: set | None = None,
        required_settings: set | None = None,
    ) -> None:
        if not plugin_name:
            raise ValueError("Plugin name must be provided")
        self.plugin_name = plugin_name
        self.defaults = defaults or {}
        self.import_strings = import_strings or set()
        self.required_settings = required_settings or set()
        self._cached_attrs = set()
        self.validate()

    def __getattr__(self, attr) -> Any:
        if attr not in self.defaults:
            raise AttributeError("Invalid setting: '%s'" % attr)

        # Try to find the setting from user settings, then from environment variables
        val = self.defaults[attr]
        try:
            val = self.user_settings[attr]
        except KeyError:
            try:
                val = env(attr, cast=type(val))
            except environ.ImproperlyConfigured:
                # Fall back to defaults
                pass

        # Coerce import strings into classes
        if attr in self.import_strings:
            val = perform_import(val, attr)

        self._cached_attrs.add(attr)
        setattr(self, attr, val)
        return val

    @property
    def user_settings(self) -> dict:
        if not hasattr(self, "_user_settings"):
            self._user_settings = getattr(settings, "PLUGIN_CONFIGS", {}).get(
                self.plugin_name, {}
            )
        return self._user_settings

    def validate(self) -> None:
        """
        This method handles the validation of the plugin settings.
        It could be overridden to provide custom validation logic.

        the base implementation checks if all the required settings are truthy.
        """
        for setting in self.required_settings:
            if not getattr(self, setting):
                raise ImproperlyConfigured(
                    f'The "{setting}" setting is required. '
                    f'Please set the "{setting}" in the environment or the {PLUGIN_NAME} plugin config.'
                )

        if getattr(self, "API_PROVIDER") not in ("openai", "azure"):
            raise ImproperlyConfigured(
                'Invalid value for "API_PROVIDER". '
                'Please set the "API_PROVIDER" to "openai" or "azure".'
            )

        if getattr(self, "API_PROVIDER") == "azure":
            for setting in ("AZURE_API_VERSION", "AZURE_ENDPOINT"):
                if not getattr(self, setting):
                    raise ImproperlyConfigured(
                        f'The "{setting}" setting is required when using Azure API. '
                        f'Please set the "{setting}" in the environment or the {PLUGIN_NAME} plugin config.'
                    )

    def reload(self) -> None:
        """
        Deletes the cached attributes so they will be recomputed next time they are accessed.
        """
        for attr in self._cached_attrs:
            delattr(self, attr)
        self._cached_attrs.clear()
        if hasattr(self, "_user_settings"):
            delattr(self, "_user_settings")


REQUIRED_SETTINGS = {
    "TRANSCRIBE_SERVICE_PROVIDER_API_KEY",
    "AUDIO_MODEL_NAME",
    "CHAT_MODEL_NAME",
    "API_PROVIDER",
}

DEFAULTS = {
    "TRANSCRIBE_SERVICE_PROVIDER_API_KEY": "",
    "AUDIO_MODEL_NAME": "whisper-1",
    "CHAT_MODEL_NAME": "gpt-4o",
    "API_PROVIDER": "openai",
    "AZURE_API_VERSION": "",
    "AZURE_ENDPOINT": "",
}

plugin_settings = PluginSettings(
    PLUGIN_NAME, defaults=DEFAULTS, required_settings=REQUIRED_SETTINGS
)


@receiver(setting_changed)
def reload_plugin_settings(*args, **kwargs) -> None:
    setting = kwargs["setting"]
    if setting == "PLUGIN_CONFIGS":
        plugin_settings.reload()
