# Care Scribe

[![Release Status](https://img.shields.io/pypi/v/care_scribe.svg)](https://pypi.python.org/pypi/care_scribe)
[![Build Status](https://github.com/ohcnetwork/care_scribe/actions/workflows/build.yaml/badge.svg)](https://github.com/ohcnetwork/care_scribe/actions/workflows/build.yaml)

Care Scribe is a plugin for care to add voice auto fill support using external services like OpenAI whisper and Google Speech to Text.


## Features

- Voice auto fill support for care
- Support for OpenAI whisper and Google Speech to Text

## Installation

https://care-be-docs.ohc.network/pluggable-apps/configuration.html

https://github.com/ohcnetwork/care/blob/develop/plug_config.py


To install care scribe, you can add the plugin config in [care/plug_config.py](https://github.com/ohcnetwork/care/blob/develop/plug_config.py) as follows:

```python
...

scribe_plug = Plug(
    name="care_scribe",
    package_name="git+https://github.com/ohcnetwork/care_scribe.git",
    version="@master",
    configs={
        "SCRIBE_API_PROVIDER": "openai",  # or "azure" or "google"
        "SCRIBE_AUDIO_MODEL_NAME": "",  # model name for speech to text - Not required for Google
        "SCRIBE_CHAT_MODEL_NAME": "",  # model name for chat completion
        "SCRIBE_TNC" : "", # Terms and Conditions that will be shown to the user before using the plugin. Supports HTML tags.

        # OpenAI Configs
        "SCRIBE_OPENAI_API_KEY": "my-api-key",

        # Azure OpenAI Configs
        "SCRIBE_AZURE_API_VERSION": "",
        "SCRIBE_AZURE_ENDPOINT": "",
        "SCRIBE_AZURE_API_KEY": "",

        # Google Configs
        "SCRIBE_GOOGLE_PROJECT_ID": "my-gcp-project",
        "SCRIBE_GOOGLE_LOCATION": "us-central1",

    },
)
plugs = [scribe_plug]
...
```

The plugin will try to find the API key from the config first and then from the environment variable.

## License

This project is licensed under the terms of the [MIT license](LICENSE).


---
This plugin was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) using the [ohcnetwork/care-plugin-cookiecutter](https://github.com/ohcnetwork/care-plugin-cookiecutter).
