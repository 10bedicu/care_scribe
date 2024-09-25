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
        "TRANSCRIBE_SERVICE_PROVIDER_API_KEY": "secret",
        "API_PROVIDER": "openai",  # or "azure"
        "AZURE_API_VERSION": "",  # required if API_PROVIDER is "azure"
        "AZURE_ENDPOINT": "",  # required if API_PROVIDER is "azure"
        "AUDIO_MODEL_NAME": "",  # model name for OpenAI or custom deployment name for Azure
        "CHAT_MODEL_NAME": "",  # model name for OpenAI or custom deployment name for Azure
    },
)
plugs = [scribe_plug]
...
```

## Configuration

The following configurations variables are available for Care Scribe:

- `TRANSCRIBE_SERVICE_PROVIDER_API_KEY`: API key for the transcribe service provider (OpenAI whisper or Google Speech to Text)
- `API_PROVIDER`: The API provider to use for transcription. Can be either "openai" or "azure".
- `AZURE_API_VERSION`: The version of the Azure API to use. This is required if `API_PROVIDER` is set to "azure".
- `AZURE_ENDPOINT`: The endpoint for the Azure API. This is required if `API_PROVIDER` is set to "azure".
- `AUDIO_MODEL_NAME`: The model name for OpenAI or the custom deployment name for Azure.
- `CHAT_MODEL_NAME`: The model name for OpenAI or the custom deployment name for Azure.

The plugin will try to find the API key from the config first and then from the environment variable.

## License

This project is licensed under the terms of the [MIT license](LICENSE).


---
This plugin was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) using the [ohcnetwork/care-plugin-cookiecutter](https://github.com/ohcnetwork/care-plugin-cookiecutter).