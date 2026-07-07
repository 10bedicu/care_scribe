import base64
import datetime
import json
import logging
import io
import re
import textwrap
from time import perf_counter
from celery import shared_task
from openai import OpenAI, AzureOpenAI
from care_scribe.models.scribe_quota import ScribeQuota
from care_scribe.models.scribe import Scribe
from care_scribe.models.scribe_file import ScribeFile
from care_scribe.settings import plugin_settings
from google.genai import types
from google import genai
from google.oauth2 import service_account

from care_scribe.utils import hash_string

logger = logging.getLogger(__name__)

TRANSCRIPT_ONLY_TRANSCRIBE_TEMPERATURE = 0.1

# Google Speech-to-Text v2 synchronous ``recognize`` accepts at most ~60 seconds
# of audio per request. Longer recordings are transcoded to 16 kHz mono PCM and
# split into chunks slightly under this limit, transcribed sequentially and
# stitched back together.
CHIRP_SYNC_CHUNK_SECONDS = 55
CHIRP_TARGET_SAMPLE_RATE = 16000

# Chirp (Speech-to-Text v2) bills by audio duration and Translate v2 by
# character count; neither returns token usage. Scribe quota is tracked in
# tokens, so these fixed rates convert each billed unit into equivalent tokens
# so Chirp scribes still consume quota.
CHIRP_TOKENS_PER_AUDIO_SECOND = 889
TRANSLATE_TOKENS_PER_CHARACTER = 67

def _google_credentials():
    b64_credentials = plugin_settings.SCRIBE_GOOGLE_APPLICATION_CREDENTIALS_B64
    if not b64_credentials:
        return None

    try:
        decoded = base64.b64decode(b64_credentials, validate=True).decode("utf-8")
    except Exception as e:
        raise Exception(
            "Scribe credential error: SCRIBE_GOOGLE_APPLICATION_CREDENTIALS_B64 is not valid base64. "
            f"({e})"
        ) from e
    try:
        info = json.loads(decoded)
    except Exception as e:
        raise Exception(
            "Scribe credential error: SCRIBE_GOOGLE_APPLICATION_CREDENTIALS_B64 did not decode to valid JSON. "
            f"({e})"
        ) from e
    try:
        return service_account.Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
    except Exception as e:
        raise Exception(
            "Scribe credential error: SCRIBE_GOOGLE_APPLICATION_CREDENTIALS_B64 is not a valid "
            "service-account key (private_key could not be parsed). "
            f"({e})"
        ) from e

def _normalize_google_transcription_usage(usage_metadata):
    if usage_metadata is None:
        return None
    details = usage_metadata.prompt_tokens_details or []
    audio_tokens = sum(
        d.token_count for d in details
        if d.modality == types.MediaModality.AUDIO and d.token_count is not None
    )
    text_tokens = sum(
        d.token_count for d in details
        if d.modality == types.MediaModality.TEXT and d.token_count is not None
    )
    return {
        "input_tokens": usage_metadata.prompt_token_count,
        "audio_input_tokens": audio_tokens or None,
        "text_input_tokens": text_tokens or None,
        "output_tokens": usage_metadata.candidates_token_count,
        "total_tokens": usage_metadata.total_token_count,
        "cached_tokens": usage_metadata.cached_content_token_count,
    }


def _normalize_openai_transcription_usage(usage):
    if usage is None:
        return None
    details = getattr(usage, "input_token_details", None)
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "audio_input_tokens": getattr(details, "audio_tokens", None) if details else None,
        "text_input_tokens": getattr(details, "text_tokens", None) if details else None,
        "output_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "cached_tokens": None,
    }


def _google_llm_transcribe(audio_file_object, model_name, temperature=0):
    """Transcribe a single audio file using a Google Gemini model.

    The audio is sent to the configured Gemini model with a prompt instructing
    it to return ONLY the transcribed text. If ``SCRIBE_TRANSCRIBE_LANGUAGE``
    is set, the model is asked to translate into that language; otherwise the
    transcript is returned in the original spoken language.

    Returns a dict with ``text``, ``prompt``, ``usage`` and ``id`` keys.
    """
    target_language = (plugin_settings.SCRIBE_TRANSCRIBE_LANGUAGE or "").strip()

    _, audio_data = audio_file_object.files_manager.file_contents(audio_file_object)
    fmt = audio_file_object.internal_name.split(".")[-1]

    client = ai_client("google")
    if target_language:
        prompt = (
            "You are an audio transcription engine. Transcribe the provided "
            f"audio and translate the transcript into the language with BCP-47 "
            f"code '{target_language}'.\n"
            "Strict output rules:\n"
            f"- Output ONLY the final transcript in '{target_language}'.\n"
            "- Do NOT include the original-language transcription.\n"
            "- Do NOT include both languages or any side-by-side text.\n"
            "- Do NOT add explanations, labels, preambles, quotes, or markdown.\n"
            "- If the audio is empty or unintelligible, or contains no speech, instead of outputting a blank string, output the reason why you could not transcribe the audio with a prefix \"|>\" (e.g., '|> Audio is empty', '|> Audio is unintelligible', '|> No speech detected')."
        )
    else:
        prompt = (
            "You are an audio transcription engine. Transcribe the provided "
            "audio in the original spoken language. Do not translate.\n"
            "Strict output rules:\n"
            "- Output ONLY the transcript text.\n"
            "- Do NOT add explanations, labels, preambles, quotes, or markdown.\n"
            "- If the audio is empty or unintelligible, or contains no speech, instead of outputting a blank string, output the reason why you could not transcribe the audio with a prefix \"|>\" (e.g., '|> Audio is empty', '|> Audio is unintelligible', '|> No speech detected')."
        )

    # Cap output length as a hard safety net against runaway token-repetition
    audio_length_ms = audio_file_object.meta.get("length", 0) or 0
    max_output_tokens = (
        int(audio_length_ms / 1000 * 5) if audio_length_ms else None
    )
    response = client.models.generate_content(
        model=model_name,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(
                        data=audio_data,
                        mime_type=f"audio/{fmt}",
                    ),
                ],
            )
        ],
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_config=(
                types.ThinkingConfig(thinking_budget=0)
                if "2.5" in model_name and "pro" not in model_name
                else None
            ),
        ),
    )
    text = (response.text or "").strip()
    # When the model cannot transcribe, it returns the reason prefixed with
    # "|>" (e.g. "|> No speech detected"). Treat these as empty transcripts.
    if text.startswith("|>"):
        text = ""
    return {
        "text": text,
        "prompt": prompt,
        "usage": _normalize_google_transcription_usage(response.usage_metadata),
        "id": response.response_id,
        "allotted_output_tokens": max_output_tokens,
    }


def _is_chirp_model(model_name):
    """Return ``True`` when the Google model is a Chirp speech-to-text model."""
    return bool(model_name) and model_name.lower().startswith("chirp")


def _resolve_transcribe_from_languages():
    """Parse ``SCRIBE_TRANSCRIBE_FROM_LANGUAGES`` into a list of BCP-47 codes.

    Returns ``["auto"]`` when the setting is blank/unset so that Chirp performs
    automatic language detection.
    """
    raw = (plugin_settings.SCRIBE_TRANSCRIBE_FROM_LANGUAGES or "").strip()
    languages = [lang.strip() for lang in raw.split(",") if lang.strip()]
    return languages or ["auto"]


def google_speech_client():
    """Build a Google Cloud Speech-to-Text v2 client for Chirp models.

    Chirp is served from regional endpoints, so the client is pinned to
    ``{location}-speech.googleapis.com`` whenever a non-global location is set.

    The REST transport is used instead of the default gRPC transport because
    google-genai (Gemini) initialises gRPC state that corrupts a subsequent gRPC
    Speech call in the same worker process (fails with ``TSI_DATA_CORRUPTED`` /
    ``SSLV3_ALERT_BAD_RECORD_MAC``). REST avoids gRPC entirely.
    """
    from google.api_core.client_options import ClientOptions
    from google.cloud.speech_v2 import SpeechClient

    location = (plugin_settings.SCRIBE_GOOGLE_LOCATION or "").strip()
    client_options = None
    if location and location != "global":
        client_options = ClientOptions(
            api_endpoint=f"{location}-speech.googleapis.com"
        )
    return SpeechClient(
        credentials=_google_credentials(),
        client_options=client_options,
        transport="rest",
    )


def _google_translate_text(text, target_language, source_language=None):
    """Translate ``text`` into ``target_language`` using the Google Translate v2 API.

    ``target_language`` is a BCP-47 tag (e.g. ``en-US``); Translate v2 expects the
    primary sub-tag (e.g. ``en``). When ``source_language`` is given (also a BCP-47
    tag, e.g. the language Chirp detected for a segment) it is passed explicitly so
    Translate does not have to guess. This matters for code-mixed transcripts:
    auto-detection picks a single dominant language for the whole input and leaves
    everything else untranslated, whereas an explicit per-segment source forces the
    intended translation. Returns ``(translated_text, meta)``.

    Translate v2 is billed per character and returns no token usage, so ``meta``
    reports ``characters`` (the billed unit), the source language (explicit or
    detected) and the wall-clock ``time`` of the call.
    """
    from google.cloud import translate_v2 as translate

    target = target_language.split("-")[0]
    source = source_language.split("-")[0] if source_language else None
    translate_start = perf_counter()
    client = translate.Client(credentials=_google_credentials())
    translate_kwargs = {"target_language": target, "format_": "text"}
    if source:
        translate_kwargs["source_language"] = source
    result = client.translate(text, **translate_kwargs)
    meta = {
        "target_language": target,
        "source_language": source,
        "detected_source_language": result.get("detectedSourceLanguage"),
        "characters": len(text),
        "time": perf_counter() - translate_start,
    }
    return result["translatedText"], meta


def _frame_pcm_bytes(frame):
    """Return the exact little-endian s16 PCM bytes for a mono ``AudioFrame``.

    FFmpeg may pad a decoded frame's plane buffer for alignment, so the buffer
    is sliced to the precise ``samples * 2`` byte length (2 bytes per s16
    sample, single channel).
    """
    return bytes(frame.planes[0])[: frame.samples * 2]


def _iter_pcm_chunks(audio_data, chunk_seconds, sample_rate):
    """Decode ``audio_data`` and yield mono PCM chunks of at most ``chunk_seconds``.

    The audio (in any FFmpeg-decodable container/codec, e.g. webm/opus, mp4/aac,
    mp3, ogg or wav) is decoded and resampled to ``sample_rate`` Hz mono signed
    16-bit little-endian PCM, then split into contiguous chunks no longer than
    ``chunk_seconds`` seconds. Each yielded value is raw PCM bytes suitable for a
    Speech-to-Text v2 ``ExplicitDecodingConfig`` (LINEAR16) request, keeping every
    request under the synchronous ``recognize`` audio-length limit.
    """
    try:
        import av
    except ModuleNotFoundError as e:  # pragma: no cover - dependency guard
        raise Exception(
            "Chirp long-audio transcription requires the 'av' (PyAV) package. "
            "Install it with 'pip install av'."
        ) from e

    max_chunk_bytes = int(chunk_seconds * sample_rate * 2)
    pending = bytearray()

    def _emit_full():
        while len(pending) >= max_chunk_bytes:
            yield bytes(pending[:max_chunk_bytes])
            del pending[:max_chunk_bytes]

    try:
        container = av.open(io.BytesIO(audio_data))
    except Exception as e:
        raise Exception(
            f"Could not decode the audio file for transcription: {e}"
        ) from e

    with container:
        audio_streams = container.streams.audio
        if not audio_streams:
            raise Exception("The uploaded audio file contains no audio stream.")
        resampler = av.AudioResampler(
            format="s16", layout="mono", rate=sample_rate
        )
        for frame in container.decode(audio_streams[0]):
            for resampled_frame in resampler.resample(frame):
                pending += _frame_pcm_bytes(resampled_frame)
            yield from _emit_full()
        # Flush any samples buffered inside the resampler.
        for resampled_frame in resampler.resample(None):
            pending += _frame_pcm_bytes(resampled_frame)
        yield from _emit_full()

    if pending:
        yield bytes(pending)


def _google_chirp_transcribe(audio_file_object, model_name, temperature=0):
    """Transcribe a single audio file using a Google Chirp model (STT v2).

    The audio is decoded to 16 kHz mono PCM and split into chunks of at most
    ``CHIRP_SYNC_CHUNK_SECONDS`` seconds so each stays under the synchronous
    ``recognize`` audio-length limit; the per-chunk transcripts are stitched
    back together in order.

    The source languages come from ``SCRIBE_TRANSCRIBE_FROM_LANGUAGES`` (at most
    two BCP-47 codes, or ``auto`` when unset). When ``SCRIBE_TRANSCRIBE_LANGUAGE``
    is set, each Chirp segment is translated into that language separately via the
    Google Translate v2 API — passing Chirp's detected language as the source when
    it differs from the target and auto-detecting otherwise — so code-mixed audio
    is translated per segment instead of Translate guessing one dominant language
    for the whole transcript.

    Returns the same dict shape as the other transcribers, plus independent
    metadata keys (no wrapper object). Speech-to-Text v2 (Chirp) is billed by
    audio duration and returns NO token usage, so the result carries
    ``stt_billed_audio_seconds`` (the billed unit, summed over chunks) and
    ``stt_time`` alongside the source/detected languages and ``stt_chunk_count``,
    plus — when translation runs — the Translate v2 ``translation_*`` fields
    (target/detected-source language, characters, time). ``temperature`` is
    accepted for a consistent signature but is unused by STT.
    """
    from google.cloud.speech_v2.types import cloud_speech

    _, audio_data = audio_file_object.files_manager.file_contents(
        audio_file_object
    )
    language_codes = _resolve_transcribe_from_languages()

    project_id = plugin_settings.SCRIBE_GOOGLE_PROJECT_ID
    location = (plugin_settings.SCRIBE_GOOGLE_LOCATION or "").strip() or "global"
    recognizer = f"projects/{project_id}/locations/{location}/recognizers/_"

    # Long recordings exceed the synchronous ``recognize`` audio-length limit, so
    # the audio is decoded once to 16 kHz mono PCM and transcribed in chunks. The
    # explicit decoding config matches the PCM produced by ``_iter_pcm_chunks``.
    config = cloud_speech.RecognitionConfig(
        explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
            encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=CHIRP_TARGET_SAMPLE_RATE,
            audio_channel_count=1,
        ),
        language_codes=language_codes,
        model=model_name,
    )

    client = google_speech_client()

    segments = []
    detected_languages = []
    request_ids = []
    billed_duration_total = 0.0
    has_billed_duration = False
    stt_time_total = 0.0
    chunk_count = 0

    for chunk in _iter_pcm_chunks(
        audio_data, CHIRP_SYNC_CHUNK_SECONDS, CHIRP_TARGET_SAMPLE_RATE
    ):
        chunk_count += 1
        request = cloud_speech.RecognizeRequest(
            recognizer=recognizer,
            config=config,
            content=chunk,
        )
        stt_start = perf_counter()
        response = client.recognize(request=request)
        stt_time_total += perf_counter() - stt_start

        for result in response.results:
            language_code = getattr(result, "language_code", "") or ""
            if result.alternatives:
                text = result.alternatives[0].transcript
                if text and text.strip():
                    # Keep each segment paired with the language Chirp detected
                    # for it so translation can be done per-language (below).
                    segments.append((text, language_code))
            if language_code and language_code not in detected_languages:
                detected_languages.append(language_code)

        metadata = getattr(response, "metadata", None)
        if metadata is not None:
            if metadata.total_billed_duration is not None:
                billed_duration_total += metadata.total_billed_duration.total_seconds()
                has_billed_duration = True
            request_id = getattr(metadata, "request_id", None) or None
            if request_id:
                request_ids.append(request_id)

    transcript = " ".join(
        text.strip() for text, _ in segments if text and text.strip()
    ).strip()

    billed_duration = billed_duration_total if has_billed_duration else None
    # STT v2 returns one request id per chunk; the first identifies the file.
    request_id = request_ids[0] if request_ids else None

    # Chirp only transcribes, so any requested translation happens here via
    # Translate v2, one Chirp segment at a time. Every segment is translated
    # separately (never skipped): when Chirp reports a source language that
    # differs from the target it is passed explicitly, otherwise Translate
    # auto-detects. Translating per segment — and re-detecting when Chirp claims
    # the target language — recovers segments Chirp mislabels (e.g. Hindi tagged
    # ``en-IN``), which a single whole-transcript call would leave untranslated.
    translation_target = None
    translation_sources = []
    translation_characters = 0
    translation_time = 0.0

    target_language = (plugin_settings.SCRIBE_TRANSCRIBE_LANGUAGE or "").strip()
    if target_language and segments:
        target_primary = target_language.split("-")[0]
        translated_parts = []
        for text, segment_language in segments:
            segment_text = text.strip() if text else ""
            if not segment_text:
                continue
            segment_primary = (segment_language or "").split("-")[0]
            explicit_source = (
                segment_language
                if segment_primary and segment_primary != target_primary
                else None
            )
            translated_text, translation_meta = _google_translate_text(
                segment_text,
                target_language,
                source_language=explicit_source,
            )
            translated_parts.append(translated_text)
            translation_target = translation_meta["target_language"]
            translation_characters += translation_meta["characters"]
            translation_time += translation_meta["time"]
            source = (
                translation_meta["source_language"]
                or translation_meta["detected_source_language"]
            )
            if source and source not in translation_sources:
                translation_sources.append(source)
        transcript = " ".join(
            part.strip() for part in translated_parts if part and part.strip()
        ).strip()

    return {
        "text": transcript,
        "prompt": None,
        "usage": None,
        "id": request_id,
        "allotted_output_tokens": None,
        "stt_from_languages": language_codes,
        "stt_detected_languages": detected_languages,
        "stt_billed_audio_seconds": billed_duration,
        "stt_time": stt_time_total,
        "stt_chunk_count": chunk_count,
        "translation_target_language": translation_target,
        "translation_detected_source_language": translation_sources or None,
        "translation_characters": translation_characters or None,
        "translation_time": translation_time or None,
    }


def _merge_transcription_metadata(processing, result):
    """Store Chirp STT / Translate v2 metadata from a transcribe ``result`` as
    independent, aggregated keys on ``processing`` (no wrapper object).

    Neither API returns token usage, so only the billed units (audio seconds,
    characters), timings and languages are recorded. Numeric values are summed
    across audio files; languages are collected without duplicates. Non-Chirp
    results (OpenAI/Gemini) carry none of these keys and are left untouched.
    """
    def _add(key, value):
        if value:
            processing[key] = (processing.get(key) or 0) + value

    def _collect(key, value):
        if not value:
            return
        bucket = processing.setdefault(key, [])
        for item in value if isinstance(value, list) else [value]:
            if item and item not in bucket:
                bucket.append(item)

    if result.get("stt_from_languages") is not None:
        processing["transcription_from_languages"] = result["stt_from_languages"]
    _collect("transcription_detected_languages", result.get("stt_detected_languages"))
    _add("transcription_billed_audio_seconds", result.get("stt_billed_audio_seconds"))
    _add("transcription_stt_time", result.get("stt_time"))
    _add("transcription_chunk_count", result.get("stt_chunk_count"))
    if result.get("translation_target_language"):
        processing["translation_target_language"] = result["translation_target_language"]
    _collect(
        "translation_detected_source_languages",
        result.get("translation_detected_source_language"),
    )
    _add("translation_characters", result.get("translation_characters"))
    _add("translation_time", result.get("translation_time"))


def _chirp_quota_tokens(processing):
    """Return quota-equivalent tokens for Chirp STT + Translate v2 usage.

    Chirp (Speech-to-Text v2) is billed by audio seconds and Translate v2 by
    translated characters; neither returns token usage, but scribe quota is
    tracked in tokens. ``_merge_transcription_metadata`` has already summed the
    billed audio seconds and translated characters across audio files onto
    ``processing``, so they are converted here with fixed per-unit rates. Runs
    that used no Chirp/Translate (OpenAI/Gemini) record neither metric and
    yield 0.
    """
    seconds = processing.get("transcription_billed_audio_seconds") or 0
    characters = processing.get("translation_characters") or 0
    return round(
        (seconds * CHIRP_TOKENS_PER_AUDIO_SECOND)
        + (characters * TRANSLATE_TOKENS_PER_CHARACTER)
    )


def transcribe_audio_file(audio_file_object, provider, audio_model, temperature=0):
    """Transcribe a single audio file using the configured provider.

    Returns a dict with ``text``, ``prompt``, ``usage`` and ``id`` keys.
    ``prompt`` and ``usage`` may be ``None`` when the underlying provider does
    not expose them (e.g. ``whisper-1``).
    """
    if provider == "google":
        if _is_chirp_model(audio_model):
            return _google_chirp_transcribe(
                audio_file_object, audio_model, temperature=temperature
            )
        return _google_llm_transcribe(
            audio_file_object, audio_model, temperature=temperature
        )

    client = ai_client(provider)
    _, audio_file_data = audio_file_object.files_manager.file_contents(
        audio_file_object
    )
    fmt = audio_file_object.internal_name.split(".")[-1]
    buffer = io.BytesIO(audio_file_data)
    buffer.name = "file." + fmt
    # Only whisper-1 supports the /audio/translations endpoint.
    # Newer models (gpt-4o-transcribe, gpt-4o-mini-transcribe, etc.) are
    # transcription-only and must use /audio/transcriptions.
    if audio_model == "whisper-1":
        transcription = client.audio.translations.create(
            model=audio_model, file=buffer, temperature=temperature
        )
    else:
        transcription = client.audio.transcriptions.create(
            model=audio_model, file=buffer, temperature=temperature
        )
    return {
        "text": transcription.text,
        "prompt": None,
        "usage": _normalize_openai_transcription_usage(
            getattr(transcription, "usage", None)
        ),
        "id": getattr(transcription, "_request_id", None),
        "allotted_output_tokens": None,
    }


def _parse_provider_model(value: str):
    """Split a 'provider/model-name' string into (provider, model).

    The model portion may itself contain '/' characters (kept intact).
    """
    if not value or "/" not in value:
        raise ValueError(
            f"Expected 'provider/model-name' format, got: {value!r}"
        )
    provider, model = value.split("/", 1)
    if provider == "openai" and plugin_settings.SCRIBE_AZURE_API_KEY:
        provider = "azure"
    return provider, model


def ai_client(provider):
    if provider == "azure":
        AiClient = AzureOpenAI(
            api_key=plugin_settings.SCRIBE_AZURE_API_KEY,
            api_version=plugin_settings.SCRIBE_AZURE_API_VERSION,
            azure_endpoint=plugin_settings.SCRIBE_AZURE_ENDPOINT,
        )
    elif provider == "openai":
        AiClient = OpenAI(
            api_key=plugin_settings.SCRIBE_OPENAI_API_KEY,
        )

    elif provider == "google":
        AiClient = genai.Client(
            vertexai=True,
            project=plugin_settings.SCRIBE_GOOGLE_PROJECT_ID,
            location=plugin_settings.SCRIBE_GOOGLE_LOCATION,
            credentials=_google_credentials(),
        )

    else:
        raise Exception("Invalid api provider")
    return AiClient

def chat_message(provider, role="user", text=None, file_object=None, file_type="audio"):
    """ Generates a chat message compatible with the given AI provider client."""
    if file_object:
        _, file_data = file_object.files_manager.file_contents(file_object)
        format = file_object.internal_name.split(".")[-1]
        buffer = io.BytesIO(file_data)
        buffer.name = "file." + format

        if provider == "google":
            return types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=f"{file_type} : "),
                    types.Part.from_bytes(
                        data=file_data,
                        mime_type=f"{file_type}/" + format,
                    ),
                ],
            )
        else:
            encoded_string = base64.b64encode(file_data).decode("utf-8")

            return {
                "role": role,
                "content": [{
                    "type": f"{file_type}_url",
                    f"{file_type}_url": {"url": f"data:{file_type}/{format};base64,{encoded_string}"},
                }]
            }

    else:
        if provider == "google":
            return types.Content(role="user", parts=[types.Part.from_text(text=text)])
        else:
            return {"role": role, "content": [{"type": "text", "text": text}]}

@shared_task
def process_ai_form_fill(external_id):

    form = Scribe.objects.get(external_id=external_id, status=Scribe.Status.READY)

    processing = {
        "created_date" : datetime.datetime.now().isoformat(),
    }

    base_prompt = textwrap.dedent(
        """
        You will receive a patient's encounter in the form of text, audio, or image. Your task is to extract all relevant data and populate the specified form fields accordingly. Follow the instructions and rules meticulously to ensure accuracy and compliance.

        Instructions:
        1. Analyze the encounter content thoroughly to identify and extract valid data.
        2. Use readable terms for coded entries (e.g., convert “A32Q Brain Hemorrhage” to “Brain Hemorrhage”).
        3. If the encounter contains non-English content, translate it to English before processing.
        4. If the audio or image contains no relevant data, return an empty string for the transcription field, and do not assume any context or information.
        5. You do not have to fill all fields. Only fill the fields that are relevant to the encounter. Let the rest have a null value.

        Notes Handling:
        - Populate the `note` field only if there is additional context that cannot be captured in the `value`.
        - For example, if the encounter states, “Patient's SPO2 is 20%, but had spiked to 50% an hour ago,” then you should fill `value: 20%` and `note: Spiked to 50% an hour ago`.
        - If the encounter simply states, “Patient's SPO2 is 20%,” set note as null.
        - If additional context does not exist beyond the value, set `note` field to null.

        Current Date and Time: {current_date_time}
    """
    )
    if form.prompt:
        base_prompt = form.prompt
    base_prompt = base_prompt.replace("{current_date_time}", datetime.datetime.now().isoformat())

    is_benchmark = form.meta.get("benchmark", False)

    # Verify if the user/facility has not exceeded their quota and has accepted the terms and conditions
    user_quota = None
    facility_quota = None

    if not form.audio_file_ids and not form.document_file_ids:
        processing["error"] = "No audio or documents associated with the Scribe. Your upload might have failed."
        form.meta["processings"] = [
            *form.meta.get("processings", []),
            processing
        ]
        form.status = Scribe.Status.FAILED
        form.save()
        return

    if not is_benchmark:
        user_quota = ScribeQuota.objects.filter(user=form.requested_by, facility=form.requested_in_facility).first()
        facility_quota = ScribeQuota.objects.filter(user=None, facility=form.requested_in_facility).first()

        logger.info(f"=== Found user quota {user_quota.external_id if user_quota else 'None'} ===")
        logger.info(f"=== Found facility quota {facility_quota.external_id if facility_quota else 'None'} ===")

        # recalculate used quota. This prevents edge cases where quota was exceeded last month and this is the first request this month
        if facility_quota:
            facility_quota.calculate_used()
        if user_quota:
            user_quota.calculate_used()

        error = None

        if not facility_quota:
            error = "Facility does not have a scribe quota."

        if not user_quota:
            error = "User does not have a scribe quota."

        tnc = plugin_settings.SCRIBE_TNC
        tnc_hash = hash_string(tnc)

        if user_quota.tnc_hash != tnc_hash:
            error = "User has not accepted the latest terms and conditions."
            logger.info(f"User TNC hash: {user_quota.tnc_hash if user_quota else 'None'}, Current TNC hash: {tnc_hash}")

        if facility_quota.used >= facility_quota.tokens:
            error = "Facility has exceeded its scribe quota."

        if user_quota.used >= facility_quota.tokens_per_user:
            error = "User has exceeded their scribe quota."

        if not facility_quota.allow_ocr and not user_quota.allow_ocr and len(form.document_file_ids) > 0:
            error = "OCR is not enabled for this user or facility."

        if error:
            processing["error"] = error
            form.meta["processings"] = [
                *form.meta.get("processings", []),
                processing
            ]
            form.status = Scribe.Status.FAILED
            form.save()
            return

    chat_provider, chat_model = _parse_provider_model(
        plugin_settings.SCRIBE_CHAT_MODEL_NAME
    )
    transcribe_provider, transcribe_model = _parse_provider_model(
        plugin_settings.SCRIBE_TRANSCRIBE_MODEL_NAME
    )
    temperature = 0

    if form.chat_model:
        chat_provider, chat_model = _parse_provider_model(form.chat_model)

    if form.audio_model:
        # Form override may be either "provider/model" or just a model name
        if "/" in form.audio_model:
            transcribe_provider, transcribe_model = _parse_provider_model(
                form.audio_model
            )
        else:
            transcribe_model = form.audio_model

    if form.chat_model_temperature is not None:
        temperature = form.chat_model_temperature

    processing["transcribe_provider"] = transcribe_provider
    processing["transcribe_model"] = (
        transcribe_model
        if form.transcript_only or chat_provider != "google"
        else None
    )
    processing["form_data"] = form.form_data

    if not form.transcript_only:
        processing["chat_provider"] = chat_provider
        processing["chat_model"] = chat_model

    audio_files = ScribeFile.objects.filter(external_id__in=form.audio_file_ids)
    total_audio_duration = sum(file.meta.get("length", 0) for file in audio_files)

    if form.transcript_only:
        logger.info(f"=== Processing transcript-only Scribe {form.external_id} ===")
        processing["transcript_only"] = True
        processing["audio_duration"] = total_audio_duration
        try:
            form.status = Scribe.Status.GENERATING_TRANSCRIPT
            form.save()
            transcript = form.transcript or ""
            if not transcript:
                transcription_start = perf_counter()
                transcription_prompt = None
                input_tokens_total = 0
                output_tokens_total = 0
                total_tokens_total = 0
                audio_input_tokens_total = 0
                text_input_tokens_total = 0
                cached_tokens_total = 0
                allotted_output_tokens_total = 0
                transcription_ids = []
                has_usage = False
                for audio_file_object in audio_files:
                    result = transcribe_audio_file(
                        audio_file_object=audio_file_object,
                        provider=transcribe_provider,
                        audio_model=transcribe_model,
                        temperature=TRANSCRIPT_ONLY_TRANSCRIBE_TEMPERATURE,
                    )
                    transcript += result["text"] or ""
                    if result.get("prompt") and transcription_prompt is None:
                        transcription_prompt = result["prompt"]
                    if result.get("id"):
                        transcription_ids.append(result["id"])
                    allotted_output_tokens_total += (
                        result.get("allotted_output_tokens") or 0
                    )
                    usage = result.get("usage")
                    if usage:
                        has_usage = True
                        input_tokens_total += usage.get("input_tokens") or 0
                        output_tokens_total += usage.get("output_tokens") or 0
                        total_tokens_total += usage.get("total_tokens") or 0
                        audio_input_tokens_total += usage.get("audio_input_tokens") or 0
                        text_input_tokens_total += usage.get("text_input_tokens") or 0
                        cached_tokens_total += usage.get("cached_tokens") or 0
                    _merge_transcription_metadata(processing, result)
                processing["transcription_time"] = perf_counter() - transcription_start
                if transcription_prompt:
                    processing["prompt"] = transcription_prompt
                if transcription_ids:
                    processing["transcription_ids"] = transcription_ids
                if allotted_output_tokens_total:
                    processing["transcription_allotted_output_tokens"] = allotted_output_tokens_total
                if has_usage:
                    processing["completion_input_tokens"] = input_tokens_total
                    processing["completion_output_tokens"] = output_tokens_total
                    processing["completion_total_tokens"] = total_tokens_total
                    processing["completion_audio_input_tokens"] = (
                        audio_input_tokens_total or None
                    )
                    processing["completion_text_input_tokens"] = (
                        text_input_tokens_total or None
                    )
                    processing["completion_cached_tokens"] = (
                        cached_tokens_total or None
                    )
                    form.chat_input_tokens = input_tokens_total
                    form.chat_output_tokens = output_tokens_total

                chirp_tokens = _chirp_quota_tokens(processing)
                if chirp_tokens:
                    processing["transcription_quota_tokens"] = chirp_tokens
                    form.chat_input_tokens = (
                        form.chat_input_tokens or 0
                    ) + chirp_tokens
            form.transcript = transcript
            processing["ai_response"] = transcript
            form.meta["processings"] = [
                *form.meta.get("processings", []),
                processing,
            ]
            form.status = Scribe.Status.COMPLETED
            form.save()
            if not is_benchmark:
                user_quota.calculate_used()
                facility_quota.calculate_used()
        except Exception as e:
            logger.error(
                f"Transcript-only processing failed at line "
                f"{e.__traceback__.tb_lineno}: {e}"
            )
            processing["error"] = str(e)
            form.meta["processings"] = [
                *form.meta.get("processings", []),
                processing,
            ]
            form.status = Scribe.Status.FAILED
            form.save()
        return

    # Instantiate the AI client once to avoid premature closure and resource management issues,
    # especially with the Google GenAI provider. Reuse this client instance throughout the function.
    try:
        client = ai_client(chat_provider)
    except Exception as e:
        logger.exception(f"Scribe {form.external_id}: failed to initialize AI client ({chat_provider}): {e}")
        processing["error"] = f"Failed to initialize AI client: {e}"
        form.meta["processings"] = [
            *form.meta.get("processings", []),
            processing,
        ]
        form.status = Scribe.Status.FAILED
        form.save()
        return

    processed_fields = {}

    def process_fields(fields: list, indent: int = 0):
        for fd in fields:
            if "fields" in fd:
                process_fields(fd["fields"], indent + 1)
            else:
                schema = fd.get("schema", {})
                field_id = fd.get("id", "")
                processed_fields[field_id] = schema

    for qn in form.form_data:
        process_fields(qn["fields"])

    processed_fields_no_keys = {f"q{i}": v for i, (k, v) in enumerate(processed_fields.items())}

    output_schema = {
        "type": "object",
        "properties": {
            **processed_fields_no_keys,
            "__scribe__transcription": {
                "type": "string",
                "description": "The transcription of the audio",
            }
        },
        "required": ["__scribe__transcription"]
    }

    initiation_time = perf_counter()

    if len(form.document_file_ids) > 0 or total_audio_duration > (3 * 60 * 1000):
        # Asking for the full transcription on longer audio would eat up too many tokens.
        output_schema["properties"]["__scribe__transcription"]["description"] = f"A short summarized transcription of the {'image' if len(form.document_file_ids) > 0 else 'audio'} content, focusing on key points and insights in English."

    if chat_provider != "google" and len(form.document_file_ids) == 0:
        # As we are transcribing using whisper, we do not need the transcription field in the output schema
        del output_schema["properties"]["__scribe__transcription"]
        output_schema["required"].remove("__scribe__transcription")

    logger.info(f"=== Processing AI form fill {form.external_id} ===")

    processing["function"] = output_schema
    processing["prompt"] = base_prompt

    messages = []

    messages.append(
        chat_message(
            provider=chat_provider,
            role="system",
            text=base_prompt,
        )
    )

    if form.text:
        messages.append(
            chat_message(
                provider=chat_provider,
                role="user",
                text=form.text,
            )
        )

    try:
        form.status = Scribe.Status.GENERATING_TRANSCRIPT
        form.save()

        transcript = ""
        if not form.transcript:
            logger.info(f"Audio file objects: {audio_files}")

            for audio_file_object in audio_files:

                if chat_provider == "google":
                    messages.append(
                        chat_message(
                            provider=chat_provider,
                            role="user",
                            file_object=audio_file_object,
                            file_type="audio",
                        )
                    )

                else:
                    logger.info(f"=== Generating transcript for AI form fill {form.external_id} ===")
                    try:
                        transcription_result = transcribe_audio_file(
                            audio_file_object=audio_file_object,
                            provider=transcribe_provider,
                            audio_model=transcribe_model,
                        )
                        transcription_text = transcription_result["text"]
                    except Exception as e:
                        logger.error(f"Error generating transcript: {e}")
                        processing["error"] = f"Error generating transcript: {e}"
                        form.meta["processings"] = [
                            *form.meta.get("processings", []),
                            processing
                        ]
                        form.status = Scribe.Status.FAILED
                        form.save()
                        return

                    transcript += transcription_text or ""

                    allotted_output_tokens = transcription_result.get(
                        "allotted_output_tokens"
                    )
                    if allotted_output_tokens is not None:
                        processing["transcription_allotted_output_tokens"] = (
                            processing.get("transcription_allotted_output_tokens", 0)
                            + allotted_output_tokens
                        )

                    _merge_transcription_metadata(processing, transcription_result)

                    transcription_time = perf_counter() - initiation_time
                    processing["transcription_time"] = transcription_time
                    form.save()

                    # Save the transcript to the form
                    form.transcript = transcript
        else:
            transcript = form.transcript

        document_file_objects = ScribeFile.objects.filter(external_id__in=form.document_file_ids)
        logger.info(f"=== Document file objects: {document_file_objects} ===")

        for document_file_object in document_file_objects:
            messages.append(
                chat_message(
                    provider=chat_provider,
                    role="user",
                    file_object=document_file_object,
                    file_type="image",
                )
            )

        if transcript != "":
            messages.append(
                chat_message(
                    provider=chat_provider,
                    role="user",
                    text=transcript,
                )
            )

        logger.info(f"=== Generating AI form fill {form.external_id} ===")
        form.status = Scribe.Status.GENERATING_AI_RESPONSE
        form.save()

        completion_start_time = perf_counter()

        if chat_provider == "google":

            output_schema_hash = hash_string(json.dumps(output_schema, sort_keys=True))
            try:
                cache_list = list(client.caches.list())
                existing_cache = next((cache for cache in cache_list if cache.display_name == f"scribe_{output_schema_hash}" and cache.model.split("/")[-1] == chat_model), None)
            except Exception as e:
                logger.error(f"Error fetching cache: {e}")
                existing_cache = None

            tools = [
                types.Tool(
                    function_declarations=[{
                        "name": "process_ai_form_fill",
                        "description": "Process the AI form fill and return the filled form data.",
                        "parameters": output_schema,
                    }]
                )
            ]

            tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.ANY
                )
            )

            if not existing_cache:
                logger.info(f"=== Creating new cache for scribe_{output_schema_hash} ===")
                try:
                    existing_cache = client.caches.create(
                        model=chat_model,
                        config=types.CreateCachedContentConfig(
                            display_name=f"scribe_{output_schema_hash}",
                            tools=tools,
                            tool_config=tool_config,
                            ttl="86400s"
                        )
                    )
                except Exception as e:
                    logger.warning(f"Error creating cache: {e}")
                    message = None
                    match = re.search(r"'message': '([^']+)'", str(e))
                    if match:
                        message = match.group(1)

                    if message and "constraint-is-too-big" in message:
                        raise Exception("The form is too large for Scribe. Please try again with a smaller form.")
                    existing_cache = None

            will_use_cache = existing_cache and existing_cache.usage_metadata.total_token_count > 1024
            if will_use_cache:
                processing["cache_name"] = existing_cache.name
                logger.info(f"CACHED TOKEN COUNT: {existing_cache.usage_metadata.total_token_count}")

            else:
                logger.info(f"Cache is not large enough, will not use it for this iteration")

            def generate_response(retry=0):
                ai_resp = client.models.generate_content(
                    model=chat_model,
                    contents=messages,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        cached_content=existing_cache.name if will_use_cache else None,
                        tool_config=tool_config if not will_use_cache else None,
                        tools=tools if not will_use_cache else None,
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=0 if "pro" not in chat_model else 1024,
                            include_thoughts=True if "pro" in chat_model else False,
                        ) if "2.5" in chat_model else None
                    ),
                )

                # Sometimes gemini creates a malformed function call on it's server, which causes a failure. Nothing we can do about it really.
                # Refer to : https://discuss.ai.google.dev/t/malformed-function-call-finish-reason-happens-too-frequently-with-vertex-ai/93630
                if ai_resp.candidates[0].finish_reason == types.FinishReason.MALFORMED_FUNCTION_CALL:
                    if retry > 0:
                        raise Exception(f"AI response was malformed, please retry : {str(ai_resp.candidates[0].finish_message)}")
                    else:
                        processing["retries"] = retry + 1
                        return generate_response(retry + 1)
                return ai_resp

            ai_response = generate_response()

            if ai_response.candidates[0].finish_reason != types.FinishReason.STOP:
                raise Exception(f"AI response did not finish successfully: {str(ai_response.candidates[0].finish_reason)} : {str(ai_response.candidates[0].finish_message)}")

            thinking = next((part for part in ai_response.candidates[0].content.parts if part.thought), None)
            processing["thinking"] = thinking.text if thinking else None

            ai_response_json = next(part.function_call.args for part in ai_response.candidates[0].content.parts if part.function_call)

            form.transcript = ai_response_json["__scribe__transcription"]

            processing["completion_id"] = ai_response.response_id
            processing["completion_input_tokens"] = ai_response.usage_metadata.prompt_token_count
            processing["completion_audio_input_tokens"] = sum(
                [detail.token_count for detail in ai_response.usage_metadata.prompt_tokens_details
                 if detail.modality == types.MediaModality.AUDIO and detail.token_count is not None]
            )
            processing["completion_image_input_tokens"] = sum([detail.token_count for detail in ai_response.usage_metadata.prompt_tokens_details if detail.modality == types.MediaModality.IMAGE])
            processing["completion_text_input_tokens"] = sum([detail.token_count for detail in ai_response.usage_metadata.prompt_tokens_details if detail.modality == types.MediaModality.TEXT])
            processing["completion_cached_tokens"] = ai_response.usage_metadata.cached_content_token_count
            processing["completion_cached_audio_tokens"] = sum([detail.token_count for detail in ai_response.usage_metadata.cache_tokens_details if detail.modality == types.MediaModality.AUDIO]) if ai_response.usage_metadata.cache_tokens_details else None
            processing["completion_cached_image_tokens"] = sum([detail.token_count for detail in ai_response.usage_metadata.cache_tokens_details if detail.modality == types.MediaModality.IMAGE]) if ai_response.usage_metadata.cache_tokens_details else None
            processing["completion_cached_text_tokens"] = sum([detail.token_count for detail in ai_response.usage_metadata.cache_tokens_details if detail.modality == types.MediaModality.TEXT]) if ai_response.usage_metadata.cache_tokens_details else None
            processing["completion_output_tokens"] = ai_response.usage_metadata.candidates_token_count
            processing["completion_thinking_tokens"] = ai_response.usage_metadata.thoughts_token_count
            processing["completion_total_tokens"] = ai_response.usage_metadata.total_token_count
            form.chat_input_tokens = ai_response.usage_metadata.prompt_token_count + (ai_response.usage_metadata.cached_content_token_count if ai_response.usage_metadata.cached_content_token_count else 0)
            form.chat_output_tokens = ai_response.usage_metadata.candidates_token_count

        else:
            # These models do not support setting a temperature
            no_temp_models = ["gpt-5", "gpt-5-mini", "gpt-5-nano"]

            ai_response = client.chat.completions.create(
                model=chat_model,
                temperature=temperature if chat_model not in no_temp_models else None,
                messages=messages,
                response_format={
                    "type" : "json_schema",
                    "json_schema" : {
                        "name" : "process_ai_form_fill",
                        "schema" : {
                            **output_schema,
                            "required" : [key for key, value in output_schema["properties"].items()],
                            "additionalProperties": False
                        },
                        "strict" : True,
                    },
                }
            )

            try:
                ai_response_json = json.loads(ai_response.choices[0].message.content)

            except Exception as e:
                raise e

            if not form.transcript and not transcript:
                form.transcript = ai_response_json["__scribe__transcription"]

            processing["completion_id"] = ai_response.id
            processing["completion_input_tokens"] = ai_response.usage.prompt_tokens
            processing["completion_output_tokens"] = ai_response.usage.completion_tokens
            processing["completion_cached_tokens"] = ai_response.usage.prompt_tokens_details.cached_tokens
            form.chat_input_tokens = ai_response.usage.prompt_tokens
            form.chat_output_tokens = ai_response.usage.completion_tokens

    except Exception as e:
        # Log the error or handle it as needed
        logger.error(f"AI form fill processing failed at line {e.__traceback__.tb_lineno}: {e}")
        processing["error"] = str(e)
        form.meta["processings"] = [
            *form.meta.get("processings", []),
            processing
        ]
        form.status = Scribe.Status.FAILED
        form.save()
        return

    processing["completion_time"] = perf_counter() - completion_start_time

    chirp_tokens = _chirp_quota_tokens(processing)
    if chirp_tokens:
        processing["transcription_quota_tokens"] = chirp_tokens
        form.chat_input_tokens = (form.chat_input_tokens or 0) + chirp_tokens

    # convert the keys back to the original field IDs
    converted_response = {k: ai_response_json.get(f"q{i}") for i,(k, v) in enumerate(processed_fields.items()) if ai_response_json.get(f"q{i}") is not None}
    form.ai_response = converted_response
    processing["ai_response"] = converted_response
    form.meta["processings"] = [
        *form.meta.get("processings", []),
        processing
    ]
    form.status = Scribe.Status.COMPLETED
    form.save()

    # Update the user and facility quotas
    if not is_benchmark:
        user_quota.calculate_used()
        facility_quota.calculate_used()
