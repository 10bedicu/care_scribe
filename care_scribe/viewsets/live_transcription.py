import logging
from datetime import UTC, datetime, timedelta

import jwt
import requests as http_requests
from django.conf import settings as django_settings
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.viewsets import ViewSet

from care.emr.models.encounter import Encounter
from care_scribe.models.scribe import Scribe
from care_scribe.models.scribe_file import ScribeFile
from care_scribe.models.scribe_quota import ScribeQuota
from care_scribe.settings import plugin_settings
from care_scribe.tasks.live_transcription import finalize_live_transcription
from care_scribe.utils import hash_string

logger = logging.getLogger(__name__)

OPENAI_REALTIME_TRANSCRIPTION_URL = "https://api.openai.com/v1/realtime/transcription_sessions"


def _get_realtime_url_and_headers():
    """
    Return (url, headers) for the Realtime transcription session endpoint.
    Prefers Azure if credentials are configured, otherwise falls back to OpenAI.
    Returns None if neither is configured.
    """
    if plugin_settings.SCRIBE_AZURE_API_KEY:
        endpoint = plugin_settings.SCRIBE_AZURE_ENDPOINT.rstrip("/")
        api_version = plugin_settings.SCRIBE_AZURE_API_VERSION
        url = f"{endpoint}/openai/realtime/transcription_sessions?api-version={api_version}"
        headers = {
            "api-key": plugin_settings.SCRIBE_AZURE_API_KEY,
            "Content-Type": "application/json",
        }
        return url, headers

    if plugin_settings.SCRIBE_OPENAI_API_KEY:
        headers = {
            "Authorization": f"Bearer {plugin_settings.SCRIBE_OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        return OPENAI_REALTIME_TRANSCRIPTION_URL, headers

    return None


def _validate_quota(user, facility_id):
    """
    Validate quota, TNC acceptance, and live transcription enablement.
    Returns (facility_quota, user_quota) on success, or a Response on failure.
    """
    facility_quota = ScribeQuota.objects.filter(
        facility__external_id=facility_id, user=None
    ).first()

    if not facility_quota:
        return Response(
            {"detail": "Facility does not have a scribe quota."}, status=400
        )

    user_quota = ScribeQuota.objects.filter(
        user=user, facility__external_id=facility_id
    ).first()

    if not user_quota:
        return Response(
            {"detail": "User does not have a scribe quota."}, status=400
        )

    # Check TNC acceptance
    tnc = plugin_settings.SCRIBE_TNC
    tnc_hash = hash_string(tnc)
    if user_quota.tnc_hash != tnc_hash:
        return Response(
            {"detail": "User has not accepted the latest terms and conditions."},
            status=403,
        )

    # Check live transcription is enabled
    if not facility_quota.enable_live_transcription and not user_quota.enable_live_transcription:
        return Response(
            {"detail": "Live transcription is not enabled for this user or facility."},
            status=403,
        )

    # Recalculate to get fresh usage numbers
    facility_quota.calculate_used()
    user_quota.calculate_used()

    # Check facility quota not exhausted
    if facility_quota.used >= facility_quota.tokens:
        return Response(
            {"detail": "Facility has exceeded its scribe quota."}, status=403
        )

    # Check user quota not exhausted
    if user_quota.used >= facility_quota.tokens_per_user:
        return Response(
            {"detail": "User has exceeded their scribe quota."}, status=403
        )

    return facility_quota, user_quota


class LiveTranscriptionViewSet(ViewSet):
    permission_classes = [IsAuthenticated]

    @action(detail=False, methods=["post"], url_path="token")
    def create_token(self, request):
        """
        Generate an ephemeral client token for live transcription.

        When the provider is ``openai`` or ``azure`` the response contains an
        ephemeral session payload from OpenAI / Azure Realtime API.

        When the provider is ``google`` the response returns the WebSocket URL
        of the care_scribe_middleware service together with the caller's current
        access token so the frontend can open a direct WebSocket connection.

        Request body:
            - facility_id: str (required) – facility external_id for quota validation
            - encounter_id: str (optional) – encounter external_id
            - language: str – ISO-639-1 language code (auto-detect if omitted)
        """
        user = request.user
        facility_id = request.data.get("facility_id")

        if not facility_id:
            return Response(
                {"detail": "facility_id is required."}, status=400
            )

        # --- Quota & TNC validation ---
        result = _validate_quota(user, facility_id)
        if isinstance(result, Response):
            return result
        facility_quota, user_quota = result

        # --- Resolve encounter (optional) ---
        encounter = None
        encounter_id = request.data.get("encounter_id")
        if encounter_id:
            encounter = Encounter.objects.filter(external_id=encounter_id).first()
            if not encounter:
                return Response({"detail": "Encounter not found."}, status=404)

        # --- Determine provider ---
        provider = plugin_settings.SCRIBE_REALTIME_TRANSCRIPTION_PROVIDER

        # --- Create tracking session ---
        session = Scribe.objects.create(
            requested_by=user,
            requested_in_facility=facility_quota.facility,
            requested_in_encounter=encounter,
            live=True,
            meta={"provider": provider},
        )

        if provider == "google":
            return self._create_google_session(request, session)

        return self._create_openai_session(request, session)

    @action(detail=False, methods=["post"], url_path="complete")
    def complete_session(self, request):
        """
        Report usage for a completed live transcription session.

        The frontend must upload the recorded audio as a ScribeFile
        (with associating_id=session_id) before calling this endpoint.

        Tokens are calculated server-side based on the audio duration and provider:
        - Google: 60 tokens per second of audio
        - OpenAI: 30 tokens per second of audio

        Request body:
            - session_id: str (required) – external_id of the Scribe
            - transcript: str (required) – final assembled transcript text
        """
        required_fields = ["session_id", "transcript"]
        missing = [f for f in required_fields if f not in request.data]
        if missing:
            return Response({"detail": f"Missing required fields: {', '.join(missing)}"}, status=400)

        session_id = request.data["session_id"]

        session = Scribe.objects.filter(
            external_id=session_id,
            requested_by=request.user,
            live=True,
            status=Scribe.Status.CREATED,
        ).first()

        if not session:
            return Response({"detail": "Active session not found."}, status=404)

        # Verify audio was uploaded
        audio_files = ScribeFile.objects.filter(
            associating_id=str(session.external_id),
            file_type=ScribeFile.FileType.SCRIBE_AUDIO,
            upload_completed=True,
        )
        if not audio_files.exists():
            return Response({"detail": "No uploaded audio found. Upload audio before completing the session."}, status=400)

        # Calculate audio duration from uploaded file metadata
        audio_duration_ms = sum(f.meta.get("length", 0) for f in audio_files)

        transcript = request.data["transcript"]

        if not isinstance(transcript, str):
            return Response({"detail": "transcript must be a string."}, status=400)

        # Calculate tokens based on audio duration and provider
        audio_duration_seconds = audio_duration_ms / 1000
        provider = session.meta.get("provider", "openai")
        if provider == "google":
            tokens = int(60 * audio_duration_seconds)
        else:
            tokens = int(30 * audio_duration_seconds)

        processing = {
            "provider": provider,
            "transcription_time": audio_duration_seconds,
            "completion_output_tokens": tokens,
            "transcription_model": plugin_settings.SCRIBE_REALTIME_TRANSCRIPTION_MODEL,
        }

        session.meta["processings"] = [
            *session.meta.get("processings", []),
            processing,
        ]
        session.chat_input_tokens = 0
        session.chat_output_tokens = tokens
        session.transcript = transcript
        session.status = Scribe.Status.COMPLETED
        session.save()

        finalize_live_transcription.delay(str(session.external_id))

        return Response({"detail": "Session completed successfully."})

    def _create_google_session(self, request, session):
        """Return the middleware WebSocket URL + a short-lived scoped JWT for Google STT."""
        middleware_url = plugin_settings.SCRIBE_MIDDLEWARE_URL
        if not middleware_url:
            session.status = Scribe.Status.FAILED
            session.save()
            return Response(
                {"detail": "Google live transcription requires SCRIBE_MIDDLEWARE_URL to be configured."},
                status=400,
            )

        # Mint a short-lived, purpose-scoped JWT for the middleware.
        # This ensures only users who passed quota validation can connect.
        now = datetime.now(UTC)
        token_payload = {
            "user_id": request.user.id,
            "session_id": str(session.external_id),
            "purpose": "scribe_middleware",
            "iat": now,
            "exp": now + timedelta(seconds=60),
        }
        token = jwt.encode(
            token_payload,
            django_settings.SECRET_KEY,
            algorithm="HS256",
        )

        language = request.data.get("language") or None

        return Response({
            "provider": "google",
            "url": middleware_url.rstrip("/") + "/ws/transcribe",
            "token": token,
            "session_id": str(session.external_id),
            "config": {
                **({"language": language} if language else {}),
            },
        })

    def _create_openai_session(self, request, session):
        """Create an ephemeral session via OpenAI / Azure Realtime API."""
        # --- Build session config ---
        result = _get_realtime_url_and_headers()
        if not result:
            session.status = Scribe.Status.FAILED
            session.save()
            return Response(
                {"detail": "Live transcription requires OpenAI or Azure credentials to be configured."},
                status=400,
            )
        url, headers = result

        model = plugin_settings.SCRIBE_REALTIME_TRANSCRIPTION_MODEL
        language = request.data.get("language") or None

        transcription_config = {"model": model}
        if language:
            transcription_config["language"] = language

        session_config = {
            "input_audio_format": "pcm16",
            "input_audio_transcription": transcription_config,
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 500,
            },
            "input_audio_noise_reduction": {"type": "near_field"},
        }

        # --- Create session via HTTP ---
        try:
            resp = http_requests.post(url, headers=headers, json=session_config, timeout=30)
            resp.raise_for_status()
        except http_requests.HTTPError:
            logger.error("Failed to create realtime transcription session [%s]: %s", resp.status_code, resp.text)
            session.status = Scribe.Status.FAILED
            session.save()
            return Response(
                {"detail": "Failed to create live transcription session.", "upstream_error": resp.json()},
                status=502,
            )
        except http_requests.RequestException as e:
            logger.error("Network error creating realtime transcription session: %s", e)
            session.status = Scribe.Status.FAILED
            session.save()
            return Response(
                {"detail": "Failed to reach transcription service."},
                status=502,
            )

        data = resp.json()
        data["provider"] = session.meta.get("provider", "openai")
        data["session_id"] = str(session.external_id)
        return Response(data)
