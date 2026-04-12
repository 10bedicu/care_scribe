import logging

import requests as http_requests
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.viewsets import ViewSet

from care_scribe.settings import plugin_settings

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

        Request body (all optional):
            - facility_id: str – facility external_id for quota validation
            - language: str – ISO-639-1 language code (auto-detect if omitted)
            - model: str – transcription model name (default from settings)
        """
        # TODO: Re-enable quota/TNC validation
        # user = request.user
        # facility_id = request.data.get("facility_id")
        #
        # if facility_id:
        #     facility_quota = ScribeQuota.objects.filter(
        #         facility__external_id=facility_id, user=None
        #     ).first()
        #     user_quota = ScribeQuota.objects.filter(
        #         user=user, facility__external_id=facility_id
        #     ).first()
        #
        #     if not facility_quota:
        #         return Response(
        #             {"detail": "Facility does not have a scribe quota."}, status=400
        #         )
        #     if not user_quota:
        #         return Response(
        #             {"detail": "User does not have a scribe quota."}, status=400
        #         )
        #
        #     facility_quota.calculate_used()
        #     user_quota.calculate_used()
        #
        #     tnc = plugin_settings.SCRIBE_TNC
        #     tnc_hash = hash_string(tnc)
        #     if user_quota.tnc_hash != tnc_hash:
        #         return Response(
        #             {"detail": "User has not accepted the latest terms and conditions."},
        #             status=403,
        #         )
        #
        #     if facility_quota.used >= facility_quota.tokens:
        #         return Response(
        #             {"detail": "Facility has exceeded its scribe quota."}, status=403
        #         )
        #
        #     if user_quota.used >= facility_quota.tokens_per_user:
        #         return Response(
        #             {"detail": "User has exceeded their scribe quota."}, status=403
        #         )

        # --- Determine provider ---
        provider = plugin_settings.SCRIBE_REALTIME_TRANSCRIPTION_PROVIDER

        if provider == "google":
            return self._create_google_session(request)

        return self._create_openai_session(request)

    def _create_google_session(self, request):
        """Return the middleware WebSocket URL + JWT token for Google STT."""
        middleware_url = plugin_settings.SCRIBE_MIDDLEWARE_URL
        if not middleware_url:
            return Response(
                {"detail": "Google live transcription requires SCRIBE_MIDDLEWARE_URL to be configured."},
                status=400,
            )

        # Extract the JWT access token from the Authorization header
        auth_header = request.META.get("HTTP_AUTHORIZATION", "")
        token = auth_header.split(" ", 1)[1] if " " in auth_header else ""

        language = request.data.get("language") or None
        model = request.data.get("model") or None

        return Response({
            "provider": "google",
            "url": middleware_url.rstrip("/") + "/ws/transcribe",
            "token": token,
            "config": {
                **({"language": language} if language else {}),
                **({"model": model} if model else {}),
            },
        })

    def _create_openai_session(self, request):
        """Create an ephemeral session via OpenAI / Azure Realtime API."""
        # --- Build session config ---
        result = _get_realtime_url_and_headers()
        if not result:
            return Response(
                {"detail": "Live transcription requires OpenAI or Azure credentials to be configured."},
                status=400,
            )
        url, headers = result

        model = request.data.get("model", plugin_settings.SCRIBE_REALTIME_TRANSCRIPTION_MODEL)
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
            return Response(
                {"detail": "Failed to create live transcription session.", "upstream_error": resp.json()},
                status=502,
            )
        except http_requests.RequestException as e:
            logger.error("Network error creating realtime transcription session: %s", e)
            return Response(
                {"detail": "Failed to reach transcription service."},
                status=502,
            )

        return Response(resp.json())
