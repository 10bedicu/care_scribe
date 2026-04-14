import logging

from celery import shared_task

from care_scribe.models.scribe import Scribe
from care_scribe.models.scribe_quota import ScribeQuota

logger = logging.getLogger(__name__)


@shared_task
def finalize_live_transcription(session_external_id):
    session = Scribe.objects.filter(
        external_id=session_external_id,
        live=True,
        status=Scribe.Status.COMPLETED,
    ).first()

    if not session:
        logger.error("Live transcription Scribe %s not found or not completed.", session_external_id)
        return

    user_quota = ScribeQuota.objects.filter(
        user=session.requested_by, facility=session.requested_in_facility
    ).first()
    facility_quota = ScribeQuota.objects.filter(
        user=None, facility=session.requested_in_facility
    ).first()

    if user_quota:
        user_quota.calculate_used()
    if facility_quota:
        facility_quota.calculate_used()

    logger.info(
        "Finalized live transcription %s: input_tokens=%d, output_tokens=%d",
        session_external_id,
        session.chat_input_tokens or 0,
        session.chat_output_tokens or 0,
    )
