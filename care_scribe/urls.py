from rest_framework.routers import DefaultRouter

from care_scribe.viewsets.scribe_quota import ScribeQuotaViewSet
from care_scribe.viewsets.scribe import ScribeViewset
from care_scribe.viewsets.scribe_file import FileUploadViewSet
from care_scribe.viewsets.live_transcription import LiveTranscriptionViewSet

router = DefaultRouter()
router.register("scribe", ScribeViewset)
router.register("quota", ScribeQuotaViewSet)
router.register("scribe_file", FileUploadViewSet)
router.register("live-transcription", LiveTranscriptionViewSet, basename="live-transcription")

urlpatterns = router.urls
