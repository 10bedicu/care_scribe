from care.facility.models.file_upload import BaseFileUpload
from django.contrib.auth import get_user_model
from django.db import models

User = get_user_model()


class ScribeFile(BaseFileUpload):
    class FileType(models.IntegerChoices):
        OTHER = 0
        SCRIBE_AUDIO = 1
        SCRIBE_DOCUMENT = 2

    file_type = models.IntegerField(choices=FileType.choices, default=FileType.SCRIBE_AUDIO)
    uploaded_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )
    