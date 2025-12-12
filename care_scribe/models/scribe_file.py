import time
from uuid import uuid4

from django.contrib.auth import get_user_model
from django.db import models
from care.utils.models.base import BaseModel
from care.emr.utils.file_manager import S3FilesManager
from care.utils.csp.config import BucketType
from care.utils.models.validators import parse_file_extension

User = get_user_model()


class ScribeFile(BaseModel):
    class FileType(models.IntegerChoices):
        OTHER = 0
        SCRIBE_AUDIO = 1
        SCRIBE_DOCUMENT = 2

    name = models.CharField(max_length=2000)
    internal_name = models.CharField(max_length=2000)
    associating_id = models.CharField(max_length=100, blank=False, null=False)
    file_type = models.IntegerField(choices=FileType.choices, default=FileType.SCRIBE_AUDIO)
    mime_type = models.CharField(max_length=200, blank=True, null=True)
    signed_url = models.CharField(max_length=2000, blank=True, null=True)
    upload_completed = models.BooleanField(default=False)

    uploaded_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )

    meta = models.JSONField(default=dict)

    files_manager = S3FilesManager(BucketType.PATIENT)

    def get_extension(self):
        extensions = parse_file_extension(self.internal_name)
        return f".{'.'.join(extensions)}" if extensions else ""

    def save(self, *args, **kwargs):
        """
        Create a random internal name to internally manage the file
        This is used as an intermediate step to avoid leakage of PII in-case of data leak
        """
        skip_internal_name = kwargs.pop("skip_internal_name", False)
        if (not self.internal_name or not self.id) and not skip_internal_name:
            internal_name = str(uuid4()) + str(int(time.time()))
            if self.internal_name and (extension := self.get_extension()):
                internal_name = f"{internal_name}{extension}"
            self.internal_name = internal_name
        return super().save(*args, **kwargs)

    def __str__(self):
        return f"ScribeFile {self.external_id} - {self.name} - Scribe ID {self.associating_id}"
