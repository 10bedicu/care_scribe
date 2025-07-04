import jsonschema
from care.utils.models.base import BaseModel
from care.facility.models.facility import Facility
from django.contrib.auth import get_user_model
from django.db import models

User = get_user_model()

form_data_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "friendlyName": {"type": "string"},
            "id": {"type": "string"},
            "current": {"type": ["number","string","boolean","object","array", "null"]},
            "description": {"type": "string"},
            "type": {"type": "string"},
            "example": {"type": "string"},
            "options": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "anyOf": [
                                {"type": "integer"},
                                {"type": "string"},
                            ]
                        },
                        "text": {"type": "string"},
                    },
                    "required": ["id", "text"],
                },
            },
        },
        "required": ["friendlyName", "id", "description", "type", "example", "current"],
    },
}


def validate_json_schema(value):
    try:
        jsonschema.validate(value, form_data_schema)
    except jsonschema.ValidationError as e:
        raise jsonschema.ValidationError(f"Invalid JSON data: {e}")


class Scribe(BaseModel):
    class Status(models.TextChoices):
        CREATED = "CREATED"
        READY = "READY"
        GENERATING_TRANSCRIPT = "GENERATING_TRANSCRIPT"
        GENERATING_AI_RESPONSE = "GENERATING_AI_RESPONSE"
        COMPLETED = "COMPLETED"
        REFUSED = "REFUSED"
        FAILED = "FAILED"

    requested_by = models.ForeignKey(User, null=True, on_delete=models.SET_NULL)
    requested_in_facility = models.ForeignKey(Facility, null=True, on_delete=models.SET_NULL)

    form_data = models.JSONField(
        validators=[validate_json_schema], null=True, blank=True
    )
    transcript = models.TextField(null=True, blank=True)
    text = models.TextField(null=True, blank=True)
    ai_response = models.TextField(null=True, blank=True)
    status = models.CharField(
        max_length=50, choices=Status.choices, default=Status.CREATED
    )
    prompt = models.TextField(null=True, blank=True)

    @property
    def audio_file_ids(self):
        from care_scribe.models.scribe_file import ScribeFile

        return ScribeFile.objects.filter(
            associating_id=self.external_id,
            file_type=ScribeFile.FileType.SCRIBE_AUDIO,
            upload_completed=True,
        ).values_list("external_id", flat=True)

    @property
    def document_file_ids(self):
        from care_scribe.models.scribe_file import ScribeFile

        return ScribeFile.objects.filter(
            associating_id=self.external_id,
            file_type=ScribeFile.FileType.SCRIBE_DOCUMENT,
            upload_completed=True,
        ).values_list("external_id", flat=True)