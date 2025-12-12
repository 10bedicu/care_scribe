import jsonschema
from care.utils.models.base import BaseModel
from care.facility.models.facility import Facility
from care.emr.models.encounter import Encounter
from django.contrib.auth import get_user_model
from django.db import models

User = get_user_model()

form_data_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "description": {"type": "string"},
            "fields": {
                "type": "array",
                "items": {
                    "anyOf": [
                        {
                            # Field object
                            "type": "object",
                            "properties": {
                                "friendlyName": {"type": "string"},
                                "id": {"type": "string"},
                                "current": {"type": ["number", "string", "boolean", "object", "array", "null"]},
                                "type": {"type": "string"},
                                "structuredType": {
                                    "anyOf": [
                                        {"type": "string"},
                                        {"type": "null"},
                                    ]
                                },
                                "repeats": {"type": "boolean"},
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
                            "required": ["friendlyName", "id", "type", "current"],
                        },
                        {
                            # Nested questionnaire object
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "description": {"type": "string"},
                                "fields": {"$ref": "#/items/properties/fields"},
                            },
                            "required": ["title", "fields"],
                        },
                    ]
                },
            },
        },
        "required": ["title", "fields"],
    },
}

meta_schema = {
    "type": "object",
    "properties": {
        "provider": {
            "type": "string",
            "enum": ["google", "openai", "azure"],
        },
        "transcription_time": {"type": "integer"},
        "completion_output_tokens": {"type": "integer"},
        "completion_input_tokens": {"type": "integer"},
        "completion_time": {"type": "integer"},
        "completion_id": {"type": "string"},
    },
}


def validate_json_schema(value):
    try:
        jsonschema.validate(value, form_data_schema)
    except jsonschema.ValidationError as e:
        raise jsonschema.ValidationError(f"Invalid JSON data: {e}")


def validate_json_schema_meta(value):
    try:
        jsonschema.validate(value, meta_schema)
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
    requested_in_encounter = models.ForeignKey(Encounter, null=True, on_delete=models.SET_NULL)

    form_data = models.JSONField(validators=[validate_json_schema], null=True, blank=True)
    transcript = models.TextField(null=True, blank=True)
    text = models.TextField(null=True, blank=True)
    ai_response = models.JSONField(null=True, blank=True, default=dict)
    status = models.CharField(max_length=50, choices=Status.choices, default=Status.CREATED)
    prompt = models.TextField(null=True, blank=True)
    meta = models.JSONField(null=True, blank=True, default=dict, validators=[validate_json_schema_meta])
    chat_model = models.CharField(max_length=100, null=True, blank=True)
    audio_model = models.CharField(max_length=100, null=True, blank=True)
    chat_model_temperature = models.FloatField(null=True, blank=True)

    is_feedback_positive = models.BooleanField(null=True, blank=True, help_text="Whether the user has given positive feedback on the AI response")
    feedback_comments = models.TextField(null=True, blank=True, help_text="Details of the feedback provided by the user")
    chat_input_tokens = models.IntegerField(null=True, blank=True, help_text="Number of tokens used in the chat input")
    chat_output_tokens = models.IntegerField(null=True, blank=True, help_text="Number of tokens used in the chat output")

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

    def __str__(self):
        return f"Scribe {self.external_id} - {self.status}"
