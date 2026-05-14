import jsonschema
from care.utils.models.base import BaseModel
from care.emr.models import Questionnaire
from django.db import models

form_data_schema = {
  "type": "object",
  "properties": {
    "instructions": {
      "type": "string"
    },
    "questions": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {
            "type": "string"
          },
          "instructions": {
            "type": "string"
          }
        },
        "required": ["id", "instructions"],
        "additionalProperties": False
      }
    }
  },
  "required": ["instructions", "questions"],
  "additionalProperties": False
}

def validate_json_schema(value):
    try:
        jsonschema.validate(value, form_data_schema)
    except jsonschema.ValidationError as e:
        raise jsonschema.ValidationError(f"Invalid JSON data: {e}")

class ScribeQuestionnaireInstruction(BaseModel):
    questionnaire = models.ForeignKey(
        Questionnaire,
        on_delete=models.CASCADE,
        related_name="scribe_questionnaires",
    )
    instructions = models.JSONField(default=dict, validators=[validate_json_schema])

    def __str__(self):
        return f"{self.questionnaire.title}"
