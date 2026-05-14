from rest_framework import serializers

from care.emr.models.questionnaire import Questionnaire
from care_scribe.models.scribe_question import ScribeQuestionnaireInstruction

class ScribeQuestionnaireInstructionsSerializer(serializers.ModelSerializer):

    questionnaire_id = serializers.CharField(source='questionnaire.external_id', read_only=True)
    add_questionnaire_id = serializers.CharField(write_only=True, required=False)
    questionnaire_title = serializers.CharField(source='questionnaire.title', read_only=True)
    questionnaire_slug = serializers.CharField(source='questionnaire.slug', read_only=True)

    class Meta:
        model = ScribeQuestionnaireInstruction
        fields = (
            "external_id",
            "questionnaire_id",
            "add_questionnaire_id",
            "questionnaire_title",
            "questionnaire_slug",
            "instructions",
            "created_date",
            "modified_date",
        )

        read_only_fields = ("external_id", "created_date", "modified_date", "questionnaire_title", "questionnaire_id", "questionnaire_slug")

    def validate(self, attrs):
        questionnaire_id = attrs.pop("add_questionnaire_id", None)

        if not self.instance:
            if not questionnaire_id:
                raise serializers.ValidationError("The 'questionnaire_id' must be provided.")

            questionnaire = Questionnaire.objects.filter(external_id=questionnaire_id).first()
            if not questionnaire:
                raise serializers.ValidationError(f"Questionnaire does not exist.")

            if ScribeQuestionnaireInstruction.objects.filter(questionnaire=questionnaire).exists():
                raise serializers.ValidationError(f"Questionnaire {questionnaire.title} already has Scribe instructions.")

            attrs["questionnaire"] = questionnaire

        return super().validate(attrs)
