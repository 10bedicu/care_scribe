from rest_framework import serializers
from care.facility.models.facility import Facility
from care.emr.models.encounter import Encounter
from care.emr.models.patient import Patient
from care.users.models import User
from care_scribe.serializers.base import FacilitySerializer
from care_scribe.models.scribe import Scribe
from care_scribe.models.scribe_file import ScribeFile
from care_scribe.serializers.scribe_file import ScribeFileUploadUpdateSerializer


class ScribePatientSerializer(serializers.ModelSerializer):
    class Meta:
        model = Patient
        fields = [
            "external_id",
            "name",
        ]


class ScribeEncounterSerializer(serializers.ModelSerializer):
    patient = ScribePatientSerializer(read_only=True)

    class Meta:
        model = Encounter
        fields = [
            "external_id",
            "patient",
        ]


class ScribeUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["first_name", "username", "last_name", "read_profile_picture_url"]


class ScribeSerializer(serializers.ModelSerializer):

    requested_in_facility_id = serializers.CharField(write_only=True, required=False)
    requested_in_encounter_id = serializers.CharField(write_only=True, required=False)
    requested_in_facility = FacilitySerializer(read_only=True)
    requested_in_encounter = ScribeEncounterSerializer(read_only=True)
    requested_by = ScribeUserSerializer(read_only=True)
    processed_ai_response = serializers.JSONField(write_only=True, required=False)
    benchmark = serializers.BooleanField(
        write_only=True,
        required=False,
        help_text="Whether the scribe request is for benchmarking purposes.",
    )

    audio = serializers.SerializerMethodField()
    documents = serializers.SerializerMethodField()

    class Meta:
        model = Scribe
        fields = [
            "external_id",
            "requested_by",
            "requested_in_facility",
            "requested_in_facility_id",
            "requested_in_encounter",
            "requested_in_encounter_id",
            "transcript",
            "ai_response",
            "status",
            "form_data",
            "audio",
            "prompt",
            "text",
            "meta",
            "created_date",
            "modified_date",
            "documents",
            "processed_ai_response",
            "benchmark",
            "chat_model",
            "audio_model",
            "chat_model_temperature",
            "live",
            "is_feedback_positive",
            "feedback_comments",
        ]
        read_only_fields = [
            "external_id",
            "requested_by",
            "requested_in_facility",
            "ai_response",
            "audio",
            "created_date",
            "modified_date",
            "documents",
            "live",
        ]

    def save(self, **kwargs):
        facility_id = self.validated_data.get("requested_in_facility_id", None)
        encounter_id = self.validated_data.get("requested_in_encounter_id", None)
        processed_ai_response = self.validated_data.pop("processed_ai_response", None)
        benchmark = self.validated_data.pop("benchmark", False) or (self.instance.meta.get("benchmark", False) if self.instance else False)

        user = self.context["request"].user

        if not user:
            raise serializers.ValidationError({"user": "User is required to create a scribe request."})

        if facility_id:
            self.validated_data["requested_in_facility"] = Facility.objects.filter(external_id=facility_id).first()
        if encounter_id:
            self.validated_data["requested_in_encounter"] = Encounter.objects.filter(external_id=encounter_id).first()
        if processed_ai_response:
            # update the latest processing entry with the processed_ai_response
            latest_processing = self.instance.meta.get("processings", [])[-1] if self.instance and self.instance.meta.get("processings", []) else {}
            latest_processing["processed_ai_response"] = processed_ai_response
            # update meta with the updated processings list
            self.validated_data["meta"] = {
                **self.instance.meta,
                "processings": [
                    *(self.instance.meta.get("processings", [])[:-1] if self.instance and self.instance.meta.get("processings", []) else []),
                    latest_processing
                ]} if self.instance else {"processings": [latest_processing]}

        if benchmark:
            if user.is_superuser:
                self.validated_data["meta"] = {**(self.instance.meta if self.instance else {}), "benchmark": benchmark}
            else:
                raise serializers.ValidationError(
                    {"benchmark": "You do not have permission to create a benchmark scribe request."}
                )

        if (self.validated_data.get("chat_model", None) or self.validated_data.get("audio_model", None) or self.validated_data.get("chat_model_temperature", None)) and not user.is_superuser:
            raise serializers.ValidationError(
                {"chat_model": "You do not have permission to set custom chat or audio models."}
            )

        if not benchmark:
            if (self.instance and not self.instance.requested_in_facility) and not self.validated_data["requested_in_facility"]:
                raise serializers.ValidationError({"requested_in_facility": "Invalid facility ID"})

            if (self.instance and not self.instance.requested_in_encounter) and not self.validated_data["requested_in_encounter"]:
                raise serializers.ValidationError({"requested_in_encounter": "Invalid encounter ID"})

        self.validated_data.pop("requested_in_facility_id", None)
        self.validated_data.pop("requested_in_encounter_id", None)

        return super().save(**kwargs)

    def get_audio(self, obj):
        audio_file_ids = obj.audio_file_ids
        if not audio_file_ids:
            return []

        audio_files = ScribeFile.objects.filter(external_id__in=audio_file_ids)

        return ScribeFileUploadUpdateSerializer(audio_files, many=True).data if audio_files else []

    def get_documents(self, obj):
        document_file_ids = obj.document_file_ids
        if not document_file_ids:
            return []

        document_files = ScribeFile.objects.filter(external_id__in=document_file_ids)

        return ScribeFileUploadUpdateSerializer(document_files, many=True).data if document_files else []
