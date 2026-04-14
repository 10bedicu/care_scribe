from rest_framework import serializers

from care.facility.models.facility import Facility
from care_scribe.serializers.base import FacilitySerializer
from care_scribe.models.scribe_quota import ScribeQuota
from care_scribe.serializers.scribe import ScribeUserSerializer

class ScribeQuotaSerializer(serializers.ModelSerializer):
    facility = FacilitySerializer(read_only=True)
    facility_external_id = serializers.CharField(write_only=True, required=False)
    user = ScribeUserSerializer(read_only=True)
    created_by = ScribeUserSerializer(read_only=True)

    class Meta:
        model = ScribeQuota
        fields = (
            "external_id",
            "user",
            "facility",
            "tokens",
            "tokens_per_user",
            "used",
            "allow_ocr",
            "enable_live_transcription",
            "created_by",
            "created_date",
            "modified_date",
            "facility_external_id",
            "tnc_hash",
            "tnc_accepted_date",
        )

        read_only_fields = ("external_id", "used", "created_by", "created_date", "modified_date", "tnc_hash", "tnc_accepted_date")

    def validate(self, attrs):
        facility_external_id = attrs.pop("facility_external_id", None)

        if not self.instance:
            if not facility_external_id:
                raise serializers.ValidationError("The 'facility_external_id' must be provided.")

            facility = Facility.objects.filter(external_id=facility_external_id).first()
            if not facility:
                raise serializers.ValidationError(f"Facility does not exist.")

            if ScribeQuota.objects.filter(facility__external_id=facility_external_id, user=None).exists():
                raise serializers.ValidationError(f"A Scribe Quota already exists for this facility.")

            attrs["facility"] = facility

        return super().validate(attrs)
