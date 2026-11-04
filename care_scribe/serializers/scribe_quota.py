from django.db import transaction
from django.utils import timezone
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
            "allow_scribe",
            "allow_notes_scribe",
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


class ScribeQuotaBulkCreateSerializer(serializers.Serializer):
    facility_external_ids = serializers.ListField(
        child=serializers.UUIDField(), allow_empty=False, max_length=500
    )
    tokens = serializers.IntegerField(min_value=1)
    tokens_per_user = serializers.IntegerField(min_value=0)
    allow_ocr = serializers.BooleanField(default=False)
    allow_scribe = serializers.BooleanField(default=True)
    allow_notes_scribe = serializers.BooleanField(default=False)

    def create(self, validated_data):
        facility_ids = list(dict.fromkeys(validated_data.pop("facility_external_ids")))

        with transaction.atomic():
            # Row locks prevent concurrent requests from creating duplicate facility quotas.
            facilities = {
                facility.external_id: facility
                for facility in Facility.objects.select_for_update(of=("self",)).filter(
                    external_id__in=facility_ids
                )
            }
            existing = set(
                ScribeQuota.objects.filter(
                    facility__in=facilities.values(), user=None
                ).values_list("facility__external_id", flat=True)
            )

            errors = {}
            for facility_id in facility_ids:
                if facility_id not in facilities:
                    errors[str(facility_id)] = "Facility does not exist."
                elif facility_id in existing:
                    errors[str(facility_id)] = "A Scribe Quota already exists for this facility."
            if errors:
                raise serializers.ValidationError({"facility_errors": errors})

            return [
                ScribeQuota.objects.create(facility=facilities[facility_id], **validated_data)
                for facility_id in facility_ids
            ]


class ScribeQuotaBulkUpdateSerializer(serializers.Serializer):
    external_ids = serializers.ListField(
        child=serializers.UUIDField(), allow_empty=False, max_length=500
    )
    tokens = serializers.IntegerField(min_value=1, required=False)
    tokens_per_user = serializers.IntegerField(min_value=0, required=False)
    allow_ocr = serializers.BooleanField(required=False)
    allow_scribe = serializers.BooleanField(required=False)
    allow_notes_scribe = serializers.BooleanField(required=False)

    def validate(self, attrs):
        if len(attrs) == 1:
            raise serializers.ValidationError("At least one field to update must be provided.")
        return attrs

    def update_quotas(self):
        changes = dict(self.validated_data)
        quota_ids = list(dict.fromkeys(changes.pop("external_ids")))
        # Bulk edit only applies to facility-level quotas, matching the quota list.
        queryset = ScribeQuota.objects.filter(user=None, external_id__in=quota_ids)

        with transaction.atomic():
            found = {quota.external_id for quota in queryset.select_for_update()}
            errors = {
                str(quota_id): "Quota does not exist."
                for quota_id in quota_ids
                if quota_id not in found
            }
            if errors:
                raise serializers.ValidationError({"quota_errors": errors})

            queryset.update(**changes, modified_date=timezone.now())

        return list(queryset)


class ScribeQuotaBulkDeleteSerializer(serializers.Serializer):
    external_ids = serializers.ListField(
        child=serializers.UUIDField(), allow_empty=False, max_length=500
    )

    def delete_quotas(self):
        quota_ids = list(dict.fromkeys(self.validated_data["external_ids"]))

        with transaction.atomic():
            quotas = {
                quota.external_id: quota
                for quota in ScribeQuota.objects.select_for_update().filter(
                    user=None, external_id__in=quota_ids
                )
            }
            errors = {
                str(quota_id): "Quota does not exist."
                for quota_id in quota_ids
                if quota_id not in quotas
            }
            if errors:
                raise serializers.ValidationError({"quota_errors": errors})

            # Per-instance delete so the model also removes the facility's user quotas.
            for quota in quotas.values():
                quota.delete()

        return len(quotas)
