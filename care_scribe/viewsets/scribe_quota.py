from rest_framework.mixins import (
    CreateModelMixin,
    ListModelMixin,
    RetrieveModelMixin,
    UpdateModelMixin,
    DestroyModelMixin
)
from rest_framework.permissions import IsAdminUser, IsAuthenticated
from rest_framework.viewsets import GenericViewSet

from care_scribe.models.scribe_quota import ScribeQuota
from care_scribe.serializers.scribe_quota import ScribeQuotaSerializer

from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import filters as rest_framework_filters
from rest_framework.pagination import LimitOffsetPagination
from rest_framework.decorators import action
from rest_framework.response import Response
from django_filters import rest_framework as filters
from care_scribe.settings import plugin_settings
from care_scribe.utils import hash_string
from care.facility.models.facility import Facility
from care.emr.models.organization import FacilityOrganizationUser
from django.utils import timezone

class ScribeQuotaFilter(filters.FilterSet):

    facility = filters.CharFilter(
        field_name="facility__name",
        lookup_expr="icontains",
        label="Facility Name",
    )

    username = filters.CharFilter(
        field_name="user__username",
        lookup_expr="icontains",
        label="Username",
    )


class ScribeQuotaViewSet(
    ListModelMixin,
    RetrieveModelMixin,
    CreateModelMixin,
    UpdateModelMixin,
    GenericViewSet,
    DestroyModelMixin
):
    queryset = ScribeQuota.objects.all()
    serializer_class = ScribeQuotaSerializer
    lookup_field = "external_id"
    permission_classes = [IsAdminUser]
    filter_backends = [
        DjangoFilterBackend,
        rest_framework_filters.OrderingFilter,
    ]
    filterset_class = ScribeQuotaFilter
    pagination_class = LimitOffsetPagination

    def get_queryset(self):
        queryset = super().get_queryset()
        facility_id = self.request.query_params.get("facility_id", None)

        if facility_id:
            # If facility_id is provided, return only quotas with linked users under that facility

            queryset = queryset.exclude(user=None).filter(facility__external_id=facility_id)
        else:
            # If no facility_id, return only quotas without linked users (facility-level quotas)
            queryset = queryset.filter(user=None)

        return queryset

    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)

    @action(detail=False, methods=["get"], url_path="my-quota", permission_classes=[IsAuthenticated])
    def my_quota(self, request):
        user = request.user
        facility_id = request.query_params.get("facility_id", None)

        if not facility_id:
            return Response({"detail": "Facility ID is required."}, status=400)

        tnc = plugin_settings.SCRIBE_TNC
        tnc_hash = hash_string(tnc)

        user_quota = ScribeQuota.objects.filter(user=user, facility__external_id=facility_id).first()
        facility_quota = ScribeQuota.objects.filter(facility__external_id=facility_id, user=None).first()
        accepted_tnc = user_quota.tnc_hash == tnc_hash if user_quota else False

        to_show = []
        if facility_quota:
            to_show.append(facility_quota)
        if user_quota:
            to_show.append(user_quota)

        serializer = self.get_serializer(to_show, many=True)
        return Response(
            {
                "quotas" : serializer.data,
                "tnc_accepted": accepted_tnc,
                "tnc" : tnc,
            })

    @action(detail=False, methods=["post"], url_path="accept-tnc", permission_classes=[IsAuthenticated])
    def accept_tnc(self, request):
        user = request.user
        facility_id = request.data.get("facility_id", None)

        if not facility_id:
            return Response({"detail": "Facility ID is required."}, status=400)

        tnc = plugin_settings.SCRIBE_TNC
        tnc_hash = hash_string(tnc)

        user_quota = ScribeQuota.objects.filter(user=user, facility__external_id=facility_id).first()

        if user_quota and user_quota.tnc_hash == tnc_hash:
            return Response({"detail": "Terms and Conditions already accepted."})

        if not user_quota:
            facility = Facility.objects.filter(external_id=facility_id).first()
            if not facility:
                return Response({"detail": "Facility does not exist."}, status=400)

            if not FacilityOrganizationUser.objects.filter(user=user, organization__facility=facility).exists():
                return Response({"detail": "User does not belong to the facility."}, status=403)

            facility_quota = ScribeQuota.objects.filter(facility=facility, user=None).first()
            if not facility_quota:
                return Response({"detail": "Facility does not have a quota."}, status=400)

            user_quota = ScribeQuota.objects.create(
                user=user,
                facility=facility,
                tokens=facility_quota.tokens_per_user,
                tnc_hash=tnc_hash,
                tnc_accepted_date=timezone.now(),
            )
        else:
            user_quota.tnc_hash = tnc_hash
            user_quota.tnc_accepted_date = timezone.now()
            user_quota.save()

        return Response({"detail": "Terms and Conditions accepted successfully."})
