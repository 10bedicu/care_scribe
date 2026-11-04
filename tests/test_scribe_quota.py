from concurrent.futures import ThreadPoolExecutor
from threading import Event

from care_scribe.models.scribe_quota import ScribeQuota
from care_scribe.serializers.scribe_quota import ScribeQuotaSerializer
from care_scribe.viewsets.scribe_quota import ScribeQuotaViewSet
from django.db import connection, connections, transaction
from django.test import TestCase, TransactionTestCase, skipUnlessDBFeature
from model_bakery import baker
from rest_framework.exceptions import ValidationError
from rest_framework.test import APIRequestFactory, force_authenticate

from care.facility.models.facility import Facility
from care.users.models import User


def quota_request(user, action, data):
    method = "patch" if action == "bulk_update" else "post"
    request = getattr(APIRequestFactory(), method)("/", data, format="json")
    force_authenticate(request, user=user)
    return ScribeQuotaViewSet.as_view({method: action})(request)


class ScribeQuotaTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.admin = baker.make(User, is_staff=True)
        cls.facility = baker.make(Facility, created_by=cls.admin)
        cls.quota = ScribeQuota.objects.create(
            facility=cls.facility, created_by=cls.admin, tokens=100, tokens_per_user=10
        )

    def test_bulk_update_rejects_token_overflow_without_writes(self):
        for field in ("tokens", "tokens_per_user"):
            for value in (2147483648, 10**30):
                with self.subTest(field=field, value=value):
                    with self.assertNumQueries(0):
                        response = quota_request(
                            self.admin,
                            "bulk_update",
                            {
                                "external_ids": [str(self.quota.external_id)],
                                field: value,
                            },
                        )
                    self.assertEqual(response.status_code, 400)
                    self.assertIn(field, response.data)
        self.quota.refresh_from_db()
        self.assertEqual(self.quota.tokens, 100)
        self.assertEqual(self.quota.tokens_per_user, 10)

    def test_bulk_create_rejects_token_overflow_without_writes(self):
        for field in ("tokens", "tokens_per_user"):
            with self.subTest(field=field):
                data = {
                    "facility_external_ids": [str(self.facility.external_id)],
                    "tokens": 100,
                    "tokens_per_user": 10,
                    field: 10**30,
                }
                with self.assertNumQueries(0):
                    response = quota_request(self.admin, "bulk", data)
                self.assertEqual(response.status_code, 400)
                self.assertIn(field, response.data)

    def test_bulk_update_accepts_token_bounds(self):
        for tokens, tokens_per_user in ((2147483647, 0), (1, 2147483647)):
            with self.subTest(tokens=tokens, tokens_per_user=tokens_per_user):
                response = quota_request(
                    self.admin,
                    "bulk_update",
                    {
                        "external_ids": [str(self.quota.external_id)],
                        "tokens": tokens,
                        "tokens_per_user": tokens_per_user,
                    },
                )
                self.assertEqual(response.status_code, 200)
                self.quota.refresh_from_db()
                self.assertEqual(self.quota.tokens, tokens)
                self.assertEqual(self.quota.tokens_per_user, tokens_per_user)

    def test_bulk_token_lower_bounds_are_preserved(self):
        for action in ("bulk", "bulk_update"):
            for field, value in (("tokens", 0), ("tokens_per_user", -1)):
                with self.subTest(action=action, field=field):
                    data = (
                        {
                            "facility_external_ids": [str(self.facility.external_id)],
                            "tokens": 100,
                            "tokens_per_user": 10,
                        }
                        if action == "bulk"
                        else {"external_ids": [str(self.quota.external_id)]}
                    )
                    data[field] = value
                    with self.assertNumQueries(0):
                        response = quota_request(self.admin, action, data)
                    self.assertEqual(response.status_code, 400)
                    self.assertIn(field, response.data)

    def test_single_create_can_replace_a_deleted_quota(self):
        self.quota.delete()
        response = quota_request(
            self.admin,
            "create",
            {"facility_external_id": str(self.facility.external_id), "tokens": 100},
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(ScribeQuota.objects.filter(facility=self.facility).count(), 1)

    def test_bulk_update_response_query_count_is_constant_for_500_quotas(self):
        facilities = Facility.objects.bulk_create(
            [
                Facility(name=f"Quota facility {index}", facility_type=2)
                for index in range(500)
            ]
        )
        quotas = ScribeQuota.objects.bulk_create(
            [
                ScribeQuota(facility=facility, created_by=self.admin, tokens=100)
                for facility in facilities
            ]
        )
        # A savepoint, row-lock query, update, release, and joined response query.
        with self.assertNumQueries(5):
            response = quota_request(
                self.admin,
                "bulk_update",
                {
                    "external_ids": [str(quota.external_id) for quota in quotas],
                    "tokens": 200,
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 500)
        for row in response.data:
            self.assertEqual(row["tokens"], 200)
            self.assertIsNotNone(row["facility"])
            self.assertEqual(row["created_by"]["username"], self.admin.username)

    def test_single_create_rechecks_uniqueness_after_validation(self):
        facility = baker.make(Facility, created_by=self.admin)
        serializer = ScribeQuotaSerializer(
            data={"facility_external_id": str(facility.external_id), "tokens": 100}
        )
        self.assertTrue(serializer.is_valid(), serializer.errors)
        response = quota_request(
            self.admin,
            "bulk",
            {
                "facility_external_ids": [str(facility.external_id)],
                "tokens": 100,
                "tokens_per_user": 10,
            },
        )
        self.assertEqual(response.status_code, 201)
        with self.assertRaises(ValidationError):
            serializer.save(created_by=self.admin)
        self.assertEqual(ScribeQuota.objects.filter(facility=facility).count(), 1)


@skipUnlessDBFeature("has_select_for_update")
class ScribeQuotaCreationConcurrencyTests(TransactionTestCase):
    def setUp(self):
        self.admin = baker.make(User, is_staff=True)
        self.facility = baker.make(Facility, created_by=self.admin)

    def assert_creation_is_serialized(self, first_action, second_action):
        lock_attempted = Event()
        single_data = {
            "facility_external_id": str(self.facility.external_id),
            "tokens": 100,
        }
        bulk_data = {
            "facility_external_ids": [str(self.facility.external_id)],
            "tokens": 100,
            "tokens_per_user": 10,
        }
        data = {"create": single_data, "bulk": bulk_data}

        def observe_lock(execute, sql, params, many, context):
            if "FOR UPDATE" in sql and Facility._meta.db_table in sql:  # noqa: SLF001
                lock_attempted.set()
            return execute(sql, params, many, context)

        def second_request():
            try:
                with connection.execute_wrapper(observe_lock), transaction.atomic():
                    return quota_request(self.admin, second_action, data[second_action])
            finally:
                connections.close_all()

        with ThreadPoolExecutor(max_workers=1) as executor:
            with transaction.atomic():
                Facility.objects.select_for_update(of=("self",)).get(
                    pk=self.facility.pk
                )
                future = executor.submit(second_request)
                self.assertTrue(
                    lock_attempted.wait(timeout=10),
                    "Concurrent creation must lock the facility before inserting.",
                )
                response = quota_request(self.admin, first_action, data[first_action])
                self.assertEqual(response.status_code, 201)
            response = future.result(timeout=10)

        self.assertEqual(response.status_code, 400)
        self.assertIn(
            "non_field_errors" if second_action == "create" else "facility_errors",
            response.data,
        )
        self.assertEqual(ScribeQuota.objects.filter(facility=self.facility).count(), 1)

    def test_single_create_waits_for_bulk_create(self):
        self.assert_creation_is_serialized("bulk", "create")

    def test_bulk_create_waits_for_single_create(self):
        self.assert_creation_is_serialized("create", "bulk")
