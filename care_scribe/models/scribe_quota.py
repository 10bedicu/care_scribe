from care import facility
from care.utils.models.base import BaseModel
from care.facility.models.facility import Facility
from django.contrib.auth import get_user_model
from django.db import models
from django.utils import timezone
from django.core.exceptions import ValidationError
from django.db.models import Sum, F, ExpressionWrapper, IntegerField
User = get_user_model()

start_of_month = timezone.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
end_of_month = (start_of_month + timezone.timedelta(days=31)).replace(day=1) - timezone.timedelta(seconds=1)

class ScribeQuota(BaseModel):
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name="created_scribe_quota", null=True, blank=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="scribe_quota", null=True, blank=True)
    facility = models.ForeignKey(Facility, on_delete=models.CASCADE, related_name="scribe_quota", null=True, blank=True)
    tokens = models.IntegerField(default=0, help_text="Total tokens available for the facility monthly")
    tokens_per_user = models.IntegerField(default=0, help_text="Tokens available per user in the facility")
    used = models.IntegerField(default=0, help_text="Tokens used by the user/facility in the current month")
    allow_ocr = models.BooleanField(default=False, help_text="Whether the user/facility is allowed to use OCR features")
    allow_scribe = models.BooleanField(default=True, help_text="Whether the user/facility is allowed to use normal (form-fill) scribes")
    allow_notes_scribe = models.BooleanField(default=False, help_text="Whether the user/facility is allowed to use transcript-only scribes")


    tnc_hash = models.CharField(max_length=255, null=True, blank=True, help_text="Hash of the terms and conditions accepted by the user")
    tnc_accepted_date = models.DateTimeField(null=True, blank=True, help_text="Date when the terms and conditions were accepted by the user")

    # either user or facility must be set, not both
    class Meta:
        verbose_name = "Scribe Quota"
        verbose_name_plural = "Scribe Quotas"

    def clean(self):
        super().clean()
        if not self.user and not self.facility:
            raise ValidationError("Either 'user' or 'facility' must be set.")
        if self.user and self.facility and ScribeQuota.objects.filter(user=self.user, facility=self.facility).exclude(pk=self.pk).exists():
            raise ValidationError("The user already has a quota for this facility.")
        if self.facility and not self.user and ScribeQuota.objects.filter(facility=self.facility, user=None).exclude(pk=self.pk).exists():
            raise ValidationError("The facility already has a quota")
        if self.user and not self.facility and ScribeQuota.objects.filter(user=self.user, facility=None).exclude(pk=self.pk).exists():
            raise ValidationError("The user already has an independent quota")


    def calculate_used(self, from_date=start_of_month, to_date=end_of_month):
        qs = self.facility.scribe_set.filter(created_date__range=(from_date, to_date))
        if self.user:
            qs = qs.filter(requested_by=self.user)
        sum_expr = ExpressionWrapper(
            F('chat_input_tokens') + F('chat_output_tokens'),
            output_field=IntegerField()
        )
        total_tokens = qs.aggregate(total_tokens=Sum(sum_expr))['total_tokens'] or 0

        self.used = total_tokens
        self.save(update_fields=["used"])

    def __str__(self):
        return f"{self.user.username if self.user else self.facility.name} - {self.tokens} tokens"

    def save(self, *args, **kwargs):
        self.full_clean()  # Ensure clean is called before saving

        if self.deleted and self.facility and not self.user:
            ScribeQuota.objects.filter(facility=self.facility).exclude(pk=self.pk).delete()

        super().save(*args, **kwargs)
