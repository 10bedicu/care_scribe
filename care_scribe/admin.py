from django.contrib import admin

from care_scribe.models.scribe_question import ScribeQuestionnaireInstruction
from care_scribe.models.scribe_quota import ScribeQuota
from care_scribe.models.scribe import Scribe
from care_scribe.models.scribe_file import ScribeFile

@admin.register(Scribe)
class ScribeAdmin(admin.ModelAdmin):
    search_fields = ["requested_by__username", "transcript", "status"]


@admin.register(ScribeFile)
class ScribeFileAdmin(admin.ModelAdmin):
    search_fields = ["file_type", "uploaded_by__username"]

@admin.register(ScribeQuota)
class ScribeQuotaAdmin(admin.ModelAdmin):
    search_fields = ["user__username", "facility__name"]

@admin.register(ScribeQuestionnaireInstruction)
class ScribeQuestionnaireInstructionsAdmin(admin.ModelAdmin):
    search_fields = ["questionnaire__title"]
