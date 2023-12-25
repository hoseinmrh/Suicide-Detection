from django.contrib import admin
from .models import SuicideTest,TestResult

class SuicideTestAdmin(admin.ModelAdmin):
    list_display = ["text"]
    list_filter = ["time"]
admin.site.register(SuicideTest,SuicideTestAdmin)

class TestResultAdmin(admin.ModelAdmin):
    list_display = ["test","suicidal_rate","non_suicidal_rate"]
admin.site.register(TestResult,TestResultAdmin)