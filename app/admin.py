from django.contrib import admin
from .models import SurveyResponse

class SurveyResponseAdmin(admin.ModelAdmin):
    # Fields to be displayed in the list view
    list_display = ('name', 'email', 'age', 'age_group', 'height', 'weight', 'food_preference', 'body_exercise', 'submitted_at')
    
    # Fields to be used for filtering data in the admin panel
    list_filter = ('age_group', 'food_preference', 'body_exercise', 'age', 'submitted_at')
    
    # Fields to allow search functionality
    search_fields = ('name', 'email', 'age', 'age_group', 'food_preference')

    # Limits the number of records shown in the list view
    list_max_show_all = 10  # You can set any number here to limit max records
    list_per_page=10
    # Optional: Fieldsets for better organization (if required)
    fieldsets = (
        (None, {
            'fields': ('name', 'email', 'age', 'age_group', 'height', 'weight', 'food_preference', 'body_exercise')
        }),
        ('Additional Information', {
            'classes': ('collapse',),
            'fields': ('marital_status', 'smoking', 'employment_status')
        }),
         
    )

# Register the model with the custom admin
admin.site.register(SurveyResponse, SurveyResponseAdmin)
