from django.urls import path
from .views import QuestionnaireWizard,dashboard,findings
from .forms import PersonalInfoForm, LifestyleForm, AdultOnlyForm
from django.views.generic import TemplateView

# Define form list for the wizard
FORMS = [
    ("personal", PersonalInfoForm),
    ("lifestyle", LifestyleForm),
    ("adult", AdultOnlyForm),
]

urlpatterns = [
      path("survey/", QuestionnaireWizard.as_view(FORMS), name="survey"),
      path("survey/success/", TemplateView.as_view(template_name="success.html"), name="survey-success"),    
      path('',  dashboard, name='dashboard'),
      path('findings/',  findings, name='findings'),
]
