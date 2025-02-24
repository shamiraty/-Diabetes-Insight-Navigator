from django import forms
from .models import SurveyResponse

class PersonalInfoForm(forms.Form):
    name = forms.CharField(
        label="Full Name", 
        max_length=100, 
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    email = forms.EmailField(
        label="Email Address", 
        required=False, 
        widget=forms.EmailInput(attrs={'class': 'form-control'})
    )  # Optional field

    age = forms.IntegerField(
        label="Age", 
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )

    def clean_email(self):
        email = self.cleaned_data.get("email")
        if email and SurveyResponse.objects.filter(email=email).exists():
            raise forms.ValidationError("This email is already used. Please use a different one.")
        return email

    def clean_age(self):
        age = self.cleaned_data["age"]
        if age < 0:
            raise forms.ValidationError("Age cannot be negative.")
        return age


class LifestyleForm(forms.Form):
    height = forms.FloatField(
        label="Height (cm)", 
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    weight = forms.FloatField(
        label="Weight (kg)", 
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )

    food_preference = forms.ChoiceField(
        label="Food Preference",
        choices=[("starch", "Starch"), ("sugar", "Sugar"), ("other", "Other (Specify)")],
        widget=forms.Select(attrs={'class': 'form-control'})
    )

    body_exercise = forms.ChoiceField(
        label="Body Exercise Frequency",
        choices=[
            ("daily", "Daily"),
            ("weekly", "Weekly"),
            ("monthly", "Monthly"),
            ("annually", "Annually"),
            ("unknown", "Unknown"),
            ("other", "Other (Specify)"),
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )


class AdultOnlyForm(forms.Form):
    marital_status = forms.ChoiceField(
        label="Marital Status", 
        choices=[
            ("single", "Single"), 
            ("married", "Married"), 
            ("divorced", "Divorced"), 
            ("widowed", "Widowed")
        ],
        required=False,
        widget=forms.Select(attrs={'class': 'form-control select2'})
    )
    
    smoking = forms.BooleanField(
        label="Do you smoke?", 
        required=False, 
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    employment_status = forms.ChoiceField(
        label="Employment Status",
        choices=[
            ("employed", "Employed"),
            ("self_employed", "Self-employed"),
            ("business", "Business"),
            ("other", "Other (Specify)"),
        ],
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
