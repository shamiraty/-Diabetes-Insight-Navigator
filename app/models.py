from django.db import models

class SurveyResponse(models.Model):
    AGE_GROUP_CHOICES = [
        ('adult', '18 and above'),
        ('child', 'Under 18'),
    ]

    EXERCISE_CHOICES = [
        ('daily', 'Daily'),
        ('weekly', 'Weekly'),
        ('monthly', 'Monthly'),
        ('annually', 'Annually'),
        ('unknown', 'Unknown'),
        ('other', 'Other'),
    ]

    EMPLOYMENT_CHOICES = [
        ('employed', 'Employed'),
        ('self_employed', 'Self-employed'),
        ('business', 'Business'),
        ('other', 'Other'),
    ]

    FOOD_CHOICES = [
        ('starch', 'Starch'),
        ('sugar', 'Sugar'),
        ('other', 'Other'),
    ]

    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True, blank=True, null=True)  # Optional but unique
    age = models.IntegerField()
    age_group = models.CharField(max_length=10, choices=AGE_GROUP_CHOICES, blank=True)

    # Fields for all age groups
    height = models.FloatField()
    weight = models.FloatField()
    food_preference = models.CharField(max_length=20, choices=FOOD_CHOICES)

    # Fields for adults only
    marital_status = models.CharField(max_length=20, blank=True, null=True)
    smoking = models.BooleanField(default=False)
    employment_status = models.CharField(max_length=20, choices=EMPLOYMENT_CHOICES, blank=True, null=True)

    # Exercise habits for all age groups
    body_exercise = models.CharField(max_length=20, choices=EXERCISE_CHOICES)

    submitted_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} - {self.age} years"
