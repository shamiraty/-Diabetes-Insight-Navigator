from django.core.management.base import BaseCommand
import random
from faker import Faker
from app.models import SurveyResponse  # Update with your app name

class Command(BaseCommand):
    help = 'Delete all existing survey response records and generate new ones'

    def handle(self, *args, **kwargs):
        fake = Faker()
        age_groups = ['adult', 'child']
        exercise_choices = ['daily', 'weekly', 'monthly', 'annually', 'unknown', 'other']
        employment_choices = ['employed', 'self_employed', 'business', 'other']
        food_choices = ['starch', 'sugar', 'other']

        num_records = 300  # Adjust as needed

        # Delete existing records
        SurveyResponse.objects.all().delete()
        self.stdout.write(self.style.WARNING('All existing records deleted.'))

        new_records = []
        for _ in range(num_records):
            name = fake.name()
            email = fake.email()
            age = random.randint(10, 80)
            age_group = 'adult' if age >= 18 else 'child'
            height = round(random.uniform(1.5, 2.0), 2)  # Height between 1.5m and 2.0m

            # Assign base weight with variation based on height
            if height < 1.6:
                weight = random.randint(45, 65)
            elif height < 1.7:
                weight = random.randint(50, 75)
            elif height < 1.8:
                weight = random.randint(55, 85)
            elif height < 1.9:
                weight = random.randint(60, 95)
            else:
                weight = random.randint(65, 100)

            food_preference = random.choice(food_choices)

            marital_status = None
            employment_status = None
            if age >= 18:
                marital_status = random.choice(['single', 'married', 'divorced', 'widowed'])
                employment_status = random.choice(employment_choices)

            body_exercise = random.choice(exercise_choices)
            smoking = random.choice([True, False])

            new_records.append(SurveyResponse(
                name=name,
                email=email,
                age=age,
                age_group=age_group,
                height=height,
                weight=weight,
                food_preference=food_preference,
                marital_status=marital_status,
                smoking=smoking,
                employment_status=employment_status,
                body_exercise=body_exercise
            ))

        # Bulk insert for better performance
        SurveyResponse.objects.bulk_create(new_records)

        self.stdout.write(self.style.SUCCESS(f'{num_records} new records generated successfully!'))
