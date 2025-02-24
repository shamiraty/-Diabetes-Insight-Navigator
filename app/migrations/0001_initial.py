# Generated by Django 5.0.6 on 2025-02-23 09:21

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='SurveyResponse',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('email', models.EmailField(max_length=254)),
                ('age', models.IntegerField()),
                ('age_group', models.CharField(blank=True, choices=[('adult', '18 and above'), ('child', 'Under 18')], max_length=10)),
                ('height', models.FloatField()),
                ('weight', models.FloatField()),
                ('food_preference', models.CharField(choices=[('starch', 'Starch'), ('sugar', 'Sugar'), ('other', 'Other')], max_length=20)),
                ('marital_status', models.CharField(blank=True, max_length=20, null=True)),
                ('smoking', models.BooleanField(default=False)),
                ('employment_status', models.CharField(blank=True, choices=[('employed', 'Employed'), ('self_employed', 'Self-employed'), ('business', 'Business'), ('other', 'Other')], max_length=20, null=True)),
                ('body_exercise', models.CharField(choices=[('daily', 'Daily'), ('weekly', 'Weekly'), ('monthly', 'Monthly'), ('annually', 'Annually'), ('unknown', 'Unknown'), ('other', 'Other')], max_length=20)),
                ('submitted_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
