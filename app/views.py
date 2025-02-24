
import numpy as np
import pandas as pd
from django.shortcuts import render
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import math
from .models import SurveyResponse
from formtools.wizard.views import SessionWizardView
from django.shortcuts import redirect
from .forms import PersonalInfoForm, LifestyleForm, AdultOnlyForm
from .models import SurveyResponse

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    log_loss,
    r2_score,
    mean_squared_error,
)

class QuestionnaireWizard(SessionWizardView):
    form_list = [PersonalInfoForm, LifestyleForm, AdultOnlyForm]
    template_name = "index.html"

    def get_form_list(self):
        """ Ondoa AdultOnlyForm kama age ≤ 18 """
        personal_data = self.storage.get_step_data("0")  # Step 0 (PersonalInfoForm)
        if personal_data:
            age = int(personal_data.get("0-age", 0))
            if age <= 18:
                return [PersonalInfoForm, LifestyleForm]  # Usihusishe AdultOnlyForm
        return super().get_form_list()

    def done(self, form_list, **kwargs):
        form_data = {key: value for form in form_list for key, value in form.cleaned_data.items()}
        age = form_data["age"]
        age_group = "adult" if age > 18 else "child"

        SurveyResponse.objects.create(
            name=form_data["name"],
            email=form_data.get("email", None),
            age=age,
            age_group=age_group,
            height=form_data["height"],
            weight=form_data["weight"],
            food_preference=form_data["food_preference"],
            body_exercise=form_data["body_exercise"],
            marital_status=form_data.get("marital_status", "") if age > 18 else "",
            smoking=form_data.get("smoking", False) if age > 18 else False,
            employment_status=form_data.get("employment_status", "") if age > 18 else "",
        )

        return redirect("survey-success")




# Function to estimate population interval with confidence interval
def interval_population_estimation(data, population_size=1000000, alpha=0.05):
    sample_mean = data.mean()
    sample_std = data.std()
    sample_size = len(data)
    standard_error = sample_std / math.sqrt(sample_size)
    margin_of_error = stats.t.ppf(1 - alpha / 2, sample_size - 1) * standard_error
    confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)
    
    fpcf = math.sqrt((population_size - sample_size) / (population_size - 1))
    adjusted_margin_of_error = margin_of_error * fpcf
    
    return sample_mean, sample_std, confidence_interval, adjusted_margin_of_error, standard_error, sample_size

def dashboard(request):
    # Fetch survey responses from the database
    responses = SurveyResponse.objects.all()

    # Prepare data for analysis
    data = pd.DataFrame(list(responses.values()))
    population_size = 1000000 
    
    # Encoding categorical features for ML
    label_encoder = LabelEncoder()
    categorical_columns = ['age_group', 'food_preference', 'marital_status', 'smoking', 'employment_status', 'body_exercise']
    
    for column in categorical_columns:
        data[column] = label_encoder.fit_transform(data[column].fillna('Unknown'))
    
    # Select features for prediction
    features = ['age', 'height', 'weight', 'food_preference', 'marital_status', 'smoking', 'employment_status', 'body_exercise']
    X = data[features]
    y = data['age_group']  # Target is age_group (adult or child), can change to diabetes risk prediction

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train Logistic Regression Model
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train)
    logistic_pred = logistic_model.predict(X_test)
    logistic_accuracy = accuracy_score(y_test, logistic_pred)
    
    logistic_accuracy = accuracy_score(y_test, logistic_pred)
    logistic_precision = precision_score(y_test, logistic_pred, average='weighted')
    logistic_recall = recall_score(y_test, logistic_pred, average='weighted')
    logistic_f1 = f1_score(y_test, logistic_pred, average='weighted')
    logistic_confusion = confusion_matrix(y_test, logistic_pred)
    logistic_probabilities = logistic_model.predict_proba(X_test)[:, 1] #get probablities for ROC/AUC, Log loss.
    logistic_auc = roc_auc_score(y_test, logistic_probabilities)
    logistic_log_loss = log_loss(y_test, logistic_probabilities)

    # Get Logistic Regression Coefficients
    logistic_coefficients = logistic_model.coef_
    logistic_intercept = logistic_model.intercept_
      
    # Train Random Forest Classifier Model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    # Train KMeans Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    data['Cluster'] = kmeans.labels_

    # Perform ANOVA for weight and exercise habits
    anova_results = stats.f_oneway(data['weight'], data['body_exercise'])
    
    # Get F-value and P-value
    anova_f_value = anova_results.statistic
    anova_p_value = anova_results.pvalue
    anova_significance = "Significant" if anova_p_value < 0.05 else "Not Significant"
    
    # Confidence interval estimation (example for 'weight')
    weight_data = data['weight']
    weight_mean, weight_std, weight_conf_interval, weight_margin_error, weight_standard_error, weight_sample_size = interval_population_estimation(weight_data, population_size)

    # Linear Regression Example (predicting weight from height)
    height_data = data['height'].values.reshape(-1, 1)
    weight_data_regression = data['weight'].values.reshape(-1, 1)
    
    linear_model = LinearRegression()
    linear_model.fit(height_data, weight_data_regression)
    weight_pred_regression = linear_model.predict(height_data)
    
    r2 = r2_score(weight_data_regression, weight_pred_regression)
    mse = mean_squared_error(weight_data_regression, weight_pred_regression)
    slope = linear_model.coef_[0][0]
    intercept = linear_model.intercept_[0]
    
    sse = mse * len(weight_data_regression)

    # Pass height and weight lists to the template
    data_height_list = data['height'].tolist()
    data_weight_list = data['weight'].tolist()

    # Render dashboard with all metrics
    return render(request, 'dashboard.html', {
        
        'logistic_accuracy': logistic_accuracy,
        'logistic_precision': logistic_precision,
        'logistic_recall': logistic_recall,
        'logistic_f1': logistic_f1,
        'logistic_confusion': logistic_confusion.tolist(), # Convert NumPy array to list
        'logistic_auc': logistic_auc,
        'logistic_log_loss': logistic_log_loss,
        'logistic_coefficients': logistic_coefficients.tolist(), # Convert NumPy array to list
        'logistic_intercept': logistic_intercept.tolist(),# Convert NumPy array to list
        
        
        'logistic_accuracy': logistic_accuracy,
        'rf_accuracy': rf_accuracy,
        'kmeans_clusters': data['Cluster'].value_counts().to_dict(),
        'anova_f_value': anova_f_value,
        'anova_p_value': anova_p_value,
        'anova_significance': anova_significance,
        'weight_mean': weight_mean,
        'weight_std': weight_std,
        'weight_conf_interval': weight_conf_interval,
        'weight_margin_error': weight_margin_error,
        'weight_standard_error': weight_standard_error,
        'weight_sample_size': weight_sample_size,
        'population_size': population_size,
        'r2': r2,
        'mse': mse,
        'slope': slope,
        'intercept': intercept,
        'sse': sse,
        'data_height_list': data_height_list,
        'data_weight_list': data_weight_list,
    })
    
    
def findings(request):
    return render(request,'findings.html')