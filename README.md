## DIABETES INSIGHT NAVIGATOR:
## COMPREHENSIVE REAL-TIME HEALTH ANALYTICS & RISK PREDICTION SYSTEM
### AUTOMATED SCIENTIFIC RESEARCH TRENDS, AND PREDICTION 

  <img src="https://github.com/user-attachments/assets/90e6bb44-455c-4079-9a4d-59feadce6fdb" alt="logo7" width="100" height="100" />
### Contact Information

- **Location**: Dodoma, Tanzania
- **Email**: [sashashamsia@gmail.com](mailto:sashashamsia@gmail.com)
- **WhatsApp**: [+255675839840](https://wa.me/255675839840)
  <!--- **Youtube**: [Videos](https://www.youtube.com/channel/UCjepDdFYKzVHFiOhsiVVffQ)-->


| Icon | Rank | Professional Target Audience                                |
|------|------|------------------------------------------------------------|
| 🔬  | 1    | Healthcare Researchers (Epidemiologists, Clinical Researchers) |
| 🩺  | 2    | Physicians (General Practitioners and Specialists)            |
| 🏛️  | 3    | Public Health Officials (Health Department Personnel, Policy Makers) |
| 🏢  | 4    | Hospital Administrators                                      |
| 👩‍⚕️ | 5    | Nurses                                                     |
| 📊  | 6    | Data Analysts in Healthcare                                   |
| 👨‍💼 | 7    | Medical Directors                                          |
| 💊  | 8    | Pharmacists                                                 |
| 🏥  | 9    | Clinic Administrators                                        |
| 🎓  | 10   | Medical Educators (Professors of medicine)                   |

## Disclaimer

> The patient records used to train and test the **Symptom-Matcher AI** model are **not real patient data**. They are **fictitious data** that have been generated for educational purposes only. These records have no connection to any actual individuals or real-life medical conditions. 

> The diseases and symptoms displayed in the application are **for learning and demonstration purposes**. They do not represent actual medical diagnoses and should not be interpreted as such. The use of these simulated cases is intended solely for academic and training purposes.

> This application and its content are not intended to diminish or disrespect the real-world medical field, institutions, or individuals. The information presented is purely hypothetical and should not be used for making medical decisions. The primary goal of this project is to provide a platform for learning and development in the field of machine learning and healthcare technology.

# Introduction

This application is designed to collect data from respondents through a form and analyze it in real-time. It is intended for quantitative research with a specific focus on a case study related to diabetes. The data collected includes personal details, lifestyle habits, and other relevant factors that may influence health outcomes. The analysis uses various statistical methods and machine learning models to derive insights from the collected data.

The goal of this study is to analyze the collected data in real-time using various statistical and machine learning techniques. By doing this, we hope to:
- Identify key risk factors for diabetes,
- Understand early warning signs and symptoms,
- Provide insights for early intervention and prevention strategies.

# Problem 

Diabetes is increasingly becoming a major health challenge in our community. Despite growing awareness about its impact, many individuals remain at risk due to a combination of unhealthy lifestyle choices, poor dietary habits, physical inactivity, and socio-economic factors. The problem is compounded by the fact that many people are unaware of the early warning signs and risk factors, leading to delayed diagnoses and severe complications.

Key challenges include:

- **High Prevalence:** Diabetes rates are rising, affecting a significant portion of the population, which places a heavy burden on individuals, families, and healthcare systems.
- **Unhealthy Lifestyle:** Sedentary behavior, poor nutritional choices, and lack of regular exercise contribute heavily to the development of diabetes.
- **Lack of Awareness:** Many individuals do not recognize the early symptoms or understand the risk factors that contribute to the onset of diabetes.
- **Delayed Intervention:** Without early detection and timely intervention, diabetes can lead to serious health complications, increasing both the personal and economic costs.

This application, **Diabetes Insight Navigator: Comprehensive Real-Time Health Analytics & Risk Prediction System**, addresses these challenges by collecting real-time data from respondents through structured forms. By analyzing factors such as age, weight, height, food preferences, exercise habits, and other lifestyle indicators, the system aims to:

- Identify and monitor key risk factors for diabetes.
- Provide insights into the early signs and potential causes of diabetes.
- Support healthcare providers and policymakers in designing targeted prevention and intervention strategies.


# Main Purpose

The primary objective of this application is to gather and process data from respondents to estimate and predict health outcomes, such as the risk of developing diabetes. Through the analysis of factors like age, body mass index (BMI), lifestyle habits, and exercise frequency, the app will provide actionable insights into health risks.

---

# Methodology

The development of **Diabetes Insight Navigator: Comprehensive Real-Time Health Analytics & Risk Prediction System** involves a blend of modern web development, data processing, machine learning, statistical analysis, and responsive UI design. The following table summarizes the key technologies and libraries used in this project:

| Technology/Library         | Description                                                      | Usage in the Project                                          |
|----------------------------|------------------------------------------------------------------|---------------------------------------------------------------|
| **NumPy**                  | A library for numerical computing and array operations.          | Performs mathematical calculations and manages numerical data. |
| **Pandas**                 | A library for data manipulation and analysis with DataFrames.      | Processes survey responses and transforms data for analysis.  |
| **Django**                 | A high-level Python web framework for building web applications.   | Manages the web interface, form submissions, and data routing.  |
| **Django FormTools**       | Utilities for creating multi-step forms.                           | Implements multi-step survey forms to collect user data.       |
| **Scikit-Learn**           | A machine learning library offering various algorithms.            | Builds and evaluates models (Logistic Regression, Random Forest, Linear Regression, KMeans Clustering) and computes evaluation metrics. |
| **SciPy**                  | A library for scientific computing and statistical analysis.       | Performs statistical tests (e.g., ANOVA) and calculates confidence intervals. |
| **Math (Python's built-in)** | Provides basic mathematical functions.                             | Handles calculations like margin of error and square roots.    |
| **HTML**                   | The standard markup language for web pages.                        | Used to structure the web interface and display forms, data, and results. |
| **JavaScript**             | A programming language for dynamic and interactive content.        | Enables real-time updates on the page (AJAX requests for data fetching, dynamic form handling, etc.). |
| **Bootstrap**              | A front-end framework for responsive web design.                   | Provides styling and layout components (forms, buttons, cards, etc.) for an aesthetically pleasing and user-friendly interface. |
| **SQLite**                 | A lightweight, serverless SQL database engine.                     | Stores and manages survey responses and any other data required for the project. |

## Detailed Process

1. **Data Collection:**  
   - The system utilizes Django forms to gather information from respondents via a multi-step form approach.  
   - Forms are divided into sections (personal information, lifestyle details, and adult-specific questions) to ensure comprehensive data collection.

2. **Data Preparation:**  
   - Collected data is structured into Pandas DataFrames for easy manipulation and analysis.  
   - Categorical variables are encoded using Scikit-Learn's `LabelEncoder`, preparing the data for machine learning.

3. **Model Building and Evaluation:**  
   - **Logistic Regression** and **Random Forest** classifiers are implemented to predict diabetes risk (or categorize respondents based on risk factors).  
   - These models are evaluated using metrics such as accuracy, precision, recall, F1-score, AUC, and log loss.
   - **Linear Regression** is used to explore the relationship between height and weight.
   - **KMeans Clustering** identifies natural groupings within the data.

4. **Statistical Analysis:**  
   - An ANOVA test (using SciPy) examines the relationship between weight and exercise habits.  
   - Confidence intervals and margins of error are calculated to estimate population parameters from the sample data.

5. **Dashboard and Reporting:**  
   - The processed data and analysis results are displayed on a dashboard using Django's templating system.  
   - This dashboard provides real-time insights, enabling users to understand key trends and relationships in the data.

6. **Responsive Design & User Interface:**  
   - **HTML** structures the content of the web pages, including forms, tables, and charts.  
   - **Bootstrap** ensures the website is mobile-friendly and visually appealing, offering components like buttons, cards, and modals.  
   - **JavaScript** enables dynamic content updates (such as real-time survey submissions, AJAX fetching, and instant data visualization).

7. **Database Management:**  
   - **SQLite** is used as a lightweight, serverless database to store survey responses and user-generated data. The database ensures that all collected data is persisted, processed, and accessible for analysis.

By combining these methodologies, the system leverages advanced web and data science technologies to deliver comprehensive insights into diabetes risk factors. This integration helps healthcare providers, researchers, and policymakers make informed decisions for early intervention and prevention strategies.

### Forms for Data Collection

1. **Personal Information Form**  
   This form captures basic personal details, such as the respondent's name, email, and age. It includes validation checks for the email address to ensure uniqueness, as well as checks for age to ensure it is a valid value.

2. **Lifestyle Form**  
   This form collects information about the respondent's height, weight, food preferences, and body exercise frequency. These factors can significantly influence health outcomes and are essential for the study of diabetes-related risk factors.

3. **Adult-Only Form**  
   This form is targeted at respondents over the age of 18 and collects additional data about marital status, smoking habits, and employment status. These factors are also considered significant in determining the likelihood of diabetes or other related health conditions.

---

### Data Processing and Analysis

1. **Logistic Regression Model**  
   Logistic regression is used to predict the likelihood of respondents being categorized as "adult" or "child" based on collected data such as age, height, weight, and lifestyle choices. The model calculates metrics such as accuracy, precision, recall, and F1-score to evaluate the prediction performance.

2. **Random Forest Classifier**  
   A random forest model is used for classification tasks. It is trained on the data collected from the respondents to predict the age group or other health-related outcomes. The model's performance is also evaluated based on its accuracy.

3. **KMeans Clustering**  
   KMeans clustering is applied to the data to identify patterns and group similar respondents together. This unsupervised machine learning technique helps identify clusters of individuals who share similar health behaviors or characteristics.

4. **ANOVA Test**  
   ANOVA (Analysis of Variance) is performed to test whether there is a statistically significant difference between different groups of respondents based on variables like weight and exercise habits.

5. **Linear Regression Model**  
   Linear regression is applied to predict one variable (weight) based on another (height). This helps identify relationships between variables and can be used to estimate values for respondents with missing data or for future predictions.

6. **Interval Population Estimation**  
   The interval estimation method is used to calculate a confidence interval for a population parameter, such as the average weight of respondents. This method helps estimate the range within which the true population parameter likely falls, providing insights into health trends across larger populations.

---

### Dashboard

The dashboard provides a summary of the analysis, including:

- Logistic Regression metrics (accuracy, precision, recall, F1-score, etc.)
- Random Forest Classifier accuracy
- KMeans clustering results
- ANOVA test results (F-value, p-value, significance)
- Confidence intervals for health metrics like weight
- Linear regression results (slope, intercept, R2 score, etc.)

The data displayed on the dashboard enables users to explore relationships between different health factors, identify trends, and make informed decisions regarding health management, especially with a focus on diabetes prevention.

---

By integrating real-time data collection and advanced statistical analysis, this application offers valuable insights into the health risks associated with lifestyle choices and provides a foundation for further research into diabetes and related conditions.

# Diabetes Risk Prediction – Understanding the Results

We built a model to predict the likelihood of diabetes using different machine learning techniques. Below is a simplified explanation of what we found.

---

## **1. Logistic Regression (Predicting Diabetes Risk)**

- **Accuracy: 100%**  
  The model correctly predicted every case, meaning it made no mistakes. While this sounds great, such perfect accuracy is highly unusual and could indicate an issue with the dataset or model.

- **Precision, Recall, and F1-Score: 1.00**  
  These are all perfect scores, meaning the model made no false predictions. Again, this is rare and suggests we should check the dataset for errors, data leakage, or an overly simple dataset.

- **AUC (Area Under the Curve): 1.00**  
  The model perfectly separates those with diabetes from those without, reinforcing its unusual accuracy.

- **Log Loss: 0.00**  
  This means the model’s confidence in its predictions was perfect, another sign of a potential issue.

- **Intercept: 15.98**  
  This number represents the model’s baseline prediction when all other factors are zero. However, because our model is too perfect, this number doesn’t tell us much by itself.

---

## **2. Random Forest (Alternative Prediction Model)**

- **Accuracy: 100%**  
  Just like logistic regression, this model also predicted every case correctly. This further raises concerns about possible data leakage or overfitting.

---

## **3. K-Means Clustering (Finding Patterns in the Data)**

- The data was grouped into **three clusters**:  
  - **Cluster 2:** 109 responses  
  - **Cluster 0:** 105 responses  
  - **Cluster 1:** 86 responses

✅ This means the algorithm found natural groupings in the data, but we need to analyze how these clusters relate to diabetes risk.

---

## **4. ANOVA Test (Relationship Between Weight and Exercise)**

- **F-Value: 7577.23**  
  This is a very high number, showing a strong relationship between weight and exercise.

- **P-Value: 0.000**  
  A p-value below 0.05 means the relationship is statistically significant. In simple terms, weight and exercise habits are closely related.

---

## **5. Weight Confidence Interval (How Reliable Is the Weight Data?)**

- **Mean Weight:** 70.37 kg  
  The average weight of people in the study.

- **Standard Deviation:** 13.38 kg  
  This tells us how much individual weights vary from the average.

- **Confidence Interval:** 68.85 kg to 71.89 kg  
  We are **95% confident** that the true average weight in the population falls within this range.

- **Margin of Error:** 1.52 kg  
  This is the potential difference between the sample mean and the true population mean.

- **Sample Size:** 300  
  The number of people used in this analysis.

- **Population Size:** 1,000,000  
  The total number of people we are trying to make predictions for.

✅ Since the sample size is relatively small compared to the total population, we should be cautious when generalizing the results.

---

## **6. Regression Analysis (Relationship Between Height and Weight)**

- **R² (Coefficient of Determination):** 0.516  
  This means height explains **about 51.6%** of the variation in weight. There is a **moderate** relationship between the two.

- **Slope (β₁):** 65.90  
  For every **1-unit** increase in height, weight increases by **65.90 units** on average.

- **Intercept:** -45.74  
  This is the point where the line crosses the y-axis.

- **Regression Equation:**  
  **y = 65.90 * x - 45.74**  
  This is the formula we can use to predict weight based on height.

- **Sum of Squared Errors (SSE):** 25,912.47  
  This measures the total error in the model.

- **Mean Squared Error (MSE):** 86.37  
  This is the average squared error in predictions.

✅ The model shows a moderate correlation between height and weight, meaning taller people tend to weigh more, but height alone does not fully predict weight.

---

## **Key Takeaways & Concerns**

🚩 **Perfect Accuracy in Diabetes Prediction Models (100%)**

- This is highly suspicious and requires investigation.
- Possible reasons:
  1. **Data Leakage** – The model may have accidentally "seen" the correct answers during training.
  2. **Overfitting** – The model may have memorized the data instead of learning useful patterns.
  3. **Too Simple Data** – If there’s an obvious pattern in the dataset, the model might easily separate cases of diabetes and non-diabetes.

✅ **Height and Weight Have a Moderate Relationship**

- Height explains about **51.6%** of the variation in weight.
- Other factors like diet, lifestyle, and genetics likely play a role.

✅ **Weight and Exercise Are Strongly Related**

- The ANOVA test confirms that weight and exercise habits are statistically linked.

✅ **The Study Sample is Small**

- The dataset has **300** responses, but we are trying to predict for **1,000,000** people.
- A larger sample size would give more reliable predictions.

---

## **Final Thoughts**

The results suggest that while our model is technically "perfect," something may be wrong with the data or methodology. Before using this model in real-world applications, we must:

1. **Double-check the dataset** for issues like duplicate entries or leaked information.
2. **Re-evaluate the training process** to ensure proper data splitting.
3. **Test the model on completely unseen data** to confirm its real-world accuracy.

While the height-weight relationship and weight-exercise findings seem reasonable, the diabetes prediction models require further validation before they can be trusted.

---

