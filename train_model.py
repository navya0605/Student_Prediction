# train_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Step 1: Data Collection
print("Step 1: Data Collection - Loading OULAD dataset files")

# Step 2: Load Dataset
print("\nStep 2: Load Dataset")
student_info = pd.read_csv('studentInfo.csv')
student_registration = pd.read_csv('studentRegistration.csv')
student_assessment = pd.read_csv('studentAssessment.csv')
assessments = pd.read_csv('assessments.csv')
courses = pd.read_csv('courses.csv')
vle = pd.read_csv('vle.csv')
student_vle = pd.read_csv('studentVle.csv', encoding='latin1')

# Step 3: Data Preprocessing
print("\nStep 3: Data Preprocessing")

# 3.1: Data Exploration
print("3.1: Data Exploration")
print("studentInfo shape:", student_info.shape)
print("studentInfo columns:", student_info.columns.tolist())

# 3.2: Handling Missing Values
print("\n3.2: Handling Missing Values")
def handle_missing_values(df, numerical_cols, categorical_cols):
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df

student_info_numerical = ['num_of_prev_attempts', 'studied_credits']
student_info_categorical = ['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability', 'final_result']
student_info = handle_missing_values(student_info, student_info_numerical, student_info_categorical)

student_registration_numerical = ['date_registration', 'date_unregistration']
student_registration_categorical = []
student_registration = handle_missing_values(student_registration, student_registration_numerical, student_registration_categorical)

student_assessment_numerical = ['score']
student_assessment_categorical = []
student_assessment = handle_missing_values(student_assessment, student_assessment_numerical, student_assessment_categorical)

assessments_numerical = ['date', 'weight']
assessments_categorical = ['assessment_type']
assessments = handle_missing_values(assessments, assessments_numerical, assessments_categorical)

courses_numerical = ['module_presentation_length']
courses_categorical = []
courses = handle_missing_values(courses, courses_numerical, courses_categorical)

vle_numerical = ['week_from', 'week_to']
vle_categorical = ['activity_type']
vle = handle_missing_values(vle, vle_numerical, vle_categorical)

student_vle_numerical = ['date', 'sum_click']
student_vle_categorical = []
student_vle = handle_missing_values(student_vle, student_vle_numerical, student_vle_categorical)

# 3.3: Handling Duplicates
print("\n3.3: Handling Duplicates")
student_info = student_info.drop_duplicates(subset=['id_student', 'code_module', 'code_presentation'])
student_registration = student_registration.drop_duplicates(subset=['id_student', 'code_module', 'code_presentation'])
student_assessment = student_assessment.drop_duplicates(subset=['id_student', 'id_assessment'])
assessments = assessments.drop_duplicates(subset=['id_assessment'])
courses = courses.drop_duplicates(subset=['code_module', 'code_presentation'])
vle = vle.drop_duplicates(subset=['id_site'])

student_vle = student_vle.groupby(['id_student', 'code_module', 'code_presentation', 'id_site', 'date'])['sum_click'].sum().reset_index()

# 3.4: Data Info
print("\n3.4: Data Info")
print("studentInfo types:\n", student_info.dtypes)

# 3.5: Summary Statistics
print("\n3.5: Summary Statistics")
print("studentInfo summary:\n", student_info.describe())

# 3.6: Handling Outliers
print("\n3.6: Handling Outliers")
def handle_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df

student_assessment = handle_outliers_iqr(student_assessment, 'score')
student_vle = handle_outliers_iqr(student_vle, 'sum_click')

# 3.7: Generating New Columns
print("\n3.7: Generating New Columns")
student_vle_agg = student_vle.groupby(['id_student', 'code_module', 'code_presentation'])['sum_click'].sum().reset_index()
student_vle_agg = student_vle_agg.rename(columns={'sum_click': 'total_clicks'})

student_registration['days_to_start'] = student_registration['date_registration'].apply(lambda x: abs(x) if x < 0 else 0)

# Merge Datasets
print("\nMerging Datasets")
df = student_info.copy()
df = pd.merge(df, student_registration, on=['id_student', 'code_module', 'code_presentation'], how='left')

assessment_data = pd.merge(student_assessment, assessments, on='id_assessment', how='left')
assessment_agg = assessment_data.groupby(['id_student', 'code_module', 'code_presentation'])['score'].mean().reset_index()
assessment_agg = assessment_agg.rename(columns={'score': 'avg_score'})
df = pd.merge(df, assessment_agg, on=['id_student', 'code_module', 'code_presentation'], how='left')

vle_data = pd.merge(student_vle, vle, on='id_site', how='left')
df = pd.merge(df, student_vle_agg, on=['id_student', 'code_module', 'code_presentation'], how='left')

df = pd.merge(df, courses, on=['code_module', 'code_presentation'], how='left')

df['avg_score'] = df['avg_score'].fillna(df['avg_score'].median())
df['total_clicks'] = df['total_clicks'].fillna(0)
df['days_to_start'] = df['days_to_start'].fillna(df['days_to_start'].median())

# Step 4: Data Visualization
print("\nStep 4: Data Visualization")
plt.figure(figsize=(10, 6))
sns.histplot(df['total_clicks'], kde=True)
plt.title('Histogram & KDE of Total Clicks')
plt.savefig('static/total_clicks_hist.png')
plt.close()

plt.figure(figsize=(8, 5))
sns.countplot(x='final_result', data=df)
plt.title('Distribution of Final Result')
plt.savefig('static/final_result_dist.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(x='final_result', y='avg_score', data=df)
plt.title('Average Score vs Final Result')
plt.savefig('static/avg_score_vs_result.png')
plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(df[['avg_score', 'total_clicks', 'studied_credits', 'days_to_start']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('static/correlation_heatmap.png')
plt.close()

# Step 5: Prepare Data for Modeling
print("\nStep 5: Prepare Data for Modeling")

# Define all possible categories for categorical columns
gender_categories = ['M', 'F']
disability_categories = ['Y', 'N']
highest_education_categories = [
    'No Formal quals', 'Lower Than A Level', 'A Level or Equivalent',
    'HE Qualification', 'Post Graduate Qualification'
]
age_band_categories = ['0-35', '35-55', '55<=']

# Initialize OrdinalEncoder with predefined categories
encoders = {
    'gender': OrdinalEncoder(categories=[gender_categories]),
    'disability': OrdinalEncoder(categories=[disability_categories]),
    'highest_education': OrdinalEncoder(categories=[highest_education_categories]),
    'age_band': OrdinalEncoder(categories=[age_band_categories]),
    'final_result': OrdinalEncoder() 
}

# Encode categorical columns
df['gender'] = encoders['gender'].fit_transform(df[['gender']].astype(str))
df['disability'] = encoders['disability'].fit_transform(df[['disability']].astype(str))
df['highest_education'] = encoders['highest_education'].fit_transform(df[['highest_education']].astype(str))
df['age_band'] = encoders['age_band'].fit_transform(df[['age_band']].astype(str))
df['final_result'] = encoders['final_result'].fit_transform(df[['final_result']].astype(str))

features = ['avg_score', 'total_clicks', 'studied_credits', 'days_to_start', 'gender', 'disability', 'highest_education', 'age_band']
X = df[features]
y = df['final_result']

scaler = StandardScaler()
X[['avg_score', 'total_clicks', 'studied_credits', 'days_to_start']] = scaler.fit_transform(
    X[['avg_score', 'total_clicks', 'studied_credits', 'days_to_start']]
)


# Save the preprocessed dataset
print("\nSaving Preprocessed Dataset")
preprocessed_df = X.copy()
preprocessed_df['final_result'] = y
preprocessed_df['id_student'] = df['id_student']  # Keep student ID for lookup
joblib.dump(preprocessed_df, 'models/preprocessed_data.pkl')
print("Preprocessed dataset saved as 'models/preprocessed_data.pkl'")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Model Selection
print("\nStep 6: Model Selection")
model = RandomForestClassifier(random_state=42)

# Step 7: Model Performance Evaluation
print("\nStep 7: Model Performance Evaluation")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 8: Parameter Hyper-tuning
print("\nStep 8: Parameter Hyper-tuning")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print("\nUpdated Model Accuracy:", accuracy_score(y_test, y_pred_best))
print("\nUpdated Classification Report:\n", classification_report(y_test, y_pred_best))

# Step 9: Model Saving
print("\nStep 9: Model Saving")
joblib.dump(best_model, 'models/oulad_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(encoders, 'models/encoders.pkl')  
print("Model, scaler, and encoders saved successfully")