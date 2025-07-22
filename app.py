
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("ds_salaries.csv")
    return df

df = load_data()

st.title("💼 Advanced Employee Salary Predictor")
st.subheader("Using DS Salaries Sample Dataset")

st.markdown("Select employee details below to predict the salary (in USD):")

# Sidebar inputs
experience = st.selectbox("Experience Level", df['experience_level'].unique())
job = st.selectbox("Job Title", sorted(df['job_title'].unique()))
company_size = st.selectbox("Company Size", df['company_size'].unique())
remote = st.slider("Remote Work (%)", 0, 100, 100, step=50)

# Preprocessing
features = ['experience_level', 'job_title', 'company_size', 'remote_ratio']
target = 'salary_in_usd'

X = df[features]
y = df[target]

# Preprocess categorical features
categorical_features = ['experience_level', 'job_title', 'company_size']
numerical_features = ['remote_ratio']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='passthrough')

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train model
X['remote_ratio'] = df['remote_ratio']
model.fit(X, y)

# Predict
input_df = pd.DataFrame([{
    'experience_level': experience,
    'job_title': job,
    'company_size': company_size,
    'remote_ratio': remote
}])
predicted_salary = model.predict(input_df)[0]
# Convert to INR (Assuming 1 USD = 83.0 INR)
usd_to_inr = 83.0
salary_in_inr = predicted_salary * usd_to_inr

st.success(f"💰 Estimated Salary: **${predicted_salary:,.2f} USD**")
