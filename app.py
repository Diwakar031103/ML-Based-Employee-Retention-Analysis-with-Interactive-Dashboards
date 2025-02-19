import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("rf.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Employee Retention Prediction")

# Input fields for 9 features
satisfaction_level = st.number_input("Satisfaction Level", min_value=0.0, max_value=1.0, step=0.01)
last_evaluation = st.number_input("Last Evaluation", min_value=0.0, max_value=1.0, step=0.01)
number_project = st.number_input("Number of Projects", min_value=1, max_value=10, step=1)
average_montly_hours = st.number_input("Average Monthly Hours", min_value=50, max_value=350, step=1)
time_spend_company = st.number_input("Time Spent in Company", min_value=1, max_value=20, step=1)
work_accident = st.selectbox("Work Accident", [0, 1])
promotion_last_5years = st.selectbox("Promotion in Last 5 Years", [0, 1])
department = st.selectbox("Department", ["sales", "technical", "support", "IT", "hr", "accounting", "marketing", "product_mng", "management", "RandD"])
salary = st.selectbox("Salary Level", ["low", "medium", "high"])

# Encode department and salary manually (assuming they were label encoded during training)
department_mapping = {"sales": 0, "technical": 1, "support": 2, "IT": 3, "hr": 4,
                      "accounting": 5, "marketing": 6, "product_mng": 7, "management": 8, "RandD": 9}
salary_mapping = {"low": 0, "medium": 1, "high": 2}

department_encoded = department_mapping[department]
salary_encoded = salary_mapping[salary]

# Predict button
if st.button("Predict"):
    features = np.array([[satisfaction_level, last_evaluation, number_project, average_montly_hours,
                          time_spend_company, work_accident, promotion_last_5years, department_encoded, salary_encoded]])
    prediction = model.predict(features)
    st.write("Prediction:", "Will Leave" if prediction[0] == 1 else "Will Stay")


