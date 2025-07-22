import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load("salary_model.pkl")

# Set page configuration
st.set_page_config(page_title="Salary Prediction", layout="centered")

# Apply custom background color using CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center;'>Salary Prediction</h1>", unsafe_allow_html=True)

# Sidebar inputs
with st.form(key="prediction_form"):
    age = st.number_input("Age:", min_value=18, max_value=90, step=1)
    hours = st.number_input("Working Hours Per Week:", min_value=1, max_value=100, step=1)

    workclass = st.selectbox("Work Class:", [
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", 
        "State Government", "Without-pay", "Never Worked"
    ])

    country = st.selectbox("Country:", [
        "United States", "India", "Mexico", "Germany", "Philippines", "England"
    ])

    submit = st.form_submit_button("Predict Salary")

# Mapping categories to integers (simplified dummy encoding)
workclass_mapping = {
    "Private": 0, "Self-emp-not-inc": 1, "Self-emp-inc": 2,
    "Federal-gov": 3, "Local-gov": 4, "State Government": 5,
    "Without-pay": 6, "Never Worked": 7
}

country_mapping = {
    "United States": 0, "India": 1, "Mexico": 2,
    "Germany": 3, "Philippines": 4, "England": 5
}

# Prediction logic
if submit:
    try:
        workclass_val = workclass_mapping[workclass]
        country_val = country_mapping[country]

        # Create input DataFrame
        user_data = pd.DataFrame([[age, hours, workclass_val, country_val]],
                                 columns=["age", "hours-per-week", "workclass", "native-country"])


        # Feature scaling (optional depending on training)
        scaler = StandardScaler()
        user_data_scaled = scaler.fit_transform(user_data)

        # Predict salary class
        prediction = model.predict(user_data_scaled)
        salary_class = ">50K" if prediction[0] == 1 else "<=50K"

        # Display result
        st.markdown(f"<h4 style='color:green; text-align: center;'>Predicted Salary Class: {salary_class}</h4>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")
