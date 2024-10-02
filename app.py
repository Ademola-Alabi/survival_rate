import streamlit as st
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('best_gradient_boosting_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app definition
def main():
    st.title("Survival Prediction using Gradient Boosting")
    st.write("This app predicts survival status based on several clinical and sociodemographic features.")
    
    # Input fields for user to enter data
    age = st.number_input("Age Baseline", min_value=0, max_value=120, value=30)
    
    # One-hot encoding for 'sex' (male or female)
    sex = st.selectbox("Sex", ["Male", "Female"])
    sex_male = 1 if sex == "Male" else 0  # Binary encoding
    
    # One-hot encoding for 'education_baseline'
    education = st.selectbox("Education Level", ["2.upper secondary and vocational training", "1.less than upper secondary", "3.tertiary", "o.other"])
    education_upper_sec = 1 if education == "2.upper secondary and vocational training" else 0
    education_less_sec = 1 if education == "1.less than upper secondary" else 0
    education_tertiary = 1 if education == "3.tertiary" else 0
    education_other = 1 if education == "o.other" else 0
    
    # One-hot encoding for 'marital_status_baseline'
    marital_status = st.selectbox("Marital Status", ["1.married", "7.widowed", "5.divorced", "8.never married", "3.partnered", "4.separated"])
    married = 1 if marital_status == "1.married" else 0
    widowed = 1 if marital_status == "7.widowed" else 0
    divorced = 1 if marital_status == "5.divorced" else 0
    never_married = 1 if marital_status == "8.never married" else 0
    partnered = 1 if marital_status == "3.partnered" else 0
    separated = 1 if marital_status == "4.separated" else 0
    
    # Binary inputs for Yes/No columns
    mltc_bp_baseline = st.selectbox("Blood Pressure Baseline", ["yes", "no"])
    mltc_bp_baseline = 1 if mltc_bp_baseline == "yes" else 0
    
    mltc_diab_baseline = st.selectbox("Diabetes Baseline", ["yes", "no"])
    mltc_diab_baseline = 1 if mltc_diab_baseline == "yes" else 0
    
    mltc_cancer_baseline = st.selectbox("Cancer Baseline", ["yes", "no"])
    mltc_cancer_baseline = 1 if mltc_cancer_baseline == "yes" else 0
    
    mltc_lungdisease_baseline = st.selectbox("Lung Disease Baseline", ["yes", "no"])
    mltc_lungdisease_baseline = 1 if mltc_lungdisease_baseline == "yes" else 0
    
    medication = st.number_input("Medication Baseline", min_value=0, max_value=100, value=0)
    
    survived = st.selectbox("Survived", ["yes", "no"])
    survived = 1 if survived == "yes" else 0
    
    dead = st.selectbox("Dead", ["yes", "no"])
    dead = 1 if dead == "yes" else 0
    
    progfree_interval = st.number_input("Progression-Free Interval", min_value=0, value=0)
    
    progression_status = st.selectbox("Progression Status", ["Progressing", "Stable"])
    progression_status = 1 if progression_status == "Progressing" else 0
    
    survival_interval = st.number_input("Survival Interval", min_value=0, value=0)
    
    # Creating the input array with exactly 20 features (remove 2 extra features)
    input_data = np.array([
        age,
        sex_male,
        education_upper_sec,
        education_less_sec,
        education_tertiary,
        education_other,
        married,
        widowed,
        divorced,
        never_married,
        partnered,
        mltc_bp_baseline,
        mltc_diab_baseline,
        mltc_cancer_baseline,
        mltc_lungdisease_baseline,
        medication,
        survived,
        dead,
        progfree_interval,
        progression_status
        # Removed 2 extra features
    ]).reshape(1, -1)

    # Scale the input data
    scaled_input = scaler.transform(input_data)

    # Make the prediction
    prediction = model.predict(scaled_input)

    # Display the prediction
    if st.button("Predict"):
        st.write(f"The predicted survival status is: {prediction[0]}")

# Run the app
if __name__ == "__main__":
    main()
