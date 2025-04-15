
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models and scaler
st.write("🔄 Loading models...")
rf_model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
grid_model = joblib.load("logistic_regression_model.pkl")
st.success("✅ Models loaded!")

# Define features
features = ["Pclass", "Sex", "Age", "Fare_log", "FamilySize", "IsAlone"]

# UI
st.title("Titanic Survival Prediction")
st.markdown("Enter passenger details to predict survival using a logistic regression model.")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0.0, max_value=100.0, step=1.0)
fare = st.number_input("Fare", min_value=0.0, step=1.0)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, step=1)
parch = st.number_input("Parents/Children Aboard", min_value=0, step=1)

if st.button("Predict Survival"):
    try:
        sex_val = 0 if sex == "male" else 1
        fare_log = np.log1p(fare)
        family_size = sibsp + parch + 1
        is_alone = 1 if (sibsp + parch) == 0 else 0

        input_data = pd.DataFrame([[pclass, sex_val, age, fare_log, family_size, is_alone]], columns=features)
        input_scaled = scaler.transform(input_data)
        prediction = grid_model.predict(input_scaled)[0]

        if prediction == 1:
            st.success("✅ Survived")
        else:
            st.error("❌ Did Not Survive")

    except Exception as e:
        st.warning(f"⚠️ Prediction Error: {str(e)}")
