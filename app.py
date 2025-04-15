import joblib
import numpy as np
import pandas as pd
import gradio as gr

# Load models and scaler once for performance
print("🔄 Loading models...")
rf_model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
grid_model = joblib.load('logistic_regression_model.pkl')
print("✅ Models loaded!")

# Define the features expected by the model
features = ["Pclass", "Sex", "Age", "Fare_log", "FamilySize", "IsAlone"]

def predict_survival(pclass, sex, age, fare, sibsp, parch):
    print("🚀 Received input:", pclass, sex, age, fare, sibsp, parch)

    try:
        # Sanitize and prepare input
        sex_val = 0 if sex == "male" else 1
        fare = max(0, fare or 0)
        fare_log = np.log1p(fare)
        family_size = sibsp + parch + 1
        is_alone = 1 if (sibsp + parch) == 0 else 0

        input_data = pd.DataFrame([[pclass, sex_val, age, fare_log, family_size, is_alone]],
                                  columns=features)
        print("📦 Input DataFrame:", input_data.to_dict(orient='records'))

        # Transform and predict
        input_scaled = scaler.transform(input_data)
        prediction = grid_model.predict(input_scaled)[0]
        print("✅ Prediction:", prediction)

        return "✅ Survived" if prediction == 1 else "❌ Did Not Survive"

    except Exception as e:
        print("❌ Error during prediction:", str(e))
        return f"⚠️ Prediction Error: {str(e)}"

# Build Gradio interface
interface = gr.Interface(
    fn=predict_survival,
    inputs=[
        gr.Dropdown([1, 2, 3], label="Passenger Class"),
        gr.Radio(["male", "female"], label="Sex"),
        gr.Number(label="Age"),
        gr.Number(label="Fare"),
        gr.Number(label="Siblings/Spouses Aboard"),
        gr.Number(label="Parents/Children Aboard")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Titanic Survival Prediction",
    description="Enter passenger details to predict survival."
)

# Launch the app with queueing (but no sharing UI)
if __name__ == "__main__":
    interface.launch(share=False)
