import joblib
import pandas as pd

# Load models and encoders
model = joblib.load("models/xgboost_model.pkl")
encoders = joblib.load("models/encoders.pkl")

# ✅ Replace with your custom input
test_input = {
    'Gender': 'Female',
    'Department': 'Engineering',
    'Job_Title': 'Engineer',
    'Education_Level': 'Bachelor',
    'Location': 'Austin',
    'Experience_Years': 3,
    'Age': 26
}

# Create DataFrame
df = pd.DataFrame([test_input])

# Encode categorical fields
for col in encoders:
    df[col] = encoders[col].transform(df[col])

# ✅ Ensure feature order matches training
expected_features = ['Age', 'Gender', 'Department', 'Job_Title', 'Experience_Years', 'Education_Level', 'Location']
df = df[expected_features]

# Predict salary
predicted_salary = model.predict(df)[0]
print(f"💰 Predicted Salary: ₹{predicted_salary:,.2f}")
