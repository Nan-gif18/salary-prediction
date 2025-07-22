import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

#  Step 1: Load and preprocess dataset
print(" Loading and preprocessing dataset...")
df = pd.read_csv("Employers_data.csv")

#  Drop unnecessary columns if they exist
drop_cols = ['Name', 'Employee_ID', 'Email']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

#  Separate target and features
X = df.drop("Salary", axis=1)
y = df["Salary"]

# Encode categorical columns
encoders = {}
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Save encoders
os.makedirs("models", exist_ok=True)
joblib.dump(encoders, "models/encoders.pkl")

#  Train-test split
print(" Splitting data into train/test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Helper to evaluate and save model
def train_and_evaluate_model(name, model):
    print(f"\n Training {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    # For compatibility with older scikit-learn (use squared=True for RMSE calculation manually)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"✅ {name} RMSE: {rmse:.2f}")
    print(f"✅ {name} R² Score: {r2:.2f}")

    joblib.dump(model, f"models/{name.lower()}_model.pkl")

    #  Plot actual vs predicted
    plt.figure(figsize=(6,4))
    plt.scatter(y_test, preds, alpha=0.6, color='royalblue')
    plt.xlabel("Actual Salary")
    plt.ylabel("Predicted Salary")
    plt.title(f"{name} Prediction")
    plt.grid(True)
    plt.savefig(f"models/{name.lower()}_plot.png")
    plt.close()

# Train all models
train_and_evaluate_model("DecisionTree", DecisionTreeRegressor(random_state=42))
train_and_evaluate_model("RandomForest", RandomForestRegressor(random_state=42))
train_and_evaluate_model("XGBoost", XGBRegressor(random_state=42, verbosity=0))

print("\n✅ All models trained and saved in 'models/' folder.")
