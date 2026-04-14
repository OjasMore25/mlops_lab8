import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


ROLL_NO = "2022bcd0043"
NAME = "Ojas Ganesh More"

# Load dataset
data = pd.read_csv("data/housing.csv")

# Drop Address column (not useful for regression directly)
if "Address" in data.columns:
    data = data.drop(columns=["Address"])

# Fill missing values
data = data.fillna(data.mean(numeric_only=True))

# Target column
target_col = "Price"

# Split features and target
X = data.drop(columns=[target_col])
y = data[target_col]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

# Save metrics
os.makedirs("results", exist_ok=True)

metrics = {
    "name": NAME,
    "roll_no": ROLL_NO,
    "dataset_rows": int(data.shape[0]),
    "train_samples": int(X_train.shape[0]),
    "test_samples": int(X_test.shape[0]),
    "rmse": float(rmse),
    "r2_score": float(r2)
}

with open("results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("=======================================")
print(f"Name: {NAME}")
print(f"Roll No: {ROLL_NO}")
print("=======================================")
print(f"Dataset Rows: {data.shape[0]}")
print(f"RMSE: {rmse}")
print(f"R2 Score: {r2}")
print("Training completed successfully!")
print("Model saved to models/model.pkl")
print("Metrics saved to results/metrics.json")