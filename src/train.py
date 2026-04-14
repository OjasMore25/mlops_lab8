import os
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


ROLL_NO = "2022bcd0043"
NAME = "Ojas Ganesh More"


def main():
    # Load dataset
    df = pd.read_csv("data/housing.csv")

    # Target column (California Housing dataset target)
    target_col = "median_house_value"

    if target_col not in df.columns:
        raise Exception(f"Target column '{target_col}' not found in dataset!")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Print details for logs
    print("=========================================")
    print(f"Name: {NAME}")
    print(f"Roll No: {ROLL_NO}")
    print("=========================================")
    print(f"Dataset Size (Total Rows): {df.shape[0]}")
    print(f"Training Samples: {X_train.shape[0]}")
    print(f"Testing Samples: {X_test.shape[0]}")
    print("-----------------------------------------")
    print(f"RMSE: {rmse}")
    print(f"R2 Score: {r2}")
    print("=========================================")

    # Save metrics.json
    os.makedirs("results", exist_ok=True)

    metrics_data = {
        "roll_no": ROLL_NO,
        "name": NAME,
        "dataset_rows": int(df.shape[0]),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "rmse": float(rmse),
        "r2_score": float(r2)
    }

    with open("results/metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=4)

    print("Saved metrics to results/metrics.json")

    # GitHub Actions Summary Output
    if "GITHUB_STEP_SUMMARY" in os.environ:
        summary_path = os.environ["GITHUB_STEP_SUMMARY"]
        with open(summary_path, "a") as f:
            f.write(f"## Training Report (Lab 8)\n")
            f.write(f"**Name:** {NAME}\n\n")
            f.write(f"**Roll No:** {ROLL_NO}\n\n")
            f.write(f"### Dataset Info\n")
            f.write(f"- Total Rows: {df.shape[0]}\n")
            f.write(f"- Train Samples: {X_train.shape[0]}\n")
            f.write(f"- Test Samples: {X_test.shape[0]}\n\n")
            f.write(f"### Metrics\n")
            f.write(f"- RMSE: {rmse}\n")
            f.write(f"- R2 Score: {r2}\n\n")

        print("GitHub Summary updated successfully!")


if __name__ == "__main__":
    main()