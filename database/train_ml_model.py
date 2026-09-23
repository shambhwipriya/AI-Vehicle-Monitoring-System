import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("traffic_ml_dataset.csv")

print("\nTraffic ML Dataset")
print("-" * 60)
print(df)

# Input and target
X = df[["previous_count"]]
y = df["vehicle_count"]

# Create Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X, y)

# Save model
joblib.dump(model, "traffic_prediction_model.pkl")

print("\nML Model Trained Successfully!")
print("Model: Random Forest Regressor")
print("Input: Previous minute vehicle count")
print("Target: Current minute vehicle count")
print("Saved as: traffic_prediction_model.pkl")

# Test prediction
test_value = [[160]]
prediction = model.predict(test_value)

print(f"\nExample:")
print(f"Previous minute vehicles: 160")
print(f"Predicted next-minute vehicles: {prediction[0]:.0f}")