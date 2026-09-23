import sqlite3
import pandas as pd

DATABASE_PATH = "vehicles.db"

connection = sqlite3.connect(DATABASE_PATH)

df = pd.read_sql_query(
    """
    SELECT timestamp, vehicle_count
    FROM traffic_data
    ORDER BY id
    """,
    connection
)

connection.close()

df["timestamp"] = pd.to_datetime(df["timestamp"])

# Previous minute vehicle count
df["previous_count"] = df["vehicle_count"].shift(1)

# Remove first row because it has no previous value
df = df.dropna()

df.to_csv(
    "traffic_ml_dataset.csv",
    index=False
)

print("\nML Dataset Created")
print("-" * 60)
print(df)
print("-" * 60)
print(f"Total ML records: {len(df)}")
print("\nSaved as: traffic_ml_dataset.csv")