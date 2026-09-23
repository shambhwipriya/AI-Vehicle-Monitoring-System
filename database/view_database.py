import sqlite3

DATABASE_PATH = "vehicles.db"


connection = sqlite3.connect(DATABASE_PATH)
cursor = connection.cursor()


cursor.execute("""
    SELECT id, vehicle_type, track_id, confidence, timestamp
    FROM vehicles
    ORDER BY id
""")


records = cursor.fetchall()


print("\nVehicle Detection Records")
print("-" * 80)

for record in records:
    print(
        f"ID: {record[0]} | "
        f"Vehicle: {record[1]} | "
        f"Track ID: {record[2]} | "
        f"Confidence: {record[3]:.2f} | "
        f"Time: {record[4]}"
    )


print("-" * 80)
print(f"Total Records: {len(records)}")


connection.close()