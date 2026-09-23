import sqlite3
from datetime import datetime

DATABASE_PATH = "database/vehicles.db"


# ==============================
# CREATE DATABASE
# ==============================

def create_database():

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    # Vehicle table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_type TEXT NOT NULL,
            track_id INTEGER,
            confidence REAL,
            timestamp TEXT NOT NULL
        )
    """)

    # Number plate table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS number_plates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL,
            vehicle_type TEXT,
            track_id INTEGER,
            plate_confidence REAL,
            timestamp TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()

    print("Database created successfully!")


# ==============================
# SAVE VEHICLE
# ==============================

def save_vehicle(vehicle_type, track_id, confidence):

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    cursor.execute("""
        INSERT INTO vehicles
        (vehicle_type, track_id, confidence, timestamp)
        VALUES (?, ?, ?, ?)
    """, (
        vehicle_type,
        track_id,
        confidence,
        timestamp
    ))

    connection.commit()
    connection.close()

    print(
        f"Saved to database: "
        f"{vehicle_type} | ID: {track_id}"
    )


# ==============================
# SAVE NUMBER PLATE
# ==============================

def save_number_plate(
    plate_number,
    vehicle_type,
    track_id,
    plate_confidence
):

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    cursor.execute("""
        INSERT INTO number_plates
        (
            plate_number,
            vehicle_type,
            track_id,
            plate_confidence,
            timestamp
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        plate_number,
        vehicle_type,
        track_id,
        plate_confidence,
        timestamp
    ))

    connection.commit()
    connection.close()

    print(
        f"Plate saved to database: "
        f"{plate_number} | "
        f"{vehicle_type} | "
        f"ID: {track_id}"
    )


# ==============================
# RUN DIRECTLY
# ==============================

if __name__ == "__main__":
    create_database()

