from ultralytics import YOLO
import cv2
import sys
import os
import time
import sqlite3


# Add project root folder to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)


# Import database functions
from database.vehicle_database import create_database, save_vehicle


# Database path
DATABASE_PATH = os.path.join(
    PROJECT_ROOT,
    "database",
    "vehicles.db"
)


# Load YOLO model
model = YOLO("yolo11n.pt")


# Create database/table if it does not exist
create_database()


# Vehicle classes
VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}


# Store IDs already counted in the current session
counted_ids = set()


# Total unique vehicles
total_count = 0


# Start time of current 1-minute interval
interval_start_time = time.time()

# Vehicles detected during current minute
interval_vehicle_count = 0


# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()


print("Camera started successfully.")
print("Traffic data will be recorded every minute.")
print("Press Q to stop.")


while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break


    # Track vehicles
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[2, 3, 5, 7],
        verbose=False
    )

    result = results[0]


    if result.boxes is not None:

        for box in result.boxes:

            # Skip if tracking ID is unavailable
            if box.id is None:
                continue


            track_id = int(box.id[0])
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])


            # Get bounding box
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # Vehicle name
            vehicle_name = VEHICLE_CLASSES.get(
                class_id,
                "Unknown"
            )


            # Count each tracking ID only once
            if track_id not in counted_ids:

                counted_ids.add(track_id)

                total_count += 1

                interval_vehicle_count += 1


                # Save individual vehicle information
                save_vehicle(
                    vehicle_name,
                    track_id,
                    confidence
                )


                print(
                    f"Vehicle Detected: {vehicle_name} | "
                    f"ID: {track_id} | "
                    f"Confidence: {confidence:.2f} | "
                    f"Total: {total_count}"
                )


            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # Label
            label = (
                f"{vehicle_name} | "
                f"ID: {track_id} | "
                f"{confidence:.2f}"
            )


            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )


    # Check whether one minute has passed
    current_time = time.time()

    if current_time - interval_start_time >= 60:

        # Save minute-level traffic observation
        connection = sqlite3.connect(DATABASE_PATH)

        cursor = connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS traffic_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                vehicle_count INTEGER NOT NULL
            )
        """)

        cursor.execute("""
            INSERT INTO traffic_data
            (timestamp, vehicle_count)
            VALUES (datetime('now', 'localtime'), ?)
        """, (
            interval_vehicle_count,
        ))

        connection.commit()
        connection.close()


        print(
            f"\nTraffic Observation Saved: "
            f"{interval_vehicle_count} vehicles in the last minute"
        )


        # Reset minute counter
        interval_vehicle_count = 0

        interval_start_time = current_time


    # Display total count
    cv2.putText(
        frame,
        f"Total Vehicles: {total_count}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )


    # Display current minute count
    cv2.putText(
        frame,
        f"Current Minute: {interval_vehicle_count}",
        (20, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )


    # Show camera
    cv2.imshow(
        "AI Vehicle Monitoring System",
        frame
    )


    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release resources
cap.release()

cv2.destroyAllWindows()


print("\nFinal Vehicle Count:", total_count)