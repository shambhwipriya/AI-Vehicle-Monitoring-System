from ultralytics import YOLO
import cv2
import easyocr
import re
import sys
import os

# ==============================
# PROJECT PATH
# ==============================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

sys.path.insert(0, PROJECT_ROOT)

from database.vehicle_database import (
    create_database,
    save_number_plate
)

# ==============================
# LOAD MODELS
# ==============================

print("Loading vehicle YOLO model...")
model = YOLO("yolo11n.pt")

print("Loading number plate model...")
plate_model = YOLO("license-plate-finetune-v1n.pt")

print("Loading EasyOCR...")
reader = easyocr.Reader(["en"])

# Create database and tables
create_database()

print("Models loaded successfully!")

# ==============================
# VEHICLE CLASSES
# ==============================

VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# ==============================
# WEBCAM
# ==============================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Camera started successfully.")
print("Press Q to stop.")

# ==============================
# DETECTED PLATES
# ==============================

detected_plates = set()

frame_counter = 0

# Run plate detection + OCR every 15 frames
OCR_INTERVAL = 15

# ==============================
# MAIN LOOP
# ==============================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break

    # ==============================
    # VEHICLE DETECTION + TRACKING
    # ==============================

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

            if box.id is None:
                continue

            track_id = int(box.id[0])
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            vehicle_name = VEHICLE_CLASSES.get(
                class_id,
                "Unknown"
            )

            # Keep coordinates inside frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            # ==============================
            # VEHICLE CROP
            # ==============================

            vehicle_crop = frame[
                y1:y2,
                x1:x2
            ]

            plate_text = ""

            # ==============================
            # NUMBER PLATE DETECTION
            # ==============================

            if (
                frame_counter % OCR_INTERVAL == 0
                and vehicle_crop.size > 0
            ):

                plate_results = plate_model.predict(
                    vehicle_crop,
                    conf=0.25,
                    verbose=False
                )

                plate_result = plate_results[0]

                if plate_result.boxes is not None:

                    for plate_box in plate_result.boxes:

                        px1, py1, px2, py2 = map(
                            int,
                            plate_box.xyxy[0]
                        )

                        plate_confidence = float(
                            plate_box.conf[0]
                        )

                        # Keep plate coordinates inside vehicle crop
                        px1 = max(0, px1)
                        py1 = max(0, py1)

                        px2 = min(
                            vehicle_crop.shape[1],
                            px2
                        )

                        py2 = min(
                            vehicle_crop.shape[0],
                            py2
                        )

                        plate_crop = vehicle_crop[
                            py1:py2,
                            px1:px2
                        ]

                        if plate_crop.size == 0:
                            continue

                        # ==============================
                        # OCR
                        # ==============================

                        enlarged = cv2.resize(
                            plate_crop,
                            None,
                            fx=3,
                            fy=3,
                            interpolation=cv2.INTER_CUBIC
                        )

                        ocr_results = reader.readtext(
                            enlarged,
                            detail=1
                        )

                        possible_plates = []

                        for detection in ocr_results:

                            text = detection[1]
                            ocr_confidence = detection[2]

                            cleaned = re.sub(
                                r"[^A-Za-z0-9]",
                                "",
                                text
                            ).upper()

                            if (
                                6 <= len(cleaned) <= 12
                                and ocr_confidence >= 0.30
                            ):
                                possible_plates.append(
                                    cleaned
                                )

                        # ==============================
                        # SAVE DETECTED PLATE
                        # ==============================

                        if possible_plates:

                            plate_text = possible_plates[0]

                            if plate_text not in detected_plates:

                                detected_plates.add(
                                    plate_text
                                )

                                # Save plate to SQLite
                                save_number_plate(
                                    plate_number=plate_text,
                                    vehicle_type=vehicle_name,
                                    track_id=track_id,
                                    plate_confidence=plate_confidence
                                )

                                print()
                                print(
                                    "NUMBER PLATE DETECTED"
                                )

                                print(
                                    "Plate:",
                                    plate_text
                                )

                                print(
                                    "Vehicle:",
                                    vehicle_name
                                )

                                print(
                                    "Track ID:",
                                    track_id
                                )

                                print(
                                    "Vehicle Confidence:",
                                    round(
                                        confidence,
                                        2
                                    )
                                )

                                print(
                                    "Plate Confidence:",
                                    round(
                                        plate_confidence,
                                        2
                                    )
                                )

                                print(
                                    "Saved to database!"
                                )

                        # ==============================
                        # DRAW PLATE BOX
                        # ==============================

                        cv2.rectangle(
                            vehicle_crop,
                            (px1, py1),
                            (px2, py2),
                            (255, 0, 0),
                            2
                        )

            # ==============================
            # DRAW VEHICLE BOX
            # ==============================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            vehicle_label = (
                f"{vehicle_name} | "
                f"ID: {track_id} | "
                f"{confidence:.2f}"
            )

            cv2.putText(
                frame,
                vehicle_label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

            # ==============================
            # SHOW PLATE TEXT
            # ==============================

            if plate_text:

                cv2.putText(
                    frame,
                    f"Plate: {plate_text}",
                    (
                        x1,
                        min(
                            y2 + 25,
                            frame.shape[0] - 10
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 0, 0),
                    2
                )

    frame_counter += 1

    # ==============================
    # PLATE COUNTER
    # ==============================

    cv2.putText(
        frame,
        f"Plates Found: {len(detected_plates)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # ==============================
    # SHOW WEBCAM
    # ==============================

    cv2.imshow(
        "AI Vehicle + Number Plate Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==============================
# CLOSE
# ==============================

cap.release()
cv2.destroyAllWindows()

print()
print("Camera stopped.")

print()
print("Detected Plates:")

for plate in detected_plates:
    print("-", plate)

print()
print("Plate data has been saved to SQLite database.")

