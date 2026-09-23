#  AI Vehicle Monitoring System

An AI-based vehicle monitoring project that uses computer vision to detect and count vehicles, stores vehicle data in a database, and includes a Machine Learning pipeline for traffic prediction.

##  About the Project

The **AI Vehicle Monitoring System** is designed to monitor vehicles from video or webcam input.

The project uses **YOLO11n** for vehicle detection. It identifies vehicles such as cars, motorcycles, buses, trucks, and bicycles and records their counts.

The collected vehicle information is stored using **SQLite** and can also be used to create datasets for the Machine Learning part of the project.

A **Random Forest** model is trained using the collected traffic data for traffic prediction.

The project also includes a separate license plate detection component using a custom YOLO model.

##  Features

*  Real-time vehicle detection from webcam/video input
*  Vehicle counting
*  Detection of different vehicle types
*  Vehicle data storage using SQLite
*  Vehicle and traffic dataset creation
*  Random Forest-based traffic prediction
*  License plate detection using a custom YOLO model
*  Separate modules for detection, database management, and ML training

## AI & Machine Learning

The project uses AI/ML in two main areas.

### 1. Vehicle Detection

The project uses the **YOLO11n** pre-trained object detection model through Ultralytics.

The vehicle classes used in the project include:

* Car
* Motorcycle
* Bus
* Truck
* Bicycle

The model processes video frames and detects the vehicles present in them.

### 2. Traffic Prediction

Vehicle-count observations collected during the monitoring process are used for the Machine Learning pipeline.

The workflow is:

```text
Video / Webcam
      ↓
YOLO Vehicle Detection
      ↓
Vehicle Counting
      ↓
Data Collection
      ↓
ML Dataset
      ↓
Random Forest Training
      ↓
Traffic Prediction
```

The trained model is stored in:

```text
database/traffic_prediction_model.pkl
```

##  Technologies Used

### Programming

* Python

### Computer Vision

* YOLO
* Ultralytics
* OpenCV

### Machine Learning

* Scikit-learn
* Random Forest
* Pandas
* NumPy

### Database

* SQLite

### Tools

* VS Code
* Git
* GitHub

##  Project Structure

```text
AI-Vehicle-Monitoring-System/
│
├── database/
│   ├── create_ml_dataset.py
│   ├── traffic_ml_dataset.csv
│   ├── traffic_prediction_model.pkl
│   ├── train_ml_model.py
│   ├── vehicle_database.py
│   ├── vehicle_ml_dataset.csv
│   ├── view_database.py
│   └── __init__.py
│
├── src/
│   └── detection.py
│
├── plate_test.py
├── plate_webcam.py
│
├── yolo11n.pt
├── license-plate-finetune-v1n.pt
├── .gitignore
└── README.md
```

## ⚙️ How It Works

### 1. Vehicle Detection

Video or webcam frames are processed using YOLO11n to detect vehicles.

### 2. Vehicle Counting

The detected vehicles are classified according to their vehicle type and counted.

### 3. Data Storage

Vehicle-related information is stored in an SQLite database for later access and analysis.

### 4. Dataset Creation

Collected vehicle-count information is used to create datasets for the Machine Learning workflow.

The dataset creation script is:

```text
database/create_ml_dataset.py
```

### 5. Model Training

The traffic prediction model is trained using the collected traffic data.

The training script is:

```text
database/train_ml_model.py
```

The trained Random Forest model is saved as:

```text
database/traffic_prediction_model.pkl
```

### 6. Traffic Prediction

The trained model can be used as part of the traffic prediction workflow using vehicle-count information.

## 📊 Dataset

The project contains datasets created from vehicle-count and traffic observations:

```text
database/traffic_ml_dataset.csv
database/vehicle_ml_dataset.csv
```

These datasets are used in the Machine Learning workflow.

## 🔍 License Plate Detection

The project also includes a separate license plate detection component.

The custom YOLO model is:

```text
license-plate-finetune-v1n.pt
```

Related scripts include:

```text
plate_webcam.py
plate_test.py
```

## 💻 Running the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/shambhwipriya/AI-Vehicle-Monitoring-System.git
cd AI-Vehicle-Monitoring-System
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

For Windows:

```bash
.venv\Scripts\activate
```

### 4. Install the Required Libraries

```bash
pip install ultralytics opencv-python pandas numpy scikit-learn
```

### 5. Run Vehicle Detection

```bash
python src/detection.py
```

### 6. Run License Plate Detection

```bash
python plate_webcam.py
```

##  Notes

* `yolo11n.pt` is a pre-trained YOLO model used for vehicle detection.
* `traffic_prediction_model.pkl` contains the trained Random Forest traffic prediction model.
* SQLite is used for local vehicle-data storage.
* Webcam-based features require access to a local camera.
* The current version has been developed and tested primarily in a local Python environment.

##  Future Improvements

Some possible improvements for the project include:

* Web-based traffic monitoring dashboard
* Real-time traffic visualization
* Traffic congestion classification
* Training with a larger and more diverse traffic dataset
* Historical traffic analysis
* Alerts for high traffic conditions
* Further improvements to license plate detection
* Optimization for web-based deployment

##  Author

**Shambhwi Priya**

B.Tech — Computer Science & Data Science

GitHub:
https://github.com/shambhwipriya

Feel free to explore the project and its implementation.
