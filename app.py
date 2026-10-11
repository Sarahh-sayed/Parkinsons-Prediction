from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load trained model
model_data = joblib.load("model/parkinson_model.pkl")

model = model_data["model"]
features = model_data["features"]


# Demo patient information
patients = {
    "P001": {
        "name": "Aarav Mehta",
        "age": 64,
        "gender": "Male"
    },
    "P002": {
        "name": "Priya Sharma",
        "age": 58,
        "gender": "Female"
    },
    "P003": {
        "name": "Rohan Desai",
        "age": 71,
        "gender": "Male"
    },
    "P004": {
        "name": "Ananya Patel",
        "age": 55,
        "gender": "Female"
    },
    "P005": {
        "name": "Vikram Joshi",
        "age": 67,
        "gender": "Male"
    },
    "P006": {
        "name": "Meera Shah",
        "age": 61,
        "gender": "Female"
    },
    "P007": {
        "name": "Kabir Rao",
        "age": 69,
        "gender": "Male"
    },
    "P008": {
        "name": "Ishita Kapoor",
        "age": 57,
        "gender": "Female"
    },
    "P009": {
        "name": "Aditya Nair",
        "age": 73,
        "gender": "Male"
    },
    "P010": {
        "name": "Nisha Verma",
        "age": 62,
        "gender": "Female"
    },
    "P011": {
        "name": "Rahul Malhotra",
        "age": 54,
        "gender": "Male"
    },
    "P012": {
        "name": "Sneha Kulkarni",
        "age": 48,
        "gender": "Female"
    },
    "P013": {
        "name": "Arjun Kapoor",
        "age": 60,
        "gender": "Male"
    },
    "P014": {
        "name": "Neha Iyer",
        "age": 51,
        "gender": "Female"
    },
    "P015": {
        "name": "Karan Shah",
        "age": 56,
        "gender": "Male"
    },
    "P016": {
        "name": "Riya Mehta",
        "age": 46,
        "gender": "Female"
    },
    "P017": {
        "name": "Sahil Deshmukh",
        "age": 59,
        "gender": "Male"
    },
    "P018": {
        "name": "Tanya Rao",
        "age": 53,
        "gender": "Female"
    },
    "P019": {
        "name": "Manish Gupta",
        "age": 65,
        "gender": "Male"
    },
    "P020": {
        "name": "Simran Joshi",
        "age": 50,
        "gender": "Female"
    }
}

# Home page
@app.route("/")
def home():
    return render_template(
        "index.html",
        patients=patients
    )


# Load selected patient
@app.route("/patient/<patient_id>")
def load_patient(patient_id):

    if patient_id not in patients:
        return "Patient not found."

    file_path = f"patient_data/{patient_id}.csv"

    if not os.path.exists(file_path):
        return "Patient data file not found."

    data = pd.read_csv(file_path)

    patient_info = patients[patient_id]

    measurements = {}

    for feature in features:
        measurements[feature] = data.iloc[0][feature]

    return render_template(
        "index.html",
        patients=patients,
        selected_patient=patient_id,
        patient_info=patient_info,
        measurements=measurements
    )


# Prediction
@app.route("/predict", methods=["POST"])
def predict():

    patient_id = request.form["patient_id"]

    if patient_id not in patients:
        return "Invalid patient."

    file_path = f"patient_data/{patient_id}.csv"

    if not os.path.exists(file_path):
        return "Patient data file not found."

    data = pd.read_csv(file_path)

    # Select only the 22 features required by the model
    input_data = data[features]

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Calculate probability
    probability = None

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0][1] * 100

    if prediction == 1:
        result = "Parkinson's Disease Detected"
    else:
        result = "No Parkinson's Disease Detected"

    return render_template(
        "result.html",
        patient=patients[patient_id],
        patient_id=patient_id,
        result=result,
        probability=probability
    )


@app.route("/model-performance")
def model_performance():
    model_results = [
        {
            "name": "Logistic Regression",
            "accuracy": 92.31,
            "precision": 93.33,
            "recall": 96.55,
            "f1": 94.92
        },
        {
            "name": "SVM",
            "accuracy": 92.31,
            "precision": 90.62,
            "recall": 100.00,
            "f1": 95.08
        },
        {
            "name": "Random Forest",
            "accuracy": 92.31,
            "precision": 93.33,
            "recall": 96.55,
            "f1": 94.92
        }
    ]

    return render_template(
        "performance.html",
        model_results=model_results
    )
if __name__ == "__main__":
    app.run(debug=True)