import pandas as pd
import os

# Load original Parkinson's dataset
data = pd.read_csv("dataset/parkinsons.data")

# Create patient_data folder
os.makedirs("patient_data", exist_ok=True)

# Separate the two classes
parkinson_records = data[data["status"] == 1].reset_index(drop=True)
healthy_records = data[data["status"] == 0].reset_index(drop=True)

# Fictional patient information
patients = [
    ("P001", "Aarav Mehta", 64, "Male", parkinson_records.iloc[0]),
    ("P002", "Priya Sharma", 58, "Female", healthy_records.iloc[0]),
    ("P003", "Rohan Desai", 71, "Male", parkinson_records.iloc[1]),
    ("P004", "Ananya Patel", 55, "Female", healthy_records.iloc[1]),
    ("P005", "Vikram Joshi", 67, "Male", parkinson_records.iloc[2]),
    ("P006", "Meera Shah", 61, "Female", healthy_records.iloc[2]),
    ("P007", "Kabir Rao", 69, "Male", parkinson_records.iloc[3]),
    ("P008", "Ishita Kapoor", 57, "Female", healthy_records.iloc[3]),
    ("P009", "Aditya Nair", 73, "Male", parkinson_records.iloc[4]),
    ("P010", "Nisha Verma", 62, "Female", healthy_records.iloc[4]),

    ("P011", "Rahul Malhotra", 54, "Male", parkinson_records.iloc[5]),
    ("P012", "Sneha Kulkarni", 48, "Female", healthy_records.iloc[5]),
    ("P013", "Arjun Kapoor", 60, "Male", parkinson_records.iloc[6]),
    ("P014", "Neha Iyer", 51, "Female", healthy_records.iloc[6]),
    ("P015", "Karan Shah", 56, "Male", parkinson_records.iloc[7]),
    ("P016", "Riya Mehta", 46, "Female", healthy_records.iloc[7]),
    ("P017", "Sahil Deshmukh", 59, "Male", parkinson_records.iloc[8]),
    ("P018", "Tanya Rao", 53, "Female", healthy_records.iloc[8]),
    ("P019", "Manish Gupta", 65, "Male", parkinson_records.iloc[9]),
    ("P020", "Simran Joshi", 50, "Female", healthy_records.iloc[9])
]


# Create individual CSV files
for patient_id, name, age, gender, record in patients:

    row = record.copy()

    # Remove original dataset name and target
    row = row.drop(labels=["name", "status"])

    # Convert to DataFrame
    patient_data = pd.DataFrame([row])

    # Add patient information
    patient_data.insert(0, "Patient_ID", patient_id)
    patient_data.insert(1, "Patient_Name", name)
    patient_data.insert(2, "Age", age)
    patient_data.insert(3, "Gender", gender)

    # Save CSV
    filename = f"patient_data/{patient_id}.csv"
    patient_data.to_csv(filename, index=False)

    print(f"Created: {filename}")


print("\n20 patient records created successfully!")
print("10 Parkinson's + 10 non-Parkinson's records mixed together.")