# Parkinson's Disease Prediction

A machine learning-based web application for predicting Parkinson's disease using voice measurement features.

## 📌 Project Overview

Parkinson's disease is a neurological disorder that can affect speech and voice characteristics. This project uses machine learning to analyze voice-related measurements and predict whether a sample indicates Parkinson's disease.

The application provides a simple web interface where users can select a demo patient, view their voice measurements, and generate a prediction using a trained machine learning model.

> **Note:** The patient names, ages, genders, and IDs used in the application are fictional demo information. They do not represent real patients.

---

## 🎯 Objectives

- Analyze voice-related features associated with Parkinson's disease.
- Train and compare multiple machine learning algorithms.
- Select a suitable model for prediction.
- Develop a Flask-based web application.
- Provide an easy-to-use patient prediction interface.

---

## 📊 Dataset

The project uses the **UCI Parkinson's Disease Dataset**.

The dataset contains:

- **195 voice recordings**
- **22 voice measurement features**
- **1 target variable** (`status`)

The target variable represents:

- `1` → Parkinson's disease
- `0` → No Parkinson's disease

### Voice Features

The model uses the following 22 features:

1. MDVP:Fo(Hz)
2. MDVP:Fhi(Hz)
3. MDVP:Flo(Hz)
4. MDVP:Jitter(%)
5. MDVP:Jitter(Abs)
6. MDVP:RAP
7. MDVP:PPQ
8. Jitter:DDP
9. MDVP:Shimmer
10. MDVP:Shimmer(dB)
11. Shimmer:APQ3
12. Shimmer:APQ5
13. MDVP:APQ
14. Shimmer:DDA
15. NHR
16. HNR
17. RPDE
18. DFA
19. spread1
20. spread2
21. D2
22. PPE

---

## 🤖 Machine Learning Models

Three machine learning algorithms were trained and compared:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest

The dataset was divided into:

- **80% training data**
- **20% testing data**

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score

### Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 92.31% | 93.33% | 96.55% | 94.92% |
| SVM | 92.31% | 90.62% | 100.00% | 95.08% |
| Random Forest | 92.31% | 93.33% | 96.55% | 94.92% |

The current application uses **Logistic Regression** as the selected model.

---

## 🖥️ Application Features

- Patient selection using a dropdown.
- 20 fictional demo patient profiles.
- Automatic loading of patient voice measurements.
- Display of 22 voice features.
- Machine learning-based prediction.
- Parkinson's probability score.
- Model performance comparison page.
- Responsive web interface.
- Separate prediction result page.

---

## 📁 Project Structure

```text
Parkinsons-Prediction/
│
├── app.py
├── train_model.py
├── create_patients.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── dataset/
│   └── parkinsons.data
│
├── model/
│   └── parkinson_model.pkl
│
├── patient_data/
│   ├── P001.csv
│   ├── P002.csv
│   ├── ...
│   └── P020.csv
│
└── templates/
    ├── index.html
    ├── result.html
    └── performance.html


⚙️ Technologies Used
- Python
- Flask
- Pandas
- Scikit-learn
- Joblib
- HTML
- CSS
- JavaScript