import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("dataset/parkinsons.data")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)


# ==========================================
# 2. PREPROCESS DATA
# ==========================================

# Remove patient/recording name
data = data.drop(columns=["name"])

# Separate input features and target
X = data.drop(columns=["status"])
y = data["status"]

print("Number of features:", X.shape[1])


# ==========================================
# 3. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 4. DEFINE MODELS
# ==========================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(probability=True))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


# ==========================================
# 5. TRAIN AND COMPARE MODELS
# ==========================================

results = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)

    results[name] = accuracy

    print("\n" + "=" * 40)
    print(name)
    print("=" * 40)

    print("Accuracy :", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall   :", round(recall * 100, 2), "%")
    print("F1 Score :", round(f1 * 100, 2), "%")


# ==========================================
# 6. SELECT BEST MODEL
# ==========================================

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("\n" + "=" * 40)
print("BEST MODEL:", best_model_name)
print("BEST ACCURACY:", round(results[best_model_name] * 100, 2), "%")
print("=" * 40)


# ==========================================
# 7. SAVE BEST MODEL
# ==========================================

model_data = {
    "model": best_model,
    "features": X.columns.tolist()
}

joblib.dump(model_data, "model/parkinson_model.pkl")

print("\nBest model saved successfully!")
print("Location: model/parkinson_model.pkl")