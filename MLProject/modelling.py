import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# 1. PERSIAPAN DATA (LOAD DATA)

# Diatur ke folder root karena file train_clean.csv akan diletakkan di dalam folder MLProject
dataset_path = "train_clean.csv"

if not os.path.exists(dataset_path):
    print(f"❌ Error: File tidak ditemukan di {dataset_path}")
    exit()

print(f"✅ Memuat data dari: {dataset_path}")
df = pd.read_csv(dataset_path)

# Sesuai dengan hasil ekspor preprocessing sebelumnya
target_col = 'Loan_Status_Target' 

if target_col not in df.columns:
    print(f"❌ Error: Kolom target '{target_col}' tidak ditemukan.")
    exit()

X = df.drop(columns=[target_col])
y = df[target_col]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 2. SETUP MLFLOW & TRAINING

mlflow.set_experiment("Eksperimen_Loan_Prediction_Otomatis")

with mlflow.start_run():
    print(" Memulai Training dengan Hyperparameter Tuning via MLflow Run...")

    rf = RandomForestClassifier(random_state=42)
    
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    }
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    y_pred = best_model.predict(X_test)

    # Hitung Metrik
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f" Best Params: {best_params}")
    print(f" Accuracy: {accuracy:.4f}")

    
    # 3. LOGGING KE MLFLOW
    
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Simpan Model
    mlflow.sklearn.log_model(best_model, "model_random_forest_tuned")

    print("✅ Model dan metrik berhasil disimpan ke MLflow Artifacts.")

