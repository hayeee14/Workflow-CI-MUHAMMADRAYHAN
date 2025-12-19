import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==========================================
# 1. PERSIAPAN DATA (LOAD DATA)
# ==========================================
# Ini bagian penting: Kita arahkan ke folder 'LoanPrediction'
dataset_path = os.path.join("LoanPrediction", "train_clean.csv")

# Cek apakah file ada (untuk debugging)
if not os.path.exists(dataset_path):
    print(f"❌ Error: File tidak ditemukan di {dataset_path}")
    print("Pastikan nama folder adalah 'LoanPrediction' dan file 'train_clean.csv' ada di dalamnya.")
    exit()

print(f"✅ Memuat data dari: {dataset_path}")
df = pd.read_csv(dataset_path)

# Pisahkan Fitur (X) dan Target (y)
# Asumsi kolom target bernama 'Loan_Status' (sesuaikan jika beda, misal 'target')
target_col = 'Loan_Status' 
if target_col not in df.columns:
    # Fallback jika nama kolom beda, ambil kolom terakhir
    target_col = df.columns[-1]

X = df.drop(columns=[target_col])
y = df[target_col]

# Encoding sederhana (Jaga-jaga jika masih ada data string)
X = pd.get_dummies(X, drop_first=True)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 2. SETUP MLFLOW & TRAINING (SKILLED)
# ==========================================
# Set nama eksperimen
mlflow.set_experiment("Eksperimen_Loan_Prediction_Otomatis")

with mlflow.start_run():
    print("🚀 Memulai Training dengan Hyperparameter Tuning...")

    # Definisi Model & Parameter untuk Tuning
    rf = RandomForestClassifier(random_state=42)
    
    # Grid Search (Mencari settingan terbaik)
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    # Ambil model terbaik
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Prediksi
    y_pred = best_model.predict(X_test)

    # Hitung Metrik
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"🏆 Best Params: {best_params}")
    print(f"📊 Accuracy: {accuracy:.4f}")

    # ==========================================
    # 3. LOGGING KE MLFLOW
    # ==========================================
    # Log Parameter Terbaik
    mlflow.log_params(best_params)

    # Log Metrik
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Log Model (PENTING: Folder ini yang nanti diambil GitHub Actions)
    mlflow.sklearn.log_model(best_model, "model_random_forest_tuned")

    print("✅ Model berhasil disimpan ke MLflow Artifacts.")