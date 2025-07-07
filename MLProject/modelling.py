# modelling.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import mlflow
import mlflow.sklearn

# Memulai run MLflow secara eksplisit
with mlflow.start_run():
    # Dapatkan ID dari run yang sedang aktif
    run_id = mlflow.active_run().info.run_id
    print(f"MLflow Run ID: {run_id}")

    # Muat data dan scaler
    df = pd.read_csv('telco-customer-churn_preprocessing/churn_data_processed.csv')
    scaler = joblib.load('scaler.joblib')

    # Pisahkan fitur dan target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Gunakan scaler yang sudah di-fit
    X_scaled = scaler.transform(X)

    # Latih model dengan parameter terbaik
    final_model = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=4, random_state=42)
    final_model.fit(X_scaled, y)

    # Simpan model di dalam run yang aktif
    mlflow.sklearn.log_model(final_model, "model")
    print("Model telah dilatih dan dilog ke MLflow sebagai 'model'")
