# modelling.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import mlflow
import mlflow.sklearn
import os

# Tentukan folder output yang pasti
output_path = "output/model"

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

# Simpan model langsung ke folder output dalam format MLflow
mlflow.sklearn.save_model(
    sk_model=final_model,
    path=output_path
)

print(f"Model telah dilatih dan disimpan di folder: {output_path}")
