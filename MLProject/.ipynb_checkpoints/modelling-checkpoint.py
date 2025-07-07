# modelling.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Muat data dan scaler
df = pd.read_csv('telco-customer-churn_preprocessing/churn_data_processed.csv')
scaler = joblib.load('scaler.joblib')

# Pisahkan fitur dan target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Gunakan scaler yang sudah di-fit
X_scaled = scaler.transform(X)

# Latih model dengan parameter terbaik dari Kriteria 2
# Ganti parameter jika Anda menemukan yang lebih baik
final_model = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=4, random_state=42)
final_model.fit(X_scaled, y)

# Simpan model yang sudah dilatih
joblib.dump(final_model, 'final_model.joblib')

print("Model telah dilatih dan disimpan sebagai final_model.joblib")