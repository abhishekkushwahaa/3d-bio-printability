import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt
import os
import re

def clean_numeric(value, default=0.0):
    """Convert string with special characters to float"""
    if pd.isna(value):
        return default
    try:
        cleaned = str(value).replace('–', '').replace('—', '').replace('−', '').strip()
        return float(cleaned) if cleaned else default
    except:
        return default

# Create directories if they don't exist
os.makedirs("models", exist_ok=True)

# Load and clean data
df = pd.read_excel("data/raw_data.xlsx", sheet_name="Dataset")

# Extract percentages from Composition
df[["Silk_%", "Gelatin_%"]] = df["Composition"].str.extract(r'Silk (\d+)%, Gelatin (\d+)%').astype(float)

# Clean target variable
df["Printable"] = df["Printable"].map({"No": 0, "Yes": 1, "-": 0, None: 0, "": 0})
df = df.dropna(subset=["Printable"])

# Feature engineering with consistent names
df["Crosslinker"] = df["Crosslinker"].apply(lambda x: 0 if str(x).strip() in ["None", ""] else 1)
df["Needle_Gauge"] = df["Gauge"].apply(clean_numeric, default=22).astype(int)
df["LH_mm"] = df["LH (mm)"].apply(clean_numeric)
df["Pressure_psi"] = df["Pressure (kPa)"].apply(clean_numeric) * 0.145038
df["Temp_C"] = df["TG (°C)"].apply(clean_numeric)

# Final feature selection
features = ["Silk_%", "Gelatin_%", "Crosslinker", "Needle_Gauge", "LH_mm", "Pressure_psi", "Temp_C"]
X = df[features]
y = df["Printable"].astype(int)

# Verify data
print("\nData Summary:")
print(f"Total samples: {len(df)}")
print(f"Printable samples: {y.sum()} ({(y.sum()/len(y))*100:.1f}%)")
print("\nFeature ranges:")
print(X.describe())

# Train and save model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
joblib.dump(model, "models/model.pkl")
joblib.dump(features, "models/features.pkl")  # Save feature names

# Save visualization
pd.Series(model.feature_importances_, index=features).plot(kind="barh")
plt.title("Feature Importance")
plt.savefig("models/feature_importance.png", bbox_inches="tight")
print("\nModel trained and saved successfully!")