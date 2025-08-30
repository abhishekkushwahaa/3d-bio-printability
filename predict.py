import joblib
import pandas as pd

model = joblib.load("models/model.pkl")

def predict_printability():
    print("\n3D Bioprinting Printability Predictor (CLI)")
    print("Enter the following parameters:\n")
    
    data = {
        "Silk_%": float(input("Silk (% w/v): ")),
        "Gelatin_%": float(input("Gelatin (% w/v): ")),
        "Crosslinker": 1 if input("Crosslinker present? (y/n): ").lower() == "y" else 0,
        "Needle_Gauge": int(input("Needle Gauge (0-30): ")),
        "LH_(mm)": float(input("Layer Height (mm): ")),
        "Pressure_(psi)": float(input("Pressure (psi): ")),
        "Temp_(°C)": float(input("Temperature (°C): "))
    }
    
    prediction = model.predict(pd.DataFrame([data]))[0]
    probability = model.predict_proba(pd.DataFrame([data]))[0][1]
    
    print(f"\nResult: {'Printable' if prediction else 'Not Printable'}")
    print(f"probability: {probability:.0%}")

if __name__ == "__main__":
    predict_printability()