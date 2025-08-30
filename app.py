from flask import Flask, request, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load model and features
model = joblib.load("models/model.pkl")
features = joblib.load("models/features.pkl")

@app.route('/')
def home():
    # Define default ranges based on your dataset
    param_ranges = {
    'silk': {
        'min': 2,       # Reduced from 4 to allow experimental low concentrations
        'max': 10,      # Increased from 6 for testing higher concentrations
        'step': 0.1,
        'default': 4    # Added default value
    },
    'gelatin': {
        'min': 10,      # Reduced from 14
        'max': 20,      # Increased from 16
        'step': 0.1,
        'default': 15
    },
    'needle': {
        'min': 10,      # Reduced from 15 to allow finer needles
        'max': 35,      # Increased from 30
        'step': 1,
        'default': 22   # Most common gauge
    },
    'height': {
        'min': 0.01,    # Reduced from 0.05 for ultra-fine layers
        'max': 0.5,     # Increased from 0.2 for thicker layers
        'step': 0.01,
        'default': 0.1
    },
    'pressure': {
        'min': 1,       # Reduced from 3
        'max': 15,      # Increased from 8
        'step': 0.1,
        'default': 6.0
    },
    'temp': {
        'min': 15,      # Reduced from 22
        'max': 40,      # Increased from 28
        'step': 0.5,    # More precise temperature control
        'default': 25
    }
}
    return render_template('index.html', param_ranges=param_ranges)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {
            "Silk_%": float(request.form["silk"]),
            "Gelatin_%": float(request.form["gelatin"]),
            "Crosslinker": 1 if request.form["crosslinker"] == "yes" else 0,
            "Needle_Gauge": int(request.form["needle"]),
            "LH_mm": float(request.form["height"]),
            "Pressure_psi": float(request.form["pressure"]),
            "Temp_C": float(request.form["temp"])
        }
        
        X_pred = pd.DataFrame([input_data], columns=features)
        prediction = model.predict(X_pred)[0]
        probability = model.predict_proba(X_pred)[0][1]

        # 🌟 Add remarks based on basic rules or probability
        remarks = "Good overall structure."  # default

        if prediction == 1:
            if probability > 0.90:
                remarks = "Excellent stacking with high resolution."
            elif probability > 0.75:
                remarks = "Stable geometry and decent print fidelity."
            else:
                remarks = "Likely printable but may need optimization."
        else:
            if input_data["Pressure_psi"] < 3:
                remarks = "Pressure too low for proper extrusion."
            elif input_data["LH_mm"] > 0.2:
                remarks = "Layer height too thick; may affect resolution."
            else:
                remarks = "Formulation lacks required cohesion."

        return render_template('index.html',
                               prediction=f"Printable: {'YES' if prediction else 'NO'}",
                               probability=f"Probability: {probability:.0%}",
                               remarks=remarks,
                               form_data=request.form,
                               param_ranges={
                                   'silk': {'min': 4, 'max': 6, 'step': 0.1},
                                   'gelatin': {'min': 14, 'max': 16, 'step': 0.1},
                                   'needle': {'min': 15, 'max': 30, 'step': 1},
                                   'height': {'min': 0.05, 'max': 0.2, 'step': 0.01},
                                   'pressure': {'min': 3, 'max': 8, 'step': 0.1},
                                   'temp': {'min': 22, 'max': 28, 'step': 1}
                               })
    
    except Exception as e:
        return render_template('index.html',
                               prediction=f"Error: {str(e)}",
                               form_data=request.form)


if __name__ == '__main__':
    os.makedirs("models", exist_ok=True)
    app.run(debug=True)