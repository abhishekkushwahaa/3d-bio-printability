# 🧪 Bio-Ink Prediction Platform

This guide walks you through **every step** needed to get this project up and running.

---

## 🔧 Step 1: Install Python

1. Go to the official website: https://www.python.org/downloads/windows/
2. Download the **latest Python 3.x installer** for Windows.
3. Run the installer:
   - ✅ Check **"Add Python to PATH"**
   - ✅ Click **"Install Now"**

To verify installation:
- Press `Win + R`, type `cmd`, press Enter
- In the black terminal window (Command Prompt), type:

```bash
python --version
```

You should see something like `Python 3.x.x`

---

## 📁 Step 2: Download the Project Folder

If you've received this project as a ZIP file:

1. Right-click on it → Extract All → Choose a folder
2. Navigate into the extracted folder.

---

## 🪟 Step 3: Open Command Prompt in the Project Folder

1. Open the folder where you extracted this project.
2. Click on the address bar (where the folder path is shown).
3. Type `cmd` and press Enter.
   - This opens the terminal directly in this folder!

---

## 📦 Step 4: Install Required Libraries

In the Command Prompt, run:

```bash
pip install -r requirements.txt
```

This installs all the required packages like `streamlit`, `pandas`, `scikit-learn`, etc.

---

## 🧠 Step 5: Train the Machine Learning Models

### 🔹 A. Train Printability Model

```bash
python main.py
```

This will:
- Read the dataset
- Preprocess the data
- Train the model
- Save the model files inside `outputs/models/`

### 🔹 B. Train Degradation Prediction Model

```bash
python degradation_project/main.py
```

This will:
- Read the degradation dataset
- Train regression models
- Save the model files in `degradation_project/models/`

---

## 🌐 Step 6: Run the Web App

Once both models are trained, you can run the interactive app with:

```bash
streamlit run frontend/app.py
```

This will:
- Launch the app in your web browser
- Let you choose between:
  - **Printability Prediction**
  - **Degradation Prediction**

---

## 🚪 Step 7: Stop the App

To stop the app, go back to the Command Prompt and press:

```
Ctrl + C
```

---

## 📁 Folder Structure

```
bioink_prediction/
├── frontend/
│   ├── app.py            ← Main Streamlit app
│   ├── assets/           ← GIFs and icons
│
├── degradation_project/
│   ├── main.py           ← Trains degradation model
│   ├── models/           ← Saves trained model
│   ├── src/
│       └── predict.py    ← Prediction logic
│
├── data/
│   └── ...csv files
│
├── outputs/
│   └── models/
│       └── ...trained models
│
├── src/
│   ├── predict.py        ← Printability prediction logic
│   └── main.py           ← Trains printability model
│
├── requirements.txt      ← Python libraries to install
└── README.md             ← You're reading this file!
```

---

## 🧠 About

This app was created as part of a summer internship project. It uses **machine learning** to assist biomedical researchers in optimizing their **bio-ink** and **scaffold formulations** for better performance and experimental efficiency.

---
