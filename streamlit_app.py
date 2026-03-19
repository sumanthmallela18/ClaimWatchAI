# ==============================
# ClaimWatch AI - Clean Version (Top-Tier)
# ==============================

import pandas as pd
import streamlit as st
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# ==============================
# Important Features (Reduced Inputs)
# ==============================

important_features = [
    "months_as_customer",
    "age",
    "policy_deductable",
    "policy_annual_premium",
    "umbrella_limit",
    "insured_sex",
    "insured_education_level",
    "insured_occupation",
    "incident_type",
    "incident_severity",
    "number_of_vehicles_involved",
    "total_claim_amount"
]

# ==============================
# Streamlit Title
# ==============================

st.title("🚀 ClaimWatch AI - Smart Fraud Detection")

# ==============================
# Load Dataset
# ==============================

@st.cache_data
def load_data():
    return pd.read_csv("insurance_claims.csv", encoding='latin1')

data = load_data()

# ==============================
# MODEL LOAD / TRAIN
# ==============================

if os.path.exists("fraud_model.pkl"):
    model = joblib.load("fraud_model.pkl")
    imputer = joblib.load("imputer.pkl")
    label_encoders = joblib.load("encoders.pkl")
    st.success("✅ Model Loaded")

else:
    st.warning("⚙ Training model...")

    X = data[important_features]
    y = data["fraud_reported"].map({'Y': 1, 'N': 0})

    label_encoders = {}

    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "fraud_model.pkl")
    joblib.dump(imputer, "imputer.pkl")
    joblib.dump(label_encoders, "encoders.pkl")

    st.success("✅ Model Trained & Saved")

# ==============================
# DEMO DATA
# ==============================

def generate_demo_data(fraud=False):
    if fraud:
        rows = data[data["fraud_reported"] == "Y"]
    else:
        rows = data[data["fraud_reported"] == "N"]

    sample = rows.sample(1)[important_features]

    demo = {}
    for col in sample.columns:
        if col in label_encoders:
            encoded = label_encoders[col].transform([str(sample.iloc[0][col])])[0]
            demo[col] = encoded
        else:
            demo[col] = float(sample.iloc[0][col])

    return demo

# ==============================
# Buttons
# ==============================

col1, col2 = st.columns(2)

with col1:
    if st.button("🚨 Fraud Example"):
        st.session_state.demo_data = generate_demo_data(True)

with col2:
    if st.button("🎯 Genuine Example"):
        st.session_state.demo_data = generate_demo_data(False)

# ==============================
# USER INPUT UI (CLEAN)
# ==============================

st.subheader("📝 Enter Claim Details")

user_input = {}

for col in important_features:

    if col in label_encoders:
        classes = label_encoders[col].classes_

        default_index = 0
        if "demo_data" in st.session_state:
            encoded_val = st.session_state.demo_data.get(col, 0)
            decoded_val = label_encoders[col].inverse_transform([int(encoded_val)])[0]
            default_index = list(classes).index(decoded_val)

        selected = st.selectbox(col, classes, index=default_index)
        user_input[col] = label_encoders[col].transform([selected])[0]

    else:
        default_value = 1000.0
        if "demo_data" in st.session_state:
            default_value = float(st.session_state.demo_data.get(col, 1000))

        user_input[col] = st.number_input(col, value=default_value)

# ==============================
# PREDICTION
# ==============================

if st.button("🔍 Predict Fraud"):

    input_df = pd.DataFrame([user_input])
    input_df = imputer.transform(input_df)

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("Result")

    if pred == 1:
        st.error("⚠ Fraudulent Claim")
    else:
        st.success("✅ Legitimate Claim")

    st.write(f"Fraud Probability: {prob:.2f}")
    # ==============================
# FEATURE IMPORTANCE GRAPH
# ==============================

st.subheader("📊 Feature Importance")

importance = model.feature_importances_

feature_names = important_features

# Create DataFrame
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

# Plot
fig, ax = plt.subplots()
ax.barh(importance_df["Feature"], importance_df["Importance"])
ax.invert_yaxis()

st.pyplot(fig)