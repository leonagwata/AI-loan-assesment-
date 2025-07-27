from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from passlib.hash import bcrypt
import numpy as np
import mysql.connector
from mysql.connector import Error
import joblib
import os

app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "loan_app_db"
}

# Load ML model, scaler, and label encoder
def load_model():
    try:
        model = joblib.load("loan_approval_model.pkl")
        scaler = joblib.load("scaler.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        return model, scaler, label_encoder
    except Exception as e:
        print(f"[MODEL LOAD ERROR] {e}")
        raise

model, scaler, label_encoder = None, None, None
try:
    model, scaler, label_encoder = load_model()
except Exception as e:
    print(f"[CRITICAL] Failed to load model, scaler, or label encoder: {e}")

# Database connection
def get_db_connection():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Error as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

# Pydantic Models
class UserRegistration(BaseModel):
    full_name: str
    email: str
    phone: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class LoanApplication(BaseModel):
    cibil_score: float
    loan_amount: float
    loan_term: float
    income: float
    user_id: int
    age: int
    employment_status: str
    existing_debt: float
    marital_status: str
    num_dependents: int
    education_level: str
    home_ownership_status: str
    credit_utilization: float
    recent_loan_history: str

# Register endpoint
@app.post("/api/register")
def register_user(user: UserRegistration):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM users WHERE email = %s", (user.email,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Email already exists")

        hashed_pw = bcrypt.hash(user.password)
        cursor.execute(
            "INSERT INTO users (full_name, email, phone, password) VALUES (%s, %s, %s, %s)",
            (user.full_name, user.email, user.phone, hashed_pw)
        )
        conn.commit()

        cursor.execute("SELECT LAST_INSERT_ID()")
        user_id = cursor.fetchone()[0]

        return {
            "message": "User registered successfully",
            "user": {
                "id": user_id,
                "full_name": user.full_name,
                "email": user.email,
                "phone": user.phone
            }
        }
    finally:
        cursor.close()
        conn.close()

# Login endpoint
@app.post("/api/login")
def login_user(user: UserLogin):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id, full_name, email, phone, password FROM users WHERE email = %s", (user.email,))
        db_user = cursor.fetchone()
        if not db_user or not bcrypt.verify(user.password, db_user[4]):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        return {
            "user": {
                "id": db_user[0],
                "full_name": db_user[1],
                "email": db_user[2],
                "phone": db_user[3]
            }
        }
    finally:
        cursor.close()
        conn.close()

# Prediction endpoint
@app.post("/api/predict")
def predict_loan(data: dict):
    if not model or not scaler:
        raise HTTPException(status_code=500, detail="Model or Scaler not loaded")

    try:
        # Define the expected feature names
        FEATURE_NAMES = [
            'no_of_dependents', 'education', 'self_employed', 'income_annum',
            'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
            'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'
        ]

        # Ensure the model expects the correct number of features
        n_features = scaler.n_features_in_
        if n_features != len(FEATURE_NAMES):
            raise HTTPException(status_code=500, detail=f"Expected {len(FEATURE_NAMES)} features, but found {n_features}")

        # Prepare the input data
        input_full = np.zeros((1, len(FEATURE_NAMES)), dtype=float)
        try:
            input_full[0, 0] = data['no_of_dependents']
            input_full[0, 1] = data['education']
            input_full[0, 2] = data['self_employed']
            input_full[0, 3] = data['income_annum']
            input_full[0, 4] = data['loan_amount']
            input_full[0, 5] = data['loan_term']
            input_full[0, 6] = data['cibil_score']
            input_full[0, 7] = data['residential_assets_value']
            input_full[0, 8] = data['commercial_assets_value']
            input_full[0, 9] = data['luxury_assets_value']
            input_full[0, 10] = data['bank_asset_value']
        except KeyError as e:
            raise HTTPException(status_code=400, detail=f"Missing required feature: {e}")

        # Scale the input data
        scaled = scaler.transform(input_full)

        # Make predictions
        pred = int(model.predict(scaled)[0])
        prob = float(model.predict_proba(scaled)[0][1]) if hasattr(model, "predict_proba") else None

        # Save the prediction to the database
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO loan_applications
                (no_of_dependents, education, self_employed, income_annum, loan_amount, loan_term, cibil_score,
                 residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value,
                 prediction, probability_approved)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                data['no_of_dependents'],
                data['education'],
                data['self_employed'],
                data['income_annum'],
                data['loan_amount'],
                data['loan_term'],
                data['cibil_score'],
                data['residential_assets_value'],
                data['commercial_assets_value'],
                data['luxury_assets_value'],
                data['bank_asset_value'],
                pred,
                prob
            ))
            conn.commit()
        finally:
            cursor.close()
            conn.close()

        return {
            "prediction": pred,
            "probability": prob,
            "eligibility": "Approved" if pred == 1 else "Rejected"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Unhandled error in /api/predict: {e}")
        raise HTTPException(status_code=500, detail=f"Unhandled server error: {e}")

# Retrieve user loan applications
@app.get("/api/user/{user_id}/applications")
def get_user_applications(user_id: int):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM loan_applications WHERE user_id = %s ORDER BY created_at DESC", (user_id,))
        return {"applications": cursor.fetchall()}
    finally:
        cursor.close()
        conn.close()

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}
