
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from fastapi.responses import FileResponse
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "credit_risk_model.pkl"
THRESHOLD_PATH = BASE_DIR / "best_threshold.pkl"
STATIC_DIR = BASE_DIR / "static"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ml_model = {}


# --------------------------------------------------
# Application Lifespan: Load model on startup
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

        if not THRESHOLD_PATH.is_file():
            raise FileNotFoundError(
                f"Threshold file not found: {THRESHOLD_PATH}"
            )

        model = joblib.load(MODEL_PATH)
        threshold = float(joblib.load(THRESHOLD_PATH))

        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0 and 1.")

        if not hasattr(model, "predict_proba"):
            raise ValueError("Model must support predict_proba().")

        ml_model["model"] = model
        ml_model["threshold"] = threshold

        logger.info("Credit risk model loaded successfully.")
        logger.info("Decision threshold: %.4f", threshold)

        yield

    except Exception:
        logger.exception("Application startup failed.")
        raise

    finally:
        ml_model.clear()
        logger.info("Model resources cleared.")


# --------------------------------------------------
# FastAPI Application
# --------------------------------------------------

app = FastAPI(
    title="Credit Risk Analysis API",
    description=(
        "API for estimating loan default probability "
        "and classifying credit risk."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# CORS Configuration
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# --------------------------------------------------
# Request Validation
# --------------------------------------------------

class LoanApplication(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    person_age: int = Field(ge=18, le=100)
    person_income: float = Field(gt=0, allow_inf_nan=False)

    person_home_ownership: Literal[
        "RENT", "OWN", "MORTGAGE", "OTHER"
    ]

    person_emp_length: float = Field(
        ge=0, le=100, allow_inf_nan=False
    )

    loan_intent: Literal[
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION",
    ]

    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"]

    loan_amnt: float = Field(gt=0, allow_inf_nan=False)

    loan_int_rate: float = Field(
        ge=0, le=100, allow_inf_nan=False
    )

    loan_percent_income: float = Field(
        gt=0, le=1, allow_inf_nan=False
    )

    cb_person_default_on_file: Literal["Y", "N"]

    cb_person_cred_hist_length: int = Field(
        ge=0, le=100
    )


# --------------------------------------------------
# Health and Information Endpoints
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Welcome to the Credit Risk Analysis API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": "model" in ml_model,
        "threshold_loaded": "threshold" in ml_model,
    }


# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(data: LoanApplication):
    if "model" not in ml_model or "threshold" not in ml_model:
        raise HTTPException(
            status_code=503,
            detail="The prediction model is not available.",
        )

    try:
        # Preserve the original feature names expected by the model.
        input_df = pd.DataFrame([data.model_dump()])

        model = ml_model["model"]
        threshold = ml_model["threshold"]

        probabilities = model.predict_proba(input_df)

        # Identify the probability column for class 1 explicitly.
        classes = list(model.classes_)

        if 1 not in classes:
            raise ValueError("Model does not contain class 1.")

        default_probability = float(
            probabilities[0, classes.index(1)]
        )

        prediction = int(default_probability >= threshold)

        risk_level = (
            "High Risk" if prediction == 1 else "Low Risk"
        )

        return {
            "default_probability": round(default_probability, 6),
            "default_probability_percent": round(
                default_probability * 100, 2
            ),
            "default_prediction": prediction,
            "threshold": round(threshold, 6),
            "risk_level": risk_level,
            "message": (
                "The application is classified as high risk."
                if prediction == 1
                else "The application is classified as low risk."
            ),
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("Credit risk prediction failed.")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please check the model and input features.",
        )
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": "model" in ml_model,
        "threshold_loaded": "threshold" in ml_model,
    }

# --------------------------------------------------
# Serve Static Frontend (if available)
# -------------------------------------------------
# Serve the frontend
@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse(STATIC_DIR / "index.html")


app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)
