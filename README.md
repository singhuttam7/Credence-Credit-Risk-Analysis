# Credence --- Credit Risk Analysis

An explainable machine-learning application for credit-risk assessment.
Credence uses borrower and loan attributes to estimate the likelihood of
loan default and support risk-aware decision-making.

> **Disclaimer:** Credence is an educational and decision-support
> project. Predictions are estimates, not guarantees, and must not be
> the sole basis for lending decisions. Validate the model for the
> intended population and use case before any real-world deployment.

## Features

-   **XGBoost classification** --- a gradient-boosted tree model for
    loan-default prediction.
-   **Data preprocessing and feature engineering** --- prepares borrower
    and loan attributes for model input.
-   **Cross-validation** --- evaluates performance across multiple data
    splits to help assess generalization.
-   **Hyperparameter tuning** --- supports searching for better model
    configurations.
-   **Probability calibration** --- aims to make predicted probabilities
    better reflect observed event frequencies.
-   **SHAP explainability** --- helps explain how input features
    contribute to model predictions, where SHAP is enabled in the
    application.
-   **Threshold optimization** --- uses a selected decision threshold
    rather than assuming 0.50 is always appropriate.
-   **Risk classification** --- presents model outputs as risk-oriented
    decision-support information.
-   **Web interface and API** --- provides a frontend and Python backend
    for interacting with the model.

## Technology stack

-   Python
-   pandas and NumPy
-   scikit-learn
-   XGBoost
-   SHAP
-   FastAPI (if configured in `main.py`)
-   HTML, CSS, and JavaScript
-   Jupyter Notebook for experimentation

The exact packages and versions are listed in
[`requirements.txt`](requirements.txt).

## Project structure

``` text
Credit_risk_Analysis/
├── static/
│   ├── index.html
│   ├── script.js
│   └── style.css
├── best_threshold.pkl
├── credit_risk_dataset.csv
├── credit_risk_model.pkl
├── credit_risk1.ipynb
├── main.py
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
```

Files may differ depending on the version of the project. Keep model
artifacts that the application needs, and do not publish private
datasets, credentials, or other sensitive information.

## Run locally

### 1. Clone the repository

Replace `YOUR-USERNAME` with the GitHub account that owns the
repository.

``` bash
git clone https://github.com/YOUR-USERNAME/Credence-Credit-Risk-Analysis.git
cd Credence-Credit-Risk-Analysis
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the application

If `main.py` exposes a FastAPI application object named `app`, run:

``` bash
uvicorn main:app --reload
```

Open the local URL printed by Uvicorn. For FastAPI, interactive API
documentation is usually available at `/docs`.

If your entry point or ASGI object has a different name, use the command
that matches your implementation.

## Model workflow

A typical workflow for this project is:

1.  Load and inspect the credit-risk dataset.
2.  Clean and preprocess the data.
3.  Prepare features and the target variable.
4.  Split data into training and evaluation sets.
5.  Train and tune the classifier.
6.  Evaluate with suitable classification metrics, such as precision,
    recall, F1-score, ROC-AUC, and a confusion matrix where applicable.
7.  Calibrate predicted probabilities when the calibration workflow is
    enabled.
8.  Select a decision threshold according to the evaluation objective.
9.  Use SHAP to inspect global feature importance and, where
    implemented, individual predictions.
10. Save the model and any associated preprocessing, calibration, or
    threshold artifacts required for inference.

This describes the intended workflow; use the notebook and application
code as the source of truth for the steps currently implemented.

## Model artifacts

-   `credit_risk_model.pkl` --- saved model artifact.
-   `best_threshold.pkl` --- saved threshold artifact.

Only use these files with trusted code and compatible library versions.
Pickle files can execute code when loaded, so do not load untrusted
artifacts.

## Deployment

The project can be deployed to a Python web-service platform such as
Render if the application is configured for production.

For a Render web service, a common start command for a FastAPI app is:

``` bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Confirm that `main.py` exposes `app`, that the required model artifacts
are included or downloaded during deployment, and that the frontend
paths and API URLs work in production. Configure any secrets through the
hosting provider's environment-variable settings; never commit them to
Git.

## Responsible use and limitations

-   A model score is not a guarantee that a borrower will or will not
    default.
-   Dataset quality, sampling, label definitions, and distribution
    shifts can affect performance.
-   Evaluate calibration and classification metrics on held-out data.
-   Review potential bias, fairness, privacy, and applicable lending
    regulations before real-world use.
-   SHAP explanations describe model behavior; they do not establish
    causation.
-   Do not use this project as the sole basis for consequential
    financial decisions.

## Contributing

Contributions, bug reports, and suggestions are welcome. Open an issue
describing the problem or proposed improvement. For code changes, create
a focused branch and submit a pull request with a clear explanation and
relevant test results.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE)
for details.

------------------------------------------------------------------------

**Author:** Uttam Kumar Singh\
**Project:** Credence --- Credit Risk Analysis
