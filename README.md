# Credence --- Credit Risk Analysis

Credence is a machine-learning project that estimates the likelihood of
loan default from applicant and loan attributes. The project combines
exploratory data analysis, data validation, preprocessing pipelines,
model comparison, hyperparameter tuning, probability calibration, SHAP
explanations, decision-threshold selection, and error analysis.

> **Important:** This is an educational machine-learning and
> decision-support project, not financial advice or a production-ready
> lending decision system. A prediction is an estimate, not a guarantee.
> Any real-world use requires independent validation, fairness and
> privacy review, and compliance with applicable laws and regulations.

## Table of contents

-   [Project goals](#project-goals)
-   [Features used by the model](#features-used-by-the-model)
-   [Machine-learning workflow](#machine-learning-workflow)
-   [Models and evaluation](#models-and-evaluation)
-   [Probability calibration and threshold
    selection](#probability-calibration-and-threshold-selection)
-   [SHAP explainability](#shap-explainability)
-   [Project structure](#project-structure)
-   [Installation and local run](#installation-and-local-run)
-   [Saved artifacts](#saved-artifacts)
-   [Limitations and responsible use](#limitations-and-responsible-use)
-   [License](#license)

## Project goals

The notebook `credit_risk1.ipynb` develops and examines a loan-default
classification workflow. The target column is `loan_status`, and the
workflow treats the task as binary classification. It compares Logistic
Regression with XGBoost, with attention to class imbalance and metrics
beyond accuracy.

## Features used by the model

The notebook separates seven numerical features and four categorical
features.

### Numerical features

  Dataset column                 Meaning
  ------------------------------ ---------------------------------------
  `person_age`                   Applicant's age
  `person_income`                Applicant's income
  `person_emp_length`            Length of employment
  `loan_amnt`                    Requested loan amount
  `loan_int_rate`                Loan interest rate
  `loan_percent_income`          Loan amount as a proportion of income
  `cb_person_cred_hist_length`   Length of credit history

### Categorical features

  -----------------------------------------------------------------------
  Dataset column                      Meaning
  ----------------------------------- -----------------------------------
  `person_home_ownership`             Home-ownership category

  `loan_intent`                       Stated purpose or intent for the
                                      loan

  `loan_grade`                        Loan grade

  `cb_person_default_on_file`         Historical default indicator in the
                                      dataset
  -----------------------------------------------------------------------

**Target:** `loan_status` --- the binary label the model learns to
predict. Confirm the dataset's label definitions before interpreting
class `0` and class `1` in a deployed interface.

## Machine-learning workflow

### 1. Exploratory data analysis

The notebook uses pandas, NumPy, Matplotlib, and Seaborn to inspect the
dataset, dimensions, data types, missing values, target-class
distribution, numerical summaries, outliers, and correlations.

### 2. Data validation and cleaning

The notebook creates a working copy, removes duplicate rows, restricts
age to the range 18--100, removes rows where employment length exceeds
age, and excludes employment lengths above 60. It also filters out
non-positive loan amounts and inspects the interest-rate range.

Review these cleaning rules against the actual dataset and intended use.
The notebook's printed interest-rate range is an inspection note; it
should not be treated as an implemented filtering rule unless code
explicitly filters on it.

### 3. Train/test split

The data is separated into input features (`X`) and target (`y`). The
notebook uses an 80/20 train/test split with `random_state=42` and
stratification on `loan_status`.

### 4. Preprocessing pipelines

A `ColumnTransformer` and scikit-learn pipelines apply different
transformations to numerical and categorical data.

-   **Logistic Regression:** median imputation and standard scaling for
    numerical features; constant-value imputation (`Missing`) and
    one-hot encoding for categorical features.
-   **XGBoost:** median imputation for numerical features without
    scaling; constant-value imputation and one-hot encoding for
    categorical features.
-   `OneHotEncoder(handle_unknown="ignore")` is used so unseen
    categories do not cause the encoder to fail.

### 5. Class imbalance

The notebook calculates a positive-class weight from the training labels
(`negative_count / positive_count`) and supplies it to XGBoost as
`scale_pos_weight`. Logistic Regression is configured with
`class_weight="balanced"` in the baseline and cross-validation pipeline.

### 6. Cross-validation and hyperparameter tuning

The notebook defines **5-fold StratifiedKFold** cross-validation with
shuffling and `random_state=42`. It evaluates ROC-AUC, accuracy,
precision, recall, and F1.

For XGBoost, `RandomizedSearchCV` searches combinations of parameters
including:

-   Number of estimators
-   Maximum tree depth
-   Learning rate
-   Row subsampling
-   Column subsampling
-   Minimum child weight
-   Gamma

The search uses 150 sampled parameter combinations, five-fold
cross-validation, and `average_precision` as the search scoring metric.

## Models and evaluation

### Logistic Regression baseline

Logistic Regression serves as a baseline for comparison. Its
preprocessing pipeline imputes and scales numerical values, one-hot
encodes categorical values, and uses balanced class weights.

### XGBoost classifier

XGBoost is the tree-based model explored in the notebook. Its pipeline
uses median imputation, categorical one-hot encoding, and a
positive-class weight to account for imbalance. Hyperparameter search is
used to identify a candidate estimator.

### Evaluation metrics

The evaluation helper reports:

-   **Accuracy:** overall fraction of correct predictions.
-   **Precision:** fraction of predicted positives that are actually
    positive.
-   **Recall:** fraction of actual positives identified by the model.
-   **F1-score:** harmonic mean of precision and recall.
-   **Confusion matrix and classification report:** breakdown of
    classification outcomes.
-   **Precision--recall curve:** shows the precision/recall trade-off
    over probability thresholds.

For imbalanced default data, accuracy alone can be misleading. Consider
recall, precision, F1, average precision, ROC-AUC, and the operational
cost of different errors together. Actual metric values should be
generated by running the notebook; they are not hard-coded in this
README.

## Probability calibration and threshold selection

The notebook calibrates the best estimator returned by the randomized
XGBoost search using scikit-learn's `CalibratedClassifierCV` with
`method="sigmoid"` and five-fold calibration.

It then compares uncalibrated and calibrated probabilities using
calibration curves. A calibrated probability is intended to better
correspond to observed event frequency, but calibration quality must be
checked on held-out data.

The notebook also searches for a threshold that maximizes F1 on the
test-set precision--recall curve and saves the selected threshold to
`best_threshold.pkl`. Because selecting a threshold on the test set can
make the reported test performance optimistic, a robust production
workflow should choose the threshold using validation data and reserve
the test set for final evaluation.

## SHAP explainability

The notebook uses SHAP with `shap.TreeExplainer` to explain an XGBoost
classifier.

-   **Global explanation:** a SHAP summary plot highlights features that
    contribute most across the analyzed test examples and shows the
    direction and spread of their contributions.
-   **Local explanation:** a SHAP waterfall plot breaks down feature
    contributions for one selected applicant.
-   **Feature names:** names are obtained from the fitted preprocessing
    transformer so one-hot encoded categories can be identified in the
    plots.

SHAP explains how a model produces its predictions; it does not prove
that a feature causes default. Ensure the explainer is applied to the
same fitted model whose predictions are being explained.

## Project structure

The files shown in the project folder are:

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

The notebook is the source for the analysis and training workflow.
`main.py` is the application entry point shown in the project folder;
verify its API framework, route names, and run command against the
current source code before deploying. The static directory contains the
HTML, JavaScript, and CSS frontend.

## Installation and local run

### 1. Clone the repository

Replace `YOUR-USERNAME` with your GitHub username.

``` bash
git clone https://github.com/YOUR-USERNAME/Credence-Credit-Risk-Analysis.git
cd Credence-Credit-Risk-Analysis
```

### 2. Create a virtual environment

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

The notebook imports packages including pandas, NumPy, Seaborn,
Matplotlib, scikit-learn, XGBoost, SHAP, and joblib. Confirm that
`requirements.txt` includes every dependency needed by both the notebook
and application.

### 4. Run the notebook

Start Jupyter in the project root:

``` bash
jupyter notebook credit_risk1.ipynb
```

Run the cells in order. The notebook reads `credit_risk_dataset.csv`
from the current working directory.

### 5. Run the web application

The precise command depends on how `main.py` defines the application.
Inspect the file first. If it exposes a FastAPI object named `app`, a
typical local command is:

``` bash
uvicorn main:app --reload
```

For deployment on Render, a typical FastAPI start command is:

``` bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Use these commands only if `main.py` actually exposes a FastAPI ASGI
object named `app`. Make sure the model and threshold artifacts are
available at the paths expected by the application and that frontend API
requests use the deployed backend URL.

## Saved artifacts

-   `credit_risk_model.pkl` --- the notebook saves the calibrated
    classifier to this file using `joblib`.
-   `best_threshold.pkl` --- the notebook saves the selected decision
    threshold using `joblib`.

Load serialized model files only from trusted sources and keep the
required scikit-learn/XGBoost versions compatible between training and
deployment. Pickle-based files can execute code when loaded, so do not
load untrusted artifacts.

## Limitations and responsible use

-   The model's output is an estimate, not a guarantee of future
    repayment or default.
-   Dataset bias, missing values, label quality, and distribution shifts
    can affect predictions.
-   Probability calibration should be evaluated on data that was not
    used to fit the calibrator.
-   Threshold selection should be separated from final test evaluation
    to avoid optimistic results.
-   SHAP values describe model behavior, not causal effects.
-   Review privacy, fairness, explainability, security, and applicable
    lending requirements before any consequential use.
-   Do not use this project as the sole basis for a lending decision.

## Contributing

Issues and suggestions are welcome. When submitting a change, explain
the problem or improvement and include relevant validation or test
results. Do not commit credentials or data that you are not permitted to
share.

## License

This project is distributed under the MIT License. See
[`LICENSE`](LICENSE) for the full license text.

------------------------------------------------------------------------

**Author:** Uttam Kumar Singh\
**Project:** Credence --- Credit Risk Analysis
