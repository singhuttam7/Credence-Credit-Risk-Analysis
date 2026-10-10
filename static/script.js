"use strict";

"use strict";

// Automatically support local development and Render deployment.
const API_BASE_URL = (() => {
  const isLocalHost = ["localhost", "127.0.0.1"].includes(
    window.location.hostname,
  );

  const isLiveServer = isLocalHost && window.location.port === "5500";

  // VS Code Live Server: frontend on port 5500,
  // FastAPI backend on port 8000.
  if (isLiveServer) {
    return "http://127.0.0.1:8000";
  }

  // FastAPI serves both the frontend and API locally
  // on port 8000, or on the deployed Render domain.
  return window.location.origin;
})();

const API_URL = `${API_BASE_URL}/predict`;

const form = document.getElementById("loanForm");
const analyzeButton = document.getElementById("analyzeButton");
const resetButton = document.getElementById("resetButton");
const formError = document.getElementById("formError");
const connectionText = document.getElementById("connectionText");
const ratioPreview = document.getElementById("ratioPreview");
const ratioInput = document.getElementById("loanPercentIncome");
const toast = document.getElementById("toast");

const emptyState = document.getElementById("emptyState");
const loadingState = document.getElementById("loadingState");
const predictionState = document.getElementById("predictionState");

const profiles = {
  stable: {
    person_age: 42,
    person_income: 75000,
    person_home_ownership: "OWN",
    person_emp_length: 15,
    loan_intent: "VENTURE",
    loan_grade: "B",
    loan_amnt: 8000,
    loan_int_rate: 10.2,
    cb_person_default_on_file: "N",
    cb_person_cred_hist_length: 17,
  },

  young: {
    person_age: 22,
    person_income: 24000,
    person_home_ownership: "RENT",
    person_emp_length: 1,
    loan_intent: "EDUCATION",
    loan_grade: "C",
    loan_amnt: 9000,
    loan_int_rate: 14.5,
    cb_person_default_on_file: "N",
    cb_person_cred_hist_length: 3,
  },

  risky: {
    person_age: 31,
    person_income: 42000,
    person_home_ownership: "RENT",
    person_emp_length: 2,
    loan_intent: "MEDICAL",
    loan_grade: "G",
    loan_amnt: 15000,
    loan_int_rate: 22,
    cb_person_default_on_file: "Y",
    cb_person_cred_hist_length: 6,
  },
};

let lastResult = null;
let toastTimer = null;
let requestInProgress = false;

// --------------------------------------------------
// Utility functions
// --------------------------------------------------

function getField(name) {
  return form.elements.namedItem(name);
}

function showToast(message, isError = false) {
  toast.textContent = message;
  toast.classList.toggle("error", isError);
  toast.classList.add("show");

  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toast.classList.remove("show");
  }, 3000);
}

function showFormError(message) {
  formError.textContent = message;
  formError.hidden = !message;
}

function setLoading(isLoading) {
  requestInProgress = isLoading;
  analyzeButton.disabled = isLoading;
  resetButton.disabled = isLoading;

  analyzeButton.querySelector(".button-text").textContent = isLoading
    ? "Analyzing application..."
    : "Analyze credit risk";

  connectionText.textContent = isLoading ? "Analyzing..." : "AI engine ready";

  if (isLoading) {
    emptyState.hidden = true;
    predictionState.hidden = true;
    loadingState.hidden = false;
  } else {
    loadingState.hidden = true;
  }
}

function showEmptyState() {
  loadingState.hidden = true;
  predictionState.hidden = true;
  emptyState.hidden = false;
}

function formatNumber(value) {
  return new Intl.NumberFormat("en-IN", {
    maximumFractionDigits: 2,
  }).format(value);
}

// --------------------------------------------------
// Live loan-to-income ratio
// --------------------------------------------------

function updateRatio() {
  const income = Number(getField("person_income").value);
  const amount = Number(getField("loan_amnt").value);

  if (income > 0 && amount >= 0) {
    const ratio = amount / income;

    ratioInput.value = ratio;
    ratioPreview.textContent = `${(ratio * 100).toFixed(2)}%`;

    ratioPreview.style.color = ratio > 0.4 ? "#bb4a4a" : "";
  } else {
    ratioInput.value = "";
    ratioPreview.textContent = "—";
    ratioPreview.style.color = "";
  }
}

getField("person_income").addEventListener("input", updateRatio);
getField("loan_amnt").addEventListener("input", updateRatio);

// --------------------------------------------------
// Client-side validation
// --------------------------------------------------

function validateApplication(data) {
  if (data.person_age < 18 || data.person_age > 100) {
    return "Applicant age must be between 18 and 100.";
  }

  if (data.person_income <= 0 || !Number.isFinite(data.person_income)) {
    return "Enter a valid positive annual income.";
  }

  if (data.person_emp_length < 0 || data.person_emp_length > 100) {
    return "Employment length must be between 0 and 100 years.";
  }

  if (data.person_emp_length > data.person_age - 14) {
    return "Employment length looks inconsistent with applicant age. Please verify the details.";
  }

  if (data.loan_amnt <= 0 || !Number.isFinite(data.loan_amnt)) {
    return "Enter a valid positive loan amount.";
  }

  if (data.loan_int_rate < 0 || data.loan_int_rate > 100) {
    return "Interest rate must be between 0% and 100%.";
  }

  if (data.loan_percent_income <= 0 || data.loan_percent_income > 1) {
    return "Loan amount divided by annual income must be greater than 0 and no more than 100%.";
  }

  if (
    data.cb_person_cred_hist_length < 0 ||
    data.cb_person_cred_hist_length > 100
  ) {
    return "Credit history length must be between 0 and 100 years.";
  }

  return "";
}

function readApplication() {
  const data = {
    person_age: Number(getField("person_age").value),
    person_income: Number(getField("person_income").value),
    person_home_ownership: getField("person_home_ownership").value,
    person_emp_length: Number(getField("person_emp_length").value),
    loan_intent: getField("loan_intent").value,
    loan_grade: getField("loan_grade").value,
    loan_amnt: Number(getField("loan_amnt").value),
    loan_int_rate: Number(getField("loan_int_rate").value),
    loan_percent_income: Number(getField("loan_percent_income").value),
    cb_person_default_on_file: getField("cb_person_default_on_file").value,
    cb_person_cred_hist_length: Number(
      getField("cb_person_cred_hist_length").value,
    ),
  };

  return data;
}

// --------------------------------------------------
// Demo profiles
// --------------------------------------------------

document.querySelectorAll("[data-profile]").forEach((button) => {
  button.addEventListener("click", () => {
    if (requestInProgress) return;

    const profile = profiles[button.dataset.profile];
    if (!profile) return;

    Object.entries(profile).forEach(([key, value]) => {
      getField(key).value = value;
    });

    updateRatio();
    showFormError("");

    document.querySelectorAll("[data-profile]").forEach((item) => {
      item.classList.toggle("selected", item === button);
    });

    showToast("Demo profile loaded. Select Analyze credit risk to continue.");
  });
});

// --------------------------------------------------
// Risk assessment presentation
// --------------------------------------------------

function renderResult(result, application) {
  const probability = Number(result.default_probability);
  const percent = probability * 100;
  const threshold = Number(result.threshold);
  const thresholdPercent = threshold * 100;
  const predictedClass = Number(result.default_prediction);

  if (
    !Number.isFinite(probability) ||
    probability < 0 ||
    probability > 1 ||
    !Number.isFinite(threshold) ||
    threshold < 0 ||
    threshold > 1 ||
    ![0, 1].includes(predictedClass)
  ) {
    throw new Error("The API returned an invalid prediction.");
  }

  lastResult = {
    ...result,
    default_probability: probability,
    threshold,
    application,
  };

  loadingState.hidden = true;
  emptyState.hidden = true;
  predictionState.hidden = false;

  const highRisk = predictedClass === 1;
  const riskBadge = document.getElementById("riskBadge");
  const probabilityValue = document.getElementById("probabilityValue");
  const probabilityBar = document.getElementById("probabilityBar");
  const thresholdValue = document.getElementById("thresholdValue");
  const classificationValue = document.getElementById("classificationValue");
  const distanceValue = document.getElementById("distanceValue");
  const resultSummary = document.getElementById("resultSummary");

  riskBadge.textContent = highRisk ? "HIGH RISK" : "LOW RISK";
  riskBadge.className = `risk-badge ${highRisk ? "high" : ""}`;

  probabilityValue.textContent = percent.toFixed(2);

  probabilityBar.className = `probability-fill ${highRisk ? "high" : ""}`;

  // Reset before animating the new prediction.
  probabilityBar.style.transition = "none";
  probabilityBar.style.width = "0%";

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      probabilityBar.style.transition = "";
      probabilityBar.style.width = `${percent}%`;
    });
  });

  thresholdValue.textContent = `${thresholdPercent.toFixed(2)}%`;

  classificationValue.textContent = `${predictedClass} — ${highRisk ? "High Risk" : "Low Risk"}`;

  const distance = (probability - threshold) * 100;
  distanceValue.textContent = `${distance > 0 ? "+" : ""}${distance.toFixed(2)} pp`;

  resultSummary.className = `result-summary ${highRisk ? "high" : ""}`;

  if (highRisk) {
    resultSummary.textContent =
      `The model estimated a ${percent.toFixed(2)}% probability of default, ` +
      `which is at or above the ${thresholdPercent.toFixed(2)}% decision threshold. ` +
      `Review the application's supporting information before making any decision.`;
  } else {
    resultSummary.textContent =
      `The model estimated a ${percent.toFixed(2)}% probability of default, ` +
      `which is below the ${thresholdPercent.toFixed(2)}% decision threshold. ` +
      `A low-risk classification does not guarantee repayment.`;
  }

  renderInsights(application, result);

  document.getElementById("resultPanel").scrollIntoView({
    behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
      ? "auto"
      : "smooth",
    block: window.innerWidth < 700 ? "start" : "nearest",
  });
}

// --------------------------------------------------
// Contextual review suggestions
// --------------------------------------------------

function renderInsights(data, result) {
  const probability = Number(result.default_probability);
  const threshold = Number(result.threshold);
  const distance = Math.abs(probability - threshold);
  const ratio = data.loan_percent_income;

  const insights = [];

  if (ratio >= 0.3) {
    insights.push({
      title: "Review loan affordability",
      text: `The requested loan is ${(ratio * 100).toFixed(1)}% of annual income. Verify income and repayment capacity.`,
    });
  } else {
    insights.push({
      title: "Verify repayment capacity",
      text: "Review existing obligations and recurring expenses alongside the reported income.",
    });
  }

  if (data.cb_person_default_on_file === "Y") {
    insights.push({
      title: "Review credit history",
      text: "A previous default is recorded. Verify its accuracy, timing, and subsequent repayment history.",
    });
  } else {
    insights.push({
      title: "Verify credit records",
      text: "Confirm that the reported credit history is complete and accurate.",
    });
  }

  if (distance <= 0.05) {
    insights.push({
      title: "Prediction is near the threshold",
      text: "The probability is within five percentage points of the configured threshold. Review the case carefully.",
    });
  } else {
    insights.push({
      title: "Review the model output",
      text: "Compare the prediction with the applicant's verified documents and your applicable assessment criteria.",
    });
  }

  const list = document.getElementById("insightsList");
  list.replaceChildren();

  insights.forEach((insight, index) => {
    const article = document.createElement("div");
    article.className = "insight-item";

    const number = document.createElement("span");
    number.className = "insight-number";
    number.textContent = String(index + 1).padStart(2, "0");

    const content = document.createElement("div");
    const title = document.createElement("strong");
    const description = document.createElement("p");

    title.textContent = insight.title;
    description.textContent = insight.text;

    content.append(title, description);
    article.append(number, content);
    list.append(article);
  });
}

// --------------------------------------------------
// Submit application to FastAPI
// --------------------------------------------------

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (requestInProgress) return;

  showFormError("");

  if (!form.reportValidity()) return;

  const application = readApplication();
  const validationMessage = validateApplication(application);

  if (validationMessage) {
    showFormError(validationMessage);
    return;
  }

  document.querySelectorAll("[data-profile]").forEach((button) => {
    button.classList.remove("selected");
  });

  setLoading(true);

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(application),
    });

    const responseData = await response.json().catch(() => null);

    if (!response.ok) {
      if (response.status === 422 && responseData?.detail) {
        const details = Array.isArray(responseData.detail)
          ? responseData.detail
              .map((item) => {
                const field = item.loc?.slice(1).join(".");
                return `${field ? field + ": " : ""}${item.msg}`;
              })
              .join("; ")
          : String(responseData.detail);

        throw new Error(`Invalid application: ${details}`);
      }

      throw new Error(
        responseData?.detail ||
          `Request failed with status ${response.status}.`,
      );
    }

    if (!responseData || typeof responseData !== "object") {
      throw new Error("The server returned an invalid response.");
    }

    renderResult(responseData, application);
    connectionText.textContent = "AI engine ready";
    showToast("Credit risk assessment completed.");
  } catch (error) {
    console.error("Prediction request failed:", error);

    loadingState.hidden = true;
    showEmptyState();

    const message =
      error instanceof TypeError
        ? "Unable to reach the API. Check that FastAPI is running and that you opened this page through the backend."
        : error.message;

    showFormError(message);
    connectionText.textContent = "Connection issue";
    showToast("Assessment failed. Check the error message.", true);
  } finally {
    setLoading(false);
  }
});

// --------------------------------------------------
// Reset and new assessment
// --------------------------------------------------

function resetAssessment() {
  if (requestInProgress) return;

  form.reset();
  updateRatio();
  showFormError("");
  lastResult = null;

  document.querySelectorAll("[data-profile]").forEach((button) => {
    button.classList.remove("selected");
  });

  const bar = document.getElementById("probabilityBar");
  bar.style.transition = "none";
  bar.style.width = "0%";

  showEmptyState();
  document.getElementById("insightsList").replaceChildren();

  const defaultInsights = [
    [
      "Verify applicant details",
      "Ensure income, employment, and credit history are accurate.",
    ],
    [
      "Review affordability",
      "Compare the requested loan amount with annual income.",
    ],
    [
      "Use the result responsibly",
      "A model estimate should support, not replace, a full credit review.",
    ],
  ];

  defaultInsights.forEach(([title, text], index) => {
    const article = document.createElement("div");
    article.className = "insight-item";

    const number = document.createElement("span");
    number.className = "insight-number";
    number.textContent = String(index + 1).padStart(2, "0");

    const content = document.createElement("div");
    const heading = document.createElement("strong");
    const paragraph = document.createElement("p");

    heading.textContent = title;
    paragraph.textContent = text;

    content.append(heading, paragraph);
    article.append(number, content);
    document.getElementById("insightsList").append(article);
  });

  showToast("Form reset. Ready for a new assessment.");
}

resetButton.addEventListener("click", resetAssessment);
document
  .getElementById("newAssessmentButton")
  .addEventListener("click", resetAssessment);

// --------------------------------------------------
// Copy prediction summary
// --------------------------------------------------

document
  .getElementById("copyResultButton")
  .addEventListener("click", async () => {
    if (!lastResult) {
      showToast("Run an assessment before copying.", true);
      return;
    }

    const result = lastResult;
    const summary = [
      "CREDENCE — CREDIT RISK ASSESSMENT",
      `Predicted default probability: ${(result.default_probability * 100).toFixed(2)}%`,
      `Risk classification: ${result.risk_level || (result.default_prediction === 1 ? "High Risk" : "Low Risk")}`,
      `Predicted class: ${result.default_prediction}`,
      `Decision threshold: ${(result.threshold * 100).toFixed(2)}%`,
      "",
      "Model output is an estimate, not a guarantee of repayment or default.",
    ].join("\n");

    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(summary);
      } else {
        const textArea = document.createElement("textarea");
        textArea.value = summary;
        textArea.style.position = "fixed";
        textArea.style.opacity = "0";
        document.body.appendChild(textArea);
        textArea.select();

        const copied = document.execCommand("copy");
        textArea.remove();

        if (!copied) throw new Error("Clipboard copy is unavailable.");
      }

      showToast("Assessment summary copied.");
    } catch (error) {
      console.error(error);
      showToast("Unable to copy. Check browser clipboard permissions.", true);
    }
  });

// --------------------------------------------------
// Initialize
// --------------------------------------------------

updateRatio();
showEmptyState();

// Check API availability without sending an applicant record.
fetch(`${API_BASE_URL}/health`)
  .then((response) => {
    if (!response.ok) throw new Error("Health check failed.");
    return response.json();
  })
  .then((health) => {
    const ready =
      health.status === "healthy" &&
      health.model_loaded === true &&
      health.threshold_loaded === true;

    connectionText.textContent = ready
      ? "AI engine ready"
      : "Model unavailable";
  })
  .catch(() => {
    connectionText.textContent = "Connection issue";
  });
