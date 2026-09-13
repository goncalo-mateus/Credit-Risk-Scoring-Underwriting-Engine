# Credit Risk Scoring & Underwriting Engine

Run this project interactively in Google Colab: https://colab.research.google.com/drive/1pIXLxjN1z2OcGk76hUccPJ1Qa2QNW1aW?usp=sharing

An end-to-end credit risk analytics and automated underwriting engine built in Python that evaluates loan applications using machine learning and portfolio risk benchmarking. The tool generates synthetic historical credit data, benchmarks predictive classification models (Logistic Regression vs. XGBoost), and provides a real-time decisioning HUD for institutional loan approval.

Designed for credit risk management, quantitative underwriting, and FinTech decision-making workflows.

## Key Capabilities & Features

* **Dynamic Synthetic Data Synthesis:** Generates historical credit datasets reflecting realistic financial distributions (Income, Debt, Delinquencies, Utilization).
* **Machine Learning Model Benchmarking:** Evaluates baseline Logistic Regression against an XGBoost Classifier using industry-standard risk metrics (ROC-AUC and Gini Coefficient).
* **Feature Importance Analysis:** Identifies and visualizes key drivers of default risk to ensure explainability and regulatory compliance (XAI).
* **Automated Credit Decision Engine:** Translates individual borrower profiles into calculated credit scores (300–850 range), Probability of Default (PD %), and policy-based underwriting decisions (APPROVED / REFER / DECLINED).
* **Colab Form HUD Integration:** Features interactive UI controls (inputs, sliders) for real-time risk scoring and visual comparison against portfolio-wide risk benchmarks.

## Tech Stack & Tools

* **Language:** Python 3
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning & Modeling:** Scikit-Learn, XGBoost
* **Data Visualization:** Matplotlib, Seaborn
* **Interactive HUD / UI:** IPyWidgets / Google Colab Forms

## Quantitative Framework

The engine operates on the following mathematical core:

* **Debt-to-Income (DTI) Ratio:**
  $$\text{DTI} = \frac{\text{Total Monthly Debt}}{\frac{\text{Annual Income}}{12}}$$

* **Loan-to-Income (LTI) Ratio:**
  $$\text{LTI} = \frac{\text{Loan Amount Requested}}{\text{Annual Income}}$$

* **Gini Coefficient:**
  $$\text{Gini} = 2 \times \text{ROC-AUC} - 1$$

* **Probability of Default (PD):**
  $$P(Y = 1 | X) = \sigma(W^T X + b) \quad \text{or} \quad \text{XGBoost Tree Ensemble}$$

* **Credit Score Mapping:**
  $$\text{Score} = \text{Base Score} - (\text{PD} \times \text{Scaling Factor})$$

## How to Run in Google Colab

1. Open the interactive notebook: https://colab.research.google.com/drive/1pIXLxjN1z2OcGk76hUccPJ1Qa2QNW1aW?usp=sharing
2. Configure borrower profile inputs (Income, Loan Amount, Monthly Debt, Delinquencies) in the interactive HUD.
3. Run all cells sequentially to train the models, generate performance charts, and view the automated underwriting decision.
