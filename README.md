# Customer Churn Prediction: Machine Learning with Python

**Portfolio Project 3 — Predictive Analytics**

> *Projects 1 & 2 answered "what happened." This project answers "what will happen next — and who is at risk?"*

---

## Business Problem

Customer churn — when a subscriber cancels or stops using a service — is one of the costliest problems in the telecom industry. Every churned customer represents **~$500 in lost lifetime revenue** (conservative estimate based on ~$65/month × 12 months average tenure before exit). With industry churn rates of 20–26%, a 7,000-customer base can lose **$700,000–$900,000 in annual recurring revenue** to churn alone.

Retention is far more cost-effective than acquisition: winning back or retaining one customer costs approximately **$50**, while acquiring a new customer costs **$200–$350**. The business imperative is clear: identify at-risk customers *before* they leave so targeted retention campaigns can intervene.

**This project builds a machine learning model that does exactly that.**

---

## Dataset

| Attribute | Detail |
|---|---|
| Source | IBM Telco Customer Churn (Kaggle) |
| Customers | 7,043 |
| Features | 20 |
| Target | `Churn` — Yes (1) / No (0) |
| Churn rate | ~26.5% |

**Key features:** `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`, `PaymentMethod`, `OnlineSecurity`, `TechSupport`, `PaperlessBilling`, `SeniorCitizen`, and more.

---

## Technical Approach

```
Raw Data → EDA → Feature Engineering → SMOTE → 3 ML Models → SHAP Explainability → Business Recommendations
```

### 1. Exploratory Data Analysis (6 charts)
Identified key churn drivers before modelling:
- Month-to-month contracts: **43% churn rate** vs 3% for two-year contracts
- Fiber optic customers: **~42% churn rate** vs 19% for DSL
- Electronic check payment: **~45% churn rate** vs 15% for auto bank transfer
- New customers (< 12 months tenure): dramatically higher churn risk

### 2. Data Cleaning & Feature Engineering
- Corrected `TotalCharges` (stored as string with blank values) → numeric, median-filled
- Converted `Churn` Yes/No → binary 1/0
- One-hot encoded all categorical features (`pd.get_dummies`, `drop_first=True` to avoid multicollinearity)
- Dropped `customerID` (unique identifier, zero predictive value)
- **30 features** in final model input

### 3. Class Imbalance Handling — SMOTE
The dataset is imbalanced (~74% retained, ~26% churned). A naive model predicting "no churn" always would score 74% accuracy but identify **zero churners** — useless. Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to the *training data only* to create a 1:1 balanced training set without data leakage.

### 4. Three Models Trained & Compared

| Model | Why Included |
|---|---|
| Logistic Regression | Interpretable linear baseline; coefficients show feature direction |
| Random Forest | Ensemble bagging; handles non-linearity; robust to outliers |
| Gradient Boosting | Sequential ensemble; often highest accuracy on tabular data |

### 5. SHAP Explainability
Used SHAP (SHapley Additive exPlanations) to explain *individual* predictions — not just global feature rankings. SHAP reveals the direction and magnitude of each feature's impact per customer, enabling personalised retention messaging.

---

## Best Model & Performance Metrics

**Winner: Logistic Regression** — selected on ROC-AUC (primary metric for imbalanced classification)

| Metric | Logistic Regression | Random Forest | Gradient Boosting |
|---|---|---|---|
| **Accuracy** | 73.8% | 77.4% | 78.2% |
| **Precision** | 50.4% | 57.2% | 57.9% |
| **Recall** | **79.7%** | 59.4% | 65.5% |
| **F1** | **0.618** | 0.583 | 0.615 |
| **ROC-AUC** | **0.8403** | 0.8231 | 0.8362 |

**Why ROC-AUC over Accuracy?** With a 74/26 class imbalance, accuracy is misleading. ROC-AUC measures ranking ability — the probability that the model ranks a churner above a non-churner — across all decision thresholds. An AUC of **0.84** means the model correctly identifies the higher-risk customer 84% of the time. For a retention campaign, good ranking = good targeting.

**Why Logistic Regression won despite lower accuracy?** It achieved the highest Recall (79.7%) — meaning it catches the most actual churners — and the highest ROC-AUC. In churn prediction, **missing a churner is more costly than a false alarm**. Gradient Boosting has better precision but leaves 34.5% of churners undetected.

---

## Key Findings: Top 5 Churn Predictors

Based on SHAP values (direction shows how each feature affects churn probability):

| Rank | Feature | SHAP Direction | Business Meaning |
|---|---|---|---|
| 1 | `tenure` | Negative ↓ | Longer-tenured customers are far less likely to churn — loyalty compounds over time |
| 2 | `MonthlyCharges` | Positive ↑ | Higher monthly bills significantly increase churn risk — value perception is critical |
| 3 | `Contract_Two year` | Negative ↓ | Two-year contracts strongly protect against churn — lock-in creates loyalty |
| 4 | `InternetService_Fiber optic` | Positive ↑ | Fiber customers churn more — possibly unmet speed/value expectations |
| 5 | `PaymentMethod_Electronic check` | Positive ↑ | Manual payment methods signal low digital engagement and higher churn |

---

## Business Recommendations

### Top 3 At-Risk Customer Profiles
1. **New + flexible**: Tenure < 12 months, month-to-month contract, no add-on services
2. **Expensive fiber**: Fiber optic + >$80/month + month-to-month contract
3. **Disengaged payer**: Electronic check payment + first year of service

### 3 Targeted Retention Strategies
1. **Contract upgrade campaign** — Offer 15–20% discount to upgrade month-to-month customers to annual contracts in months 3–5 of tenure
2. **Fiber value assurance** — Proactive customer success outreach for Fiber customers > $80/month; bundle TechSupport/OnlineSecurity at discount
3. **Digital engagement conversion** — Incentivise ($10–$20 bill credit) electronic check payers to switch to auto-pay

### ROI Estimate
With the model's 70% recall rate, targeting 7,043 customers:

| | Value |
|---|---|
| Churners identified by model | ~1,306 |
| Successfully retained (30% conversion) | ~392 |
| Revenue saved | **$196,000** |
| Campaign cost ($50 × 1,306) | $65,300 |
| **Net ROI** | **$130,700 (200% return)** |

A blanket campaign (no model) targeting all 7,043 would cost $352,150 — **5.4× more spend** for the same retention.

---

## Tools & Technologies

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| pandas | Data loading, cleaning, feature engineering |
| scikit-learn | ML models, train/test split, StandardScaler, metrics |
| imbalanced-learn | SMOTE for class imbalance |
| SHAP | Explainable AI — feature impact direction & magnitude |
| seaborn / matplotlib | Visualisation (15 charts) |
| joblib | Model serialisation |
| Jupyter Notebook | Interactive analysis environment |

---

## Folder Structure

```
customer-churn-prediction/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   ← IBM Telco dataset (Kaggle)
├── notebooks/
│   └── churn_analysis.ipynb                    ← Main analysis (11 sections)
├── outputs/
│   ├── charts/                                 ← 15 saved visualisations
│   │   ├── 01_churn_by_contract.png
│   │   ├── 02_churn_by_internet_service.png
│   │   ├── 03_monthly_charges_distribution.png
│   │   ├── 04_tenure_distribution.png
│   │   ├── 05_churn_by_payment_method.png
│   │   ├── 06_correlation_heatmap.png
│   │   ├── 07_model_comparison.png
│   │   ├── 08_feature_importance.png
│   │   ├── 09_shap_summary.png
│   │   ├── 09_shap_bar.png
│   │   ├── cm_logistic_regression.png
│   │   ├── cm_random_forest.png
│   │   └── cm_gradient_boosting.png
│   └── models/
│       ├── best_model.pkl                      ← Trained Logistic Regression
│       └── scaler.pkl                          ← Fitted StandardScaler
├── venv/                                       ← Python virtual environment
└── README.md
```

---

## How to Run Locally

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd customer-churn-prediction

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter \
            openpyxl shap imbalanced-learn

# 4. Launch Jupyter
jupyter notebook notebooks/churn_analysis.ipynb

# 5. Run All Cells (Kernel → Restart & Run All)
```

---

## Load the Saved Model & Make Predictions

```python
import joblib
import pandas as pd
import numpy as np

# Load model and scaler
model  = joblib.load('outputs/models/best_model.pkl')
scaler = joblib.load('outputs/models/scaler.pkl')

# Create a customer profile (must match the 30 encoded features)
# Feature names from the trained model:
feature_names = model.feature_names_in_  # if sklearn >= 1.0

# Example: high-risk customer
customer = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
customer['tenure']         = 2     # 2 months — new customer
customer['MonthlyCharges'] = 90.0  # high monthly bill
customer['TotalCharges']   = 180.0

# Scale and predict
scaled        = scaler.transform(customer)
churn_prob    = model.predict_proba(scaled)[0][1]
churn_pred    = model.predict(scaled)[0]

print(f'Churn probability: {churn_prob*100:.1f}%')
print(f'Prediction: {"CHURN RISK" if churn_pred == 1 else "LIKELY TO STAY"}')
```

---

## Certifications

This project was built to demonstrate skills from:

- **Google Data Analytics Professional Certificate** — data cleaning, EDA, visualisation, SQL, spreadsheets
- **Google Advanced Data Analytics Professional Certificate** — machine learning, Python, statistical analysis, predictive modelling, model evaluation

---

## Portfolio

- **Project 1:** [Descriptive Analytics — Link](https://github.com/yourusername/project-1)
- **Project 2:** [Descriptive Analytics — Link](https://github.com/yourusername/project-2)
- **Project 3:** This project — Predictive ML with Python
