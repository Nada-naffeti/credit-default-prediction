# Credit Card Default Prediction

**Predicting credit card payment defaults using machine learning**  
EDA · WOE/IV · SMOTE · Logistic Regression · UCI Dataset

---

## Overview

Predicting whether a client will default on their credit card payment is a critical challenge in banking risk management. This project delivers a full end-to-end pipeline from exploratory analysis to modeling on a real-world dataset of **30,000 clients** from a Taiwanese bank.

| | |
|---|---|
| **Dataset** | [UCI Credit Card Default Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) |
| **Target variable** | `default.payment.next.month` (1 = default, 0 = no default) |
| **Class imbalance** | 77% non-default / 23% default |
| **Best model** | Logit + StandardScaler : Accuracy: 82.2% · Gini: 0.528 |

---

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Univariate analysis of categorical variables (`SEX`, `EDUCATION`, `MARRIAGE`, `PAY_i`) and continuous variables (`LIMIT_BAL`, `AGE`, `BILL_AMT_i`, `PAY_AMT_i`)
- Class imbalance detection and visualization
- Statistical tests: **Chi-square** for categorical variables, **ANOVA** for continuous ones

### 2. Feature Selection — WOE / Information Value
- Computed **Weight of Evidence (WOE)** and **Information Value (IV)** for each feature
- Most predictive variables: `PAY_0`, `PAY_2` → `PAY_6`, `LIMIT_BAL`, `PAY_AMT1`, `PAY_AMT3`
- Multicollinearity analysis using **Cramér's V coefficient**

### 3. Preprocessing
- Feature recoding and category grouping
- Continuous variable discretization (binning)
- Comparison of normalization strategies: `StandardScaler`, `MinMaxScaler`, `RobustScaler`
- Class rebalancing with **SMOTE**

### 4. Modeling
- Logistic Regression tested across multiple configurations
- Evaluation metrics: ROC curve, Gini index, confusion matrix, classification report

---

## Results

| Model | Accuracy | Gini Index |
|-------|----------|------------|
| Logit  | 77.7% | 0.301 |
| Logit + StandardScaler | **82.2%** | **0.528** |
| Logit + RobustScaler | 82.2% | 0.528 |
| Logit + SMOTE + StandardScaler | 72.5% | 0.586 |

> **Note:** The SMOTE model trades off overall accuracy for significantly better detection of defaulting clients (minority class) a key priority in credit risk management.

---

## Tech Stack

```
pandas · numpy · matplotlib · seaborn
scikit-learn · imbalanced-learn (SMOTE) · scipy
```

---

## Project Structure

```
credit-default-prediction/
├── README.md
├── pfa-analyse-de-transactions-de-cartes-de-cr-dit.ipynb
├── data/
│   └── README.md        # link to UCI/Kaggle dataset
└── requirements.txt
```

---

## Getting Started

```bash
git clone https://github.com/Nada-naffeti/credit-default-prediction.git
cd credit-default-prediction
pip install -r requirements.txt
jupyter notebook pfa-analyse-de-transactions-de-cartes-de-cr-dit.ipynb
```

---

## Author

**Nada Naffeti** — Data Science & AI Engineering Student @ [ESSAI](https://www.essai.tn/)

[LinkedIn](https://linkedin.com/in/nada-naffeti) · [GitHub](https://github.com/Nada-naffeti)
