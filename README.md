#  Credit Card Default Prediction

Prédiction des défauts de paiement par carte de crédit à partir du dataset UCI — un projet complet de bout en bout combinant analyse exploratoire, sélection de variables par WOE/IV, gestion du déséquilibre de classes et modélisation.

---

##  Contexte

Dans le secteur bancaire, anticiper le défaut de paiement d'un client est un enjeu majeur pour la gestion du risque de crédit. Ce projet s'appuie sur un dataset réel de 30 000 clients d'une banque taïwanaise pour construire un modèle de scoring prédictif.

**Dataset :** [UCI Credit Card Default Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)  
**Variable cible :** `default.payment.next.month` (1 = défaut, 0 = pas de défaut)

---

##  Démarche

### 1. Analyse exploratoire (EDA)
- Analyse univariée des variables qualitatives (SEX, EDUCATION, MARRIAGE, PAY_i) et quantitatives (LIMIT_BAL, AGE, BILL_AMT_i, PAY_AMT_i)
- Détection du déséquilibre de classes : **77% non-défaut / 23% défaut**
- Tests statistiques : **chi² pour les variables catégorielles**, **ANOVA pour les variables continues**

### 2. Sélection de variables par WOE / Information Value
- Calcul du Weight of Evidence (WOE) et de l'Information Value (IV) pour chaque variable
- Variables les plus prédictives identifiées : `PAY_0`, `PAY_2`, `PAY_3`, `PAY_4`, `PAY_5`, `PAY_6`, `LIMIT_BAL`, `PAY_AMT1`, `PAY_AMT3`
- Analyse de la multicolinéarité entre variables PAY_i via le **coefficient V de Cramér**

### 3. Preprocessing
- Recodage et regroupement des modalités
- Discrétisation des variables continues par intervalles (binning)
- Comparaison de plusieurs stratégies de normalisation : **StandardScaler, MinMaxScaler, RobustScaler**
- Rééchantillonnage par **SMOTE** pour corriger le déséquilibre de classes

### 4. Modélisation
- **Régression Logistique** — plusieurs configurations testées :
  - Sans scaling
  - Avec StandardScaler → **Accuracy : 82.2% | Gini : 0.528**
  - Avec MinMaxScaler → Accuracy : 82.1% | Gini : 0.523
  - Avec RobustScaler → Accuracy : 82.2% | Gini : 0.528
  - Avec SMOTE + StandardScaler → Accuracy : 72.5% | **Gini : 0.586**
- Évaluation via : **courbe ROC, indice de Gini, matrice de confusion, classification report**

---

## 📊 Résultats clés

| Modèle | Accuracy | Gini Index |
|---|---|---|
| Logit sans scaling | 77.7% | 0.301 |
| Logit + StandardScaler | 82.2% | 0.528 |
| Logit + RobustScaler | 82.2% | 0.528 |
| Logit + SMOTE | 72.5% | **0.586** |

> Le modèle avec SMOTE sacrifie légèrement l'accuracy globale mais améliore significativement la détection des clients défaillants (classe minoritaire).

---

##  Stack technique

```
Python 3.10
pandas | numpy | matplotlib | seaborn
scikit-learn | imbalanced-learn (SMOTE)
scipy
```

---

## 📁 Structure du repo

```
credit-default-prediction/
├── README.md
├── notebook/
│   └── credit_default_analysis.ipynb
├── data/
│   └── README.md        # lien vers le dataset Kaggle/UCI
└── requirements.txt
```

---

##  Reproduire le projet

```bash
git clone https://github.com/ton-username/credit-default-prediction
cd credit-default-prediction
pip install -r requirements.txt
jupyter notebook notebook/credit_default_analysis.ipynb
```

---

## 👩‍💻 Auteure

**Nada Naffeti** — Élève-ingénieure en Statistiques et Intelligence Artificielle à l'ESSAI  
[LinkedIn](https://linkedin.com/in/ton-profil) · [GitHub](https://github.com/ton-username)
