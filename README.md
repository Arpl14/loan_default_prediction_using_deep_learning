# loan_default_prediction_using_deep_learning

# 🏦 Home Credit Default Risk Prediction

## 📌 Project Overview

This project aims to predict **loan default risk** using a comprehensive dataset from the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/) Kaggle competition. By leveraging **machine learning** and **deep learning models**, the goal is to identify customers most likely to **default on a loan**, enabling smarter, safer lending practices and promoting **financial inclusion** for underbanked populations.

---

## 🎯 Purpose

Home Credit seeks to expand credit access for individuals with limited credit history. To ensure that loans are offered responsibly, it's crucial to develop **predictive models** that assess default risk using diverse and complex borrower information.

---

## 🗂️ Data Overview

The project utilizes **multiple relational tables** containing over 2.7 GB of data. Major datasets include:

- `application_train` / `application_test`: Applicant info & target variable (`TARGET`)
- `bureau` & `bureau_balance`: Previous credit history from other institutions
- `previous_application`: Records of past loan applications
- `POS_CASH_balance`: Point-of-sale cash loan behavior
- `credit_card_balance`: Monthly credit card data
- `installments_payments`: Payment history on previous loans

These datasets contain rich, yet often **noisy and imbalanced** data. Each table contributes unique insights into the financial behavior of applicants.

---

## ⚠️ Key Challenges

- **High dimensionality and volume** of data (millions of rows, hundreds of columns)
- **Missing values** and inconsistent formats across datasets
- **Imbalanced target** variable (fewer defaults)
- **Merging multiple tables** with complex joins
- **Avoiding data leakage** during preprocessing and training

---

## 🧠 Approach Overview

The project was executed in four structured phases:

---

## 📦 Phase 1: Dataset Loading & Setup

- Downloaded the full dataset from Kaggle.
- Unzipped and organized the files for analysis.
- Ensured integrity and consistency across all CSV files.

---

## 🔍 Phase 2: Data Understanding, Baseline Modeling

### 🔹 Goals:
- Explore the `application_train` dataset.
- Identify missing values, column types, and distributions.
- Create a baseline machine learning model.

### 🔹 Key Steps:
- Visualized key numeric and categorical features.
- Conducted correlation analysis to drop low-impact or highly collinear features.
- Imputed missing values with median/mode as appropriate.
- Built and evaluated a **Logistic Regression** and **Random Forest** model.

### 🔹 Results:
- Baseline accuracy: ~91%
- Logistic Regression slightly outperformed Random Forest in identifying defaulters.
- Highlighted importance of features like `EXT_SOURCE_2`, `EXT_SOURCE_3`, `DAYS_BIRTH`.

---

## 🔄 Phase 3: Multi-Table Merging, Feature Engineering & Model Tuning

### 🔹 Goals:
- Enrich feature space using additional tables.
- Engineer meaningful features and optimize models using pipelines and GridSearchCV.

### 🔹 Key Steps:
1. **Merging Tables:**
   - Merged `bureau`, `bureau_balance`, `previous_application`, `POS_CASH_balance`, `credit_card_balance`, and `installments_payments` into the base dataset using `SK_ID_CURR`.

2. **Data Cleaning:**
   - Dropped irrelevant and highly missing columns based on correlation and EDA.
   - Applied aggregation (`mean`, `sum`, `count`) to reduce table size and retain signal.

3. **Feature Engineering:**
   - Created interaction features: `credit/income ratio`, `employment % of age`, etc.
   - Combined domain-specific and behavioral features for higher prediction power.

4. **Pipeline Creation:**
   - Built numeric and categorical pipelines using `ColumnTransformer` and `Pipeline` from `sklearn`.

5. **Modeling & Tuning:**
   - Applied `GridSearchCV` to tune Logistic Regression, Random Forest, and Gradient Boosting models using `ROC-AUC` as scoring.

### 🔹 Results:
- **Best model**: Gradient Boosting with ROC-AUC ≈ **0.79**
- Logistic Regression remained competitive and interpretable.
- Pipelines ensured clean, reusable transformations.

---

## 🤖 Phase 4: Deep Learning Model with PyTorch

### 🔹 Goals:
- Build a deep learning model on the final merged dataset using PyTorch.
- Compare performance with traditional ML models.

### 🔹 Key Steps:
1. **Neural Network Setup:**
   - Built a Multi-Layer Perceptron (MLP) using PyTorch.
   - Used ReLU activations and sigmoid output for binary classification.

2. **Architectures Tried:**
   - Architecture 1: 256 → 128 → 64 (Best performer)
   - Architecture 2: 64 → 64 → 64
   - Architecture 3: 128 → 64 → 32

3. **Training & Optimization:**
   - Used Adam/SGD optimizers and trained for 5–20 epochs.
   - Visualized results using TensorBoard for accuracy/loss tracking.

4. **Evaluation:**
   - Evaluated using ROC-AUC, Precision, Recall, F1-Score.
   - Ensured no data leakage across splits.

### 🔹 Results:
- **Architecture 1** achieved best validation accuracy and generalization.
- Deep learning slightly outperformed Gradient Boosting, especially in detecting minority class defaults.

---

## 🧪 Final Model and Evaluation Summary

| Model                | ROC-AUC | Recall (Defaults) | Remarks                     |
|---------------------|---------|-------------------|-----------------------------|
| Logistic Regression | 0.76    | 0.60              | Interpretable baseline      |
| Random Forest       | 0.75    | 0.58              | Higher accuracy, lower recall |
| Gradient Boosting   | 0.79    | 0.64              | Tuned model with best tradeoff |
| **Neural Network**  | **0.81**| **0.67**          | Best recall, no overfitting |

---

## 🧾 Conclusion

- The project successfully built a full **ML pipeline** from raw data to deep learning.
- Through step-by-step feature enrichment and model optimization, we achieved a strong predictive model to **identify likely loan defaulters**.
- Deep learning emerged as the top performer in terms of **recall** and **ROC-AUC**, making it the final model of choice.
- This solution is deployable, explainable, and scalable for real-world credit risk assessment.

