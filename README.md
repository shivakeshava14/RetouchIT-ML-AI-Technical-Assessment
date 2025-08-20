# RetouchIT-ML-AI-Technical-Assessment
Technical Assessment that ensures candidates demonstrates practical expertise rather than just theoretical knowledge.


# 🕵️‍♂️ Phase 1: Structured Data Fraud Detection (scikit-learn)

> **Real-world context**: Financial institutions process billions of transactions daily. Your model will flag fraudulent transactions while minimizing false positives (blocking legitimate payments costs $3M/hour in lost revenue).

## 📊 Project Overview
Build a fraud detection model using **structured transaction data** with scikit-learn. This phase focuses exclusively on tabular data analysis - **no deep learning required**. You'll demonstrate mastery of scikit-learn's ecosystem for imbalanced classification problems.

### ⚠️ Critical Constraints
- **Class imbalance**: Only 0.2% of transactions are fraudulent
- **Time limit**: Full pipeline must complete in ≤ 15 minutes
- **Business impact**: False positives cost 5x more than false negatives

## 📥 Dataset
- **Source**: [Synthetic Financial Transactions](https://www.kaggle.com/datasets/ealaxi/paysim1) (subset provided)
- **Location**: `data/transactions.csv`
- **Key fields**:
  ```csv
  step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,
  nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud
  ```

Critical note: isFlaggedFraud is NOT a valid feature (leakage!)

✅ Deliverables

Submit these files in your PR:

src/preprocessing.py
	
Data pipeline
	
Must use
ColumnTransformer
and handle imbalance

src/model_comparison.py
	
Model training
	
3+ scikit-learn models with
GridSearchCV

models/fraud_model.pkl
	
Final model
	
Includes preprocessing + classifier

reports/analysis.md
	
Business report
	
Max 1 page explaining key decisions

Task Requirements 
1. Data Preprocessing (40%)

# MUST implement:
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['amount', 'oldbalanceOrg', ...]),
        ('cat', OneHotEncoder(), ['type'])
    ])


Handle missing values (if any)
Address class imbalance using TWO methods:
Technique 1: Algorithmic (e.g., class_weight='balanced')
Technique 2: Resampling (e.g., SMOTE or NearMiss)
         
Explain in report: Why you chose these specific methods

2. Model Comparison (30%) 

Train and compare 3 scikit-learn models: 

    Logistic Regression (with L1/L2 regularization)
    Random Forest
    XGBoost (using scikit-learn API)
     

For each model: 

    Use GridSearchCV with precision-recall AUC as scoring metric
     

3. Model Selection & Export (30%) 

    Select ONE final model based on:
        Precision-recall tradeoffs (not accuracy!)
        Business impact of false positives
        SHAP-based feature importance
         
    Export complete pipeline (preprocessing + model) using joblib or pickle
   
    Critical: Justify why deep learning would be inappropriate here

🚫 Common Pitfalls (Avoid These!)

- Using accuracy as primary metric
- Ignoring `isFlaggedFraud` leakage (will fail validation!)
- Building separate preprocessing for train/test
- Using `RandomForest` without `class_weight`
- Trying to use TensorFlow here (overkill!)

🛠️ Getting Started 

    Clone repo
    Install requirements

Use RandomizedSearchCV instead of GridSearchCV for faster tuning

Calculate business impact: `(false_positives * false_negatives)

For SHAP: Focus on top 3 features driving fraud predictions

Remember: In finance, explainability > 0.5% accuracy gain
