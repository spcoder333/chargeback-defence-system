import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# 1. Load dataset
df = pd.read_csv('data/razorpay_transactions.csv')

# 2. Preprocess Categorical Variables
df_encoded = pd.get_dummies(df.drop(columns=['transaction_id']), columns=['payment_mode', 'merchant_category', 'dispute_reason_code'], drop_first=True)

X = df_encoded.drop(columns=['is_chargeback_fraud'])
y = df_encoded['is_chargeback_fraud']

feature_names = X.columns.tolist()

# 3. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

test_raw = df.loc[X_test.index].copy()
test_raw.to_csv('data/test_sample.csv', index=False)

# 4. Train XGBoost Model
scale_pos_weight = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)
xgb_model = XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

# 5. Evaluate
preds = xgb_model.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

COST_PER_FP = 5
COST_PER_FN = 500
total_cost = (fp * COST_PER_FP) + (fn * COST_PER_FN)

# 6. Save Artifacts
importances = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
importances.head(10).to_csv('models/feature_importance.csv')

joblib.dump(xgb_model, 'models/model.pkl')
joblib.dump(feature_names, 'models/feature_names.pkl')
joblib.dump({
    'precision_recall': classification_report(y_test, preds, output_dict=True),
    'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
    'total_cost': total_cost
}, 'models/metrics.pkl')

print("XGBoost training completed successfully. Artifacts saved to models/")