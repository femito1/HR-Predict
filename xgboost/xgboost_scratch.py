import torch
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
import xgboost as xgb
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, log_loss, precision_recall_curve, roc_auc_score, classification_report
import matplotlib.pyplot as plt

data = pd.read_csv("data/HR.csv")
data = data.rename(columns={"sales": "department"})
print(data.columns)
data = pd.get_dummies(data)
categorical = ['department_IT', 'department_RandD', 'department_accounting', 'department_hr', 'department_management', 'department_marketing', 'department_product_mng', 'department_sales', 'department_support', 'department_technical', 'salary_high', 'salary_low', 'salary_medium']
data[categorical] = data[categorical].astype(float)
print(data.columns)

data.head()

y = data['left']
X = data.drop(columns=['left'])

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# create training and validation sets (80% train, 20% validation of remaining data)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
)

print("Training set size:", X_train.shape[0])
print("Validation test size:", X_val.shape[0])
print("Test set size:", X_test.shape[0])

feature_names = X_train.columns.tolist()
X_train_val = X_train_val.to_numpy()
X_train = X_train.to_numpy()
X_val = X_val.to_numpy()
X_test = X_test.to_numpy()

X_train_val_tensor = torch.tensor(X_train_val, dtype=torch.float32)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

y_train_val_tensor = torch.tensor(y_train_val.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)


params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.1,
    'max_depth': 6,
    'alpha': 0.1,
    'lambda': 10,
    'subsample': 0.8
}
num_rounds = 100
xgb_model = xgb.XGBClassifier(**params)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb_model, X_train_val_tensor, y_train_val_tensor, cv=skf, scoring='roc_auc')
print(f"Cross-validation AUC scores: {cv_scores}")
print(f"Mean AUC: {cv_scores.mean():.4f}, Std: {cv_scores.std():.4f}")

train_dmatrix = xgb.DMatrix(X_train_val_tensor, label=y_train_val_tensor, feature_names=feature_names)
test_dmatrix = xgb.DMatrix(X_test_tensor, label=y_test_tensor, feature_names=feature_names)

cv_results = xgb.cv(
    params=params,
    dtrain=train_dmatrix,
    num_boost_round=1000,
    nfold=5,
    stratified=True,
    metrics=['auc', 'logloss'],
    early_stopping_rounds=50,
    seed=42,
    verbose_eval=100
)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(cv_results['train-logloss-mean'], label='Training Loss')
plt.plot(cv_results['test-logloss-mean'], label='Validation Loss')
plt.xlabel('Boosting Rounds')
plt.ylabel('Log Loss')
plt.title('Learning Curves - Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(cv_results['train-auc-mean'], label='Training AUC')
plt.plot(cv_results['test-auc-mean'], label='Validation AUC')
plt.xlabel('Boosting Rounds')
plt.ylabel('AUC')
plt.title('Learning Curves - AUC')
plt.legend()
plt.tight_layout()
plt.show()

best_rounds = cv_results['test-logloss-mean'].argmin()
print(f"Optimal number of boosting rounds: {best_rounds}")

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

xgb_model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=best_rounds,
    evals=[(dtrain, 'train'), (dval, 'validation')],
    early_stopping_rounds=50,
    verbose_eval=100
)

preds = xgb_model.predict(dtest)
y_pred = (preds > 0.5).astype(int)

print("\nTest Set Evaluation:")
print(f"AUC: {roc_auc_score(y_test, preds):.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Log Loss: {log_loss(y_test, preds):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

threshold = 0
f1_scores = []
while threshold < 1:
    threshold += 0.001
    binary_preds = [1 if p >= threshold else 0 for p in preds]
    f1 = f1_score(y_test, binary_preds)
    f1_scores.append((threshold,f1))

best_treshold = max(f1_scores, key=lambda x: x[1])[0]
print(f"Best threshold: {best_treshold}")