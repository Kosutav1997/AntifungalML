
import pandas as pd
import numpy as np
import joblib
import random
import matplotlib.pyplot as plt
#from scipy import interp  # For interpolation of TPR values (if using older numpy, otherwise use np.interp)
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, classification_report,
                             matthews_corrcoef, balanced_accuracy_score, make_scorer)
random.seed(42) 
# --------------------------
# Data Preparation
# --------------------------
df = pd.read_csv('combined_data.csv')

# Select only numerical features for scaling
numerical_features = ['MolecularWeight', 'LogP', 'NumHDonors', 'NumHAcceptors']
X = df[numerical_features].values
y = df.iloc[:, -1].values

# Split data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

# Scale numerical features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# --------------------------
# Model and Cross Validation Metrics
# --------------------------
classifier = RandomForestClassifier(n_estimators=100, random_state=0)

# Define custom scorer for MCC
mcc_scorer = make_scorer(matthews_corrcoef)

# Define scoring metrics including balanced accuracy
scoring = {
    'accuracy': 'accuracy',
    'balanced_accuracy': 'balanced_accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc',
    'mcc': mcc_scorer
}

# Perform 5-fold cross validation on the training set
cv_results = cross_validate(classifier, X_train, y_train, cv=5, scoring=scoring, return_train_score=False)

# Print metrics for each fold
print("5-Fold Cross Validation Metrics:")
for i in range(5):
    print(f"\nFold {i+1}:")
    print(f"  Accuracy: {cv_results['test_accuracy'][i]:.4f}")
    print(f"  Balanced Accuracy: {cv_results['test_balanced_accuracy'][i]:.4f}")
    print(f"  Precision: {cv_results['test_precision'][i]:.4f}")
    print(f"  Recall: {cv_results['test_recall'][i]:.4f}")
    print(f"  F1 Score: {cv_results['test_f1'][i]:.4f}")
    print(f"  ROC AUC: {cv_results['test_roc_auc'][i]:.4f}")
    print(f"  MCC: {cv_results['test_mcc'][i]:.4f}")

# Calculate and print mean metrics across folds
mean_accuracy = np.mean(cv_results['test_accuracy'])
mean_bal_accuracy = np.mean(cv_results['test_balanced_accuracy'])
mean_precision = np.mean(cv_results['test_precision'])
mean_recall = np.mean(cv_results['test_recall'])
mean_f1 = np.mean(cv_results['test_f1'])
mean_roc_auc = np.mean(cv_results['test_roc_auc'])
mean_mcc = np.mean(cv_results['test_mcc'])

print("\nMean Cross Validation Metrics:")
print(f"  Mean Accuracy: {mean_accuracy:.4f}")
print(f"  Mean Balanced Accuracy: {mean_bal_accuracy:.4f}")
print(f"  Mean Precision: {mean_precision:.4f}")
print(f"  Mean Recall: {mean_recall:.4f}")
print(f"  Mean F1 Score: {mean_f1:.4f}")
print(f"  Mean ROC AUC: {mean_roc_auc:.4f}")
print(f"  Mean MCC: {mean_mcc:.4f}")

# --------------------------
# ROC Curve Plot for Each Fold & Mean ROC
# --------------------------
# We use StratifiedKFold to generate ROC curves per fold
cv = StratifiedKFold(n_splits=5)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

plt.figure(figsize=(15, 15))
i = 0

for train_index, test_index in cv.split(X_train, y_train):
    X_cv_train, X_cv_test = X_train[train_index], X_train[test_index]
    y_cv_train, y_cv_test = y_train[train_index], y_train[test_index]
    
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X_cv_train, y_cv_train)
    y_cv_proba = clf.predict_proba(X_cv_test)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_cv_test, y_cv_proba)
    roc_auc = roc_auc_score(y_cv_test, y_cv_proba)
    aucs.append(roc_auc)
    
    # Interpolate tpr at common mean_fpr values
    tpr_interp = np.interp(mean_fpr, fpr, tpr)
    tpr_interp[0] = 0.0
    tprs.append(tpr_interp)
    
    plt.plot(fpr, tpr, lw=1, linestyle='--', label=f'Fold {i+1} (AUC = {roc_auc:.2f})')
    i += 1

# Plot mean ROC curve
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = roc_auc_score(y, classifier.fit(X_train, y_train).predict_proba(X)[:, 1])
plt.plot(mean_fpr, mean_tpr, color='b', lw=2, label=f'Mean ROC (AUC = {np.mean(aucs):.2f})', alpha=0.8)

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=0.8)
plt.xlabel('False Positive Rate', fontsize=30)
plt.ylabel('True Positive Rate', fontsize=30)
#plt.title('ROC Curves for 5-Fold Cross Validation')
plt.legend(loc="lower right", fontsize=30)
plt.grid(alpha=0.3)
#plt.show()
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=800, pil_kwargs={'compression':'png_lzw'} )
plt.close()
# --------------------------
# Final Model Training and Test Evaluation
# --------------------------
# Train classifier on the full training set
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)
y_pred_proba = classifier.predict_proba(X_test)[:, 1]  # For ROC AUC

# Compute metrics on test set
test_acc = accuracy_score(y_test, y_pred)
test_bal_acc = balanced_accuracy_score(y_test, y_pred)
test_prec = precision_score(y_test, y_pred)
test_rec = recall_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)
test_auc = roc_auc_score(y_test, y_pred_proba)
test_mcc = matthews_corrcoef(y_test, y_pred)

print("\nTest Set Evaluation Metrics:")
print(f"  Accuracy: {test_acc:.4f}")
print(f"  Balanced Accuracy: {test_bal_acc:.4f}")
print(f"  Precision: {test_prec:.4f}")
print(f"  Recall: {test_rec:.4f}")
print(f"  F1 Score: {test_f1:.4f}")
print(f"  ROC AUC: {test_auc:.4f}")
print(f"  MCC: {test_mcc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --------------------------
# Save the final trained model
# --------------------------
joblib.dump(classifier, 'random_forest_model.joblib')
print("\nModel saved as 'random_forest_model.joblib'.")
