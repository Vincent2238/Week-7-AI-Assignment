# Install required libraries if not already installed
# !pip install aif360 numpy pandas matplotlib seaborn scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from aif360.datasets import CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the COMPAS dataset
dataset = CompasDataset()

# Define privileged and unprivileged groups (race)
privileged_groups = [{'race': 1}]      # Caucasian
unprivileged_groups = [{'race': 0}]    # African-American

# Initial fairness metric before training
metric_orig = BinaryLabelDatasetMetric(dataset,
                                       unprivileged_groups=unprivileged_groups,
                                       privileged_groups=privileged_groups)

print("Difference in mean outcomes (FPR):", metric_orig.mean_difference())

# Reweighing to mitigate bias
RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
dataset_transf = RW.fit_transform(dataset)

# Split into features and labels
X = dataset_transf.features
y = dataset_transf.labels.ravel()

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train logistic regression
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Wrap predictions in AIF360 BinaryLabelDataset
from aif360.datasets import BinaryLabelDataset
dataset_pred = dataset.copy()
dataset_pred.labels = y_pred

# Evaluate fairness metrics after prediction
classified_metric = ClassificationMetric(dataset,
                                         dataset_pred,
                                         unprivileged_groups=unprivileged_groups,
                                         privileged_groups=privileged_groups)

print("Disparate Impact:", classified_metric.disparate_impact())
print("False Positive Rate Difference:", classified_metric.false_positive_rate_difference())

# Visualization
fpr_diff = classified_metric.false_positive_rate_difference()
groups = ['Before Reweighing', 'After Reweighing']
fpr_values = [metric_orig.mean_difference(), fpr_diff]

plt.figure(figsize=(8, 5))
sns.barplot(x=groups, y=fpr_values, palette='viridis')
plt.axhline(0, color='black', linestyle='--')
plt.title('False Positive Rate Difference by Race')
plt.ylabel('FPR Difference (African-American - Caucasian)')
plt.tight_layout()
plt.show()