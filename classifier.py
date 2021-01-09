import magpy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import pickle

bandgap_dataset = pd.read_csv("bandgap-example.csv")

_, compositions, y_reg = bandgap_dataset.T.values

y_clf = np.where(y_reg == 0, 0, 1)


N = y_clf.shape[0]
N_gap = y_clf.sum()
N_cond = N - N_gap
print("The total number of samples is {} of which {}"
      " are conductors and {} have bandgaps".format(N, N_cond, N_gap))


X = magpy.core.descriptors(compositions, embedding_file="embeddings/elem_embedding.json",
                           operations=["wmean", "wstd", "max", "min"])

# keys = X
# values = y_reg

# forward_dict = dict(list(zip(keys, values)))

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_clf, test_size=0.2)


# Scale the input features to improve
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Define and fit the model
clf = SVC(probability=True, C=10, gamma=0.001).fit(X_train, y_train)
filename = 'svc_model.sav'
pickle.dump(clf, open(filename, 'wb'))

# Make predictions on the test set
y_pred = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)[:, 1]

print(confusion_matrix(y_test, y_pred))
print('\n')
print(classification_report(y_test, y_pred))


"""
Grid search
# old search
param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
grid = GridSearchCV(SVC(), param_grid, verbose=3)


grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_estimator_)


grid_predictions = grid.predict(X_test)

print(confusion_matrix(y_test, grid_predictions))
print('\n')
print(classification_report(y_test, grid_predictions))
"""


# Evaluate the performance vs target values
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
auc_ = auc(fpr, tpr)
pre, re, f1, sup = precision_recall_fscore_support(y_test, y_pred)

plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC-AUC score is' + str(auc_))
plt.show()

print("The ROC-AUC score is {} \n"
      "The Precision score is {} \n"
      "The Recall score is {} \n"
      "The F1 score is {}".format(auc_, pre, re, f1))


y_pred = clf.predict(X_test)
