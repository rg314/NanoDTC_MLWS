# -*- coding: utf-8 -*-
"""
Created on Thu May  2 20:05:16 2019

@author: Ryan
"""

import magpy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import pickle
import magpy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam
import seaborn as sns
import keras
from scipy import stats


bandgap_dataset = pd.read_csv("faketest.csv")

_, compositions, y_full = bandgap_dataset.T.values

y_clf = np.where(y_full == 0, 0, 1)


N = y_clf.shape[0]
N_gap = y_clf.sum()
N_cond = N - N_gap
print("The total number of samples is {} of which {}"
      " are conductors and {} have bandgaps".format(N, N_cond, N_gap))


X = magpy.core.descriptors(compositions, embedding_file="elem_embedding.json",
                           operations=["wmean", "wstd", "max", "min"])


filename = "svc_model.sav"
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X, y_clf)
y_pred = loaded_model.predict(X)

preds_clf = y_pred.copy()

nonzerogap_condition = (y_clf != 0)
zerogap_condition = (y_clf == 0)

X_nonzerogap = X[nonzerogap_condition]
y_nonzerogap = y_full[nonzerogap_condition]

X_zerogap = X[zerogap_condition]
y_zerogap = y_full[zerogap_condition]

nonzerogappred_condition = (y_pred != 0)
zerogappred_condition = (y_pred == 0)

X_nonzerogappred = X[nonzerogappred_condition]
y_nonzerogappred = y_full[nonzerogappred_condition]

X_zerogappred = X[zerogappred_condition]
y_zerogappred = y_full[zerogappred_condition]


new_model = keras.models.load_model('model_mse.h5')
new_model.summary()


preds_reg_zeros = new_model.predict(X_nonzerogap)
preds_reg = new_model.predict(X_nonzerogappred)


#predicted1 = []
#ytested1 = []
#for value in preds_clf_zeros:
#    predicted1.append(float(value))
#
#for value in preds_reg_zeros:
#    predicted1.append(float(value))
#
#for value in y_nonzerogappred:
#    ytested1.append(float(value))
#
#for value in y_zerogappred:
#    ytested1.append(float(value))

predicted2 = []
ytested2 = []
for value in preds_clf:
    predicted2.append(float(value))

for value in preds_reg:
    predicted2.append(float(value))

for value in y_nonzerogap:
    ytested2.append(float(value))

for value in y_zerogap:
    ytested2.append(float(value))

def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2




#r2 = str(r2(predicted1, ytested1))
r2 = str(r2(predicted2, ytested2))

#mse1 = np.mean((np.array(predicted1) - np.array(ytested1))**2)
mse2 = np.mean((np.array(predicted2) - np.array(ytested2))**2)


fig, axs =plt.subplots(1,2)
axs = axs.flatten()
#ax = sns.regplot(x=predicted1, y=ytested1, ax=axs[0])
#ax.set(xlabel='pred', ylabel='test', title="zeros \n r2 = "+str(r2_zeros)+"\n mse = "+str(mse1))
ax = sns.regplot(x=predicted2, y=ytested2, ax=axs[1])
ax.set(xlabel='pred', ylabel='test', title="zeros dropped \n r2 = "+str(r2)+"\n mse = "+str(mse2))
plt.show()

