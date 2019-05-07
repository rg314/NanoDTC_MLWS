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

bandgap_dataset = pd.read_csv("bandgap-example_filter.csv")

_, compositions, y_reg = bandgap_dataset.T.values

y_clf = np.where(y_reg == 0, 0, 1)


N = y_clf.shape[0]
N_gap = y_clf.sum()
N_cond = N - N_gap
print("The total number of samples is {} of which {}"
      " are conductors and {} have bandgaps".format(N, N_cond, N_gap))


X = magpy.core.descriptors(compositions, embedding_file="elem_embedding.json",
                           operations=["wmean", "wstd", "max", "min"])

print(len(X[0]))
X = pd.DataFrame(X, columns=['arr{}'.format(i + 1) for i in range(544)])
y = pd.DataFrame(y_reg)

X = np.asarray(X)
y = np.asarray(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

new_model = keras.models.load_model('model.h5')

new_model.summary()

preds = new_model.predict(X_test)

diff = preds.flatten() - y_test
percentDiff = (diff / y_test) * 100
absPercentDiff = np.abs(percentDiff)
print("Abs error diff" + str(absPercentDiff))

mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

print("mean = " + str(mean) + " std = " + str(std))


predicted = []
ytested = []
for value in preds:
    predicted.append(float(value))

for value in y_test:
    ytested.append(float(value))


def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2


print(r2(predicted, ytested))
sns.regplot(x=predicted, y=ytested)
plt.show()
