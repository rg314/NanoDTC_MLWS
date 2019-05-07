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
from keras.layers import Dropout


bandgap_dataset = pd.read_csv("bandgap_dataset.csv")

_, compositions, y_reg = bandgap_dataset.T.values

y_clf = np.where(y_reg == 0, 0, 1)


y_reg = (y_reg - y_reg.mean()) / y_reg.std()
print(y_reg)

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


def network(dim, regress=False):
    model = Sequential()
    model.add(Dense(400, input_dim=544, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(200, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.2))
    if regress:
        model.add(Dense(1, activation="linear"))

    return model


model = network(X_train.shape[1], regress=True)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_squared_error", optimizer=opt, metrics=['accuracy'])


model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000, batch_size=100)
model.save('model_mse_dropout_normal.h5')


# preds = model.predict(X_test)

# diff = preds.flatten() - y_test
# percentDiff = (diff / y_test) * 100
# absPercentDiff = np.abs(percentDiff)
# print("Abs error diff" + str(absPercentDiff))

# mean = np.mean(absPercentDiff)
# std = np.std(absPercentDiff)

# print("mean = " + str(mean) + " std = " + str(std))


# def r2(x, y):
#     return stats.pearsonr(x, y)[0] ** 2


# print(r2(preds, y_test))

# # sns.jointplot(preds, y_test, kind="reg")
# # print(r2(preds, y_test))
# # plt.show()
