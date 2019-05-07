import magpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import keras
from scipy import stats


bandgap_dataset = pd.read_csv("bandgap-example.csv")

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
y_clfpred = loaded_model.predict(X)


new_model = keras.models.load_model('model_mse_dropout.h5')
new_model.summary()

y_regpred = new_model.predict(X)


nonzerogappred_condition = (y_clfpred != 0)
zerogappred_condition = (y_clfpred == 0)


y_nonzerogappred = y_regpred[nonzerogappred_condition]
y_zerogappred = y_regpred[zerogappred_condition]

y_nonzerogap = y_full[nonzerogappred_condition]
y_zerogap = y_full[zerogappred_condition]


predicted = []
ytested = []
for value in y_nonzerogappred:
    predicted.append(float(value))

for value in y_zerogappred:
    predicted.append(float(0))

for value in y_nonzerogap:
    ytested.append(float(value))

for value in y_zerogap:
    ytested.append(float(value))


pred = predicted
y_full = ytested

def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2




#r2_zeros = str(r2(predicted1, ytested1))
r2 = str(r2(pred, y_full))
mse1 = np.mean((np.array(pred) - np.array(y_full))**2)
#mse1 = np.mean((np.array(predicted1) - np.array(ytested1))**2)
#mse2 = np.mean((np.array(predicted2) - np.array(ytested2))**2)


fig, axs =plt.subplots(1,2)
axs = axs.flatten()
#ax = sns.regplot(x=predicted1, y=ytested1, ax=axs[0])
#ax.set(xlabel='pred', ylabel='test', title="zeros \n r2 = "+str(r2_zeros)+"\n mse = "+str(mse1))
ax = sns.regplot(x=pred, y=y_full, ax=axs[1])
ax.set(xlabel='pred', ylabel='test', title="r2 = "+str(r2)+"\n mse = "+str(mse1))
plt.show()

