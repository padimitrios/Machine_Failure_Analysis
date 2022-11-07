import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sksurv.ensemble import RandomSurvivalForest

set_config(display="text")

Xt = pd.read_csv("data_2.csv")
y = []

for i,j in zip(Xt['Machine failure'],Xt['Tool wear [min]']):
  y.append((bool(i),j))

#drop axreiastwn cols
Xt = Xt.drop(['UDI','Product ID','Machine failure','Tool wear [min]','TWF','HDF','PWF','OSF','RNF'], axis=1)

#make array compatible with train_test_split
y = np.asarray(y,dtype="bool,i4")

random_state = 20

X_train, X_test, y_train, y_test = train_test_split(
    Xt[:3000], y[:3000], test_size=0.25, random_state=random_state)

rsf = RandomSurvivalForest(n_estimators=1000,
                           min_samples_split=10,
                           min_samples_leaf=15,
                           n_jobs=-1,
                           random_state=random_state)
rsf.fit(X_train, y_train)
print(rsf.score(X_test, y_test))

X_test_sorted = X_test.sort_values(by=["Air temperature [K]", "Process temperature [K]","Rotational speed [rpm]","Torque [Nm]"])
X_test_sel = pd.concat((X_test_sorted.head(100), X_test_sorted.tail(100)))

surv = rsf.predict_survival_function(X_test_sel, return_array=True)

                ##### Compare survival curves on plot #####
# for i, s in enumerate(surv):
#     plt.step(rsf.event_times_, s, where="post", label=str(i))
# plt.ylabel("Survival probability")
# plt.xlabel("Time in minutes")
# plt.legend()
# plt.grid(True)
# plt.show()

result = permutation_importance(
    rsf, X_test, y_test, n_repeats=15, random_state=random_state
)

print(pd.DataFrame(
    {k: result[k] for k in ("importances_mean", "importances_std",)},
    index=X_test.columns
).sort_values(by="importances_mean", ascending=False))