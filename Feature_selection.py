import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from yellowbrick.model_selection import rfecv

Xt = pd.read_csv("C:/Users/Dimitris/Desktop/data_2.csv")
y = Xt['Machine failure']

Xt = Xt.drop(['UDI','Product ID','Machine failure','Tool wear [min]','TWF','HDF','PWF','OSF','RNF'], axis=1)

cv = StratifiedKFold(5)
visualizer = rfecv(RandomForestClassifier(), X=Xt, y=y, cv=cv, scoring='f1_weighted')