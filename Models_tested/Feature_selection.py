import numpy as np
import pandas as pd
from numpy import mean
from numpy import std

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier

from yellowbrick.model_selection import rfecv

############################################### Visualize ##################################################
Xt = pd.read_csv("data.csv")
y = Xt['Machine failure']

Xt = Xt.drop(['UDI','Product ID','Machine failure','Tool wear [min]','TWF','HDF','PWF','OSF','RNF'], axis=1)

cv = StratifiedKFold(5)
visualizer = rfecv(RandomForestClassifier(), X=Xt, y=y, cv=cv, scoring='f1_weighted')

######################################### Predicted Features ################################################
# create pipeline
rfe = RFECV(estimator=DecisionTreeClassifier())
model = DecisionTreeClassifier()
pipeline = Pipeline(steps=[('s',rfe),('m',model)])

# evaluate model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(pipeline, Xt, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

# define RFE
rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=2)
# fit RFE
rfe.fit(Xt, y)
# summarize all features
for i in range(Xt.shape[1]):
	print('Column: %d, Selected %s, Rank: %.3f' % (i, rfe.support_[i], rfe.ranking_[i]))

