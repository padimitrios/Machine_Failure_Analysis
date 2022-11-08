import numpy as np
import pandas as pd
from yellowbrick.features import PCA


Xt = pd.read_csv("C:/Users/Dimitris_Pap/Desktop/data_2.csv")
y = Xt['Machine failure']

Xt = Xt.drop(['UDI','Product ID','Machine failure','Tool wear [min]','TWF','HDF','PWF','OSF','RNF'], axis=1)

visualizer = PCA(scale=True,classes = ['Failure', 'Working'], proj_features=True)
visualizer.fit_transform(Xt, y)
visualizer.show()