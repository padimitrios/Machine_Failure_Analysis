import numpy as np
import pandas as pd
from yellowbrick.features.radviz import radviz



Xt = pd.read_csv("data.csv")
y = Xt['Machine failure']

Xt = Xt.drop(['UDI','Product ID','Machine failure','Tool wear [min]','TWF','HDF','PWF','OSF','RNF'], axis=1)

# Specify the target classes
classes = ["Functioning", "Machine failure"]

# Instantiate the visualizer
radviz(Xt, y, classes=classes)
