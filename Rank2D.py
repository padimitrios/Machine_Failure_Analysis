import numpy as np
import pandas as pd
from yellowbrick.features import Rank2D


Xt = pd.read_csv("data.csv")
y = Xt['Machine failure']

Xt = Xt.drop(['UDI','Product ID','Machine failure','Tool wear [min]','TWF','HDF','PWF','OSF','RNF'], axis=1)

# Instantiate the visualizer with the Pearson ranking algorithm
visualizer = Rank2D(algorithm='pearson')

visualizer.fit(Xt, y)           # Fit the data to the visualizer
visualizer.transform(Xt)        # Transform the data
visualizer.show()    