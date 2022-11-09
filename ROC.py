import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from yellowbrick.classifier import ROCAUC
from yellowbrick.datasets import load_spam

Xt = pd.read_csv("C:/Users/Dimitris/Desktop/data_2.csv")
y = Xt['Machine failure']

Xt = Xt.drop(['UDI','Product ID','Machine failure','Tool wear [min]','TWF','HDF','PWF','OSF','RNF'], axis=1)

# Create the training and test data
X_train, X_test, y_train, y_test = train_test_split(Xt, y, random_state=42)

# Instantiate the visualizer with the classification model
model = LogisticRegression(multi_class="auto", solver="liblinear")
visualizer = ROCAUC(model, classes=["Working", "Machine failure"])

visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
visualizer.score(X_test, y_test)        # Evaluate the model on the test data
visualizer.show()      