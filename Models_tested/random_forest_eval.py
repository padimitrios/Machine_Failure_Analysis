import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sksurv.metrics import concordance_index_censored

#import the dataset
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv")

# Drop unnecessary columns
data = data.drop(["UDI",'Product ID'], axis=1)

# Convert target variable to structured array
status = data["Machine failure"].astype(bool)
time = np.where(status, data["Tool wear [min]"], data["Tool wear [min]"].max())
y = np.array([status, time]).T

# Split into features and target variables
X = data.drop(["Machine failure",'TWF','HDF','PWF','OSF','RNF'], axis=1)

# Split into training and testing sets
train_fraction = 0.8
train_size = int(train_fraction * len(data))
train_X, train_y = X.iloc[:train_size], y[:train_size]
test_X, test_y = X.iloc[train_size:], y[train_size:]

# Initialize the random forest model
rf = RandomForestRegressor(n_estimators=100)

# Train the model
rf.fit(train_X, train_y)

# Make predictions on the testing set
test_predictions = rf.predict(test_X)

event = test_y[:,0].astype(bool)

# Calculate the concordance index
cindex = concordance_index_censored(event, test_y[:,1], test_predictions[:,0])
print("Concordance index:",cindex)