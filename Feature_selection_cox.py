import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

# import the dataset
data = pd.read_csv("C:/Users/Dimitris/Desktop/data_2.csv")

# Drop unnecessary columns
data = data.drop(["UDI",'Product ID'], axis=1)

# Convert target variable to structured array
status = data["Machine failure"].astype(bool)
time = np.where(status, data["Tool wear [min]"], data["Tool wear [min]"].max())
y = np.array(list(zip(status, time)), dtype=[('status', bool), ('time', float)])

# Split into features and target variables
X = data.drop(["Machine failure",'TWF','HDF','PWF','OSF','RNF'], axis=1)

# Split into training and testing sets
train_fraction = 0.6
train_size = int(train_fraction * len(data))
train_X, train_y = X.iloc[:train_size], y[:train_size]
test_X, test_y = X.iloc[train_size:], y[train_size:]

# Define the feature selection pipeline using Rank 2D and RFE
pipe = make_pipeline(MinMaxScaler(), RFE(estimator=CoxPHSurvivalAnalysis(), n_features_to_select=6), MinMaxScaler())

# Fit the pipeline on the training data and transform both the training and testing data
train_X_selected = pipe.fit_transform(train_X, train_y)
test_X_selected = pipe.transform(test_X)

# Initialize the CoxPH model
coxph = CoxPHSurvivalAnalysis()

# Train the model
coxph.fit(train_X_selected, train_y)

# Make predictions on the testing set
test_predictions = coxph.predict(test_X_selected)

event = test_y['status']

# Calculate the concordance index
cindex = concordance_index_censored(event, test_y['time'], test_predictions)
print("Concordance index:",cindex)