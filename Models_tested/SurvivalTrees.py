import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored
from sksurv.tree import SurvivalTree

# import the dataset
data = pd.read_csv("C:/Users/Dimitris/Desktop/data_2.csv")

# Drop unnecessary columns
data = data.drop(["UDI", "Product ID"], axis=1)

# Split into features and target variables
X = data.drop(["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"], axis=1)
event = data["RNF"].astype(bool)
time = np.where(event, data["Tool wear [min]"], data["Tool wear [min]"].max())

# Split into training and testing sets
train_X, test_X, train_event, test_event, train_time, test_time = train_test_split(X, event, time,
                                                                                    test_size=0.2,
                                                                                    random_state=42)

# Convert to structured arrays
train_y = np.array(list(zip(train_event, train_time)),
                   dtype=[('status', bool), ('time', float)])
test_y = np.array(list(zip(test_event, test_time)),
                  dtype=[('status', bool), ('time', float)])

# Initialize the decision tree model
tree = SurvivalTree()

# Train the model
tree.fit(train_X, train_y)

# Make predictions on the testing set
test_predictions = tree.predict(test_X)

# Calculate the concordance index
cindex = concordance_index_censored(test_y['status'], test_y['time'], test_predictions)
print("Concordance index:", cindex)