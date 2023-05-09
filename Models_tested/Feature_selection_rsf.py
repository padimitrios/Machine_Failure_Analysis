import pandas as pd
import numpy as np
from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sklearn.ensemble import RandomForestRegressor
from sksurv.metrics import concordance_index_censored

set_config(display="text")

# Load the data
data = pd.read_csv("C:/Users/Dimitris_Pap/Desktop/data_2.csv")

# Extract the target variable
y = [(bool(i), j) for i, j in zip(data["Machine failure"], data["Tool wear [min]"])]

# Remove unnecessary columns
X = data.drop(["UDI", "Product ID", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"], axis=1)

# Convert target variable to structured array
y = np.array(y, dtype=[("failure", bool), ("time", float)])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=20)

# Define the feature selection pipeline using Rank 2D and RFE
pipe = make_pipeline(MinMaxScaler(), RFE(RandomForestClassifier(n_estimators=100, random_state=20), n_features_to_select=4), MinMaxScaler())

# Fit the pipeline on the training data and transform both the training and testing data
X_train_selected = pipe.fit_transform(X_train, Surv.from_arrays(y_train["failure"], y_train["time"]))
X_test_selected = pipe.transform(X_test)

# Define the RandomSurvivalForest model with selected features
rsf = RandomSurvivalForest(n_estimators=1000, min_samples_split=10, min_samples_leaf=15, n_jobs=-1, random_state=20)
rsf.fit(X_train_selected, y_train)

# Print the model score on the testing data
print(rsf.score(X_test_selected, y_test))