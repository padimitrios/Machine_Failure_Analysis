import pandas as pd

# Load the dataset
df = pd.read_csv("data.csv")

# Calculate the number of censored and non-censored instances
num_censored = df[df['Machine failure'] == 0].shape[0]
num_non_censored = df[df['Machine failure'] == 1].shape[0]

# Calculate the proportion of censored and non-censored instances
prop_censored = num_censored / df.shape[0]
prop_non_censored = num_non_censored / df.shape[0]

# Print the results
print("Number of censored instances: ", num_censored)
print("Number of non-censored instances: ", num_non_censored)
print("Proportion of censored instances: ", prop_censored)
print("Proportion of non-censored instances: ", prop_non_censored)