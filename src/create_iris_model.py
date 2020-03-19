import pandas as pd

# Load CSV data set into a pandas dataframe
iris_df = pd.read_csv('../iris.csv')

# Perform one-hot encoding on the class field
encoded_iris_df = pd.get_dummies(iris_df, columns=['class'])
