import pandas as pd

# Load CSV data set into a pandas dataframe
iris_df = pd.read_csv('../iris.csv')

# Perform one-hot encoding on the class field
encoded_iris_df = pd.get_dummies(iris_df, columns=['class'])

# Create separate output dataframes for each iris class
setosa_values_df = encoded_iris_df['class_Iris-setosa']
versicolor_values_df = encoded_iris_df['class_Iris-versicolor']
virginica_values_df = encoded_iris_df['class_Iris-virginica']

# Remove output fields from the data set
del encoded_iris_df['class_Iris-setosa']
del encoded_iris_df['class_Iris-versicolor']
del encoded_iris_df['class_Iris-virginica']

# Create arrays for the features and each of the output classes
X = encoded_iris_df.values

y_setosa = setosa_values_df.values
y_versicolor = versicolor_values_df.values
y_virginica = virginica_values_df.values
