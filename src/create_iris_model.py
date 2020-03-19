import pandas as pd

from sklearn import model_selection
from sklearn import ensemble

import joblib


def train_model(iris_name, model, X, y):
    """
    Trains the given model based on the given features and outputs

    iris_name -- Name of the iris flower that the model is trained to classify
    model -- The model to train
    X -- The feature data values
    y -- The output values
    """
    model.fit(X, y)

    pkl_filename = "../models/trained_iris_{}_model.pkl".format(iris_name)
    joblib.dump(model, pkl_filename)


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

# Shuffle each output data set and split them into training and test data sets
X_setosa_train, X_setosa_test, y_setosa_train, y_setosa_test = model_selection.train_test_split(X, y_setosa, test_size=0.3)
X_versicolor_train, X_versicolor_test, y_versicolor_train, y_versicolor_test = model_selection.train_test_split(X, y_versicolor, test_size=0.3)
X_virginica_train, X_virginica_test, y_virginica_train, y_virginica_test = model_selection.train_test_split(X, y_virginica, test_size=0.3)

# Create a model for each iris class
setosa_model = ensemble.GradientBoostingRegressor(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=6,
    min_samples_leaf=9,
    max_features=0.1,
    loss='huber'
)

versicolor_model = ensemble.GradientBoostingRegressor(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=6,
    min_samples_leaf=9,
    max_features=0.1,
    loss='huber'
)

virginica_model = ensemble.GradientBoostingRegressor(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=6,
    min_samples_leaf=9,
    max_features=0.1,
    loss='huber'
)

train_model("setosa", setosa_model, X_setosa_train, y_setosa_train)
train_model("versicolor", versicolor_model, X_versicolor_train, y_versicolor_train)
train_model("virginica", virginica_model, X_virginica_train, y_virginica_train)
