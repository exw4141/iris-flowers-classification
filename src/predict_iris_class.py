from sklearn.externals import joblib

setosa_model = joblib.load('../models/trained_iris_setosa_model.pkl')
versicolor_model = joblib.load('../models/trained_iris_versicolor_model.pkl')
virginica_model = joblib.load('../models/trained_iris_virginica_model.pkl')

# Create sample iris data
setosa_iris = [
    5.0,
    3.3,
    1.4,
    0.2
]

versicolor_iris = [
    6.7,
    3.2,
    4.6,
    1.45
]

virginica_iris = [
    6.05,
    3.0,
    5.55,
    2.2
]

# Check the predictions the models make when using the sample data
irises_to_classify = [setosa_iris, versicolor_iris, virginica_iris]

setosa_classification = setosa_model.predict(irises_to_classify)
versicolor_classification = versicolor_model.predict(irises_to_classify)
virginica_classification = virginica_model.predict(irises_to_classify)

print(setosa_classification)
print(versicolor_classification)
print(virginica_classification)
