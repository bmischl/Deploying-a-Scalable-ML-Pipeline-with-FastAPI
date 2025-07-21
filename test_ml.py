import pytest
# TODO: add necessary import
import numpy as np
import pandas as pd

from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
]

# TODO: implement the first test. Change the function name and input as needed
def test_model_returns_randomforest():
    """
    Test 1 Description: Test if the train_model function actually gives back a RandomForestClassifier. 
    """
    # Your code here
    
    data = pd.DataFrame({
        'age': [22, 34],
        'workclass': ['Private', 'Self-emp-not-inc'],
        'fnlgt': [123456, 654321],
        'education': ['Bachelors', 'Masters'],
        'education-num': [13, 14],
        'marital-status': ['Never-married', 'Married-civ-spouse'],
        'occupation': ['Tech-support', 'Exec-managerial'],
        'relationship': ['Not-in-family', 'Husband'],
        'race': ['White', 'Black'],
        'sex': ['Male', 'Female'],
        'capital-gain': [0, 0],
        'capital-loss': [0, 0],
        'hours-per-week': [40, 50],
        'native-country': ['United-States', 'Jamaica'],
        'salary': ['<=50K', '>50K']
    })
    X, y, encoder, lb = process_data(data, categorical_features=cat_features, label="salary", training=True)
    model = train_model(X, y)
    from sklearn.ensemble import RandomForestClassifier
    assert isinstance(model, RandomForestClassifier)


# TODO: implement the second test. Change the function name and input as needed
def test_inference_returns_numpy():
    """
    Test 2 Description: Make sure that inference gives us a numpy array of predictions.  
    """
    # Your code here
    
    data = pd.DataFrame({
        'age': [22, 44],
        'workclass': ['Private', 'Self-emp-not-inc'],
        'fnlgt': [123456, 789012],
        'education': ['Bachelors', 'Masters'],
        'education-num': [13, 14],
        'marital-status': ['Never-married', 'Married-civ-spouse'],
        'occupation': ['Tech-support', 'Exec-managerial'],
        'relationship': ['Not-in-family', 'Husband'],
        'race': ['White', 'Black'],
        'sex': ['Male', 'Female'],
        'capital-gain': [0, 0],
        'capital-loss': [0, 0],
        'hours-per-week': [40, 60],
        'native-country': ['United-States', 'United-States'],
        'salary': ['<=50K', '>50K']
    })
    X, y, encoder, lb = process_data(data, categorical_features=cat_features, label="salary", training=True)
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)


# TODO: implement the third test. Change the function name and input as needed
def test_data_shape():
    """
    Test 3 Description: Check that the sum of train and test sized equals the original. 
    """
    # Your code here
    
    data = pd.DataFrame({
        'age': [22, 44, 30, 50],
        'workclass': ['Private', 'Self-emp-not-inc', 'Private', 'Private'],
        'fnlgt': [123456, 789012, 654321, 111111],
        'education': ['Bachelors', 'Masters', 'HS-grad', 'HS-grad'],
        'education-num': [13, 14, 9, 9],
        'marital-status': ['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-civ-spouse'],
        'occupation': ['Tech-support', 'Exec-managerial', 'Sales', 'Sales'],
        'relationship': ['Not-in-family', 'Husband', 'Not-in-family', 'Husband'],
        'race': ['White', 'Black', 'White', 'Black'],
        'sex': ['Male', 'Female', 'Male', 'Female'],
        'capital-gain': [0, 0, 0, 0],
        'capital-loss': [0, 0, 0, 0],
        'hours-per-week': [40, 60, 20, 40],
        'native-country': ['United-States', 'United-States', 'United-States', 'United-States'],
        'salary': ['<=50K', '>50K', '<=50K', '>50K']
    })
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(data, test_size=0.5, random_state=42)
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert train.shape[0] + test.shape[0] == data.shape[0]

# TODO: implement the forth test. Change the function name and input as needed
def test_compute_metrics():
    """
    Test 4 Description: Test if computing metrics return the expected value. Precision, recall, and F1 should all be 1.0.
    """
    # Your code here
    
    data = pd.DataFrame({
        'age': [22, 44],
        'workclass': ['Private', 'Self-emp-not-inc'],
        'fnlgt': [123456, 789012],
        'education': ['Bachelors', 'Masters'],
        'education-num': [13, 14],
        'marital-status': ['Never-married', 'Married-civ-spouse'],
        'occupation': ['Tech-support', 'Exec-managerial'],
        'relationship': ['Not-in-family', 'Husband'],
        'race': ['White', 'Black'],
        'sex': ['Male', 'Female'],
        'capital-gain': [0, 0],
        'capital-loss': [0, 0],
        'hours-per-week': [40, 60],
        'native-country': ['United-States', 'United-States'],
        'salary': ['<=50K', '>50K']
    })
    X, y, encoder, lb = process_data(data, categorical_features=cat_features, label="salary", training=True)
    p, r, f = compute_model_metrics(y, y)
    assert p == pytest.approx(1.0)
    assert r == pytest.approx(1.0)
    assert f == pytest.approx(1.0)