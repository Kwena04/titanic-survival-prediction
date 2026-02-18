# Titanic Survival Prediction

A machine learning model that predicts passenger survival on the Titanic based on class, age, sex, and other features.

## Files

- `titanic_notebook.ipynb` - The Jupyter notebook with all code
- `titanic_model.pkl` - The trained Random Forest model
- `titanic_scaler.pkl` - The data scaler

## How to Use

1. Load the model:
```python
   import pickle
   model = pickle.load(open('titanic_model.pkl', 'rb'))
   scaler = pickle.load(open('titanic_scaler.pkl', 'rb'))
```

2. Make a prediction:
```python
   passenger = [[1, 0, 25, 1, 0, 55.0, 0]]  # First-class, Female, Age 25
   prediction = model.predict(scaler.transform(passenger))
```

## Model Performance

- Accuracy: ~82%
- Precision: ~81%
- Recall: ~73%
