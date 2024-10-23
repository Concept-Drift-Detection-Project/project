import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from river.datasets import synth

def test_regression_models():
    # Define the drift points
    first_point = 7000
    second_point = 11000

    # Generate synthetic dataset
    dataset = synth.FriedmanDrift(
        drift_type='gra',
        position=(first_point, second_point),
        seed=42
    )

    # Initialize data container
    data = []
    for i, (x, y) in enumerate(dataset):
        x_values = list(x.values())
        data.append(x_values + [y])
        if i >= second_point:
            break

    # Create DataFrame
    column_names = [f'x{i}' for i in range(1, len(x_values) + 1)] + ['y']
    df = pd.DataFrame(data, columns=column_names)

    # Split into training and streaming data
    train = df.iloc[:(first_point // 2)]
    stream = df.iloc[(first_point // 2):]

    X_train = train.drop(columns='y').values
    y_train = train['y'].values

    # Define models
    models = {
        "Linear Regressor": LinearRegression(),
        "SVM Regressor": SVR(),
        "Decision Tree Regressor": DecisionTreeRegressor(),
        "Random Forest Regressor": RandomForestRegressor()
    }

    # Iterate over models to test
    for model_name, model in models.items():
        try:
            # Create and fit the pipeline
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", model)
            ])
            pipeline.fit(X=X_train, y=y_train)

            # Simulate predictions on the streaming data
            X_stream = stream.drop(columns='y').values
            predictions = pipeline.predict(X_stream)

            print(f"{model_name} executed successfully. Predictions: {predictions[:5]}")

        except Exception as e:
            print(f"Error occurred in {model_name}: {e}")

# Call the test function
test_regression_models()
