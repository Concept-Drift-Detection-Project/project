import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from river.datasets import synth
from frouros.detectors.concept_drift.streaming.statistical_process_control.ddm import DDM, DDMConfig
from frouros.detectors.concept_drift.streaming.statistical_process_control.eddm import EDDM, EDDMConfig
from frouros.detectors.concept_drift import ADWIN, ADWINConfig
from frouros.detectors.concept_drift import PageHinkley, PageHinkleyConfig
from sklearn.metrics import mean_squared_error

def test_drift_detectors():
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

    # Create and fit the regression model
    model = LinearRegression()
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])
    pipeline.fit(X=X_train, y=y_train)

    X_stream = stream.drop(columns='y').values
    y_stream = stream['y'].values

    # Initialize drift detectors
    detectors = {
        'DDM': DDM(config=DDMConfig()),
        'EDDM': EDDM(config=EDDMConfig()),
        'ADWIN': ADWIN(config=ADWINConfig()),
        'Page Hinkley': PageHinkley(config=PageHinkleyConfig())
    }

    # Dictionary to store results
    results = {}

    for detector_name, detector in detectors.items():
        detected_drifts = []
        false_alarms = 0
        detection_delays = []

        for i in range(len(X_stream)):
            X_i = X_stream[i].reshape(1, -1)
            y_i = y_stream[i].reshape(1, -1)

            # Predict and calculate the error
            y_pred = pipeline.predict(X_i)
            error = mean_squared_error(y_true=y_i, y_pred=y_pred)

            # Update the detector
            detector.update(value=error)

            # Check for detected drift
            if detector.drift:
                detected_drifts.append(i + len(train))  # Adjust for the training set

                # Check for false alarms
                if i + len(train) < first_point:  # If drift detected before the actual drift point
                    false_alarms += 1
                else:
                    detection_delays.append((i + len(train)) - first_point)  # Delay from the first drift

        # Calculate false alarm rate and average detection delay
        false_alarm_rate = (false_alarms / (first_point-len(train))) if detected_drifts else 0
        average_detection_delay = detection_delays[0] if detection_delays else None

        # Save results
        results[detector_name] = {
            'Detected Drifts': detected_drifts,
            'False Alarms': false_alarms,
            'False Alarm Rate': false_alarm_rate,
            'Average Detection Delay': average_detection_delay
        }

    # Display the results
    for detector_name, metrics in results.items():
        print(f"--- {detector_name} ---")
        print(f"Detected Drifts: {metrics['Detected Drifts']}")
        print(f"False Alarms: {metrics['False Alarms']}")
        print(f"False Alarm Rate: {metrics['False Alarm Rate']}")
        print(f"Average Detection Delay: {metrics['Average Detection Delay']}\n")

# Call the test function
test_drift_detectors()
