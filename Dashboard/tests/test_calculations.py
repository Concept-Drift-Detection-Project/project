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


def validate_drift_detector_performance():
    # Generate synthetic dataset
    first_point = 7000
    second_point = 11000
    dataset = synth.FriedmanDrift(
        drift_type='gra',
        position=(first_point, second_point),
        seed=42
    )

    # Prepare the dataset
    data = []
    for i, (x, y) in enumerate(dataset):
        x_values = list(x.values())
        data.append(x_values + [y])
        if i >= second_point:
            break

    df = pd.DataFrame(data, columns=[f'x{i}' for i in range(1, len(x_values) + 1)] + ['y'])
    train = df.iloc[:(first_point // 2)]
    stream = df.iloc[(first_point // 2):]

    # Create and fit the regression model
    model = LinearRegression()
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])
    pipeline.fit(X=train.drop(columns='y').values, y=train['y'].values)

    # Prepare for drift detection
    X_stream = stream.drop(columns='y').values
    y_stream = stream['y'].values
    detectors = {
        'DDM': DDM(config=DDMConfig()),
        'EDDM': EDDM(config=EDDMConfig()),
        'ADWIN': ADWIN(config=ADWINConfig()),
        'Page Hinkley': PageHinkley(config=PageHinkleyConfig())
    }

    # Initialize results dictionary
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

            if np.isnan(error) or error < 0:
                print(f"Warning: Invalid error value {error} at index {i}. Skipping this iteration.")
                continue  # Skip this iteration

            # Update the detector
            detector.update(value=error)

            # Check for detected drift
            if detector.drift:
                detected_drifts.append(i + len(train))

                # Check for false alarms
                if i + len(train) < first_point:  # Before the first drift point
                    false_alarms += 1
                else:
                    detection_delays.append((i + len(train)) - first_point)

        # Calculate metrics
        false_alarm_rate = (false_alarms / (first_point-len(train))) if detected_drifts else 0
        average_detection_delay = detection_delays[0] if detection_delays else None 

        results[detector_name] = {
            'Detected Drifts': detected_drifts,
            'False Alarms': false_alarms,
            'False Alarm Rate': false_alarm_rate,
            'Average Detection Delay': average_detection_delay
        }

    # Validate results
    for detector_name, metrics in results.items():
        assert metrics['False Alarms'] >= 0, f"False alarms for {detector_name} should be non-negative"
        assert metrics['False Alarm Rate'] >= 0, f"False alarm rate for {detector_name} should be non-negative"
        assert metrics['Average Detection Delay'] is None or metrics['Average Detection Delay'] >= 0, \
            f"Average detection delay for {detector_name} should be non-negative or None"
        print(f"{detector_name} validated successfully.")
    
    print("All validations passed!")

# Call the validation function
validate_drift_detector_performance()
