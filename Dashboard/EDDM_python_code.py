from river import datasets
from river import linear_model, tree
from river import drift, metrics
from river.datasets import synth
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from frouros.metrics import PrequentialError
from sklearn.pipeline import Pipeline
from frouros.detectors.concept_drift.streaming.window_based.adwin import ADWIN
from frouros.detectors.concept_drift import DDM, DDMConfig
from frouros.detectors.concept_drift.streaming.statistical_process_control.ddm import DDM
from frouros.detectors.concept_drift.streaming.statistical_process_control.eddm import EDDM
from frouros.detectors.concept_drift.streaming.change_detection.page_hinkley import PageHinkley

first_point = 5000
second_point = 10000

dataset = synth.FriedmanDrift(
    drift_type='gra',
    position=(first_point, second_point),
    seed=42
)

# Initialize the data containers
data = []
for i, (x, y) in enumerate(dataset):
    x_values = list(x.values())
    data.append(x_values + [y])
    if i >=second_point:  # Limiting to 5000 samples for simplicity
        break

# Define the column names
column_names = [f'x{i}' for i in range(1, len(x_values) + 1)] + ['y']

# Create the DataFrame
df = pd.DataFrame(data, columns=column_names)

train = df.iloc[:(first_point//2)]
stream = df.iloc[(first_point//2):]

X =train.drop(columns='y').values
y = train['y'].values

# models
concept1 = LinearRegression()
concept2 = RandomForestRegressor()
concept3 = DecisionTreeRegressor()
concept4 = SVR()

from sklearn.pipeline import Pipeline

# Define and fit model
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", concept4),
    ]
)

odp=[5000]#track the actual point where drift occures.on 5000 index onwards all are in another concept

X =stream.drop(columns='y').values
y = stream['y'].values

# Instantiate EDDM with custom configuration
detector = EDDM()

detected_drifts = []
false_alarms = 0
detection_delays = []

y_preds = []
errors = []

warning_flag = False
for i in range(len(X)):
    X_i = X[i].reshape(1, -1)
    y_i = y[i].reshape(1, -1)
    
    # Predict and calculate the error
    y_pred = pipeline.predict(X_i)#here y_pred is an array resulted from pipieline
    y_preds.append(y_pred[0])
    error = mean_squared_error(y_i, y_pred)
    errors.append(error)
    
    binary_error = 1 if error > 83.146 else 0
    
    # Update DDM with the error 
    detector.update(value=binary_error)
    
    
    # Check for detected drift
    if detector.drift:
       # print(f"Change detected at step {i + len(train)}")  # Adjusting index to match the full dataset
        detected_drifts.append(i + len(train))  # Adjusting index to match the full dataset
        
        # Determine if it's a false alarm or not
        if i + len(train) < odp[0]:  # Compare with the original drift point
            false_alarms += 1
        else:
            detection_delays.append((i + len(train)) - odp[0])
    # Check for detected warning
    if not warning_flag and detector.warning:
        print(f"Warning detected at step {i}")
        warning_flag = True
        

false_alarm_rate = false_alarms / len(detected_drifts) if detected_drifts else 0
detection_delay =(detection_delays[0]) if detection_delays else None

print(f"Detected drift points: {detected_drifts}")
print(f"False alarms: {false_alarms}")
print(f"False alarm rate: {false_alarm_rate}")
print(f"Detection delay: {detection_delay}")

# Plotting the error values
plt.figure(figsize=(14, 7))
plt.plot(range(len(errors)), errors, label="MSE Error", color='purple')
plt.xlabel("Instance Index")
plt.ylabel("Mean Squared Error")
plt.title("Behavior of Mean Squared Error Over data points")
plt.legend()
plt.grid(True)
plt.show()

# Create binary arrays for drift detection and odp
drift_indicator = np.zeros(len(data))
for drift_point in detected_drifts:
    drift_indicator[drift_point] = 1

odp_indicator = np.zeros(len(data))
for i in range(odp[0], len(data)):
    odp_indicator[i] = 1

# Plotting the drift indicator and odp indicator
plt.figure(figsize=(12, 4))
plt.plot(drift_indicator, label='Detected Drift Indicator', color='red', drawstyle='steps-post')
plt.plot(odp_indicator, label='Actual Drift Indicator (ODP)', color='green', linestyle='-', drawstyle='steps-post')
plt.xlabel('Index')
plt.ylabel('Drift Detection (0 or 1)')
plt.title('Drift Detection Indicator')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

