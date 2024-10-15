from river import datasets
from river import linear_model, tree
from river import drift, metrics
from river.datasets import synth
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from frouros.detectors.concept_drift import PageHinkley, PageHinkleyConfig
from frouros.metrics import PrequentialError

dataset = synth.FriedmanDrift(
    drift_type='gra',
    position=(2000,3000),
    seed=42)

# Initialize the data containers
data = []
for i, (x, y) in enumerate(dataset):
    x_values = list(x.values())
    data.append(x_values + [y])
    if i >=3000:  # Limiting to 5000 samples for simplicity
        break

# Define the column names
column_names = [f'x{i}' for i in range(1, len(x_values) + 1)] + ['y']

# Create the DataFrame
df = pd.DataFrame(data, columns=column_names)

train = df.iloc[:1000]
stream = df.iloc[1000:]

# Create the drift indicator list
drift_position = 1000

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
        ("model", concept1),
    ]
)
pipeline.fit(X=X, y=y)

X_test = stream.drop(columns='y').values
y_test = stream['y'].values

# Detector configuration and instantiation
config = PageHinkleyConfig(
    lambda_ = 10.0,   # initially 50
    alpha = 0.9999,  # forgetting factor property, default to 0.9999
    min_num_instances = 30,  # minimum numbers of instances to start looking for changes, default to 30
    # delta is initially 0.005 i.e confidence value, default to 0.005
)
detector = PageHinkley(config=config)

detected_drifts = []
false_alarms = 0
detection_delays = []
y_preds = []
errors = []

# Metric to compute accuracy
metric = PrequentialError(alpha=0.9)  # alpha=1.0 is equivalent to normal accuracy

from sklearn.metrics import mean_squared_error

# isFirst = True

for i in range(len(X_test)):
    X_i = X_test[i].reshape(1, -1)
    y_i = y_test[i].reshape(1, -1)

    # Predict and calculate the error
    y_pred = pipeline.predict(X_i)
    y_preds.append(y_pred[0])
    error = mean_squared_error(y_true=y_i, y_pred=y_pred)
    metric_error = metric(error_value=error)
    # error = metric(error_value=mean_squared_error(y_true=y_i, y_pred=y_pred))
    errors.append(metric_error)

    # Update Page Hinkley with the error (0 for correct, 1 for incorrect)
    # detector.update(value=1 if error > 0 else 0)

    # Step 4: Update the drift detector with the current error
    detector.update(value=error)

    # Step 5: Check for detected drift
    if detector.drift:
        # print(f"Change detected at step {i}")
        detected_drifts.append(i)

        # Determine if it's a false alarm or not
        if i < drift_position:  # Compare with the original drift point
            false_alarms += 1
        else:
            #if isFirst:
                #isFirst = False
            detection_delays.append(i - drift_position)
        detector.reset()

    # Check for detected warning
    #if not warning_flag and detector.warning:
     #   print(f"Warning detected at step {i}")
      #  warning_flag = True

false_alarm_rate = (false_alarms / (len(X_test) - drift_position)) * 100 if detected_drifts else 0
average_detection_delay = np.mean(detection_delays) if detection_delays else None

print(f"Detected drift points: {detected_drifts}")
print(f"False alarms: {false_alarms}")
print(f"False alarm rate: {false_alarm_rate} %")
print(f"Average detection delay: {average_detection_delay}")

# Generate a line chart to compare actual vs predicted values
plt.figure(figsize=(14, 5))
plt.plot(range(len(y_test)), y_test, label="Actual y values", color='blue', linestyle='-')
plt.plot(range(len(y_preds)), y_preds, label="Predicted y values", color='red', linestyle='--')
plt.xlabel("Instance Index")
plt.ylabel("y values")
plt.title("Line Chart: Actual vs Predicted y values")
plt.legend()
plt.grid(True)
plt.show()

# Plotting the error values
plt.figure(figsize=(14, 7))
plt.plot(range(len(errors)), errors, label="MSE Error", color='purple')
plt.xlabel("Instance Index")
plt.ylabel("Mean Squared Error")
plt.title("Behavior of Mean Squared Error Over data points")
plt.legend()
plt.grid(True)
plt.show()

# binary array for initial drift
drift_indicator = np.zeros(len(data) - drift_position)
first_drift_found = False

for drift_point in detected_drifts:
    # Check if drift_point is at or after drift_position and if it's the first drift point after drift_position
    if drift_point < drift_position:
        # Set drift_indicator to 1 starting from the position of the first detected drift after drift_position
        drift_indicator[drift_point] = 1  # Fill from this index onward with 1s
    elif drift_point >= drift_position and not first_drift_found:
        drift_indicator[drift_point:] = 1  # Fill from this index onward with 1s
        first_drift_found = True
    else:
        continue

drift_creator = np.ones(len(data)-drift_position)
for i in range(drift_position,-1,-1):
    drift_creator[i] = 0

import matplotlib.pyplot as plt

# Visualize the drift indicator
plt.figure(figsize=(12, 4))
plt.plot(drift_indicator, drawstyle='steps-post', color='red', label='Detection by Drift Indicator')
plt.plot(drift_creator, drawstyle='steps-post', color='blue', label='Initial Drift')
# plt.axvline(x=drift_position-1000, color='red', linestyle='--', label='Drift Point')

plt.title('Drift Indicator Visualization')
plt.xlabel('Record Index')
plt.ylabel('Drift Indicator Value')
plt.ylim(-0.2, 1.2)
plt.yticks([0, 1], labels=['0 (No Drift)', '1 (Drift)'])
plt.legend()
plt.grid(True)
plt.show()

