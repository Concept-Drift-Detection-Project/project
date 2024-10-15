# data.py

config = {
    "Linear Regressor": {
        "DDM": {
            "warning_level": 1.65,
            "drift_level": 1.7,
            "min_num_instances": 330
        },
        "EDDM": {
            "alpha": 0.90,
            "beta": 0.85,
            "level": 1.95,
            "min_num_misclassified_instances": 50
        },
        "ADWIN": {
            "clock": 91,
            "delta": 0.002,
            "min_window_size": 56,
            "min_num_instances": 10
        },
        "Page Hinkley": {
            "delta": 0.005,
            "lambda_": 32.0,
            "alpha": 0.9099,
            "min_num_instances": 88
        }
    },
    "Decision Tree Regressor": {
        "DDM": {
            "warning_level": 2.65,
            "drift_level": 2.7,
            "min_num_instances": 250
        },
        "EDDM": {
            "alpha": 0.90,
            "beta": 0.85,
            "level": 1.55,
            "min_num_misclassified_instances": 170
        },
        "ADWIN": {
            "clock": 19,
            "delta": 0.002,
            "min_window_size": 90,
            "min_num_instances": 96
        },
        "Page Hinkley": {
            "delta": 0.005,
            "lambda_": 61.0,
            "alpha": 0.7649,
            "min_num_instances": 75
        }
    },
    "RandomForest Regressor": {
        "DDM": {
            "warning_level": 5.05,
            "drift_level": 5.1,
            "min_num_instances": 30
        },
        "EDDM": {
            "alpha": 0.80,
            "beta": 0.75,
            "level": 1.85,
            "min_num_misclassified_instances": 110
        },
        "ADWIN": {
            "clock": 1,
            "delta": 0.002,
            "min_window_size": 52,
            "min_num_instances": 10
        },
        "Page Hinkley": {
            "delta": 0.005,
            "lambda_": 12.0,
            "alpha": 0.9899,
            "min_num_instances": 74
        }
    },
    "SVM Regressor": {
        "DDM": {
            "warning_level": 2.45,
            "drift_level": 2.5,
            "min_num_instances": 90
        },
        "EDDM": {
            "alpha": 1.0,
            "beta": 0.95,
            "level": 1.0,
            "min_num_misclassified_instances": 110
        },
        "ADWIN": {
            "clock": 91,
            "delta": 0.002,
            "min_window_size": 72,
            "min_num_instances": 10
        },
        "Page Hinkley": {
            "delta": 0.005,
            "lambda_": 19.0,
            "alpha": 0.9249,
            "min_num_instances": 81
        }
    }
}
