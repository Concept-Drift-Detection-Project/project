import sys
import os 
from unittest.mock import patch, MagicMock

# Add the project root directory to Python's path, so the test can find home.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import simulation 

# Mocking Streamlit components and the functions used in run_simulation
@patch('simulation.st.altair_chart')
@patch('simulation.st.slider')
@patch('simulation.st.selectbox')
@patch('simulation.st.button')
@patch('simulation.st.columns')
@patch('simulation.st.write')
@patch('simulation.st.title')
@patch('simulation.st.subheader')
def test_run_simulation(mock_subheader, mock_title, mock_write, mock_columns, mock_button, 
                        mock_selectbox, mock_slider, mock_altair_chart):
    
    # Mocking the return values for the UI elements
    mock_col1 = MagicMock()
    mock_col2 = MagicMock()
    mock_columns.return_value = (mock_col1, mock_col2)
    mock_title.return_value = None
    mock_write.return_value = None
    mock_subheader.return_value = None
    
    # Mocking the values for dropdowns and sliders
    mock_selectbox.side_effect = ["DDM", "Linear Regressor"]  # Simulate drift method and model choice
    mock_slider.side_effect = [0.01, 3.0, 1000]  # Warning level, drift level, min_num_instances
    
    # Simulate the button press
    mock_button.return_value = True  # Simulate "Check for Drift" button press
    
    # Run the function
    simulation.run_simulation()

    # Assert that Streamlit components were called correctly
    mock_title.assert_called_once_with('Concept Drift Detection in a Synthetic Dataset')
    
    mock_subheader.assert_any_call("Parameter Values")
    mock_selectbox.assert_any_call("Select Drift Detection Method:", ("DDM", "EDDM", "ADWIN", "Page Hinkley"))
    mock_selectbox.assert_any_call("Select the regression model:", ("Linear Regressor", "SVM Regressor", "Decision Tree Regressor", "Random Forest Regressor"))
    mock_slider.assert_any_call("Warning Level", 0.0, 0.1, 0.01)
    mock_slider.assert_any_call("Drift Level", 1.0, 10.0, 3.0)
    mock_slider.assert_any_call("Min Number of Instances", 1, 5000, 1000)
    
    # Ensure that a button was clicked
    mock_button.assert_called_once_with("Check for Drift")

    # Verify Altair charts were rendered
    assert mock_altair_chart.call_count == 2  # Two charts (MSE and Drift Detection Indicator)

    # Checking that st.write is called for the false alarms and other parameters
    mock_write.assert_any_call(f"False Alarms: {0}")
    mock_write.assert_any_call(f"False Alarm Rate: {0.0}")
    #mock_write.assert_any_call(f"Average Detection Delay: {0}")
     
    # Check if Average Detection Delay is handled
    assert any("Average Detection Delay:" in call[0][0] for call in mock_write.call_args_list)
 

