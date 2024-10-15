import sys
import os 
from unittest.mock import patch

# Add the project root directory to Python's path, so the test can find home.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import home

@patch('home.st.table')  # Mock st.table
@patch('home.st.subheader')  # Mock st.subheader
@patch('home.st.write')  # Mock st.write
@patch('home.st.title')  # Mock st.title
def test_home_show(mock_title, mock_write, mock_subheader, mock_table):
    # Call the home.show function to trigger Streamlit calls
    home.show()

    # Check that st.title was called once with the correct argument
    mock_title.assert_called_once_with("Best Configurations for Drift Detection Methods")
    
    # Check that st.write was called with the page description
    mock_write.assert_any_call("This page displays the best configuration values for the drift detection methods.")
    
    # Check that st.subheader was called for each regressor
    mock_subheader.assert_any_call("Linear Regressor")
    mock_subheader.assert_any_call("Decision Tree Regressor")
    mock_subheader.assert_any_call("Random Forest Regressor")
    mock_subheader.assert_any_call("Support Vector Regressor")
    
    # Check that st.table was called 4 times for the 4 different regressors
    assert mock_table.call_count == 4
