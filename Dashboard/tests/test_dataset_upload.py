import pandas as pd
import streamlit as st
from io import StringIO
import pytest

# Mock data for testing
mock_csv = """Property_Type,Property_Area,Number_of_Windows,Number_of_Doors,Furnishing,Frequency_of_Powercuts,Power_Backup,Water_Supply,Traffic_Density_Score,Crime_Rate,Dust_and_Noise,Air_Quality_Index,Neighborhood_Review,Habitability_score
Duplex,Urban,4,2,Furnished,0,Yes,Yes,3,2,1,25,4,7.5
Bungalow,Rural,3,1,Unfurnished,1,No,No,5,3,2,30,3,6.0
Apartment,Suburban,5,3,Furnished,0,Yes,Yes,4,1,1,28,5,8.0
"""

@pytest.fixture
def mock_csv_file():
    return StringIO(mock_csv)

def test_csv_upload(mock_csv_file):
    # Create a Streamlit test session
    st.session_state.clear()  # Clear any existing state before test
    st.file_uploader("Upload your dataset (CSV format)", type="csv", accept_multiple_files=False, label_visibility="visible")

    # Simulate file upload
    uploaded_file = mock_csv_file
    df = pd.read_csv(uploaded_file)
    
    # Assert that the DataFrame is not empty and has expected columns
    assert df is not None
    assert not df.empty
    assert set(df.columns) == {
        'Property_Type', 'Property_Area', 'Number_of_Windows', 
        'Number_of_Doors', 'Furnishing', 'Frequency_of_Powercuts', 
        'Power_Backup', 'Water_Supply', 'Traffic_Density_Score', 
        'Crime_Rate', 'Dust_and_Noise', 'Air_Quality_Index', 
        'Neighborhood_Review', 'Habitability_score'
    }
    
    # You can also check specific values if necessary
    assert df.shape[0] == 3  # Should have 3 rows based on the mock data
    assert df['Property_Type'].iloc[0] == 'Duplex'  # Check the first row
