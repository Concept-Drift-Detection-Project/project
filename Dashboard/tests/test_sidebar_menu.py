from unittest.mock import patch
from streamlit_option_menu import option_menu

def test_sidebar_menu():
    with patch('streamlit_option_menu.option_menu') as mock_option_menu:
        mock_option_menu.return_value = "Home"
        selected = option_menu(
            "Concept Drift Detection", 
            ["Home", "Simulation", "Default Run", "Comparison", "Choice", "Upload"],
            icons=["house", "activity", "list-task", "table", "", "cloud-upload"],
            menu_icon="cast",
            default_index=0
        )
        assert selected == "Home"
