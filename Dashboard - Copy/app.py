import streamlit as st
from streamlit_option_menu import option_menu

# Define the navigation menu
with st.sidebar:
    selected = option_menu(
        "Concept Drift Detection",  # Sidebar title
        ["Home", "Simulation", "Default Run", "Comparison", "Choice", "Upload"],  # Menu options
        icons=["house", "activity", "list-task", "table", "", "cloud-upload"],  # Icons for menu options
        menu_icon="cast",  # Icon for the menu
        default_index=0,  # Default selected option
    )

# Routing based on selected page
if selected == "Home":
    import home
    home.show()
elif selected == "Simulation":
    import simulation
    simulation.run_simulation()
elif selected == "Default Run":
    import default
    default.compare()
elif selected == "Comparison":
    import comparison
    comparison.compare()
elif selected == "Choice":
    import choice
    choice.choose()
elif selected == "Upload":
    import real_dataset
    real_dataset.detect()

