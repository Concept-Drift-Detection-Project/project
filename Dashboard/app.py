import streamlit as st

# Define the navigation menu using st.radio
selected = st.sidebar.radio(
    "Concept Drift Detection",  # Sidebar title
    ["Home", "Simulation", "Default Run", "Comparison", "Choice", "Upload"]  # Menu options
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
    import choice2
    choice2.choose() 
elif selected == "Upload":
    import real_dataset
    real_dataset.detect()
