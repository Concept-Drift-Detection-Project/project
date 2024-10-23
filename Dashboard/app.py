import streamlit as st

# Custom CSS to increase the font size of the sidebar title
st.markdown(
    """
    <style>
    .sidebar .sidebar-content h1 {
        font-size: 48px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar title with increased font size
st.sidebar.markdown("<h1>DRIFTMARK  AI</h1>", unsafe_allow_html=True)

# Sidebar navigation menu using st.radio
selected = st.sidebar.radio(
    "",  # No title for radio since we manually created a title above
    ["Home", "Simulation", "Comparison", "Choice", "Upload"]  # Menu options
)

# Routing based on selected page
if selected == "Home":
    import home
    home.show()
elif selected == "Simulation":
    import simulation
    simulation.run_simulation()
elif selected == "Comparison":
    import comparison
    comparison.compare()
elif selected == "Choice":
    import choice2
    choice2.choose() 
elif selected == "Upload":
    import real_dataset
    real_dataset.detect()
