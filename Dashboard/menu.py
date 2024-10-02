import streamlit as st

# Define a simple page routing mechanism
def show_home():
    st.title("Home Page")
    st.write("Welcome to the Concept Drift Detection Home Page!")

def run_simulation():
    st.title("Simulation Page")
    st.write("Run your simulations here.")

def compare():
    st.title("Comparison Page")
    st.write("Compare drift detection methods.")

def compare_real():
    st.title("Real Comparison Page")
    st.write("Compare real-world drift detection scenarios.")

# Sidebar navigation
page = st.sidebar.radio(
    "Concept Drift Detection",  # Sidebar title
    ["Home", "Simulation", "Comparison", "Real Comparison"]
)

# Routing based on selected page
if page == "Home":
    show_home()
elif page == "Simulation":
    run_simulation()
elif page == "Comparison":
    compare()
elif page == "Real Comparison":
    compare_real()
