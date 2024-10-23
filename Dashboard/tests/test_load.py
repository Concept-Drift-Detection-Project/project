from locust import HttpUser, TaskSet, task, between

class DashboardUser(HttpUser):
    wait_time = between(1, 3)  # Wait time between tasks

    @task(1)
    def load_home_page(self):
        self.client.get("/")  # Load the home page of the dashboard

    @task(2)
    def upload_dataset(self):
        # This is an example of how you might simulate uploading a dataset
        # Adjust the file path and data format as needed
        with open("regression_dataset_with_abrupt_drift_20.csv", "rb") as f:
            self.client.post("/upload", files={"file": f})  # Adjust endpoint if necessary

    @task(3)
    def run_simulation(self):
        # Simulate running a drift detection simulation
        self.client.post("/run-simulation", json={
            "drift_type": "sudden",
            "transition_window_size": 10,
            "model": "Linear Regression"  # Adjust parameters as needed
        })

    @task(4)
    def access_comparison_page(self):
        self.client.get("/comparison")  # Load the comparison page

    @task(5)
    def access_real_dataset_page(self):
        self.client.get("/real-dataset")  # Load the real dataset page

class UserBehavior(TaskSet):
    tasks = {DashboardUser: 1}  # Define tasks for user behavior

# Configuration to run the test
if __name__ == "__main__":
    import os
    os.system("locust -f locustfile.py --host=https://concept-drift-detection-project-project-dashboardapp-4y1o49.streamlit.app/")
