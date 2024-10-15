import unittest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
import time

class TestStreamlitComparisonUI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up the Chrome WebDriver
        cls.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
        cls.driver.get("http://localhost:8501")  # Assuming the Streamlit app is running on this URL
        cls.driver.maximize_window()

    @classmethod
    def tearDownClass(cls):
        # Close the browser after the test
        cls.driver.quit()

    def test_select_model_and_run_drift_detection(self):
        wait = WebDriverWait(self.driver, 20)

        # Wait for the "Select the regression model" dropdown and select a model
        model_dropdown = wait.until(EC.presence_of_element_located((By.XPATH, "//select")))
        model_dropdown.click()
        model_option = wait.until(EC.presence_of_element_located((By.XPATH, "//option[contains(., 'Linear Regressor')]")))
        model_option.click()

        # Wait for and input drift points
        first_drift_input = wait.until(EC.presence_of_element_located((By.XPATH, "//input[@type='number' and @value='7000']")))
        first_drift_input.clear()
        first_drift_input.send_keys("5000")  # Enter first drift point

        second_drift_input = wait.until(EC.presence_of_element_located((By.XPATH, "//input[@type='number' and @value='11000']")))
        second_drift_input.clear()
        second_drift_input.send_keys("12000")  # Enter second drift point

        # Click the "Run Drift Detection" button
        run_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Run Drift Detection')]")))
        run_button.click()

        # Wait for the drift detection results table to appear
        results_table = wait.until(EC.presence_of_element_located((By.XPATH, "//div[contains(text(), 'Drift Detection Results Table')]/following::table[1]")))
        self.assertIsNotNone(results_table, "Results table not found after running drift detection")

        # Check for False Alarm Rate chart
        false_alarm_chart = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "vega-embed")))
        self.assertTrue(false_alarm_chart.is_displayed(), "False Alarm Rate chart not displayed")

        # Check for Detection Delay chart
        detection_delay_chart = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "vega-embed")))
        self.assertTrue(detection_delay_chart.is_displayed(), "Detection Delay chart not displayed")

    def test_drift_detection_indicator_chart(self):
        wait = WebDriverWait(self.driver, 20)

        # Click the "Run Drift Detection" button again if needed to refresh the state
        run_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Run Drift Detection')]")))
        run_button.click()

        # Wait for the Drift Detection Indicator chart to appear
        drift_chart = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "vega-embed")))
        self.assertTrue(drift_chart.is_displayed(), "Drift Detection Indicator chart not displayed")

if __name__ == "__main__":
    unittest.main()
