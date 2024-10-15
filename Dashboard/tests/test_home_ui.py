import unittest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


class TestStreamlitHomePageUI(unittest.TestCase):

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

    def test_title_and_subheaders(self):
        # Wait for the title to appear and check it
        wait = WebDriverWait(self.driver, 10)
        title_element = wait.until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
        self.assertEqual(title_element.text, "Best Configurations for Drift Detection Methods")

        # Check that the subheaders for each regressor appear
        subheaders = self.driver.find_elements(By.TAG_NAME, "h2")
        expected_subheaders = ["Linear Regressor", "Decision Tree Regressor", "Random Forest Regressor", "Support Vector Regressor"]

        for i, subheader in enumerate(subheaders):
            self.assertEqual(subheader.text, expected_subheaders[i])

    def test_table_for_linear_regressor(self):
        # Wait for the Linear Regressor table to appear
        wait = WebDriverWait(self.driver, 10)
        linear_table = wait.until(EC.presence_of_element_located((By.XPATH, "//div[contains(text(), 'Linear Regressor')]/following::table[1]")))

        # Check if the table contains the expected values for the Linear Regressor
        table_text = linear_table.text
        expected_table_values = [
            "DDM", "EDDM", "ADWIN", "Page Hinkley",
            "Warning Level = 1.65", "Alpha = 0.90", "Clock = 91", "Lambda = 32.0",
            "Drift Level = 1.7", "Beta = 0.85", "Min Window Size = 56", "Alpha = 0.9099",
            "Min Instances = 330", "Level = 1.95", "Min Num Instances = 10", "Min Num Instances = 88",
            "", "Min Instances = 50", "Memory = 7", "delta = 0.005",
            "", "", "delta = 0.002", ""
        ]

        for value in expected_table_values:
            self.assertIn(value, table_text)

    def test_table_for_decision_tree_regressor(self):
        wait = WebDriverWait(self.driver, 20)  # Increase wait time to 20 seconds

        try:
            # Attempt to locate the Decision Tree Regressor table
            dtr_table = wait.until(EC.presence_of_element_located((By.XPATH, "//h2[text()='Decision Tree Regressor']/following-sibling::table[1]")))

            # Check if the table contains the expected values
            table_text = dtr_table.text
            expected_table_values = [
                "DDM", "EDDM", "ADWIN", "Page Hinkley",
                "Warning Level = 2.65", "Alpha = 0.90", "Clock = 19", "Lambda = 61.0",
                "Drift Level = 2.7", "Beta = 0.85", "Min Window Size = 90", "Alpha = 0.7649",
                "Min Instances = 250", "Level = 1.55", "Min Num Instances = 96", "Min Num Instances = 75",
                "", "Min Instances = 170", "Memory = 15", "delta = 0.005",
                "", "", "delta = 0.002", ""
            ]

            for value in expected_table_values:
                self.assertIn(value, table_text)

        except TimeoutException:
            # Capture a screenshot for debugging if the element is not found
            self.driver.save_screenshot('screenshot_decision_tree_regressor.png')
            raise


    def test_table_for_random_forest_regressor(self):
        # Wait for the Random Forest Regressor table to appear
        wait = WebDriverWait(self.driver, 10)
        rfr_table = wait.until(EC.presence_of_element_located((By.XPATH, "//div[contains(text(), 'Random Forest Regressor')]/following::table[1]")))

        # Check if the table contains the expected values for the Random Forest Regressor
        table_text = rfr_table.text
        expected_table_values = [
            "DDM", "EDDM", "ADWIN", "Page Hinkley",
            "Warning Level = 5.05", "Alpha = 0.80", "Clock = 1", "Lambda = 12.0",
            "Drift Level = 5.1", "Beta = 0.75", "Min Window Size = 52", "Alpha = 0.9899",
            "Min Instances = 30", "Level = 1.85", "Min Instances = 10", "Min Num Instances = 74",
            "", "Min Instances = 110", "Memory = 5", "delta = 0.005",
            "", "", "delta = 0.002", ""
        ]

        for value in expected_table_values:
            self.assertIn(value, table_text)

    def test_table_for_svr(self):
        # Wait for the Support Vector Regressor table to appear
        wait = WebDriverWait(self.driver, 10)
        svr_table = wait.until(EC.presence_of_element_located((By.XPATH, "//div[contains(text(), 'Support Vector Regressor')]/following::table[1]")))

        # Check if the table contains the expected values for the Support Vector Regressor
        table_text = svr_table.text
        expected_table_values = [
            "DDM", "EDDM", "ADWIN", "Page Hinkley",
            "Warning Level = 2.45", "Alpha = 1.0", "Clock = 91", "Lambda = 19.0",
            "Drift Level = 2.5", "Beta = 0.95", "Min Window Size = 72", "Alpha = 0.9249",
            "Min Instances = 90", "Level = 1.0", "Min Num Instances = 10", "Min Num Instances = 81",
            "", "Min Instances = 110", "Memory = 7", "delta = 0.005",
            "", "", "delta = 0.002", ""
        ]

        for value in expected_table_values:
            self.assertIn(value, table_text)

if __name__ == "__main__":
    unittest.main() 