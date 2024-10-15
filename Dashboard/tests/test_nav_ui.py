import unittest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

class TestStreamlitNavigationUI(unittest.TestCase):

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

    def test_navigate_to_home(self):
        # Wait for the sidebar and click on "Home"
        wait = WebDriverWait(self.driver, 20)
        home_menu = wait.until(EC.presence_of_element_located((By.XPATH, "//span[contains(text(), 'Home')]")))
        home_menu.click()

        # Check that "Best Configurations for Drift Detection Methods" is displayed on the "Home" page
        page_title = wait.until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
        self.assertEqual(page_title.text, "Best Configurations for Drift Detection Methods")

    def test_navigate_to_simulation(self):
        # Click on the "Simulation" menu item
        wait = WebDriverWait(self.driver, 20)
        simulation_menu = wait.until(EC.presence_of_element_located((By.XPATH, "//span[contains(text(), 'Simulation')]")))
        simulation_menu.click()

        # Check that the "Simulation" page is displayed
        page_title = wait.until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
        self.assertEqual(page_title.text, "Concept Drift Detection in a Synthetic Dataset")

    def test_navigate_to_default_run(self):
        # Click on the "Default Run" menu item
        wait = WebDriverWait(self.driver, 20)
        default_run_menu = wait.until(EC.presence_of_element_located((By.XPATH, "//span[contains(text(), 'Default Run')]")))
        default_run_menu.click()

        # Check that the "Default Run" page is displayed
        page_title = wait.until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
        self.assertEqual(page_title.text, "Concept Drift Detection in a Synthetic Dataset")

    def test_navigate_to_comparison(self):
        # Click on the "Comparison" menu item
        wait = WebDriverWait(self.driver, 20)
        comparison_menu = wait.until(EC.presence_of_element_located((By.XPATH, "//span[contains(text(), 'Comparison')]")))
        comparison_menu.click()

        # Check that the "Comparison" page is displayed
        page_title = wait.until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
        self.assertEqual(page_title.text, "Concept Drift Detection in a Synthetic Dataset")

    def test_navigate_to_choice(self):
        # Click on the "Choice" menu item
        wait = WebDriverWait(self.driver, 20)
        choice_menu = wait.until(EC.presence_of_element_located((By.XPATH, "//span[contains(text(), 'Choice')]")))
        choice_menu.click()

        # Check that the "Choice" page is displayed (update based on actual content)
        page_title = wait.until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
        self.assertEqual(page_title.text, "Concept Drift Detection in a Synthetic Dataset")

    def test_navigate_to_upload(self):
        # Click on the "Upload" menu item
        wait = WebDriverWait(self.driver, 20)
        upload_menu = wait.until(EC.presence_of_element_located((By.XPATH, "//span[contains(text(), 'Upload')]")))
        upload_menu.click()

        # Check that the "Upload" page is displayed (update based on actual content)
        page_title = wait.until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
        self.assertEqual(page_title.text, "Concept Drift Detection on User-Uploaded Dataset")

if __name__ == "__main__":
    unittest.main()
