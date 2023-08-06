import os
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
import unittest

url = 'http://127.0.0.1:8000'
result = requests.get(url)
assert 200 == result.status_code
if result.status_code == 200:
    print('SUCCESS', result.status_code, result.reason)
else:
    print('FAILED', result.status_code, result.reason)


class TestSelenium(unittest.TestCase):
    """emulation user behavior with support selenium.webdriver"""

    def test(self) -> None:
        driver = webdriver.Chrome()
        driver.maximize_window()
        driver.get(url)
        search_field = driver.find_element(By.ID, "upload")
        search_field.send_keys(os.getcwd() + "/prediction_app/images/dog.jfif")

        button = driver.find_element(By.ID, 'submitBtn')
        button.click()

        driver.close()


if __name__ == '__main__':
    unittest.main()
