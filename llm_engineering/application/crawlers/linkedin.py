import time
from typing import Dict, List

from bs4 import BeautifulSoup
from bs4.element import Tag
from loguru import logger
from selenium.webdriver.common.by import By

from llm_engineering.domain.documents import PostDocument
from llm_engineering.domain.exceptions import ImproperlyConfigured
from llm_engineering.settings import settings

from .base import BaseSeleniumCrawler
class LinkedInCrawler(BaseSeleniumCrawler):
    model = PostDocument

    def __init__(self, scroll_limit: int = 5, is_deprecated: bool = True) -> None:
        super().__init__(scroll_limit)

        self._is_deprecated = is_deprecated

    def set_extra_driver_options(self, options) -> None:
        options.add_experimental_option("detach", True)
    
    def login(self) -> None:
        if self._is_deprecated:
            raise DeprecationWarning(
                "As LinkedIn has updated its security measures, the login() method is no longer supported."
            )
        
        self.driver.get("https://www.linkedin.com/login")
        if not settings.LINKEDIN_USERNAME or not settings.LINKEDIN_PASSWORD:
            raise ImproperlyConfigured(
                "LinkedIn scraper requires the {LINKEDIN_USERNAME} and {LINKEDIN_PASSWORD} settings."
            )
        
        self.driver.find_element(By.ID, "username").send_keys(settings.LINKEDIN_USERNAME)
        self.driver.find_element(By.ID, "password").send_keys(settings.LINKEDIN_PASSWORD)
        self.driver.find_element(By.CSS_SELECTOR, ".login__form_action_container button").click()

    def extract(self, link, **kwargs):
        if self._is_deprecated:
            raise DeprecationWarning(
                "As LinkedIn has updated its security measures, the login() method is no longer supported."
            )
        
        logger.info(f"Starting scrapping data for profile: {link}")

        self.login()

        soup = self._get_page_content(link)