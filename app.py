# Place page config at the absolute top of the file
import streamlit as st
st.set_page_config(page_title="AI Web Scraper", page_icon="🕷️", layout="wide")

import asyncio
import json
import re
import subprocess
import sys
import time
import random
from typing import List, Dict

# ----------------------- Helper Functions -----------------------

def safe_run_command(command):
    """
    Safely run a shell command with error handling.
    """
    try:
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        st.error(f"Command failed: {' '.join(command)}")
        st.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return False

def setup_playwright():
    """
    Comprehensive Playwright setup with multiple fallback methods.
    """
    st.info("Setting up Playwright...")
    
    install_methods = [
        [sys.executable, "-m", "pip", "install", "playwright"],
        [sys.executable, "-m", "pip", "install", "--upgrade", "playwright"],
        [sys.executable, "-m", "pip", "install", "--force-reinstall", "playwright"]
    ]

    for method in install_methods:
        if safe_run_command(method):
            browser_install_commands = [
                [sys.executable, "-m", "playwright", "install"],
                ["playwright", "install"],
                [sys.executable, "-c", "from playwright.sync_api import sync_playwright; sync_playwright().install()"]
            ]
            for browser_cmd in browser_install_commands:
                try:
                    subprocess.run(browser_cmd, check=True)
                    st.success("Playwright installed successfully!")
                    return True
                except Exception as e:
                    st.warning(f"Browser installation method failed: {str(e)}")
    
    st.error("Could not install Playwright. Please check your environment.")
    return False

# Attempt to set up Playwright
if not setup_playwright():
    st.error("Playwright setup failed. Cannot proceed with web scraping.")
    st.stop()

# ----------------------- LLM Error Check -----------------------
import openai

Improved Error Page Detection Function

def is_error_page(content: str) -> bool:
    """
    Use the configured LLM to determine if webpage content is an error/rejection message.
    Returns True if the content appears to be an error page, False otherwise.
    """
    # Common error page keywords for initial filtering
    error_keywords = [
        "page not available",
        "return to homepage",
        "web archive",
        "having difficulty",
        "access denied",
        "blocked",
        "robot",
        "automated access",
        "contact support",
        "try again later",
        "temporary error",
        "cannot process",
        "detection system",
        "security check"
    ]
    
    # Quick keyword check
    content_lower = content.lower()
    keyword_matches = sum(1 for keyword in error_keywords if keyword in content_lower)
    
    # If multiple error keywords are found, consider it an error page
    if keyword_matches >= 3:
        return True

    try:
        # Use the same LLM configuration as crawl4ai
        from crawl4ai.extraction_strategy import LLMExtractionStrategy
        from pydantic import BaseModel

        class ErrorDetection(BaseModel):
            is_error: bool
            confidence: float

        prompt = """Analyze the following webpage content and determine if it's an error/rejection page rather than actual content.

        Consider it an error/rejection page if it shows these patterns:
        1. Mentions returning to homepage or main sections
        2. References web archives or previous versions
        3. Provides contact information for support/help
        4. States the page is unavailable/inaccessible
        5. Contains generic error messages
        6. Suggests alternative navigation paths
        7. Mentions access being denied or blocked
        8. References robot/crawler detection
        9. Contains anti-automation messaging
        10. Lacks specific content related to the webpage's purpose

        Consider it legitimate content if it contains:
        1. Specific article text, news, or topic-related information
        2. Actual publication content
        3. Meaningful data relevant to the webpage's purpose
        4. Detailed information about a specific topic

        Webpage content for analysis:
        {content}
        """

        error_detector = LLMExtractionStrategy(
            provider=st.secrets["MODEL"],
            api_token=st.secrets["OPENAI_API_KEY"],
            schema=ErrorDetection.model_json_schema(),
            extraction_type="schema",
            instruction=prompt,
            extra_args={"temperature": 0.0}
        )
        
        result = error_detector.extract(content)
        
        # Log the detection result for debugging
        st.info(f"Error detection - Content length: {len(content)} chars")
        st.info(f"Keyword matches: {keyword_matches}")
        st.info(f"LLM detection result: {result}")
        
        return result.get('is_error', False)
    except Exception as e:
        st.error(f"Error in LLM error check: {str(e)}")
        # If LLM fails, fall back to keyword-based detection
        return keyword_matches >= 3



# ----------------------- Proxy Collection -----------------------
# We'll use ProxyBroker to collect proxies automatically.
try:
    from proxybroker import Broker
except ImportError:
    st.error("ProxyBroker is not installed. Please install it with 'pip install proxybroker'.")
    st.stop()

async def fetch_proxies(limit: int = 10, timeout: int = 10) -> List[str]:
    """
    Use ProxyBroker to automatically find free HTTP/HTTPS proxies.
    Returns a list of proxies in the format 'host:port'.
    """
    proxies = set()

    async def save(proxy):
        if proxy and proxy.host and proxy.port:
            proxies.add(f"{proxy.host}:{proxy.port}")

    broker = Broker(save, timeout=timeout)
    await broker.find(types=['HTTP', 'HTTPS'], limit=limit)
    return list(proxies)

# ----------------------- Crawl4ai and Extraction -----------------------
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel

class ExtractedText(BaseModel):
    text: str

async def scrape_data(url: str, instruction: str, num_pages: int, all_pages: bool) -> List[Dict[str, str]]:
    try:
        # Enhance the instruction for better extraction
        enhanced_instruction = (
            f"{instruction}\n\nEnsure the extracted text is relevant and excludes cookies, legal disclaimers, "
            "advertisements, and UI elements such as navigation bars and footers. Extract meaningful page content only."
        )

        llm_strategy = LLMExtractionStrategy(
            provider=st.secrets["MODEL"],
            api_token=st.secrets["OPENAI_API_KEY"],
            schema=ExtractedText.model_json_schema(),
            extraction_type="schema",
            instruction=enhanced_instruction,
            chunk_token_threshold=1000,
            overlap_rate=0.0,
            apply_chunking=True,
            input_format="markdown",
            extra_args={"temperature": 0.0, "max_tokens": 800},
        )

        crawl_config = CrawlerRunConfig(
            extraction_strategy=llm_strategy,
            cache_mode=CacheMode.BYPASS,
            process_iframes=True,
            remove_overlay_elements=True,
            exclude_external_links=True,
        )

        # Updated browser configuration with a modern user agent and additional args to reduce detection.
        default_browser_cfg = BrowserConfig(
            headless=True, 
            verbose=True,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            args=["--disable-blink-features=AutomationControlled"]
        )

        # Automatically fetch a list of proxies to use as fallback
        st.info("Collecting free proxies...")
        proxies = await fetch_proxies(limit=10)
        if proxies:
            st.info(f"Found {len(proxies)} proxies.")
        else:
            st.warning("No proxies found. Will proceed without proxy fallback.")

        all_data = []
        exclusion_patterns = re.compile(r'cookie|privacy policy|terms of service|advertisement', flags=re.IGNORECASE)

        for page in range(1, num_pages + 1):
            # Construct page URL
            page_url = f"{url}?page={page}" if '?' in url else f"{url}/page/{page}"
            for attempt in range(3):  # Retry mechanism; first attempt without proxy, then with proxy if needed.
                try:
                    # Use default config on first attempt, then a proxy config on subsequent attempts.
                    if attempt == 0:
                        current_browser_cfg = default_browser_cfg
                    else:
                        if proxies:
                            selected_proxy = random.choice(proxies)
                            st.info(f"Attempting page {page} with proxy: {selected_proxy}")
                            current_browser_cfg = BrowserConfig(
                                headless=True, 
                                verbose=True,
                                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
                                args=["--disable-blink-features=AutomationControlled"],
                                proxy=selected_proxy  # Assumes BrowserConfig accepts a "proxy" parameter.
                            )
                        else:
                            st.warning("No proxies available; retrying without proxy.")
                            current_browser_cfg = default_browser_cfg

                    async with AsyncWebCrawler(config=current_browser_cfg) as crawler:
                        result = await crawler.arun(url=page_url, config=crawl_config)
                    
                    # Check if the fetched content is valid and not an error or rejection page.
                    if result.success and result.extracted_content and result.extracted_content.strip():
                        if is_error_page(result.extracted_content):
                            st.warning(f"LLM detected a rejection/error page on page {page} (attempt {attempt + 1}). Retrying...")
                            time.sleep(2)
                            continue

                        try:
                            data = json.loads(result.extracted_content)
                            # Filter out unwanted content
                            filtered_data = [item for item in data if not exclusion_patterns.search(item["text"])]
                            all_data.extend(filtered_data)
                            break  # Successful extraction; exit the retry loop for this page
                        except json.JSONDecodeError:
                            st.warning(f"Failed to decode JSON on page {page}. Retrying...")
                            time.sleep(2)
                    else:
                        st.warning(f"No valid content returned on page {page} (attempt {attempt + 1}). Retrying...")
                        time.sleep(2)
                
                except Exception as e:
                    st.error(f"Error processing page {page}: {str(e)}. Retrying...")
                    time.sleep(2)
            
            if not all_pages and page >= num_pages:
                break

        if not all_data:
            st.warning(f"No data extracted from {url}. Possible reasons:")
            st.info("- The page might be dynamically loaded")
            st.info("- The content might require specific browser interactions")
            st.info("- Anti-scraping measures might be in place (rejection/error page detected)")

        return all_data
    except Exception as e:
        st.error(f"Unexpected error in scrape_data: {str(e)}")
        return []

# ----------------------- Main Streamlit App -----------------------

def main():
    st.title("AI-Assisted Web Scraping")

    url_to_scrape = st.text_input("Enter the URL to scrape:")
    instruction_to_llm = st.text_area("Enter instructions for what to scrape:")
    num_pages = st.number_input("Enter the number of pages to scrape:", min_value=1, max_value=10, value=1, step=1)
    all_pages = st.checkbox("Scrape all pages")

    if st.button("Start Scraping"):
        if url_to_scrape and instruction_to_llm:
            with st.spinner("Scraping in progress..."):
                try:
                    data = asyncio.run(asyncio.wait_for(
                        scrape_data(url_to_scrape, instruction_to_llm, num_pages, all_pages),
                        timeout=180  # Adjust timeout as needed
                    ))
                    
                    if data:
                        formatted_data = "\n".join([item['text'] for item in data])
                        st.download_button("Download Data", formatted_data, "scraped_data.txt")
                        st.success(f"Successfully scraped {len(data)} items.")
                    else:
                        st.warning("No data was scraped. Please check the URL and try again.")
                except asyncio.TimeoutError:
                    st.error("Scraping timed out. The website might be slow or unresponsive.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
                    st.error("Please check your URL, API keys, and internet connection.")
        else:
            st.warning("Please enter both the URL and instructions before starting.")

if __name__ == "__main__":
    main()

