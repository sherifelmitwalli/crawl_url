import streamlit as st
import asyncio
import json
import re
import subprocess
import sys
import time
import random
from typing import List, Dict

# Comprehensive Playwright and browser installation
def setup_playwright():
    try:
        # Ensure pip is up to date
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install Playwright and its dependencies
        subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright"])
        
        # Use subprocess to run playwright install with verbose output
        result = subprocess.run([sys.executable, "-m", "playwright", "install"], 
                                capture_output=True, 
                                text=True)
        
        # Install system dependencies
        subprocess.check_call([sys.executable, "-m", "playwright", "install-deps"])
        
        st.success("Playwright and browsers installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Installation error: {e}")
        st.error(f"Stdout: {e.stdout}")
        st.error(f"Stderr: {e.stderr}")
        return False
    except Exception as e:
        st.error(f"Unexpected error during setup: {str(e)}")
        return False

# Attempt to set up Playwright before importing
if not setup_playwright():
    st.error("Failed to set up Playwright. Cannot proceed with web scraping.")
    st.stop()

# Now import required libraries
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel

# Set page config
st.set_page_config(page_title="AI Web Scraper", page_icon="🕷️", layout="wide")

class ExtractedText(BaseModel):
    text: str

async def scrape_data(url: str, instruction: str, num_pages: int, all_pages: bool) -> List[Dict[str, str]]:
    try:
        enhanced_instruction = (
            f"{instruction}\n\nEnsure the extracted text is relevant and excludes cookies, legal disclaimers,"
            " advertisements, and UI elements such as navigation bars and footers. Extract meaningful page content only."
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

        browser_cfg = BrowserConfig(
            headless=True, 
            verbose=True,  # Increased verbosity for debugging
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )

        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            all_data = []
            exclusion_patterns = re.compile(r'cookie|privacy policy|terms of service|advertisement', flags=re.IGNORECASE)

            for page in range(1, num_pages + 1):
                page_url = f"{url}?page={page}" if '?' in url else f"{url}/page/{page}"
                
                for attempt in range(3):  # Retry mechanism
                    try:
                        result = await crawler.arun(url=page_url, config=crawl_config)
                        
                        if result.success:
                            try:
                                data = json.loads(result.extracted_content)
                                filtered_data = [item for item in data if not exclusion_patterns.search(item["text"])]
                                all_data.extend(filtered_data)
                                break  # Successful, exit retry loop
                            except json.JSONDecodeError:
                                st.warning(f"Failed to decode JSON on page {page}. Retrying...")
                                time.sleep(2)
                        else:
                            st.warning(f"Error on page {page}: {result.error_message}. Retrying...")
                    
                    except Exception as e:
                        st.error(f"Error processing page {page}: {str(e)}. Retrying...")
                        time.sleep(2)
                
                if not all_pages and page >= num_pages:
                    break

            if not all_data:
                st.warning(f"No data extracted from {url}. Possible reasons:")
                st.info("- Page might be dynamically loaded")
                st.info("- Content might require specific browser interactions")
                st.info("- Potential anti-scraping measures in place")

            return all_data
    except Exception as e:
        st.error(f"Unexpected error in scrape_data: {str(e)}")
        return []

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
                    # Use asyncio.run with a timeout
                    data = asyncio.run(asyncio.wait_for(
                        scrape_data(url_to_scrape, instruction_to_llm, num_pages, all_pages), 
                        timeout=120  # 2-minute timeout
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

