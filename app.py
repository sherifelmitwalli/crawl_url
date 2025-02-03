import streamlit as st
import asyncio
import json
import re
from typing import List, Dict
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel
import time
import random

# Set page config at the very beginning of the script
st.set_page_config(page_title="AI Web Scraper", page_icon="🕷️", layout="wide")

class ExtractedText(BaseModel):
    text: str

async def scrape_data(url: str, instruction: str, num_pages: int, all_pages: bool) -> List[Dict[str, str]]:
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
        # Remove the 'timeout' parameter if it's not supported
    )

    browser_cfg = BrowserConfig(
        headless=True, 
        verbose=False,
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
                    else:
                        st.warning(f"Error on page {page}: {result.error_message}. Retrying...")
                
                except Exception as e:
                    st.error(f"Error processing page {page}: {str(e)}. Retrying...")
                
                if attempt < 2:  # Wait before retrying, except on last attempt
                    time.sleep(random.uniform(1, 3))  # Random delay between retries
            
            if not all_pages and page >= num_pages:
                break

        if not all_data:
            st.warning(f"No data extracted from {url}. Possible reasons:")
            st.info("- Page might be dynamically loaded")
            st.info("- Content might require specific browser interactions")
            st.info("- Potential anti-scraping measures in place")

        return all_data

st.title("AI-Assisted Web Scraping")

url_to_scrape = st.text_input("Enter the URL to scrape:")
instruction_to_llm = st.text_area("Enter instructions for what to scrape:")
num_pages = st.number_input("Enter the number of pages to scrape:", min_value=1, step=1)
all_pages = st.checkbox("Scrape all pages")

if st.button("Start Scraping"):
    if url_to_scrape and instruction_to_llm:
        with st.spinner("Scraping in progress..."):
            try:
                data = asyncio.run(scrape_data(url_to_scrape, instruction_to_llm, num_pages, all_pages))
                
                if data:
                    formatted_data = "\n".join([item['text'] for item in data])
                    st.download_button("Download Data", formatted_data, "scraped_data.txt")
                    st.success(f"Successfully scraped {len(data)} items.")
                else:
                    st.warning("No data was scraped. Please check the URL and try again.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                st.error("Please check your API keys and ensure all dependencies are correctly installed.")
    else:
        st.warning("Please enter both the URL and instructions before starting.")

