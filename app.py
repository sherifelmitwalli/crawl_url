import streamlit as st
import openai
import asyncio
import json
import re
from typing import List
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel

# Define the Pydantic model for extracted text
class ExtractedText(BaseModel):
    text: str

# Asynchronous scraping function
async def scrape_data(url: str, instruction: str, num_pages: int, all_pages: bool):
    """Perform asynchronous scraping with enhanced filtering and parallel processing."""
    enhanced_instruction = (
        f"{instruction}\n\nEnsure the extracted text is relevant and excludes cookies, legal disclaimers,"
        " advertisements, and UI elements such as navigation bars and footers. Extract meaningful page content only."
    )

    # Define the LLM extraction strategy
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

    # Define crawler configuration
    crawl_config = CrawlerRunConfig(
        extraction_strategy=llm_strategy,
        cache_mode=CacheMode.BYPASS,
        process_iframes=True,
        remove_overlay_elements=True,
        exclude_external_links=True,
    )

    browser_cfg = BrowserConfig(headless=True, verbose=False)

    # Use AsyncWebCrawler for scraping
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        tasks = []
        for page in range(1, num_pages + 1):
            page_url = f"{url}?page={page}"
            tasks.append(crawler.arun(url=page_url, config=crawl_config))

            # If not scraping all pages and reached the limit, stop adding tasks
            if not all_pages and page >= num_pages:
                break

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and filter irrelevant content
        all_data = []
        exclusion_patterns = re.compile(r'cookie|privacy policy|terms of service|advertisement', flags=re.IGNORECASE)
        
        for result in results:
            if isinstance(result, Exception):
                st.error(f"Error during scraping: {str(result)}")
                continue
            
            if result.success:
                try:
                    data = json.loads(result.extracted_content)
                    filtered_data = [item for item in data if not exclusion_patterns.search(item["text"])]
                    all_data.extend(filtered_data)
                except json.JSONDecodeError:
                    st.error("Failed to decode JSON from extracted content.")
            else:
                st.error(f"Scraping error: {result.error_message}")

        return all_data

# Streamlit app setup
st.title("AI-Assisted Web Scraping")

# User input fields
url_to_scrape = st.text_input("Enter the URL to scrape:")
instruction_to_llm = st.text_area("Enter instructions for what to scrape:")
num_pages = st.number_input("Enter the number of pages to scrape:", min_value=1, step=1)
all_pages = st.checkbox("Scrape all pages")

# Start scraping when button is clicked
if st.button("Start Scraping"):
    if url_to_scrape and instruction_to_llm:
        with st.spinner("Scraping in progress..."):
            try:
                # Run the async scraper using asyncio.run()
                data = asyncio.run(scrape_data(url_to_scrape, instruction_to_llm, num_pages, all_pages))
                
                if data:
                    formatted_data = "\n".join([item['text'] for item in data])
                    st.download_button("Download Data", formatted_data, "scraped_data.txt")
                else:
                    st.write("No data was scraped.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter the URL and instructions before starting.")

