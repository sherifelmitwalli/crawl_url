import streamlit as st
import openai
import asyncio
import json
import os
from typing import List
import nest_asyncio

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel

# Apply nest_asyncio to handle async operations in Streamlit
nest_asyncio.apply()

class Product(BaseModel):
    name: str
    price: str

def run_async_scraper(url: str, instruction: str, num_pages: int, all_pages: bool, levels: int):
    """Run the asynchronous web scraper synchronously in Streamlit."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(_scrape_data(url, instruction, num_pages, all_pages, levels))
    except Exception as e:
        st.error(f"Error during scraping: {str(e)}")
        return []

async def _scrape_data(url: str, instruction: str, num_pages: int, all_pages: bool, levels: int):
    """Perform asynchronous scraping."""
    llm_strategy = LLMExtractionStrategy(
        provider=st.secrets["MODEL"],
        api_token=st.secrets["OPENAI_API_KEY"],
        schema=Product.model_json_schema(),
        extraction_type="schema",
        instruction=instruction,
        chunk_token_threshold=1000,
        overlap_rate=0.0,
        apply_chunking=True,
        input_format="markdown",
        extra_args={"temperature": 0.0, "max_tokens": 800},
    )

    crawl_config = CrawlerRunConfig(
        extraction_strategy=llm_strategy,
        cache_mode=CacheMode.BYPASS,
        process_iframes=False,
        remove_overlay_elements=True,
        exclude_external_links=True,
        levels=levels,
    )

    browser_cfg = BrowserConfig(headless=True, verbose=True)

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        all_data = []
        page = 1
        while True:
            try:
                page_url = f"{url}?page={page}"
                result = await crawler.arun(url=page_url, config=crawl_config)
                if result.success:
                    try:
                        data = json.loads(result.extracted_content)
                        all_data.extend(data)
                    except json.JSONDecodeError:
                        st.error(f"Failed to decode JSON on page {page}")
                        break
                    
                    if not all_pages or page >= num_pages:
                        break
                    page += 1
                else:
                    st.write(f"Error on page {page}: {result.error_message}")
                    break
            except Exception as e:
                st.error(f"Error processing page {page}: {str(e)}")
                break
        return all_data

# Set up the OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Streamlit app
st.title("AI Assisted Crawling")

# User input
url_to_scrape = st.text_input("Enter the URL to scrape:")
instruction_to_llm = st.text_area("Enter the instructions for what to scrape:")
num_pages = st.number_input("Enter the number of pages to scrape:", min_value=1, step=1)
all_pages = st.checkbox("Scrape all pages")
levels = st.number_input("Enter the number of levels to scrape:", min_value=1, step=1)

if st.button("Start Scraping"):
    if url_to_scrape and instruction_to_llm:
        with st.spinner("Scraping in progress..."):
            try:
                data = run_async_scraper(url_to_scrape, instruction_to_llm, num_pages, all_pages, levels)
                if data:
                    formatted_data = "\n\n".join([f"Name: {item['name']}\nPrice: {item['price']}" for item in data])
                    st.download_button("Download Data", formatted_data, "scraped_data.txt")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.write("Please enter the URL, instructions, number of pages, and levels.")
