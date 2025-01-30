import streamlit as st
import openai
import asyncio
import json
import os
from typing import List
import nest_asyncio

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel, Field

# Apply nest_asyncio to handle async operations in Streamlit
nest_asyncio.apply()

class Product(BaseModel):
    name: str
    price: str

def run_async_scraper(url, instruction, num_pages, all_pages, levels):
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_scrape_data(url, instruction, num_pages, all_pages, levels))
    except Exception as e:
        st.error(f"Error during scraping: {str(e)}")
        return []

async def _scrape_data(url, instruction, num_pages, all_pages, levels):
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
        click_elements=True,
        levels=levels
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
                    data = json.loads(result.extracted_content)
                    all_data.extend(data)
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

