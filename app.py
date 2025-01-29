import streamlit as st
import openai
import asyncio
import json
import os
from typing import List

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel, Field

class Product(BaseModel):
    name: str
    price: str

async def scrape_data(url, instruction):
    llm_strategy = LLMExtractionStrategy(
        provider="openai/gpt-4",
        api_token=os.getenv("OPENAI_API_KEY"),
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
    )

    browser_cfg = BrowserConfig(headless=True, verbose=True)

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(url=url, config=crawl_config)
        return result

# Set up the OpenAI API key
openai.api_key = st.secrets["openai_api_key"]

# Streamlit app
st.title("AI assisted Crawling")

# User input
url_to_scrape = st.text_input("Enter the URL to scrape:")
instruction_to_llm = st.text_area("Enter the instructions for what to scrape:")

if st.button("Start Scraping"):
    if url_to_scrape and instruction_to_llm:
        with st.spinner("Scraping in progress..."):
            result = asyncio.run(scrape_data(url_to_scrape, instruction_to_llm))
            if result.success:
                data = json.loads(result.extracted_content)
                st.write("Extracted items:", data)
                st.download_button("Download Data", json.dumps(data), "scraped_data.json")
            else:
                st.write("Error:", result.error_message)
    else:
        st.write("Please enter both the URL and the instructions.")
