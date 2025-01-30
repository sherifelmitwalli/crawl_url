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

async def scrape_data(url, instruction, num_pages, all_pages):
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
        click_elements=True
    )

    browser_cfg = BrowserConfig(headless=True, verbose=True)

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        all_data = []
        page = 1
        while True:
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

if st.button("Start Scraping"):
    if url_to_scrape and instruction_to_llm:
        with st.spinner("Scraping in progress..."):
            data = asyncio.run(scrape_data(url_to_scrape, instruction_to_llm, num_pages, all_pages))
            formatted_data = "\n\n".join([f"Name: {item['name']}\nPrice: {item['price']}" for item in data])
            st.download_button("Download Data", formatted_data, "scraped_data.txt")
    else:
        st.write("Please enter the URL, instructions, and number of pages.")

