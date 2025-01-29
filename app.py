"""
Requirements:
streamlit
crawl4ai
openai

Deploy to Streamlit Community Cloud:
1. Create a new app on https://share.streamlit.io/
2. Add the following secrets in the Streamlit dashboard:
   - OPENAI_API_KEY: your_api_key
   - MODEL: gpt-4o-mini
"""

import streamlit as st
import asyncio
import json
from typing import Dict, Any
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel

# Set up OpenAI configuration using Streamlit secrets
if "OPENAI_API_KEY" not in st.secrets or "MODEL" not in st.secrets:
    st.error("Please set OPENAI_API_KEY and MODEL in the Streamlit secrets")
    st.markdown("""
        To set up secrets:
        1. Go to your Streamlit dashboard
        2. Navigate to your app's settings
        3. Add the following under 'Secrets':
            ```toml
            OPENAI_API_KEY = "your_api_key"
            MODEL = "gpt-4o-mini"
            ```
    """)
    st.stop()

model = st.secrets["MODEL"]

st.set_page_config(page_title="Web Crawler", page_icon="🕷️", layout="wide")
st.title("Web Crawler with AI")

# Input fields
url = st.text_input("Enter URL to crawl:", placeholder="https://example.com")
instruction = st.text_area(
    "Enter instructions for crawling:",
    placeholder="Example: Extract all product names and prices from the page"
)

# Create a generic schema for dynamic data extraction
class DynamicData(BaseModel):
    content: Dict[str, Any]

async def run_crawler(url: str, instruction: str):
    llm_strategy = LLMExtractionStrategy(
        provider=f"openai/{model}",
        api_token=st.secrets["OPENAI_API_KEY"],
        extraction_type="text",  # Using text mode for flexible extraction
        instruction=instruction,
        chunk_token_threshold=1000,
        overlap_rate=0.0,
        apply_chunking=True,
        input_format="markdown",
        extra_args={"temperature": 0.0, "max_tokens": 1000},
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

if url and instruction:
    if st.button("Start Crawling"):
        with st.spinner("Crawling the website..."):
            try:
                result = asyncio.run(run_crawler(url, instruction))
                
                if result.success:
                    st.success("Crawling completed successfully!")
                    
                    # Display the extracted data
                    data = json.loads(result.extracted_content)
                    st.json(data)
                    
                    # Create download button
                    st.download_button(
                        label="Download Results",
                        data=result.extracted_content,
                        file_name="crawled_data.json",
                        mime="application/json"
                    )
                else:
                    st.error(f"Error during crawling: {result.error_message}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Add usage instructions
with st.expander("How to use"):
    st.markdown("""
    1. Enter the URL of the website you want to crawl
    2. Provide specific instructions for what data to extract
    3. Click 'Start Crawling' and wait for the results
    4. Download the extracted data as JSON
    
    **Note**: Make sure the website allows crawling and respect their robots.txt
    """)
