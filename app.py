import streamlit as st
import asyncio
import json
from typing import Dict, Any
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel
import os

# Install Playwright browsers
os.system("playwright install")
os.system("playwright install-deps")

# Set page title and icon
st.set_page_config(page_title="Web Crawler", page_icon="🕷️", layout="wide")

st.title("🔍 Web Crawler with AI")

# Check if API keys are set
if "OPENAI_API_KEY" not in st.secrets or "MODEL" not in st.secrets:
    st.error("❌ Missing API credentials! Please set OPENAI_API_KEY and MODEL in the Streamlit secrets.")
    st.markdown("""
        **Steps to configure secrets:**
        1. Go to your Streamlit dashboard.
        2. Navigate to your app's settings.
        3. Add the following under 'Secrets':
            ```toml
            OPENAI_API_KEY = "your_api_key"
            MODEL = "gpt-4o-mini"
            ```
    """)
    st.stop()

# Load OpenAI model from secrets
MODEL = st.secrets["MODEL"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# User input fields
url = st.text_input("🌍 Enter URL to crawl:", placeholder="https://example.com")
instruction = st.text_area(
    "📝 Enter extraction instructions:",
    placeholder="Example: Extract all product names and prices from the page."
)

# Schema for extracted content
class DynamicData(BaseModel):
    content: Dict[str, Any]

# Async function to run the web crawler
async def run_crawler(url: str, instruction: str):
    try:
        llm_strategy = LLMExtractionStrategy(
            provider=f"openai/{MODEL}",
            api_token=OPENAI_API_KEY,
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

        browser_cfg = BrowserConfig(headless=True, verbose=False)

        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            result = await crawler.arun(url=url, config=crawl_config)
            return result

    except Exception as e:
        return f"❌ Error: {str(e)}"

# Handle crawling execution
if url and instruction:
    if st.button("🚀 Start Crawling"):
        with st.spinner("⏳ Crawling the website... Please wait."):
            try:
                # Streamlit doesn't support `asyncio.run()`, so we use `asyncio.new_event_loop()`
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(run_crawler(url, instruction))

                if isinstance(result, str):  # Error case
                    st.error(result)
                elif result.success:
                    st.success("✅ Crawling completed successfully!")

                    # Display extracted data
                    data = json.loads(result.extracted_content)
                    st.json(data)

                    # Create a download button
                    st.download_button(
                        label="⬇️ Download Results",
                        data=result.extracted_content,
                        file_name="crawled_data.json",
                        mime="application/json"
                    )
                else:
                    st.error(f"⚠️ Error during crawling: {result.error_message}")

            except ValueError as ve:
                st.error(f"⚠️ Value error: {ve}")
            except ConnectionError:
                st.error("❌ Network issue. Please check your connection.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")

# Add user instructions
with st.expander("ℹ️ How to use this tool"):
    st.markdown("""
    **Follow these steps:**
    1. **Enter the URL** of the website you want to crawl.
    2. **Provide detailed instructions** for data extraction.
    3. Click **'Start Crawling'** and wait for results.
    4. Download the extracted data as **JSON**.

    🔹 *Ensure the website allows crawling by checking `robots.txt`.*  
    ❗ *Crawling restricted or private websites may result in errors.*
    """)

