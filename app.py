import streamlit as st
import openai
import asyncio
import json
import os
import pandas as pd
from typing import List
import nest_asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel, Field

# Apply nest_asyncio to handle async operations in Streamlit
nest_asyncio.apply()

# Ensure Playwright browsers are installed
try:
    import subprocess
    import sys
    subprocess.run(["playwright", "install", "chromium"], check=True)
except Exception as e:
    st.error(f"Failed to install Playwright browser: {str(e)}")
    st.info("If this error persists, please contact support.")

# Define the Product model
class Product(BaseModel):
    name: str
    price: str

# Function to run asynchronous scraping
def run_async_scraper(url, instruction, num_pages, all_pages, use_sitemap=False):
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_scrape_data(url, instruction, num_pages, all_pages, use_sitemap))
    except Exception as e:
        st.error(f"Error during scraping: {str(e)}")
        return []

# Asynchronous scraping logic
async def _scrape_data(url, instruction, num_pages, all_pages, use_sitemap):
    # Initialize extraction strategy
    llm_strategy = LLMExtractionStrategy(
        provider=st.secrets["MODEL"],
        api_token=st.secrets["OPENAI_API_KEY"],
        schema=Product.model_json_schema(),
        extraction_type="schema",
        instruction=instruction,
        chunk_token_threshold=chunk_size,
        overlap_rate=overlap,
        apply_chunking=True,
        input_format="markdown",
        extra_args={"temperature": temperature, "max_tokens": 800},
    )

    # Configure crawler
    crawl_config = CrawlerRunConfig(
        extraction_strategy=llm_strategy,
        cache_mode=CacheMode.BYPASS,
        process_iframes=False,
        remove_overlay_elements=True,
        exclude_external_links=True
    )
    browser_cfg = BrowserConfig(headless=headless, verbose=True)

    # Fetch URLs from sitemap if enabled
    if use_sitemap:
        url = f"{url}/sitemap.xml"
        result = await AsyncWebCrawler(config=browser_cfg).arun(url=url, config=crawl_config)
        if not result.success:
            st.error(f"Failed to fetch sitemap: {result.error_message}")
            return []
        urls = [loc.text for loc in result.extracted_content.find_all("loc")]
    else:
        urls = [f"{url}?page={page}" for page in range(1, num_pages + 1)]

    # Scrape each URL
    all_data = []
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        for i, page_url in enumerate(urls):
            try:
                result = await crawler.arun(url=page_url, config=crawl_config)
                if result.success:
                    data = json.loads(result.extracted_content)
                    all_data.extend(data)
                else:
                    st.write(f"Error on page {i + 1}: {result.error_message}")
                if not all_pages and i >= num_pages - 1:
                    break
            except Exception as e:
                st.error(f"Error processing page {i + 1}: {str(e)}")
                break
    return all_data

# Custom CSS for styling
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #4CAF50;
    color: white;
    font-size: 16px;
    padding: 10px 24px;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# Sidebar for navigation or additional info
st.sidebar.header("About")
st.sidebar.info("This app uses AI-assisted web scraping to extract structured data from websites.")
st.sidebar.markdown("[View Documentation](https://example.com)")

# Main title and description
st.title("AI-Assisted Web Crawler")
st.subheader("Scrape structured data from websites using Crawl4AI.")

# Organize inputs into columns for better layout
col1, col2 = st.columns(2)

with col1:
    url_to_scrape = st.text_input("Enter the URL to scrape:")
    instruction_to_llm = st.text_area("Enter the instructions for what to scrape:")

with col2:
    num_pages = st.number_input("Enter the number of pages to scrape:", min_value=1, step=1)
    all_pages = st.checkbox("Scrape all pages")
    use_sitemap = st.checkbox("Use sitemap for crawling")

# Expander for advanced settings
with st.expander("Advanced Settings"):
    st.subheader("Browser Configuration")
    headless = st.checkbox("Run browser in headless mode", value=True, 
                          help="When enabled, the browser runs in the background")
    
    st.subheader("Extraction Strategy")
    chunk_size = st.number_input("Chunk token threshold", value=1000, min_value=100, step=100,
                                help="Maximum number of tokens per chunk for text processing")
    overlap = st.slider("Chunk overlap rate", value=0.0, min_value=0.0, max_value=0.5, step=0.1,
                       help="Overlap between consecutive chunks (0.0 to 0.5)")
    temperature = st.slider("LLM Temperature", value=0.0, min_value=0.0, max_value=1.0, step=0.1,
                          help="Controls randomness in LLM responses (0.0 for focused, 1.0 for creative)")

# Start scraping button
if st.button("Start Scraping"):
    if url_to_scrape and instruction_to_llm:
        with st.spinner("Scraping in progress..."):
            try:
                # Progress bar
                progress_bar = st.progress(0)
                data = run_async_scraper(url_to_scrape, instruction_to_llm, num_pages, all_pages, use_sitemap)

                # Update progress bar
                progress_bar.progress(100)

                if not data:
                    st.warning("No data was found. Please check the URL and instructions.")
                else:
                    # Display raw data in a table
                    st.subheader("Scraped Data Summary")
                    df = pd.DataFrame(data)
                    st.dataframe(df)

                    # Visualize price distribution
                    st.subheader("Price Distribution")
                    st.bar_chart(df["price"].astype(float))

                    # Download button
                    formatted_data = "\n\n".join([f"Name: {item['name']}\nPrice: {item['price']}" for item in data])
                    st.download_button("Download Data", formatted_data, "scraped_data.txt")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}. Please try again or contact support.")
    else:
        st.write("Please enter the URL and instructions to begin scraping.")

