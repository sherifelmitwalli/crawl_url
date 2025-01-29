import streamlit as st
import asyncio
import json
from typing import Dict, Any, List
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel

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
            MODEL = "gpt-4-mini"
            ```
    """)
    st.stop()

# Load OpenAI model from secrets
MODEL_NAME = st.secrets["MODEL"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# User input fields
url = st.text_input("🌍 Enter URL to crawl:", placeholder="https://example.com")
instruction = st.text_area(
    "📝 Enter extraction instructions:",
    placeholder="Example: Extract all product names and prices from the page."
)

# Additional crawler settings
with st.expander("Advanced Settings"):
    max_pages = st.number_input("Maximum pages to crawl", min_value=1, value=10)
    next_page_selector = st.text_input(
        "CSS Selector for next page button",
        placeholder=".pagination .next"
    )
    click_selector = st.text_input(
        "CSS Selector for clickable elements",
        placeholder=".response-item"
    )
    wait_time = st.slider("Wait time between actions (seconds)", 1, 10, 3)

class ExtractedData(BaseModel):
    data: Dict[str, Any]
    current_page: int
    total_pages: int

async def click_and_extract(page, selector: str, instruction: str, llm_strategy: LLMExtractionStrategy):
    elements = await page.query_selector_all(selector)
    results = []
    
    for element in elements:
        # Click the element
        await element.click()
        await asyncio.sleep(wait_time)  # Wait for content to load
        
        # Extract content after clicking
        content = await page.content()
        extracted = await llm_strategy.process(content)
        results.append(extracted)
        
        # Go back or handle the state after clicking
        await page.go_back()
        await asyncio.sleep(wait_time)  # Wait for page to reload
    
    return results

async def run_crawler(url: str, instruction: str):
    try:
        llm_strategy = LLMExtractionStrategy(
            provider=f"openai/{MODEL_NAME}",
            api_token=OPENAI_API_KEY,
            extraction_type="text",
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
        all_results = []
        current_page = 1

        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            page = await crawler.launch_browser()
            await page.goto(url)
            
            while current_page <= max_pages:
                # Extract data from current page
                if click_selector:
                    page_results = await click_and_extract(
                        page, 
                        click_selector, 
                        instruction, 
                        llm_strategy
                    )
                    all_results.extend(page_results)
                else:
                    result = await crawler.arun(url=url, config=crawl_config)
                    all_results.append(result.extracted_content)

                # Check for next page
                if next_page_selector:
                    next_button = await page.query_selector(next_page_selector)
                    if not next_button:
                        break
                    
                    await next_button.click()
                    await asyncio.sleep(wait_time)  # Wait for next page to load
                    current_page += 1
                else:
                    break

            return {
                "success": True,
                "extracted_content": json.dumps(all_results),
                "current_page": current_page,
                "total_pages": max_pages
            }

    except Exception as e:
        return f"❌ Error: {str(e)}"

# Handle crawling execution
if url and instruction:
    if st.button("🚀 Start Crawling"):
        with st.spinner("⏳ Crawling the website... Please wait."):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(run_crawler(url, instruction))

                if isinstance(result, str):  # Error case
                    st.error(result)
                elif result.get("success"):
                    st.success("✅ Crawling completed successfully!")
                    
                    # Display crawling statistics
                    st.info(f"📊 Crawled {result['current_page']} pages out of maximum {result['total_pages']} pages")

                    # Create a download button for the extracted data
                    st.download_button(
                        label="⬇️ Download Extracted Data",
                        data=result["extracted_content"],
                        file_name="extracted_data.json",
                        mime="application/json"
                    )
                else:
                    st.error(f"⚠️ Error during crawling: {result.get('error_message', 'Unknown error')}")

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
    3. Configure **Advanced Settings** if needed:
        - Set maximum pages to crawl
        - Provide CSS selectors for pagination and clickable elements
        - Adjust wait time between actions
    4. Click **'Start Crawling'** and wait for results.
    5. Download the extracted data as **JSON**.

    🔹 *Ensure the website allows crawling by checking `robots.txt`.*  
    ❗ *Crawling restricted or private websites may result in errors.*
    """)
