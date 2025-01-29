import streamlit as st
import asyncio
import json
from typing import Dict, Any, List, Tuple
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel
from playwright.async_api import Page

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
    auto_detect = st.checkbox("Auto-detect selectors", value=True)
    
    # Only show manual selector inputs if auto-detect is disabled
    if not auto_detect:
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

async def detect_selectors(result) -> Tuple[str, str]:
    """
    Detect pagination and clickable element selectors from the page content.
    """
    content = result.page_content
    
    # Common patterns to look for in the HTML
    pagination_patterns = [
        'a[rel="next"]',
        '.pagination .next',
        '.pagination-next',
        '.next-page',
        '[aria-label="Next page"]',
        '.pagination li:last-child a',
        '.page-next',
        '[data-testid="pagination-next"]'
    ]
    
    clickable_patterns = [
        '.item-card',
        '.response-item',
        '.content-card',
        '.result-item',
        '.list-item',
        'article',
        '.post',
        '.entry'
    ]
    
    # Look for pagination selector
    next_page_selector = None
    for pattern in pagination_patterns:
        if pattern.lower() in content.lower():
            next_page_selector = pattern
            break
    
    # Look for clickable elements selector
    click_selector = None
    for pattern in clickable_patterns:
        if pattern.lower() in content.lower():
            click_selector = pattern
            break
            
    return next_page_selector, click_selector

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

        browser_cfg = BrowserConfig(headless=True, verbose=False)
        all_results = []
        current_page = 1

        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            # Initial configuration
            current_url = url
            page_selectors = None
            content_selector = None
            
            # Auto-detect selectors if enabled
            if auto_detect:
                initial_result = await crawler.arun(url=url)
                if initial_result.success:
                    page_selectors, content_selector = await detect_selectors(initial_result)
                    
                    if page_selectors:
                        st.info(f"📍 Detected pagination selector: {page_selectors}")
                    if content_selector:
                        st.info(f"🎯 Detected clickable elements selector: {content_selector}")
                    
                    if not page_selectors and not content_selector:
                        st.warning("⚠️ Could not detect selectors automatically. Using default extraction.")
            else:
                page_selectors = next_page_selector if 'next_page_selector' in locals() else None
                content_selector = click_selector if 'click_selector' in locals() else None

            crawl_config = CrawlerRunConfig(
                extraction_strategy=llm_strategy,
                cache_mode=CacheMode.BYPASS,
                process_iframes=False,
                remove_overlay_elements=True,
                exclude_external_links=True,
            )

            while current_page <= max_pages:
                # Extract content from current page
                result = await crawler.arun(url=current_url, config=crawl_config)
                
                if result.success:
                    all_results.append(result.extracted_content)
                    
                    # Try to find next page link
                    if page_selectors and current_page < max_pages:
                        # Extract links from the page
                        links = result.page_links
                        next_page_url = None
                        
                        # Look for next page link
                        for link in links:
                            if any(word in link.lower() for word in ['next', 'page', 'forward']):
                                next_page_url = link
                                break
                        
                        if next_page_url:
                            current_url = next_page_url
                            current_page += 1
                            await asyncio.sleep(wait_time)
                        else:
                            break
                    else:
                        break
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
    3. Configure **Advanced Settings**:
        - Enable/disable automatic selector detection
        - Set maximum pages to crawl
        - Adjust wait time between actions
        - Manually set selectors if auto-detection is disabled
    4. Click **'Start Crawling'** and wait for results.
    5. Download the extracted data as **JSON**.

    🔹 *Auto-detection works best on structured websites with consistent layouts.*  
    🔹 *If auto-detection fails, try manual selector configuration.*  
    ❗ *Ensure the website allows crawling by checking `robots.txt`.*
    """)
