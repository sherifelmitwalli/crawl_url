import streamlit as st
import asyncio
import json
import time
import psutil
from typing import Dict, Any, Optional
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel
import gc
import logging
from contextlib import asynccontextmanager
from prometheus_client import start_http_server, Counter, Gauge

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('crawler.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize metrics
REQUEST_COUNT = Counter('crawler_requests_total', 'Total crawl requests')
ERROR_COUNT = Counter('crawler_errors_total', 'Total crawl errors')
CRAWL_DURATION = Gauge('crawler_duration_seconds', 'Crawl duration in seconds')
MEMORY_USAGE = Gauge('crawler_memory_usage_bytes', 'Memory usage in bytes')

# Start metrics server
start_http_server(8000)

# Performance Settings
st.set_page_config(
    page_title="Web Crawler",
    page_icon="🕷️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Optimize memory usage with session state
if "crawler_instance" not in st.session_state:
    st.session_state.crawler_instance = None

# Schema for extracted content
class DynamicData(BaseModel):
    content: Dict[str, Any]

# Cache the LLM strategy creation
@st.cache_resource
def create_llm_strategy(model_name: str, api_key: str, instruction: str) -> LLMExtractionStrategy:
    return LLMExtractionStrategy(
        provider=f"openai/{model_name}",
        api_token=api_key,
        extraction_type="text",
        instruction=instruction,
        chunk_token_threshold=1000,
        overlap_rate=0.0,
        apply_chunking=True,
        input_format="markdown",
        extra_args={"temperature": 0.0, "max_tokens": 1000},
    )

# Async context manager for browser management
@asynccontextmanager
async def managed_crawler(browser_cfg: BrowserConfig):
    crawler = None
    try:
        crawler = AsyncWebCrawler(config=browser_cfg)
        await crawler.start()
        yield crawler
    finally:
        if crawler:
            await crawler.close()
        gc.collect()  # Force garbage collection

# Cache browser configuration
@st.cache_resource
def get_browser_config() -> BrowserConfig:
    return BrowserConfig(
        headless=True,
        verbose=False,
        timeout=30,  # 30 seconds timeout
        args=[
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--no-sandbox",
            "--disable-setuid-sandbox",
        ]
    )

# Main async crawling function with timeout
async def run_crawler_with_timeout(url: str, instruction: str, timeout: int = 60) -> Dict:
    REQUEST_COUNT.inc()
    start_time = time.time()
    MEMORY_USAGE.set(psutil.Process().memory_info().rss)
    
    try:
        llm_strategy = create_llm_strategy(
            st.secrets["MODEL"],
            st.secrets["OPENAI_API_KEY"],
            instruction
        )

        crawl_config = CrawlerRunConfig(
            extraction_strategy=llm_strategy,
            cache_mode=CacheMode.BYPASS,
            process_iframes=False,
            remove_overlay_elements=True,
            exclude_external_links=True,
            max_depth=1,  # Limit crawl depth for better performance
            max_pages=1,  # Limit to single page for better performance
        )

        browser_cfg = get_browser_config()

        async with managed_crawler(browser_cfg) as crawler:
            try:
                result = await asyncio.wait_for(
                    crawler.arun(url=url, config=crawl_config),
                    timeout=timeout
                )
                CRAWL_DURATION.set(time.time() - start_time)
                MEMORY_USAGE.set(psutil.Process().memory_info().rss)
                return {"success": True, "data": result}
            except asyncio.TimeoutError:
                ERROR_COUNT.inc()
                CRAWL_DURATION.set(time.time() - start_time)
                MEMORY_USAGE.set(psutil.Process().memory_info().rss)
                return {"success": False, "error": "Crawling timed out. Please try again."}
    except Exception as e:
        logger.error(f"Crawling error: {str(e)}")
        ERROR_COUNT.inc()
        return {"success": False, "error": str(e)}

# Check API credentials
if "OPENAI_API_KEY" not in st.secrets or "MODEL" not in st.secrets:
    st.error("❌ Missing API credentials! Please set OPENAI_API_KEY and MODEL in the Streamlit secrets.")
    st.markdown("""
        **Steps to configure secrets:**
        1. Go to your Streamlit dashboard
        2. Navigate to your app's settings
        3. Add the following under 'Secrets':
            ```toml
            OPENAI_API_KEY = "your_api_key"
            MODEL = "gpt-4o-mini"
            ```
    """)
    st.stop()

# UI Components
st.title("🔍 High-Performance Web Crawler with AI")

with st.form(key="crawler_form"):
    url = st.text_input("🌍 Enter URL to crawl:", placeholder="https://example.com")
    instruction = st.text_area(
        "📝 Enter extraction instructions:",
        placeholder="Example: Extract all product names and prices from the page."
    )
    timeout = st.slider("⏱️ Timeout (seconds)", 30, 180, 60)
    submit_button = st.form_submit_button("🚀 Start Crawling")

if submit_button and url and instruction:
    with st.spinner("⏳ Crawling the website... Please wait."):
        try:
            # Setup asyncio loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run crawler with timeout
            result = loop.run_until_complete(
                run_crawler_with_timeout(url, instruction, timeout)
            )
            
            if result["success"]:
                crawl_result = result["data"]
                if crawl_result.success:
                    st.success("✅ Crawling completed successfully!")
                    
                    # Display extracted data
                    data = json.loads(crawl_result.extracted_content)
                    st.json(data)
                    
                    # Create download button
                    st.download_button(
                        label="⬇️ Download Results",
                        data=crawl_result.extracted_content,
                        file_name="crawled_data.json",
                        mime="application/json"
                    )
                else:
                    st.error(f"⚠️ Crawling failed: {crawl_result.error_message}")
            else:
                st.error(f"❌ Error: {result['error']}")
                
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            st.error(f"❌ An unexpected error occurred: {str(e)}")
        finally:
            loop.close()
            gc.collect()  # Force garbage collection

# Help section
with st.expander("ℹ️ How to use this tool"):
    st.markdown("""
    **Follow these steps:**
    1. **Enter the URL** of the website you want to crawl
    2. **Provide detailed instructions** for data extraction
    3. **Adjust the timeout** if needed (default: 60 seconds)
    4. Click **'Start Crawling'** and wait for results
    5. Download the extracted data as **JSON**

    **Performance Tips:**
    - The crawler is optimized for single-page extraction
    - Increase timeout for complex pages
    - Clear browser cache if experiencing issues
    
    🔹 *Ensure the website allows crawling by checking `robots.txt`*  
    ❗ *Crawling restricted or private websites may result in errors*
    """)
