import streamlit as st
import asyncio
import json
import requests
from typing import Dict, Any, List
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import aiohttp
import aiofiles
import re

# --------------------------- Helper Functions ---------------------------

def is_valid_url(url: str) -> bool:
    """Validate the URL format."""
    try:
        parsed = urlparse(url)
        return all([parsed.scheme, parsed.netloc])
    except:
        return False

def can_crawl(url: str) -> bool:
    """Check if crawling is allowed by robots.txt."""
    try:
        robots_url = urljoin(url, "/robots.txt")
        response = requests.get(robots_url, timeout=5)
        if response.status_code == 200:
            # Simple parsing, consider using `urllib.robotparser` for more robust handling
            disallow = re.findall(r'Disallow: (.*)', response.text, re.IGNORECASE)
            for rule in disallow:
                if rule.strip() == '/':
                    return False
            return True
        return True  # Assume crawlable if no robots.txt
    except:
        return False

def parse_extracted_content(content: str) -> List[Dict]:
    """Parse and normalize the extracted content."""
    try:
        data = json.loads(content) if isinstance(content, str) else content

        if isinstance(data, dict):
            keys = ['responses', 'items', 'data']
            for key in keys:
                if key in data:
                    return data[key] if isinstance(data[key], list) else [data[key]]
            return [data]
        elif isinstance(data, list):
            return data
        else:
            return [{"content": str(data)}]
    except json.JSONDecodeError:
        # Attempt to parse HTML content if JSON fails
        soup = BeautifulSoup(content, "html.parser")
        texts = [p.get_text() for p in soup.find_all('p')]
        return [{"content": text} for text in texts if text]

# --------------------------- Main App ---------------------------

# Set page title and icon
st.set_page_config(page_title="Web Crawler with AI", page_icon="🕷️", layout="wide")

st.title("🔍 Web Crawler with AI")

# Check if API keys are set
if "OPENAI_API_KEY" not in st.secrets or "MODEL" not in st.secrets:
    st.error("❌ Missing API credentials! Please set `OPENAI_API_KEY` and `MODEL` in the Streamlit secrets.")
    st.stop()

# Load OpenAI model from secrets
MODEL_NAME = st.secrets["MODEL"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# User input fields
url = st.text_input("🌍 Enter URL to crawl:", placeholder="https://example.com")
instruction = st.text_area(
    "📝 Enter extraction instructions:",
    placeholder="Example: Extract all responses with their details."
)

# Additional crawler settings
with st.expander("Advanced Settings"):
    max_pages = st.number_input("Maximum pages to crawl", min_value=1, value=10)
    debug_mode = st.checkbox("Enable debug mode", value=False)
    wait_time = st.slider("Wait time between actions (seconds)", 1, 10, 3)
    concurrency_limit = st.number_input("Maximum concurrent requests", min_value=1, max_value=10, value=3)

# Initialize session state for cancellation
if 'cancel' not in st.session_state:
    st.session_state.cancel = False

def cancel_crawl():
    st.session_state.cancel = True

async def run_crawler(url: str, instruction: str, progress_callback):
    try:
        # Initialize extraction strategy with specific focus on responses
        llm_strategy = LLMExtractionStrategy(
            provider=f"openai/{MODEL_NAME}",
            api_token=OPENAI_API_KEY,
            extraction_type="text",
            instruction=f"""
            {instruction}
            Important: Extract each response or entry as a separate item.
            Format the output as a JSON array where each item contains:
            - The response content
            - Any associated metadata (date, author, etc.)
            - Any relevant details mentioned

            Format as: {{"responses": [{{response details}}, ...]}}
            """,
            chunk_token_threshold=2000,
            overlap_rate=0.1,
            apply_chunking=True,
            input_format="markdown",
            extra_args={
                "temperature": 0.0,
                "max_tokens": 1500,
                "response_format": {"type": "json_object"}
            },
        )

        browser_cfg = BrowserConfig(
            headless=True,
            verbose=debug_mode
        )

        all_results = []
        current_page = 1
        processed_urls = set()

        semaphore = asyncio.Semaphore(concurrency_limit)

        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            current_url = url

            while current_page <= max_pages and current_url and not st.session_state.cancel:
                async with semaphore:
                    if current_url in processed_urls:
                        if debug_mode:
                            st.write(f"Already processed URL: {current_url}")
                        break

                    if debug_mode:
                        st.write(f"Processing page {current_page}: {current_url}")

                    processed_urls.add(current_url)

                    # Configure crawler for current page
                    crawl_config = CrawlerRunConfig(
                        extraction_strategy=llm_strategy,
                        cache_mode=CacheMode.BYPASS,
                        process_iframes=False,
                        remove_overlay_elements=True,
                        exclude_external_links=True,
                    )

                    # Get page content
                    result = await crawler.arun(url=current_url, config=crawl_config)

                    if result.success:
                        # Parse the extracted content
                        page_results = parse_extracted_content(result.extracted_content)

                        if debug_mode:
                            st.write(f"Extracted {len(page_results)} items from page {current_page}")

                        all_results.extend(page_results)

                        # Look for next page link in all available links
                        next_url = None
                        base_url = current_url

                        # Get all links from the current page result
                        for link in result.page_links:
                            if link not in processed_urls and any(x in link.lower() for x in ['next', 'page', '?page=', 'paged']):
                                next_url = urljoin(base_url, link)
                                break

                        if next_url and next_url != current_url:
                            if debug_mode:
                                st.write(f"Found next page: {next_url}")
                            current_url = next_url
                            current_page += 1
                            await asyncio.sleep(wait_time)
                        else:
                            if debug_mode:
                                st.write("No more pages found")
                            break
                    else:
                        if debug_mode:
                            st.write(f"Failed to process page {current_page}")
                        break

                # Update progress
                progress = current_page / max_pages
                progress_callback(progress)

            return {
                "success": True,
                "extracted_content": json.dumps(all_results, indent=2),
                "current_page": current_page,
                "total_pages": max_pages,
                "total_items": len(all_results)
            }

    except Exception as e:
        if debug_mode:
            st.exception(e)
        return {"success": False, "error_message": str(e)}

# Handle crawling execution
if url and instruction:
    if st.button("🚀 Start Crawling"):
        if not is_valid_url(url):
            st.error("❌ Invalid URL. Please enter a valid URL with proper scheme (http/https).")
        elif not can_crawl(url):
            st.error("❌ Crawling is disallowed by the website's `robots.txt`.")
        else:
            st.session_state.cancel = False
            progress_bar = st.progress(0)
            status_text = st.empty()

            async def execute_crawl():
                result = await run_crawler(url, instruction, lambda p: progress_bar.progress(p))
                return result

            try:
                with st.spinner("⏳ Crawling the website... Please wait."):
                    result = asyncio.run(execute_crawl())

                if result.get("success"):
                    st.success("✅ Crawling completed successfully!")

                    # Display crawling statistics
                    st.info(f"""
                    📊 **Crawl Statistics:**
                    - Pages crawled: {result['current_page']} of {result['total_pages']}
                    - Total items extracted: {result['total_items']}
                    """)

                    # Preview the data
                    with st.expander("👀 Preview Extracted Data"):
                        st.json(json.loads(result['extracted_content']))

                    # Create a download button for the extracted data
                    st.download_button(
                        label="⬇️ Download Extracted Data",
                        data=result["extracted_content"],
                        file_name="extracted_data.json",
                        mime="application/json"
                    )
                else:
                    st.error(f"⚠️ Error during crawling: {result.get('error_message', 'Unknown error')}")

            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")

            finally:
                progress_bar.empty()
                status_text.empty()

    # Cancellation button
    if st.button("🛑 Cancel Crawling"):
        st.session_state.cancel = True
        st.warning("⚠️ Crawling has been cancelled.")

# Add user instructions
with st.expander("ℹ️ How to use this tool"):
    st.markdown("""
    **Follow these steps:**
    1. **Enter the URL** of the website you want to crawl.
    2. **Provide detailed instructions** for what to extract, for example:
       - "Extract all responses including the content, author, and date."
       - "Collect all comments with their associated metadata."
    3. Configure **Advanced Settings**:
        - Set the maximum number of pages to crawl.
        - Adjust wait time between page requests.
        - Set the maximum number of concurrent requests.
        - Enable debug mode to see the extraction process.
    4. Click **'Start Crawling'** and wait for results.
    5. Preview and download the extracted data as JSON.

    🔹 *Enable debug mode to see detailed progress.*  
    🔹 *Increase wait time if the site loads slowly.*  
    ❗ *Ensure the website allows crawling by checking `robots.txt`.*
    """)

