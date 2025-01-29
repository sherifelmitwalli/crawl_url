import streamlit as st
import asyncio
import json
import requests
from typing import Dict, Any, List
from pydantic import BaseModel
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import logging
from urllib.robotparser import RobotFileParser

# --------------------------- Helper Functions ---------------------------

def is_valid_url(url: str) -> bool:
    """Validate the URL format."""
    try:
        parsed = urlparse(url)
        return all([parsed.scheme, parsed.netloc])
    except:
        return False

def can_crawl_robust(url: str, user_agent: str = '*') -> bool:
    """Check if crawling is allowed by robots.txt using RobotFileParser."""
    try:
        parsed_url = urlparse(url)
        robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
        rp = RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception as e:
        logger.warning(f"Could not parse robots.txt for {url}: {e}")
        return False  # Conservative approach: disallow crawling if robots.txt cannot be parsed

def is_url_accessible(url: str) -> bool:
    """Check if the URL is accessible."""
    try:
        response = requests.head(url, timeout=10)
        return response.status_code == 200
    except requests.RequestException as e:
        logger.error(f"Error accessing URL {url}: {e}")
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

# --------------------------- Logging Configuration ---------------------------

# Configure logging
logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

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
    retries = st.number_input("Number of retries on failure", min_value=0, max_value=5, value=3)
    retry_delay = st.number_input("Delay between retries (seconds)", min_value=1, max_value=10, value=5)

# Initialize session state for cancellation
if 'cancel' not in st.session_state:
    st.session_state.cancel = False

def cancel_crawl():
    st.session_state.cancel = True

async def run_crawler(url: str, instruction: str, progress_callback):
    try:
        all_results = []
        current_page = 1
        processed_urls = set()

        while current_page <= max_pages and url and not st.session_state.cancel:
            if url in processed_urls:
                if debug_mode:
                    logger.info(f"Already processed URL: {url}")
                break

            if debug_mode:
                logger.info(f"Processing page {current_page}: {url}")

            processed_urls.add(url)

            try:
                response = requests.get(url, timeout=10)
                if response.status_code != 200:
                    logger.warning(f"Failed to retrieve {url} with status code {response.status_code}")
                    return {"success": False, "error_message": f"Failed to retrieve {url}"}
                content = response.text
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                return {"success": False, "error_message": str(e)}

            # Process content based on instruction (simplified)
            # For example, extract all paragraphs
            soup = BeautifulSoup(content, "html.parser")
            paragraphs = [p.get_text() for p in soup.find_all('p')]
            items = [{"content": p} for p in paragraphs]
            all_results.extend(items)

            # Find next page link
            next_url = None
            for a in soup.find_all('a', href=True):
                if 'next' in a.text.lower():
                    next_url = urljoin(url, a['href'])
                    break

            if next_url and next_url != url:
                url = next_url
                current_page += 1
                await asyncio.sleep(wait_time)
            else:
                if debug_mode:
                    logger.info("No more pages found")
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
            logger.exception("An unexpected error occurred in run_crawler.")
        return {"success": False, "error_message": str(e)}

async def run_crawler_with_retries(url: str, instruction: str, progress_callback, retries: int, delay: int):
    for attempt in range(1, retries + 1):
        logger.info(f"Attempt {attempt} of {retries}")
        result = await run_crawler(url, instruction, progress_callback)
        if result.get("success"):
            return result
        else:
            logger.warning(f"Attempt {attempt} failed: {result.get('error_message')}")
            if attempt < retries:
                logger.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
    return {"success": False, "error_message": "All retry attempts failed."}

def synchronous_run_crawler_with_retries(url, instruction, progress_callback, retries, delay):
    try:
        return asyncio.run(run_crawler_with_retries(url, instruction, progress_callback, retries, delay))
    except BrokenPipeError as bpe:
        logger.error(f"BrokenPipeError during asyncio.run: {bpe}")
        return {"success": False, "error_message": f"BrokenPipeError: {str(bpe)}"}
    except Exception as e:
        logger.error(f"Unexpected error during asyncio.run: {e}")
        return {"success": False, "error_message": str(e)}

# Handle crawling execution
if url and instruction:
    start_crawl = st.button("🚀 Start Crawling")
    if start_crawl:
        if not is_valid_url(url):
            st.error("❌ Invalid URL. Please enter a valid URL with proper scheme (http/https).")
        elif not can_crawl_robust(url):
            st.error("❌ Crawling is disallowed by the website's `robots.txt`.")
        elif not is_url_accessible(url):
            st.error("❌ The URL is not accessible. Please check the URL and try again.")
        else:
            st.session_state.cancel = False
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                with st.spinner("⏳ Crawling the website... Please wait."):
                    result = synchronous_run_crawler_with_retries(
                        url,
                        instruction,
                        lambda p: progress_bar.progress(p),
                        retries,
                        retry_delay
                    )

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
        - Define the number of retries on failure and delay between retries.
        - Enable debug mode to see the extraction process.
    4. Click **'Start Crawling'** and wait for results.
    5. Preview and download the extracted data as JSON.

    🔹 *Enable debug mode to see detailed progress.*  
    🔹 *Increase wait time if the site loads slowly.*  
    🔹 *Set retries and delay to handle transient network issues.*  
    ❗ *Ensure the website allows crawling by checking `robots.txt`.*
    """)

