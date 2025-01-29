import streamlit as st
import asyncio
import json
from typing import Dict, Any, List, Tuple
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Set page title and icon
st.set_page_config(page_title="Web Crawler", page_icon="🕷️", layout="wide")

st.title("🔍 Web Crawler with AI")

# Check if API keys are set
if "OPENAI_API_KEY" not in st.secrets or "MODEL" not in st.secrets:
    st.error("❌ Missing API credentials! Please set OPENAI_API_KEY and MODEL in the Streamlit secrets.")
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

def parse_extracted_content(content: str) -> List[Dict]:
    """Parse and normalize the extracted content."""
    try:
        if isinstance(content, str):
            data = json.loads(content)
        else:
            data = content

        # Handle different JSON structures
        if isinstance(data, dict):
            if 'responses' in data:
                return data['responses']
            elif 'items' in data:
                return data['items']
            elif 'data' in data:
                return data['data'] if isinstance(data['data'], list) else [data['data']]
            else:
                return [data]
        elif isinstance(data, list):
            return data
        else:
            return [{"content": str(data)}]
    except json.JSONDecodeError:
        return [{"content": content}]

async def run_crawler(url: str, instruction: str):
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
        
        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            current_url = url
            
            while current_page <= max_pages and current_url:
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
                    st.info(f"""
                    📊 Crawl Statistics:
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

# Add user instructions
with st.expander("ℹ️ How to use this tool"):
    st.markdown("""
    **Follow these steps:**
    1. **Enter the URL** of the website you want to crawl.
    2. **Provide detailed instructions** for what to extract, for example:
       - "Extract all responses including the content, author, and date"
       - "Collect all comments with their associated metadata"
    3. Configure **Advanced Settings**:
        - Set maximum pages to crawl
        - Adjust wait time between pages
        - Enable debug mode to see the extraction process
    4. Click **'Start Crawling'** and wait for results.
    5. Preview and download the extracted data as JSON.

    🔹 *Enable debug mode to see detailed progress.*  
    🔹 *Increase wait time if the site loads slowly.*  
    ❗ *Ensure the website allows crawling by checking `robots.txt`.*
    """)
