import streamlit as st
import asyncio
import json
from typing import Dict, Any, List, Tuple
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel
from bs4 import BeautifulSoup

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
    placeholder="Example: Extract all product names and prices from the page."
)

# Additional crawler settings
with st.expander("Advanced Settings"):
    max_pages = st.number_input("Maximum pages to crawl", min_value=1, value=10)
    auto_detect = st.checkbox("Auto-detect selectors", value=True)
    debug_mode = st.checkbox("Enable debug mode", value=False)
    
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

async def analyze_page_structure(html_content: str) -> Tuple[str, str]:
    """
    Analyze page structure to detect content and pagination patterns
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Common content patterns (elements that likely contain responses)
    content_patterns = [
        ('article', {}),
        ('div', {'class': ['post', 'response', 'comment', 'item', 'entry']}),
        ('div', {'class': lambda x: x and any(word in x.lower() for word in ['response', 'comment', 'post', 'content'])}),
    ]
    
    # Common pagination patterns
    pagination_patterns = [
        ('a', {'rel': 'next'}),
        ('a', {'class': lambda x: x and 'next' in x.lower()}),
        ('link', {'rel': 'next'}),
        ('button', {'class': lambda x: x and 'next' in x.lower()}),
    ]
    
    # Find content elements
    content_selector = None
    max_elements = 0
    
    for tag, attrs in content_patterns:
        elements = soup.find_all(tag, attrs)
        if len(elements) > max_elements:
            # Get the CSS selector for this element type
            if elements and elements[0].get('class'):
                content_selector = f"{tag}.{'.'.join(elements[0]['class'])}"
                max_elements = len(elements)
    
    # Find pagination element
    next_page = None
    for tag, attrs in pagination_patterns:
        element = soup.find(tag, attrs)
        if element:
            if element.get('class'):
                next_page = f"{tag}.{'.'.join(element['class'])}"
            else:
                next_page = tag
            break
    
    if debug_mode:
        st.write(f"Found {max_elements} potential content elements")
        st.write(f"Content selector: {content_selector}")
        st.write(f"Next page selector: {next_page}")
    
    return next_page, content_selector

async def run_crawler(url: str, instruction: str):
    try:
        # Initialize extraction strategy with specific focus on responses
        llm_strategy = LLMExtractionStrategy(
            provider=f"openai/{MODEL_NAME}",
            api_token=OPENAI_API_KEY,
            extraction_type="text",
            instruction=f"""
            {instruction}
            Focus on extracting individual responses or entries. 
            For each response, create a separate entry in the results.
            Include all relevant details mentioned in the text.
            """,
            chunk_token_threshold=2000,  # Increased to handle larger content blocks
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
        
        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            current_url = url
            
            while current_page <= max_pages:
                if debug_mode:
                    st.write(f"Crawling page {current_page}: {current_url}")
                
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
                    try:
                        extracted_data = json.loads(result.extracted_content)
                        if debug_mode:
                            st.write(f"Extracted data from page {current_page}:", extracted_data)
                        
                        # Handle different possible JSON structures
                        if isinstance(extracted_data, dict):
                            if 'responses' in extracted_data:
                                all_results.extend(extracted_data['responses'])
                            elif 'items' in extracted_data:
                                all_results.extend(extracted_data['items'])
                            else:
                                all_results.append(extracted_data)
                        elif isinstance(extracted_data, list):
                            all_results.extend(extracted_data)
                    except json.JSONDecodeError:
                        if debug_mode:
                            st.write("Failed to parse JSON, storing raw content")
                        all_results.append({"content": result.extracted_content})
                    
                    # Analyze page structure for navigation
                    next_page_link = None
                    if result.page_content:
                        next_selector, _ = await analyze_page_structure(result.page_content)
                        if next_selector:
                            for link in result.page_links:
                                if 'next' in link.lower() or 'page' in link.lower():
                                    next_page_link = link
                                    break
                    
                    if next_page_link:
                        current_url = next_page_link
                        current_page += 1
                        await asyncio.sleep(wait_time)
                        if debug_mode:
                            st.write(f"Moving to next page: {current_url}")
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
    2. **Provide detailed instructions** for data extraction.
    3. Configure **Advanced Settings**:
        - Enable/disable automatic selector detection
        - Set maximum pages to crawl
        - Adjust wait time between actions
        - Enable debug mode for detailed logging
    4. Click **'Start Crawling'** and wait for results.
    5. Preview and download the extracted data as JSON.

    🔹 *Enable debug mode to see detailed extraction process.*  
    🔹 *Adjust wait time if the site takes longer to load.*  
    ❗ *Ensure the website allows crawling by checking `robots.txt`.*
    """)
