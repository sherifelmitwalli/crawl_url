import streamlit as st
import asyncio
import json
from typing import Dict, Any, List, Tuple
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

async def detect_selectors(crawler: AsyncWebCrawler) -> Tuple[str, str]:
    """
    Automatically detect pagination and clickable element selectors.
    Returns tuple of (next_page_selector, clickable_elements_selector)
    """
    # Common patterns for pagination and clickable elements
    pagination_patterns = [
        # Common next page button patterns
        """
        () => {
            const patterns = [
                'a[rel="next"]',
                '.pagination .next',
                '.pagination-next',
                'a:contains("Next")',
                'button:contains("Next")',
                '.next-page',
                '[aria-label="Next page"]',
                '.pagination li:last-child a',
                '.page-next',
                '[data-testid="pagination-next"]'
            ];
            
            for (let pattern of patterns) {
                const element = document.querySelector(pattern);
                if (element) return pattern;
            }
            
            // Look for links/buttons containing "next" or "→"
            const elements = Array.from(document.querySelectorAll('a, button'));
            for (let el of elements) {
                if (el.textContent.toLowerCase().includes('next') || 
                    el.textContent.includes('→') ||
                    el.getAttribute('aria-label')?.toLowerCase().includes('next')) {
                    return el.tagName.toLowerCase() + 
                           (el.className ? '.' + el.className.split(' ').join('.') : '');
                }
            }
            
            return null;
        }
        """
    ]

    clickable_patterns = [
        # Common clickable element patterns
        """
        () => {
            // Look for repeated structural patterns
            const patterns = new Map();
            
            // Find elements that appear multiple times with similar structure
            const elements = document.querySelectorAll('*');
            for (let el of elements) {
                if (!el.className) continue;
                
                const siblings = document.querySelectorAll('.' + el.className.split(' ').join('.'));
                if (siblings.length > 1) {
                    // Check if elements contain interactive content
                    const hasContent = Array.from(siblings).some(sib => {
                        const text = sib.textContent.trim();
                        const hasLinks = sib.querySelector('a');
                        const isClickable = window.getComputedStyle(sib).cursor === 'pointer';
                        return text.length > 20 && (hasLinks || isClickable);
                    });
                    
                    if (hasContent) {
                        patterns.set('.' + el.className.split(' ').join('.'), siblings.length);
                    }
                }
            }
            
            // Return the selector with the most matches
            return Array.from(patterns.entries())
                .sort((a, b) => b[1] - a[1])
                [0]?.[0] || null;
        }
        """
    ]

    next_page_selector = None
    clickable_selector = None

    # Try each pagination pattern
    for pattern in pagination_patterns:
        result = await crawler.browser_context.evaluate(pattern)
        if result:
            next_page_selector = result
            break

    # Try each clickable pattern
    for pattern in clickable_patterns:
        result = await crawler.browser_context.evaluate(pattern)
        if result:
            clickable_selector = result
            break

    return next_page_selector, clickable_selector

async def click_and_extract(crawler: AsyncWebCrawler, url: str, config: CrawlerRunConfig, selector: str):
    results = []
    
    # Get the page content and find clickable elements
    initial_result = await crawler.arun(url=url, config=config)
    if not initial_result.success:
        return []
    
    # Use JavaScript to get all matching elements
    elements = await crawler.browser_context.evaluate(f'''
        () => {{
            const elements = document.querySelectorAll("{selector}");
            return Array.from(elements).map(el => {{
                const rect = el.getBoundingClientRect();
                return {{
                    x: rect.x + rect.width / 2,
                    y: rect.y + rect.height / 2
                }};
            }});
        }}
    ''')
    
    for element_pos in elements:
        # Click the element using mouse position
        await crawler.browser_context.mouse.click(element_pos['x'], element_pos['y'])
        await asyncio.sleep(wait_time)
        
        # Extract content after clicking
        result = await crawler.arun(url=crawler.browser_context.url, config=config)
        if result.success:
            results.append(result.extracted_content)
        
        # Go back
        await crawler.browser_context.go_back()
        await asyncio.sleep(wait_time)
    
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

        browser_cfg = BrowserConfig(headless=True, verbose=False)
        all_results = []
        current_page = 1

        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            # Auto-detect selectors if enabled
            if auto_detect:
                await crawler.arun(url=url)  # Initial page load
                next_page_selector, click_selector = await detect_selectors(crawler)
                
                if next_page_selector:
                    st.info(f"📍 Detected pagination selector: {next_page_selector}")
                if click_selector:
                    st.info(f"🎯 Detected clickable elements selector: {click_selector}")
                
                if not next_page_selector and not click_selector:
                    st.warning("⚠️ Could not detect selectors automatically. Consider manual configuration.")
                    return {
                        "success": False,
                        "error_message": "No selectors detected"
                    }

            crawl_config = CrawlerRunConfig(
                extraction_strategy=llm_strategy,
                cache_mode=CacheMode.BYPASS,
                process_iframes=False,
                remove_overlay_elements=True,
                exclude_external_links=True,
                wait_for_selectors=[click_selector] if click_selector else None
            )

            current_url = url
            
            while current_page <= max_pages:
                # Extract data from current page
                if click_selector:
                    page_results = await click_and_extract(
                        crawler,
                        current_url,
                        crawl_config,
                        click_selector
                    )
                    all_results.extend(page_results)
                else:
                    result = await crawler.arun(url=current_url, config=crawl_config)
                    if result.success:
                        all_results.append(result.extracted_content)
                    else:
                        break

                # Check for next page
                if next_page_selector:
                    # Try to find and click next page button
                    next_button_exists = await crawler.browser_context.evaluate(f'''
                        () => {{
                            const nextButton = document.querySelector("{next_page_selector}");
                            if (nextButton) {{
                                nextButton.click();
                                return true;
                            }}
                            return false;
                        }}
                    ''')
                    
                    if not next_button_exists:
                        break
                    
                    await asyncio.sleep(wait_time)
                    current_url = crawler.browser_context.url
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
