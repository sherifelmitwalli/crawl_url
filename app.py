import streamlit as st
import asyncio
import json
from typing import List, Dict, Any
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

# Set page title and icon
st.set_page_config(page_title="Universal Web Scraper", page_icon="🌐", layout="wide")

st.title("🌐 Universal Web Scraper")

# User input fields
url = st.text_input("🌍 Enter URL to scrape:", placeholder="https://example.com")
instruction = st.text_area(
    "📝 Enter extraction instructions:",
    placeholder="Example: Extract all product names and prices from the page."
)

# Async function to run the web scraper
async def run_scraper(url: str, instruction: str) -> List[Dict[str, Any]]:
    extracted_data = []
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url)

            # Wait for the page to load completely
            await page.wait_for_load_state('networkidle')

            # Extract content based on user instructions
            # This is a placeholder for the extraction logic
            # You can implement custom extraction logic here based on the instruction
            content = await page.content()
            extracted_data.append({
                "url": url,
                "instruction": instruction,
                "content": content
            })

            await browser.close()
    except PlaywrightTimeoutError:
        st.error("❌ The page took too long to load. Please check your internet connection or try again later.")
    except Exception as e:
        st.error(f"❌ Error during scraping: {str(e)}")
    return extracted_data

# Handle scraping execution
if url and instruction:
    if st.button("🚀 Start Scraping"):
        with st.spinner("⏳ Scraping the website... Please wait."):
            try:
                # Run the scraper
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                scraped_data = loop.run_until_complete(run_scraper(url, instruction))

                if scraped_data:
                    st.success("✅ Scraping completed successfully!")

                    # Display extracted data
                    for data in scraped_data:
                        st.subheader(f"Data extracted from {data['url']}")
                        st.write(data['content'])

                    # Option to download the data
                    st.download_button(
                        label="⬇️ Download Results",
                        data=json.dumps(scraped_data, indent=2),
                        file_name="scraped_data.json",
                        mime="application/json"
                    )
                else:
                    st.warning("⚠️ No data extracted.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")

# Add user instructions
with st.expander("ℹ️ How to use this tool"):
    st.markdown("""
    **Follow these steps:**
    1. **Enter the URL** of the website you want to scrape.
    2. **Provide detailed instructions** for data extraction.
    3. Click **'Start Scraping'** and wait for results.
    4. View the extracted data below.
    5. Optionally, download the results as a JSON file.

    🔹 *Ensure you have an active internet connection.*  
    ❗ *This tool extracts publicly available data from the specified website.*
    """)

