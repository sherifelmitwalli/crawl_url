import streamlit as st
import asyncio
import json
from typing import List, Dict, Any
from pydantic import BaseModel
from playwright.async_api import async_playwright

# Set page title and icon
st.set_page_config(page_title="Web Crawler", page_icon="🕷️", layout="wide")

st.title("🔍 Web Crawler with AI")

# User input fields
url = st.text_input("🌍 Enter URL to crawl:", placeholder="https://consult.gov.scot/environment-forestry/single-use-vapes/consultation/published_select_respondent")
instruction = st.text_area(
    "📝 Enter extraction instructions:",
    placeholder="Example: Extract all reviewer responses from the consultation page."
)

# Schema for extracted content
class ResponseData(BaseModel):
    respondent_name: str
    response_text: str

# Async function to run the web crawler
async def run_crawler(url: str, instruction: str) -> List[Dict[str, Any]]:
    responses = []
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url)

            # Wait for the responses list to load
            await page.wait_for_selector("div.govuk-grid-column-two-thirds")

            # Extract links to individual responses
            response_links = await page.query_selector_all("div.govuk-grid-column-two-thirds a")

            for link in response_links:
                response_url = await link.get_attribute("href")
                respondent_name = await link.inner_text()

                # Navigate to the individual response page
                response_page = await browser.new_page()
                await response_page.goto(response_url)

                # Extract the response text
                response_text = await response_page.inner_text("div.govuk-grid-column-two-thirds")

                # Append the data to the list
                responses.append(ResponseData(
                    respondent_name=respondent_name,
                    response_text=response_text
                ).dict())

                await response_page.close()

            await browser.close()
    except Exception as e:
        st.error(f"❌ Error during crawling: {str(e)}")
    return responses

# Handle crawling execution
if url and instruction:
    if st.button("🚀 Start Crawling"):
        with st.spinner("⏳ Crawling the website... Please wait."):
            try:
                # Run the scraper
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                scraped_data = loop.run_until_complete(run_crawler(url, instruction))

                if scraped_data:
                    st.success("✅ Crawling completed successfully!")

                    # Display extracted data
                    for response in scraped_data:
                        st.subheader(response['respondent_name'])
                        st.write(response['response_text'])

                    # Option to download the data
                    st.download_button(
                        label="⬇️ Download Results",
                        data=json.dumps(scraped_data, indent=2),
                        file_name="consultation_responses.json",
                        mime="application/json"
                    )
                else:
                    st.warning("⚠️ No responses found.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")

# Add user instructions
with st.expander("ℹ️ How to use this tool"):
    st.markdown("""
    **Follow these steps:**
    1. **Enter the URL** of the consultation page you want to crawl.
    2. **Provide detailed instructions** for data extraction.
    3. Click **'Start Crawling'** and wait for results.
    4. View the extracted responses below.
    5. Optionally, download the results as a JSON file.

    🔹 *Ensure you have an active internet connection.*  
    ❗ *This tool extracts publicly available responses from the specified consultation page.*
    """)

