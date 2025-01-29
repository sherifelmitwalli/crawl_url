import streamlit as st
import asyncio
from playwright.async_api import async_playwright
from pydantic import BaseModel
from typing import List

# Set page title and icon
st.set_page_config(page_title="Consultation Responses Scraper", page_icon="🕷️", layout="wide")

st.title("🔍 Consultation Responses Scraper")

# URL of the consultation page
CONSULTATION_URL = "https://consult.gov.scot/environment-forestry/single-use-vapes/consultation/published_select_respondent"

# Schema for extracted content
class ResponseData(BaseModel):
    respondent_name: str
    response_text: str

# Async function to scrape responses
async def scrape_responses():
    responses = []
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(CONSULTATION_URL)

            # Wait for the responses list to load
            await page.wait_for_selector("div.response-list")

            # Extract links to individual responses
            response_links = await page.query_selector_all("div.response-list a")

            for link in response_links:
                response_url = await link.get_attribute("href")
                respondent_name = await link.inner_text()

                # Navigate to the individual response page
                response_page = await browser.new_page()
                await response_page.goto(response_url)

                # Extract the response text
                response_text = await response_page.inner_text("div.response-text")

                # Append the data to the list
                responses.append(ResponseData(
                    respondent_name=respondent_name,
                    response_text=response_text
                ))

                await response_page.close()

            await browser.close()
    except Exception as e:
        st.error(f"❌ Error during scraping: {str(e)}")
    return responses

# Streamlit UI
if st.button("🚀 Start Scraping"):
    with st.spinner("⏳ Scraping the consultation responses... Please wait."):
        try:
            # Run the scraper
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            scraped_data = loop.run_until_complete(scrape_responses())

            if scraped_data:
                st.success("✅ Scraping completed successfully!")

                # Display extracted data
                for response in scraped_data:
                    st.subheader(response.respondent_name)
                    st.write(response.response_text)

                # Option to download the data
                st.download_button(
                    label="⬇️ Download Results",
                    data="\n\n".join([f"{r.respondent_name}\n{r.response_text}" for r in scraped_data]),
                    file_name="consultation_responses.txt",
                    mime="text/plain"
                )
            else:
                st.warning("⚠️ No responses found.")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")

# Add user instructions
with st.expander("ℹ️ How to use this tool"):
    st.markdown("""
    **Instructions:**
    1. Click **'Start Scraping'** to begin extracting consultation responses.
    2. Wait for the process to complete.
    3. View the extracted responses below.
    4. Optionally, download the results as a text file.

    🔹 *Ensure you have an active internet connection.*  
    ❗ *This tool extracts publicly available responses from the specified consultation page.*
    """)
