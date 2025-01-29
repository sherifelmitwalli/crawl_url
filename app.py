import streamlit as st
import asyncio
import json
from typing import List, Dict, Any
from pydantic import BaseModel
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

# Schema for extracted content
class ResponseData(BaseModel):
    respondent_name: str
    response_text: str

# Async function to run the web crawler
async def run_crawler(url: str) -> List[Dict[str, Any]]:
    responses = []
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url)

            # Wait for the responses list to load with an increased timeout
            try:
                await page.wait_for_selector("div.govuk-grid-column-two-thirds", timeout=60000)  # Waits up to 60 seconds
            except PlaywrightTimeoutError:
                st.error("❌ The page took too long to load. Please check your internet connection or try again later.")
                return responses

            # Extract links to individual responses
            response_links = await page.query_selector_all("div.govuk-grid-column-two-thirds a")

            for link in response_links:
                response_url = await link.get_attribute("href")
                respondent_name = await link.inner_text()

                # Navigate to the individual response page
                response_page = await browser.new_page()
                await response_page.goto(response_url)

                # Extract the response text
                try:
                    await response_page.wait_for_selector("div.govuk-grid-column-two-thirds", timeout=60000)  # Waits up to 60 seconds
                    response_text = await response_page.inner_text("div.govuk-grid-column-two-thirds")
                except PlaywrightTimeoutError:
                    st.warning(f"⚠️ Unable to load the response from {respondent_name}. Skipping...")
                    await response_page.close()
                    continue

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
