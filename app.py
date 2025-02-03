import streamlit as st
import asyncio
import json
import re
from typing import List, Dict
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel
import time
import random
import pandas as pd
import plotly.express as px

# Set page config at the very beginning of the script
st.set_page_config(page_title="Advanced AI Web Scraper", page_icon="🕷️", layout="wide")

class ExtractedText(BaseModel):
    text: str

async def scrape_data(url: str, instruction: str, num_pages: int, all_pages: bool, custom_filters: List[str]) -> List[Dict[str, str]]:
    enhanced_instruction = (
        f"{instruction}\n\nEnsure the extracted text is relevant and excludes cookies, legal disclaimers,"
        " advertisements, and UI elements such as navigation bars and footers. Extract meaningful page content only."
    )

    llm_strategy = LLMExtractionStrategy(
        provider=st.secrets["MODEL"],
        api_token=st.secrets["OPENAI_API_KEY"],
        schema=ExtractedText.model_json_schema(),
        extraction_type="schema",
        instruction=enhanced_instruction,
        chunk_token_threshold=1000,
        overlap_rate=0.0,
        apply_chunking=True,
        input_format="markdown",
        extra_args={"temperature": 0.0, "max_tokens": 800},
    )

    crawl_config = CrawlerRunConfig(
        extraction_strategy=llm_strategy,
        cache_mode=CacheMode.BYPASS,
        process_iframes=True,
        remove_overlay_elements=True,
        exclude_external_links=True,
    )

    browser_cfg = BrowserConfig(
        headless=True, 
        verbose=False,
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        all_data = []
        default_exclusions = ['cookie', 'privacy policy', 'terms of service', 'advertisement']
        exclusion_patterns = re.compile('|'.join(default_exclusions + custom_filters), flags=re.IGNORECASE)

        progress_bar = st.progress(0)
        for page in range(1, num_pages + 1):
            page_url = f"{url}?page={page}" if '?' in url else f"{url}/page/{page}"
            
            for attempt in range(3):  # Retry mechanism
                try:
                    result = await crawler.arun(url=page_url, config=crawl_config)
                    
                    if result.success:
                        try:
                            data = json.loads(result.extracted_content)
                            filtered_data = [item for item in data if not exclusion_patterns.search(item["text"])]
                            all_data.extend(filtered_data)
                            break  # Successful, exit retry loop
                        except json.JSONDecodeError:
                            st.warning(f"Failed to decode JSON on page {page}. Retrying...")
                    else:
                        st.warning(f"Error on page {page}: {result.error_message}. Retrying...")
                
                except Exception as e:
                    st.error(f"Error processing page {page}: {str(e)}. Retrying...")
                
                if attempt < 2:  # Wait before retrying, except on last attempt
                    time.sleep(random.uniform(1, 3))  # Random delay between retries
            
            progress_bar.progress(page / num_pages)
            
            if not all_pages and page >= num_pages:
                break

        progress_bar.empty()

        if not all_data:
            st.warning(f"No data extracted from {url}. Possible reasons:")
            st.info("- Page might be dynamically loaded")
            st.info("- Content might require specific browser interactions")
            st.info("- Potential anti-scraping measures in place")

        return all_data

def visualize_data(data: List[Dict[str, str]]):
    if not data:
        st.warning("No data to visualize.")
        return

    df = pd.DataFrame(data)
    
    # Word count distribution
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    fig = px.histogram(df, x='word_count', title='Distribution of Word Count in Extracted Text')
    st.plotly_chart(fig)

    # Most common words
    all_words = ' '.join(df['text']).lower().split()
    word_freq = pd.Series(all_words).value_counts().head(20)
    fig = px.bar(x=word_freq.index, y=word_freq.values, title='Top 20 Most Common Words')
    st.plotly_chart(fig)

def main():
    st.title("Advanced AI-Assisted Web Scraping")

    with st.sidebar:
        st.header("Configuration")
        url_to_scrape = st.text_input("Enter the URL to scrape:")
        instruction_to_llm = st.text_area("Enter instructions for what to scrape:")
        num_pages = st.number_input("Enter the number of pages to scrape:", min_value=1, step=1)
        all_pages = st.checkbox("Scrape all pages")
        custom_filters = st.text_input("Enter custom filter words (comma-separated):").split(',')
        custom_filters = [f.strip() for f in custom_filters if f.strip()]

    if st.sidebar.button("Start Scraping"):
        if url_to_scrape and instruction_to_llm:
            with st.spinner("Scraping in progress..."):
                try:
                    data = asyncio.run(scrape_data(url_to_scrape, instruction_to_llm, num_pages, all_pages, custom_filters))
                    
                    if data:
                        st.success(f"Successfully scraped {len(data)} items.")
                        
                        # Display data
                        st.subheader("Scraped Data Preview")
                        st.dataframe(pd.DataFrame(data).head())
                        
                        # Visualize data
                        st.subheader("Data Visualization")
                        visualize_data(data)
                        
                        # Download options
                        st.subheader("Download Options")
                        formatted_data = "\n".join([item['text'] for item in data])
                        st.download_button("Download as TXT", formatted_data, "scraped_data.txt")
                        st.download_button("Download as JSON", json.dumps(data, indent=2), "scraped_data.json")
                        st.download_button("Download as CSV", pd.DataFrame(data).to_csv(index=False), "scraped_data.csv")
                    else:
                        st.warning("No data was scraped. Please check the URL and try again.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
                    st.error("Please check your API keys and ensure all dependencies are correctly installed.")
        else:
            st.warning("Please enter both the URL and instructions before starting.")

if __name__ == "__main__":
    main()


