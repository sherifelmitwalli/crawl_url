# Place page config at the absolute top of the file
import streamlit as st
st.set_page_config(page_title="AI Web Scraper Pro", page_icon="🕷️", layout="wide")

import asyncio
import json
import re
import subprocess
import sys
import time
import random
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from urllib.parse import urlparse
from functools import wraps
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ----------------------- Security & Validation -----------------------

def validate_url(url: str) -> bool:
    """Validate URL format and scheme."""
    try:
        result = urlparse(url)
        return all([
            result.scheme in ['http', 'https'],
            result.netloc,
            len(url) < 2000  # Standard URL length limit
        ])
    except Exception as e:
        logger.error(f"URL validation error: {str(e)}")
        return False

def sanitize_input(input_str: str) -> str:
    """Sanitize user input to prevent injection attacks."""
    # Remove any potentially dangerous characters
    return re.sub(r'[;<>&|]', '', input_str)

# ----------------------- Rate Limiting -----------------------

class RateLimiter:
    def __init__(self, max_per_second: float):
        self.min_interval = 1.0 / max_per_second
        self.last_time_called = 0.0
        
    async def acquire(self):
        """Acquire permission to proceed, waiting if necessary."""
        current_time = time.time()
        elapsed = current_time - self.last_time_called
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
        self.last_time_called = time.time()

# ----------------------- Circuit Breaker -----------------------

class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: int = 60,
        half_open_timeout: int = 30
    ):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_timeout = half_open_timeout
        self.last_failure_time = 0
        self.state = "closed"
        self._lock = asyncio.Lock()

    async def call(self, func, *args, **kwargs):
        async with self._lock:
            if self.state == "open":
                if time.time() - self.last_failure_time > self.reset_timeout:
                    self.state = "half-open"
                    logger.info("Circuit breaker state changed to half-open")
                else:
                    raise Exception("Circuit breaker is open")

            try:
                result = await func(*args, **kwargs)
                if self.state == "half-open":
                    self.state = "closed"
                    self.failure_count = 0
                    logger.info("Circuit breaker state changed to closed")
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                    logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
                raise e

# ----------------------- Caching -----------------------

class TimedCache:
    def __init__(self, timeout_minutes: int = 30, max_size: int = 1000):
        self.timeout = timedelta(minutes=timeout_minutes)
        self.max_size = max_size
        self.cache = {}
        self._lock = asyncio.Lock()

    async def get_or_set(self, key: str, async_func, *args, **kwargs) -> Any:
        async with self._lock:
            now = datetime.now()
            
            # Clean expired entries
            expired_keys = [k for k, (_, timestamp) in self.cache.items()
                          if now - timestamp > self.timeout]
            for k in expired_keys:
                del self.cache[k]

            # Check cache size
            if len(self.cache) >= self.max_size:
                # Remove oldest entries
                sorted_items = sorted(self.cache.items(), key=lambda x: x[1][1])
                self.cache = dict(sorted_items[len(sorted_items)//2:])

            if key in self.cache:
                result, timestamp = self.cache[key]
                if now - timestamp < self.timeout:
                    return result

            try:
                result = await async_func(*args, **kwargs)
                self.cache[key] = (result, now)
                return result
            except Exception as e:
                logger.error(f"Cache operation failed: {str(e)}")
                raise

# ----------------------- Resource Management -----------------------

@asynccontextmanager
async def get_session():
    """Create and manage an aiohttp session with proper cleanup."""
    timeout = aiohttp.ClientTimeout(total=30)
    connector = aiohttp.TCPConnector(limit=10, force_close=True)
    
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    ) as session:
        try:
            yield session
        finally:
            await connector.close()

# ----------------------- Proxy Management -----------------------

class ProxyManager:
    def __init__(self, max_proxies: int = 10):
        self.proxies = []
        self.max_proxies = max_proxies
        self.current_index = 0
        self._lock = asyncio.Lock()

    async def refresh_proxies(self):
        """Fetch and validate new proxies."""
        async with self._lock:
            try:
                from proxybroker import Broker
                proxies = set()
                
                async def save(proxy):
                    if len(proxies) < self.max_proxies:
                        proxy_str = f"{proxy.host}:{proxy.port}"
                        if await self.validate_proxy(proxy_str):
                            proxies.add(proxy_str)

                broker = Broker(save)
                await broker.find(types=['HTTP', 'HTTPS'], limit=self.max_proxies * 2)
                self.proxies = list(proxies)
                logger.info(f"Refreshed proxy list with {len(self.proxies)} valid proxies")
            except Exception as e:
                logger.error(f"Failed to refresh proxies: {str(e)}")

    async def validate_proxy(self, proxy: str) -> bool:
        """Validate proxy by testing connection."""
        try:
            async with get_session() as session:
                async with session.get(
                    'https://httpbin.org/ip',
                    proxy=f"http://{proxy}",
                    timeout=5
                ) as response:
                    return response.status == 200
        except Exception:
            return False

    async def get_next_proxy(self) -> Optional[str]:
        """Get next proxy from the pool."""
        if not self.proxies:
            await self.refresh_proxies()
        
        if self.proxies:
            self.current_index = (self.current_index + 1) % len(self.proxies)
            return self.proxies[self.current_index]
        return None

# ----------------------- Playwright Setup -----------------------

async def setup_playwright():
    """Asynchronous Playwright setup with comprehensive error handling."""
    try:
        import playwright
        logger.info("Playwright already installed")
        return True
    except ImportError:
        logger.info("Installing Playwright...")
        
        install_commands = [
            [sys.executable, "-m", "pip", "install", "playwright"],
            [sys.executable, "-m", "playwright", "install"],
        ]

        for cmd in install_commands:
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0:
                    logger.error(f"Command failed: {' '.join(cmd)}")
                    logger.error(f"Error output: {stderr.decode()}")
                    continue
                    
                logger.info(f"Successfully executed: {' '.join(cmd)}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to execute command: {str(e)}")
                continue
                
        return False

# ----------------------- Scraping Logic -----------------------

class WebScraper:
    def __init__(self):
        self.rate_limiter = RateLimiter(max_per_second=2)
        self.circuit_breaker = CircuitBreaker()
        self.cache = TimedCache()
        self.proxy_manager = ProxyManager()

    async def scrape_page(self, url: str, instruction: str) -> Dict:
        """Scrape a single page with comprehensive error handling and retries."""
        if not validate_url(url):
            raise ValueError("Invalid URL provided")

        async def _scrape_attempt():
            await self.rate_limiter.acquire()
            
            from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
            from crawl4ai.extraction_strategy import LLMExtractionStrategy
            
            browser_config = BrowserConfig(
                headless=True,
                verbose=True,
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
                ),
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                    "--disable-setuid-sandbox"
                ]
            )

            llm_strategy = LLMExtractionStrategy(
                provider=st.secrets["MODEL"],
                api_token=st.secrets["OPENAI_API_KEY"],
                extraction_type="schema",
                instruction=instruction,
                chunk_token_threshold=1000,
                overlap_rate=0.1,
                apply_chunking=True,
                input_format="markdown",
                extra_args={"temperature": 0.0, "max_tokens": 800}
            )

            crawl_config = CrawlerRunConfig(
                extraction_strategy=llm_strategy,
                process_iframes=True,
                remove_overlay_elements=True,
                exclude_external_links=True
            )

            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await crawler.arun(url=url, config=crawl_config)
                return result

        for attempt in range(3):
            try:
                # Try to get from cache first
                cache_key = f"{url}_{hash(instruction)}"
                result = await self.cache.get_or_set(
                    cache_key,
                    self.circuit_breaker.call,
                    _scrape_attempt
                )
                
                if result.success and result.extracted_content:
                    return json.loads(result.extracted_content)
                    
            except Exception as e:
                logger.error(f"Scraping attempt {attempt + 1} failed: {str(e)}")
                
                # Get new proxy for next attempt
                proxy = await self.proxy_manager.get_next_proxy()
                if proxy:
                    logger.info(f"Switching to proxy: {proxy}")
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
                
        raise Exception("All scraping attempts failed")

    async def scrape_multiple_pages(
        self,
        base_url: str,
        instruction: str,
        num_pages: int,
        all_pages: bool
    ) -> List[Dict]:
        """Scrape multiple pages with parallel processing."""
        tasks = []
        for page in range(1, num_pages + 1):
            page_url = (
                f"{base_url}?page={page}"
                if '?' in base_url
                else f"{base_url}/page/{page}"
            )
            tasks.append(self.scrape_page(page_url, instruction))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Page scraping failed: {str(result)}")
            else:
                valid_results.extend(result)

        return valid_results

# ----------------------- Streamlit Interface -----------------------

def main():
    st.title("AI-Assisted Web Scraper Pro")
    
    # Sidebar for advanced options
    with st.sidebar:
        st.header("Advanced Options")
        max_retries = st.slider("Max Retries", 1, 5, 3)
        use_proxies = st.checkbox("Use Proxy Rotation", value=True)
        cache_timeout = st.slider("Cache Timeout (minutes)", 5, 60, 30)

    # Main content
    url_to_scrape = st.text_input("Enter the URL to scrape:")
    instruction_to_llm = st.text_area(
        "Enter instructions for what to scrape:",
        help="Be specific about what content you want to extract."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        num_pages = st.number_input(
            "Number of pages to scrape:",
            min_value=1,
            max_value=10,
            value=1
        )
    with col2:
        all_pages = st.checkbox("Scrape all available pages")

    if st.button("Start Scraping", type="primary"):
        if not url_to_scrape or not instruction_to_llm:
            st.warning("Please enter both URL and instructions.")
            return
            
        if not validate_url(url_to_scrape):
            st.error("Please enter a valid HTTP or HTTPS URL.")
            return

        try:
            # Initialize progress
            progress = st.progress(0)
            status = st.empty()
            
            # Initialize scraper
            scraper = WebScraper()
            
            # Run scraping operation
            with st.spinner("Scraping in progress..."):
                data = asyncio.run(
                    scraper.scrape_multiple_pages(
                        url_to_scrape,
                        instruction_to_llm,
                        num_pages,
                        all_pages
                    )
                )

                if data:
                    # Format data for download
                    formatted_data = json.dumps(data, indent=2)
                    
                    # Create download buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "Download as JSON",
                            formatted_data,
                            "scraped_data.json",
                            mime="application/json"
                        )
                    with col2:
                        st.download_button(
                            "Download as TXT",
                            "\n".join(item['text'] for item in data),
                            "scraped_data.txt"
                        )

                    # Show summary
                    st.success(f"Successfully scraped {len(data)} items!")
                    
                    # Display sample of results
                    with st.expander("View Sample Results"):
                        st.json(data[:5] if len(data) > 5 else data)
                    
                    # Display statistics
                    st.subheader("Scraping Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Items", len(data))
                    with col2:
                        avg_length = sum(len(item['text']) for item in data) / len(data)
                        st.metric("Avg Content Length", f"{avg_length:.0f} chars")
                    with col3:
                        st.metric("Pages Scraped", num_pages)

                else:
                    st.warning("No data was scraped. This could be due to:")
                    st.info("• The website might be blocking automated access")
                    st.info("• The content might not match your instructions")
                    st.info("• The URL might be incorrect or inaccessible")
                    st.info("• The website might require JavaScript")
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please try:")
            st.info("• Checking your internet connection")
            st.info("• Verifying the URL is accessible")
            st.info("• Adjusting your scraping instructions")
            st.info("• Reducing the number of pages")
            logger.error(f"Scraping failed: {str(e)}", exc_info=True)

# ----------------------- Monitoring & Analytics -----------------------

class ScrapingMetrics:
    def __init__(self):
        self.success_count = 0
        self.failure_count = 0
        self.total_time = 0
        self.start_time = None
        self._lock = asyncio.Lock()

    async def record_start(self):
        async with self._lock:
            self.start_time = time.time()

    async def record_success(self, duration: float):
        async with self._lock:
            self.success_count += 1
            self.total_time += duration

    async def record_failure(self, duration: float):
        async with self._lock:
            self.failure_count += 1
            self.total_time += duration

    def get_metrics(self) -> Dict[str, float]:
        total_requests = self.success_count + self.failure_count
        return {
            'success_rate': self.success_count / total_requests if total_requests > 0 else 0,
            'average_time': self.total_time / total_requests if total_requests > 0 else 0,
            'total_requests': total_requests,
            'uptime': time.time() - self.start_time if self.start_time else 0
        }

# ----------------------- Health Check -----------------------

async def health_check() -> Dict[str, bool]:
    """Perform health check of critical components."""
    health_status = {
        'playwright_status': False,
        'proxy_status': False,
        'api_status': False,
        'memory_status': False
    }
    
    try:
        # Check Playwright
        if await setup_playwright():
            health_status['playwright_status'] = True
            
        # Check proxy availability
        proxy_manager = ProxyManager()
        if await proxy_manager.get_next_proxy():
            health_status['proxy_status'] = True
            
        # Check API status (assuming OpenAI)
        try:
            async with get_session() as session:
                async with session.get('https://api.openai.com/v1/models') as response:
                    if response.status == 200:
                        health_status['api_status'] = True
        except Exception:
            pass
            
        # Check memory usage
        import psutil
        if psutil.virtual_memory().percent < 90:  # Less than 90% used
            health_status['memory_status'] = True
            
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        
    return health_status

# ----------------------- Main Entry Point -----------------------

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("scraper.log"),
            logging.StreamHandler()
        ]
    )
    
    # Initialize metrics
    metrics = ScrapingMetrics()
    
    # Run health check
    asyncio.run(health_check())
    
    # Start the application
    try:
        main()
    except Exception as e:
        logger.critical(f"Application failed to start: {str(e)}", exc_info=True)
        st.error("Application failed to start. Please check the logs.")

