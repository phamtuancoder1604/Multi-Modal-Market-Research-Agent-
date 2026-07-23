import asyncio
from bs4 import BeautifulSoup

from playwright.async_api import async_playwright

from ddgs import DDGS

def web_search_tool(query: str) -> list:
    """
    Accepts a query string.
    Uses DDGS API to retrieve top 10 relevant URLs.
    """
    print(f"Searching keyword: '{query}' on DuckDuckGo...")
    urls = []
    try:
        with DDGS() as ddgs:
            # Updated to use the text method of the new ddgs library
            results = ddgs.text(query, max_results=10)
            for result in results:
                urls.append(result['href'])
    except Exception as e:
        print(f"Search failed: {str(e)}")
    return urls

async def web_scraper_tool(url: str) -> str:
    """
    Accepts a URL.
    Uses headless Playwright with evasion parameters and BeautifulSoup to extract clean text.
    """
    print(f"Scraping data from: {url} ...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-features=IsolateOrigins,site-per-process",
                "--blink-settings=imagesEnabled=false"
            ]
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080}
        )
        page = await context.new_page()
        
        await page.add_init_script("Object.defineProperty(navigator, 'webdriver', { get: () => undefined });")
        
        try:
            await page.goto(url, timeout=30000, wait_until="networkidle")
            await page.wait_for_timeout(2000)
            content = await page.content()
            
            soup = BeautifulSoup(content, 'html.parser')
            for element in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
                element.extract()
                
            clean_text = soup.body.get_text(separator="\n") if soup.body else soup.get_text(separator="\n")
            lines = (line.strip() for line in clean_text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            return "\n".join(chunk for chunk in chunks if chunk)
            
        except Exception as e:
            return f"Error processing URL {url}: {str(e)}"
        finally:
            await context.close()
            await browser.close()

if __name__ == "__main__":
    test_query = "Vietnam electric vehicle market trends 2026"
    found_urls = web_search_tool(test_query)
    print(f"Found {len(found_urls)} target links.")
    
    if found_urls:
        first_url = found_urls[0]
        print(f"Testing execution on: {first_url}")
        scraped_content = asyncio.run(web_scraper_tool(first_url))
        
        print("Clean text sample output:")
        print(scraped_content[:1000])