import asyncio
from lxml import html
import httpx

async def web_fetch(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        response = await client.get(url, timeout=15.0)
        response.raise_for_status()
    
    try:
        tree = html.fromstring(response.content)
        # Remove script and style elements
        for bad in tree.xpath("//script|//style|//nav|//header|//footer"):
            bad.drop_tree()
        text = tree.text_content()
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text[:50000]
    except Exception as e:
        return f"Error extracting text: {e}"

async def main():
    print("Testing web_fetch...")
    try:
        content = await web_fetch("https://en.wikipedia.org/wiki/List_of_missions_to_Mars")
        print(f"Content length: {len(content)}")
        print(content[:200])
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
