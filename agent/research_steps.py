from __future__ import annotations
import json
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from agent.gemini_model import gemini_model
from config.env_config import get_env

DEEP_RESEARCH_SYSTEM_PROMPT = """
You are an autonomous Deep Research Assistant.
Your objective is to comprehensively research the given topic and answer any specific questions.
Use the web_search tool to find relevant sources and web_fetch tool to read articles and documents in depth.
Iterate your search queries to gather comprehensive, high-quality, and up-to-date information.
Once you have enough information, synthesize your findings into a detailed, well-structured Markdown report.
Ensure you cite your sources (URLs) within the report.
"""

async def gather_sources(topic: str, specific_questions: list[str]) -> list[str]:
    """Gather sources for a deep research task."""
    
    api_key = get_env("GEMINI_API_KEY", required=True)
    
    research_agent = Agent(
        model=GoogleModel(
            gemini_model.get_large_model(),
            provider=GoogleProvider(api_key=api_key),
        ),
        system_prompt=DEEP_RESEARCH_SYSTEM_PROMPT,
        tools=[web_search],
    )

    prompt = f"Gather a list of relevant URLs for a research on '{topic}'"
    if specific_questions:
        prompt += " with a focus on:\n"
        for q in specific_questions:
            prompt += f"- {q}\n"
    
    prompt += "\nReturn ONLY a JSON list of strings, where each string is a URL. Do not include markdown formatting or objects."

    result = await research_agent.run(prompt, message_history=[])
    
    output = result.output.strip()
    if output.startswith("```json"):
        output = output[7:]
    elif output.startswith("```"):
        output = output[3:]
    if output.endswith("```"):
        output = output[:-3]
    output = output.strip()
    
    try:
        parsed = json.loads(output)
        if isinstance(parsed, list):
            urls = []
            for item in parsed:
                if isinstance(item, str):
                    urls.append(item)
                elif isinstance(item, dict) and "url" in item:
                    urls.append(item["url"])
                elif isinstance(item, dict) and "link" in item:
                    urls.append(item["link"])
            return urls
    except json.JSONDecodeError:
        pass

    return []

async def read_sources(urls: list[str]) -> str:
    """Read the content of a list of URLs and return a concatenated string."""
    import asyncio
    api_key = get_env("GEMINI_API_KEY", required=True)
    
    research_agent = Agent(
        model=GoogleModel(
            gemini_model.get_large_model(),
            provider=GoogleProvider(api_key=api_key),
        ),
        system_prompt=DEEP_RESEARCH_SYSTEM_PROMPT,
        tools=[web_fetch],
    )

    async def process_url(url: str) -> str:
        try:
            if url.endswith(".pdf"):
                text = await read_pdf(url)
                return f"Source: {url}\n{text}\n\n"
            else:
                prompt = f"Read the content of the following URL and extract the relevant information. Summarize the key findings related to the overall research topic.\nURL: {url}"
                result = await research_agent.run(prompt, message_history=[])
                return f"Source: {url}\n{result.output}\n\n"
        except Exception as e:
            return f"Source: {url}\nFailed to read or process: {e}\n\n"

    # Process up to 10 URLs concurrently
    tasks = [process_url(url) for url in urls[:10]]
    results = await asyncio.gather(*tasks)
        
    return "".join(results)

async def synthesize_report(topic: str, specific_questions: list[str], sources_content: str, feedback: str | None = None) -> str:
    """Synthesize a report from the content of the sources."""

    api_key = get_env("GEMINI_API_KEY", required=True)

    research_agent = Agent(
        model=GoogleModel(
            gemini_model.get_large_model(),
            provider=GoogleProvider(api_key=api_key),
        ),
        system_prompt=DEEP_RESEARCH_SYSTEM_PROMPT,
    )

    prompt = f"Synthesize a detailed, well-structured Markdown report on '{topic}'"
    if specific_questions:
        prompt += f" with a focus on:\n"
        for q in specific_questions:
            prompt += f"- {q}\n"

    if feedback:
        prompt += f"\n\nThe user provided the following feedback to guide this report. You MUST incorporate this feedback:\n{feedback}\n"

    prompt += f"\nUse the following content to write the report. Make sure to use inline citations linking back to the source URLs provided in the text:\n{sources_content}"

    result = await research_agent.run(prompt, message_history=[])
    return result.output

async def read_pdf(url: str) -> str:
    """Read the text content of a PDF file from a URL."""
    import httpx
    from pypdf import PdfReader
    import io

    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()

    pdf_file = io.BytesIO(response.content)
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    return text[:200000]

async def web_search(query: str) -> str:
    """Search the web for a given query.
    
    Priority:
    1. Gemini Google Search (Native Grounding)
    2. SearXNG (if enabled)
    3. DuckDuckGo Search (Fallback)
    """
    import asyncio
    from google.genai import types
    from agent.gemini_model import get_client, gemini_model

    # 1. Try Gemini Google Search Grounding
    try:
        client = get_client()
        response = await client.aio.models.generate_content(
            model=gemini_model.get_large_model(),
            contents=query,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            )
        )
        
        results = []
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                results.append(f"Gemini Grounded Answer: {candidate.content.parts[0].text}\n")
            
            if candidate.grounding_metadata and candidate.grounding_metadata.grounding_chunks:
                results.append("Sources:")
                for chunk in candidate.grounding_metadata.grounding_chunks:
                    if chunk.web:
                        results.append(f"- {chunk.web.title} | {chunk.web.uri}")
        
        if results:
            return "\n".join(results)
    except Exception:
        pass

    # 2. Try SearXNG if enabled
    try:
        from search.searxng import SearxngConfig, search_web_brief
        config = SearxngConfig.from_env()
        if config.enabled:
            brief = await asyncio.to_thread(search_web_brief, query)
            if brief.result.results:
                return brief.summary
    except Exception:
        pass

    # 3. Fallback to DuckDuckGo
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            # Note: DDGS.text is also sync
            results = await asyncio.to_thread(lambda: [r for r in ddgs.text(query, max_results=5)])
        return "\n".join([f"{r.get('title', 'No Title')} | {r.get('href', 'No URL')}: {r.get('body', '')}" for r in results])
    except Exception as e:
        return f"Search failed: {e}"

async def web_fetch(url: str) -> str:
    """Fetch the content of a web page and extract its text."""
    import httpx
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
            response = await client.get(url, timeout=15.0)
            response.raise_for_status()
            
        from lxml import html
        tree = html.fromstring(response.content)
        # Remove script, style, nav, header, and footer elements
        for bad in tree.xpath("//script|//style|//nav|//header|//footer"):
            bad.drop_tree()
        text = tree.text_content()
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)
        # Limit to 200000 characters
        return text[:200000]
    except Exception as e:
        return f"Failed to fetch or parse {url}: {e}"
