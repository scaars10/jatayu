# Jatayu

## SearXNG Search Integration

This repository now includes a production-oriented SearXNG integration under
`jatayu/search/searxng`.

### What it provides

- Typed configuration with environment loading
- Typed `httpx` client with retries, timeouts, and defensive parsing
- Result normalization into a stable agent-facing schema
- Optional in-memory TTL cache
- High-level `search_web(...)` and `search_web_brief(...)` helpers
- Unit tests built with `unittest` and `httpx.MockTransport`

### SearXNG requirements

The integration expects a self-hosted SearXNG instance with JSON output enabled
in `settings.yml`. SearXNG exposes `search.formats`, and `json` must be present.

Example:

```yaml
search:
  formats:
    - html
    - json
```

### Environment variables

Supported environment variables:

```env
SEARXNG_BASE_URL=http://localhost:8080
SEARXNG_TIMEOUT_SECONDS=10
SEARXNG_MAX_RESULTS=8
SEARXNG_SAFE_SEARCH=1
SEARXNG_LANGUAGE=en
SEARXNG_CATEGORIES=general,news
SEARXNG_ENGINES=duckduckgo,bing
SEARXNG_USER_AGENT=jatayu-searxng/1.0
SEARXNG_RETRY_COUNT=2
SEARXNG_BACKOFF_SECONDS=0.5
SEARXNG_VERIFY_SSL=true
SEARXNG_CACHE_TTL_SECONDS=60
SEARXNG_ENABLED=true
JATAYU_AGENT_WEB_SEARCH_ENABLED=true
```

`JATAYU_SEARXNG_*` prefixed variants are also accepted.
`JATAYU_AGENT_WEB_SEARCH_ENABLED=true` enables ChatAgent web search.

### Usage

```python
from search.searxng import SearxngConfig, SearxngSearchTool

config = SearxngConfig.from_env()
tool = SearxngSearchTool.from_config(config)

result = tool.search_web("latest mars mission", max_results=5)
for item in result.results:
    print(item.title, item.url)
```

Brief mode:

```python
from search.searxng import search_web_brief

brief = search_web_brief("python typing PEP 695")
print(brief.summary)
```

### ChatAgent web search

When `JATAYU_AGENT_WEB_SEARCH_ENABLED=true`, `ChatAgent` exposes a Gemini
function-calling tool named `search_web` backed by SearXNG. The model can call
it when it needs current or externally verifiable information, and it will use
the returned results to finish the reply.

### Docker notes

SearXNG's official docs include container-based installation guidance, and the
`searxng/searxng-docker` repository is a common self-hosting path for Docker
deployments. Configure `settings.yml`, make sure `search.formats` includes
`json`, then point `SEARXNG_BASE_URL` at the instance root URL.

Useful references:

- https://docs.searxng.org/admin/settings/settings_search.html
- https://github.com/searxng/searxng-docker
