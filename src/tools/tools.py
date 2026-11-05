import os
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import tool
from readability.readability import Document
from tavily import TavilyClient
from trafilatura import extract

load_dotenv()

_MAX_SEARCH_RESULTS = 5
_SEARCH_TIMEOUT = 30
_REQUEST_TIMEOUT = 15
_MAX_RESPONSE_BYTES = 2_000_000
_MAX_EXTRACTED_CHARS = 12_000
_MIN_USEFUL_CHARS = 80

_USER_AGENTS = (
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0"
    ),
)


def _headers_for(url: str, attempt: int = 0) -> dict[str, str]:
    """Browser-like headers. A second attempt uses a Google referer."""
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    referer = "https://www.google.com/" if attempt else f"{origin}/"
    fetch_site = "cross-site" if attempt else "none"

    return {
        "User-Agent": _USER_AGENTS[min(attempt, len(_USER_AGENTS) - 1)],
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,image/apng,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "max-age=0",
        "Connection": "keep-alive",
        "Referer": referer,
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": fetch_site,
        "Sec-Fetch-User": "?1",
        "Sec-CH-UA": '"Chromium";v="131", "Not_A Brand";v="24"',
        "Sec-CH-UA-Mobile": "?0",
        "Sec-CH-UA-Platform": '"Windows"',
    }

_client: TavilyClient | None = None


def _get_tavily_client() -> TavilyClient:
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set. Add it to your .env file.")

    _client = TavilyClient(api_key=api_key)
    return _client


def _clean_text(text: str) -> str:
    """Collapse whitespace and drop blank lines from extracted text."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?m)^#{1,6}\s*", "", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            lines.append(line)

    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return cleaned.strip()


def _truncate(text: str, limit: int = _MAX_EXTRACTED_CHARS) -> str:
    if len(text) <= limit:
        return text

    cut = text[:limit].rstrip()
    last_break = max(cut.rfind("\n"), cut.rfind(" "))
    if last_break > int(limit * 0.8):
        cut = cut[:last_break].rstrip()
    return cut + "\n\n[Content truncated]"


def _validate_url(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "Error: provide a full http or https URL."
    return None


def _extract_main_text(html: str, url: str) -> str:
    """Prefer article text, then fall back to a readability summary."""
    try:
        extracted = extract(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
            include_links=False,
            favor_precision=True,
            deduplicate=True,
        )
    except Exception:
        extracted = None

    if extracted and len(extracted.strip()) >= _MIN_USEFUL_CHARS:
        return _clean_text(extracted)

    try:
        summary_html = Document(html).summary(html_partial=True)
    except Exception:
        summary_html = html

    soup = BeautifulSoup(summary_html, "lxml")
    for tag in soup(["script", "style", "noscript", "iframe", "svg", "form"]):
        tag.decompose()

    return _clean_text(soup.get_text(separator="\n"))


def _format_search_results(query: str, response: dict) -> str:
    results = response.get("results") or []
    answer = _clean_text(str(response.get("answer") or ""))

    if not results and not answer:
        return f"No results found for: {query}"

    sections = [f"Query: {query}"]
    if answer:
        sections.append(f"Answer:\n{answer}")

    blocks = []
    for index, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue

        title = (item.get("title") or "Untitled").strip()
        url = (item.get("url") or "").strip() or "No URL"
        snippet = _clean_text(str(item.get("content") or "")) or "No snippet available."
        score = item.get("score")
        score_line = f"\nScore: {score:.2f}" if isinstance(score, (int, float)) else ""
        blocks.append(f"{index}. {title}\nURL: {url}{score_line}\n{snippet}")

    if blocks:
        sections.append("Results:\n" + "\n\n".join(blocks))

    return _truncate("\n\n".join(sections), limit=8_000)


def _read_response(response: requests.Response, url: str) -> tuple[str, str] | str:
    content_type = (response.headers.get("Content-Type") or "").lower()
    if content_type and not any(
        kind in content_type for kind in ("html", "xml", "text/plain")
    ):
        return (
            f"Error: {url} is not an HTML page "
            f"(content type: {content_type.split(';')[0].strip()})."
        )

    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=65_536):
        if not chunk:
            continue
        total += len(chunk)
        if total > _MAX_RESPONSE_BYTES:
            break
        chunks.append(chunk)

    raw = b"".join(chunks)
    encoding = response.encoding or "utf-8"
    try:
        html = raw.decode(encoding, errors="replace")
    except LookupError:
        html = raw.decode("utf-8", errors="replace")

    if not html.strip():
        return f"Error: {url} returned an empty page."

    return response.url or url, html


def _fetch_page(url: str) -> tuple[str, str] | str:
    """Return (final_url, html) or an error string.

    A 403 is retried once with a second browser profile and a Google referer.
    """
    last_status: int | str | None = None

    for attempt in range(2):
        try:
            with requests.get(
                url,
                headers=_headers_for(url, attempt),
                timeout=_REQUEST_TIMEOUT,
                allow_redirects=True,
                stream=True,
            ) as response:
                if response.status_code in {403, 429} and attempt == 0:
                    last_status = response.status_code
                    continue
                response.raise_for_status()
                return _read_response(response, url)
        except requests.Timeout:
            return f"Error: timed out while fetching {url}."
        except requests.TooManyRedirects:
            return f"Error: too many redirects for {url}."
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            last_status = status
            if status in {403, 429} and attempt == 0:
                continue
            return f"Error: failed to fetch {url} (HTTP {status})."
        except requests.ConnectionError:
            return f"Error: could not connect to {url}. Check the address and try again."
        except requests.RequestException:
            return f"Error: could not fetch {url}."

    return f"Error: failed to fetch {url} (HTTP {last_status or 403})."


def _scrape(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return "Error: URL is empty."

    invalid = _validate_url(url)
    if invalid:
        return invalid

    fetched = _fetch_page(url)
    if isinstance(fetched, str):
        return fetched

    final_url, html = fetched
    try:
        text = _extract_main_text(html, final_url)
    except Exception as exc:
        return f"Error: failed to extract text from {url} ({type(exc).__name__}: {exc})."

    if len(text) < _MIN_USEFUL_CHARS:
        return f"Error: no readable article text found at {url}."

    header = f"Source: {final_url}"
    if final_url != url:
        header += f"\nRequested: {url}"
    return _truncate(f"{header}\n\n{text}")


@tool
def tavily_search(query: str) -> str:
    """Search the web and return a short answer plus ranked sources.

    Use this for current facts or to find pages worth reading.
    Pass a specific search query, not a full conversation.
    """
    query = (query or "").strip()
    if not query:
        return "Error: search query is empty."
    if len(query) > 400:
        return "Error: search query is too long. Keep it under 400 characters."

    try:
        client = _get_tavily_client()
        response = client.search(
            query,
            max_results=_MAX_SEARCH_RESULTS,
            search_depth="basic",
            include_answer=True,
            include_raw_content=False,
            timeout=_SEARCH_TIMEOUT,
        )
    except TimeoutError:
        return "Error: Tavily search timed out. Try a shorter query."
    except Exception as exc:
        return f"Error: Tavily search failed ({type(exc).__name__}: {exc})."

    if not isinstance(response, dict):
        return "Error: Tavily returned an unexpected response."

    return _format_search_results(query, response)


@tool
def scrape_website(url: str) -> str:
    """Fetch a page and return its cleaned main text.

    Removes scripts, navigation, and boilerplate. Use this after search
    when you need the article body from a specific URL.
    """
    return _scrape(url)


@tool
def extract_text(url: str) -> str:
    """Fetch a page and return its cleaned main text."""
    return _scrape(url)


tools = [tavily_search, scrape_website, extract_text]