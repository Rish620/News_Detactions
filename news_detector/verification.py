import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from email.utils import parsedate_to_datetime
from html import unescape
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import requests


SEARCH_TIMEOUT_SECONDS = 8
MAX_QUERY_WORDS = 14
MAX_RESULTS = 5

TRUSTED_NEWS_DOMAINS = {
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "business-standard.com",
    "cnn.com",
    "economictimes.indiatimes.com",
    "hindustantimes.com",
    "indianexpress.com",
    "ndtv.com",
    "reuters.com",
    "thehindu.com",
    "timesofindia.indiatimes.com",
}


@dataclass(frozen=True)
class WebSource:
    title: str
    url: str
    domain: str
    published: str | None
    snippet: str
    trusted: bool


@dataclass(frozen=True)
class VerificationResult:
    status: str
    confidence: float
    query: str
    summary: str
    sources: list[WebSource]

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "sources": [asdict(source) for source in self.sources],
        }


def _clean_text(text: str) -> str:
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _build_query(text: str) -> str:
    cleaned = _clean_text(text)
    words = cleaned.split()
    if len(words) <= MAX_QUERY_WORDS:
        return cleaned
    return " ".join(words[:MAX_QUERY_WORDS])


def _plain_text(value: str | None) -> str:
    if not value:
        return ""
    value = re.sub(r"<[^>]+>", " ", value)
    value = unescape(value)
    return re.sub(r"\s+", " ", value).strip()


def _normalize_domain(url: str) -> str:
    domain = urlparse(url).netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def _canonical_url(url: str) -> str:
    parsed = urlparse(url)
    if "bing.com" not in parsed.netloc.lower():
        return url

    target = parse_qs(parsed.query).get("url", [""])[0]
    return unquote(target) if target else url


def _is_trusted(domain: str) -> bool:
    return domain in TRUSTED_NEWS_DOMAINS or any(
        domain.endswith(f".{trusted}") for trusted in TRUSTED_NEWS_DOMAINS
    )


def _format_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).date().isoformat()
    except (TypeError, ValueError):
        return None


def _search_bing_news(query: str) -> list[WebSource]:
    url = f"https://www.bing.com/news/search?q={quote_plus(query)}&format=rss"
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=SEARCH_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    root = ET.fromstring(response.content)
    sources = []
    for item in root.findall("./channel/item")[:MAX_RESULTS]:
        link = _canonical_url(_plain_text(item.findtext("link")))
        domain = _normalize_domain(link)
        sources.append(
            WebSource(
                title=_plain_text(item.findtext("title")),
                url=link,
                domain=domain,
                published=_format_date(item.findtext("pubDate")),
                snippet=_plain_text(item.findtext("description")),
                trusted=_is_trusted(domain),
            )
        )
    return sources


def verify_with_web(text: str) -> VerificationResult:
    query = _build_query(text)
    if len(query) < 8:
        return VerificationResult(
            status="NOT_CHECKED",
            confidence=0.0,
            query=query,
            summary="Text is too short for a useful internet verification search.",
            sources=[],
        )

    try:
        sources = _search_bing_news(query)
    except requests.RequestException as exc:
        return VerificationResult(
            status="UNAVAILABLE",
            confidence=0.0,
            query=query,
            summary=f"Live web verification is unavailable: {exc.__class__.__name__}.",
            sources=[],
        )
    except ET.ParseError:
        return VerificationResult(
            status="UNAVAILABLE",
            confidence=0.0,
            query=query,
            summary="Live web verification returned an unreadable response.",
            sources=[],
        )

    if not sources:
        return VerificationResult(
            status="NO_EVIDENCE",
            confidence=0.2,
            query=query,
            summary="No matching recent news sources were found online.",
            sources=[],
        )

    trusted_count = sum(1 for source in sources if source.trusted)
    confidence = min(0.95, 0.45 + (0.12 * len(sources)) + (0.08 * trusted_count))

    if trusted_count >= 2:
        status = "SUPPORTED"
        summary = "Multiple trusted news sources appear to cover a similar claim."
    elif trusted_count == 1:
        status = "PARTLY_SUPPORTED"
        summary = "At least one trusted news source appears to cover a similar claim."
    else:
        status = "WEAK_EVIDENCE"
        summary = "Some web results were found, but they are not from the trusted source list."

    return VerificationResult(
        status=status,
        confidence=round(confidence, 4),
        query=query,
        summary=summary,
        sources=sources,
    )
