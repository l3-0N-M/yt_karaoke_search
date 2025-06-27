"""Search providers for different video platforms."""

from .base import SearchProvider, SearchResult
from .youtube import YouTubeSearchProvider

# Optional providers that may not be available
try:
    from .bing import BingSearchProvider

    HAS_BING = True
except ImportError:
    BingSearchProvider = None
    HAS_BING = False

try:
    from .duckduckgo import DuckDuckGoSearchProvider

    HAS_DUCKDUCKGO = True
except ImportError:
    DuckDuckGoSearchProvider = None
    HAS_DUCKDUCKGO = False

__all__ = ["SearchProvider", "SearchResult", "YouTubeSearchProvider"]

if HAS_BING:
    __all__.append("BingSearchProvider")

if HAS_DUCKDUCKGO:
    __all__.append("DuckDuckGoSearchProvider")
