"""Search module for karaoke video discovery."""

from .cache_manager import CacheManager
from .fuzzy_matcher import FuzzyMatcher
from .providers.base import SearchProvider, SearchResult
from .providers.youtube import YouTubeSearchProvider
from .result_ranker import ResultRanker

__all__ = [
    "SearchProvider",
    "SearchResult",
    "YouTubeSearchProvider",
    "FuzzyMatcher",
    "ResultRanker",
    "CacheManager",
]
