"""Search module for karaoke video discovery."""

from .providers.base import SearchProvider, SearchResult
from .providers.youtube import YouTubeSearchProvider
from .fuzzy_matcher import FuzzyMatcher
from .result_ranker import ResultRanker
from .cache_manager import CacheManager

__all__ = [
    "SearchProvider",
    "SearchResult", 
    "YouTubeSearchProvider",
    "FuzzyMatcher",
    "ResultRanker",
    "CacheManager",
]