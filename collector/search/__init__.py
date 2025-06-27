"""Search module for karaoke video discovery."""

# Import original SearchEngine from the parent search.py for backward compatibility
import sys
import os

# Add parent directory to path temporarily to import the original SearchEngine
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from search import SearchEngine
except ImportError:
    # If that fails, try direct import
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("search", os.path.join(os.path.dirname(__file__), "..", "search.py"))
        search_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(search_module)
        SearchEngine = search_module.SearchEngine
    except Exception:
        # Final fallback - create a wrapper
        from .providers.youtube import YouTubeSearchProvider
        from .providers.base import SearchResult
        
        class SearchEngine:
            """Backward compatibility wrapper for original SearchEngine."""
            def __init__(self, search_config, scraping_config):
                self.search_config = search_config
                self.scraping_config = scraping_config
                self.youtube_provider = YouTubeSearchProvider(scraping_config)
                
            async def search_videos(self, query: str, max_results: int = 100) -> list:
                """Search for videos using YouTube provider."""
                results = await self.youtube_provider.search_videos(query, max_results)
                # Convert SearchResult objects to dict format for backward compatibility
                return [
                    {
                        "video_id": r.video_id,
                        "url": r.url,
                        "title": r.title,
                        "channel": r.channel,
                        "channel_id": r.channel_id,
                        "duration": r.duration,
                        "view_count": r.view_count,
                        "upload_date": r.upload_date,
                        "search_method": r.search_method,
                        "search_query": r.search_query,
                        "relevance_score": r.relevance_score,
                    }
                    for r in results
                ]
                
            async def extract_channel_info(self, channel_url: str) -> dict:
                """Extract channel information."""
                return await self.youtube_provider.extract_channel_info(channel_url)
                
            async def extract_channel_videos(self, channel_url: str, max_videos=None, after_date=None) -> list:
                """Extract videos from a channel."""
                results = await self.youtube_provider.extract_channel_videos(channel_url, max_videos, after_date)
                # Convert SearchResult objects to dict format for backward compatibility
                return [
                    {
                        "video_id": r.video_id,
                        "url": r.url,
                        "title": r.title,
                        "channel": r.channel,
                        "channel_id": r.channel_id,
                        "duration": r.duration,
                        "view_count": r.view_count,
                        "upload_date": r.upload_date,
                        "search_method": r.search_method,
                        "search_query": r.search_query,
                        "relevance_score": r.relevance_score,
                    }
                    for r in results
                ]

# Clean up sys.path
if parent_dir in sys.path:
    sys.path.remove(parent_dir)

from .providers.base import SearchProvider, SearchResult
from .providers.youtube import YouTubeSearchProvider
from .fuzzy_matcher import FuzzyMatcher
from .result_ranker import ResultRanker
from .cache_manager import CacheManager

__all__ = [
    "SearchEngine",  # Original for backward compatibility
    "SearchProvider",
    "SearchResult", 
    "YouTubeSearchProvider",
    "FuzzyMatcher",
    "ResultRanker",
    "CacheManager",
]

# Enhanced search engine is available as collector.enhanced_search.MultiStrategySearchEngine