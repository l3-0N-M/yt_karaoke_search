"""Search module for karaoke video discovery."""

# Import SearchEngine implementation from the local module.
# Older layouts expected `search.py` at the package root; we directly import the
# packaged version for reliability.
try:
    from ..search_engine import SearchEngine as SearchEngineImpl
except Exception:
    # Final fallback - minimal wrapper using the YouTube provider only.
    from .providers.base import SearchResult
    from .providers.youtube import YouTubeSearchProvider

    class SearchEngineImpl:
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

        async def extract_channel_videos(
            self, channel_url: str, max_videos=None, after_date=None
        ) -> list:
            """Extract videos from a channel."""
            results = await self.youtube_provider.extract_channel_videos(
                channel_url, max_videos, after_date
            )
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

        def _is_likely_karaoke(self, title: str) -> bool:
            karaoke_indicators = [
                "karaoke",
                "backing track",
                "instrumental",
                "sing along",
                "minus one",
                "playback",
                "accompaniment",
            ]
            has_indicator = any(indicator in title for indicator in karaoke_indicators)
            exclusions = [
                "reaction",
                "review",
                "tutorial",
                "lesson",
                "how to",
                "analysis",
                "behind the scenes",
                "interview",
                "documentary",
            ]
            has_exclusion = any(exclusion in title for exclusion in exclusions)
            return has_indicator and not has_exclusion

        def _calculate_relevance_score(self, title: str, query: str) -> float:
            title_lower = title.lower()
            query_lower = query.lower()
            score = 0.0
            if query_lower in title_lower:
                score += 1.0
            query_terms = query_lower.split()
            if query_terms:
                matching_terms = sum(1 for term in query_terms if term in title_lower)
                score += (matching_terms / len(query_terms)) * 0.5
            quality_indicators = {
                "hd": 0.2,
                "4k": 0.3,
                "high quality": 0.2,
                "studio": 0.2,
                "professional": 0.2,
                "with lyrics": 0.3,
                "guide vocals": 0.2,
            }
            for indicator, weight in quality_indicators.items():
                if indicator in title_lower:
                    score += weight
            return min(score, 2.0)

        def _parse_upload_date(self, upload_date: str):
            from datetime import datetime

            if not upload_date:
                return None
            try:
                if len(upload_date) == 8 and upload_date.isdigit():
                    return datetime.strptime(upload_date, "%Y%m%d")
                elif len(upload_date) == 10 and upload_date.count("-") == 2:
                    return datetime.strptime(upload_date, "%Y-%m-%d")
                return None
            except ValueError:
                return None

        def _is_video_after_date(self, upload_date: str, after_date: str) -> bool:
            if not after_date or not upload_date:
                return True
            video_date = self._parse_upload_date(upload_date)
            cutoff_date = self._parse_upload_date(after_date.replace("-", "")[:8])
            if not video_date or not cutoff_date:
                return True
            return video_date > cutoff_date


from .cache_manager import CacheManager
from .fuzzy_matcher import FuzzyMatcher
from .providers.base import SearchProvider, SearchResult
from .providers.youtube import YouTubeSearchProvider
from .result_ranker import ResultRanker

SearchEngine = SearchEngineImpl

__all__ = [
    "SearchEngine",  # Original for backward compatibility
    "SearchProvider",
    "SearchResult",
    "YouTubeSearchProvider",
    "FuzzyMatcher",
    "ResultRanker",
    "CacheManager",
]

# MultiStrategySearchEngine is available as collector.enhanced_search.MultiStrategySearchEngine
