"""Search engine implementation using yt-dlp for fast video discovery."""

import asyncio
import logging
import random
from typing import Dict, List

try:
    import yt_dlp  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yt_dlp = None

try:
    from tenacity import retry, stop_after_attempt, wait_exponential  # type: ignore
except ImportError:  # pragma: no cover - optional dependency

    def retry(*dargs, **dkwargs):
        def _decorator(fn):
            return fn

        return _decorator

    def stop_after_attempt(*args, **kwargs):
        return None

    def wait_exponential(*args, **kwargs):
        return None


from .config import ScrapingConfig, SearchConfig

logger = logging.getLogger(__name__)


class SearchEngine:
    """Search engine using yt-dlp with intelligent query expansion."""

    def __init__(self, search_config: SearchConfig, scraping_config: ScrapingConfig):
        self.search_config = search_config
        self.scraping_config = scraping_config
        self.yt_dlp_opts = self._setup_yt_dlp()

    def _setup_yt_dlp(self) -> Dict:
        """Called once; we'll still override UA right before each query."""
        return {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "ignoreerrors": True,
            "socket_timeout": self.scraping_config.timeout_seconds,
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def search_videos(self, query: str, max_results: int = 100) -> List[Dict]:
        """Search for videos using yt-dlp with error handling."""
        search_query = f"ytsearch{max_results}:{query}"

        try:
            # Pick a fresh UA for each call
            self.yt_dlp_opts["user_agent"] = random.choice(self.scraping_config.user_agents)

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._execute_search, search_query)
            videos = self._process_search_results(result, query)
            logger.info(f"Found {len(videos)} videos for query: '{query}'")
            return videos
        except Exception as e:
            logger.error(f"Search failed for '{query}': {e}")
            return []

    def _execute_search(self, search_query: str) -> Dict:
        """Execute yt-dlp search in a thread-safe way."""
        with yt_dlp.YoutubeDL(self.yt_dlp_opts) as ydl:
            return ydl.extract_info(search_query, download=False)

    def _process_search_results(self, search_results: Dict, original_query: str) -> List[Dict]:
        """Process and normalize search results."""
        videos = []

        for entry in search_results.get("entries", []):
            if not entry or not entry.get("id"):
                continue

            title = entry.get("title", "").lower()
            if self._is_likely_karaoke(title):
                # Add duration heuristic filter
                dur = entry.get("duration") or 0
                if not (45 <= dur <= 900):  # 45s - 15min window
                    continue

                video_data = {
                    "video_id": entry["id"],
                    "url": f"https://www.youtube.com/watch?v={entry['id']}",
                    "title": entry.get("title", ""),
                    "channel": entry.get("uploader", ""),
                    "channel_id": entry.get("uploader_id", ""),
                    "duration": entry.get("duration"),
                    "view_count": entry.get("view_count", 0),
                    "upload_date": entry.get("upload_date"),
                    "search_method": "yt_dlp",
                    "search_query": original_query,
                    "relevance_score": self._calculate_relevance_score(title, original_query),
                }
                videos.append(video_data)

        videos.sort(key=lambda x: x["relevance_score"], reverse=True)
        return videos

    def _is_likely_karaoke(self, title: str) -> bool:
        """Filter to identify likely karaoke content."""
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
        """Calculate relevance score for ranking results."""
        title_lower = title.lower()
        query_lower = query.lower()

        score = 0.0

        if query_lower in title_lower:
            score += 1.0

        query_terms = query_lower.split()
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
