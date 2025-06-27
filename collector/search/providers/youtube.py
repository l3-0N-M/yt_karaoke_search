"""YouTube search provider using yt-dlp."""

import asyncio
import logging
import random
import time
from typing import Dict, List

try:
    import yt_dlp  # type: ignore
except ImportError:
    yt_dlp = None

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, wait_random  # type: ignore
except ImportError:

    def retry(*dargs, **dkwargs):
        def _decorator(fn):
            return fn
        return _decorator

    def stop_after_attempt(*args, **kwargs):
        return None

    def wait_exponential(*args, **kwargs):
        return None

    def wait_random(*args, **kwargs):
        return None

from .base import SearchProvider, SearchResult
from ...config import ScrapingConfig

logger = logging.getLogger(__name__)


class YouTubeSearchProvider(SearchProvider):
    """YouTube search provider using yt-dlp with intelligent query expansion."""

    def __init__(self, scraping_config: ScrapingConfig):
        super().__init__()
        self.scraping_config = scraping_config
        self.yt_dlp_opts = self._setup_yt_dlp()

        # Rate limiting to prevent API blocking
        self._request_semaphore = asyncio.Semaphore(2)
        self._last_request_time = 0
        self._min_request_interval = 2.0
        self._request_count = 0
        self._rate_limit_window_start = time.time()
        self._max_requests_per_hour = 360
        self._backoff_factor = 1.0
        self._max_backoff = 60.0

    def _setup_yt_dlp(self) -> Dict:
        """Setup yt-dlp configuration."""
        return {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "ignoreerrors": True,
            "socket_timeout": self.scraping_config.timeout_seconds,
        }

    async def is_available(self) -> bool:
        """Check if YouTube search is available."""
        return yt_dlp is not None

    def get_provider_weight(self) -> float:
        """YouTube is the primary provider with highest weight."""
        return 1.0

    async def _rate_limited_request(self, request_func, *args, **kwargs):
        """Apply rate limiting to prevent YouTube API blocking."""
        async with self._request_semaphore:
            current_time = time.time()

            # Check hourly rate limit
            if current_time - self._rate_limit_window_start > 3600:
                self._request_count = 0
                self._rate_limit_window_start = current_time
                self._backoff_factor = 1.0

            if self._request_count >= self._max_requests_per_hour:
                wait_time = 3600 - (current_time - self._rate_limit_window_start)
                self.logger.warning(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                await asyncio.sleep(wait_time)
                self._request_count = 0
                self._rate_limit_window_start = time.time()

            # Apply minimum interval between requests with exponential backoff
            elapsed = current_time - self._last_request_time
            required_interval = self._min_request_interval * self._backoff_factor

            if elapsed < required_interval:
                sleep_time = required_interval - elapsed
                self.logger.debug(f"Rate limiting: sleeping {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)

            try:
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, request_func, *args, **kwargs),
                    timeout=300.0,
                )

                self._backoff_factor = max(1.0, self._backoff_factor * 0.9)
                self._request_count += 1
                self._last_request_time = time.time()

                return result

            except asyncio.TimeoutError:
                self._backoff_factor = min(self._max_backoff, self._backoff_factor * 2.0)
                self.logger.error("Request timed out after 5 minutes, increasing backoff")
                raise
            except Exception as e:
                self._backoff_factor = min(self._max_backoff, self._backoff_factor * 1.5)
                self.logger.warning(
                    f"Request failed, increasing backoff to {self._backoff_factor:.2f}x: {e}"
                )
                raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10) + wait_random(0, 2),
    )
    async def search_videos(self, query: str, max_results: int = 100) -> List[SearchResult]:
        """Search for videos using yt-dlp with error handling."""
        start_time = time.time()
        search_query = f"ytsearch{max_results}:{query}"

        try:
            self.yt_dlp_opts["user_agent"] = random.choice(self.scraping_config.user_agents)

            result = await self._rate_limited_request(self._execute_search, search_query)
            results = self._process_search_results(result, query)
            
            # Filter for karaoke content
            filtered_results = self.filter_karaoke_content(results)
            
            response_time = time.time() - start_time
            self.update_statistics(True, len(filtered_results), response_time)
            
            self.logger.info(f"Found {len(filtered_results)} karaoke videos for query: '{query}'")
            return filtered_results
            
        except Exception as e:
            response_time = time.time() - start_time
            self.update_statistics(False, 0, response_time)
            self.logger.error(f"Search failed for '{query}': {e}")
            return []

    def _execute_search(self, search_query: str) -> Dict:
        """Execute yt-dlp search in a thread-safe way."""
        if yt_dlp is None:
            raise RuntimeError("yt-dlp not available")
        with yt_dlp.YoutubeDL(self.yt_dlp_opts) as ydl:
            info = ydl.extract_info(search_query, download=False)
            return info or {}

    def _process_search_results(self, search_results: Dict, original_query: str) -> List[SearchResult]:
        """Process and normalize search results."""
        results = []

        for entry in search_results.get("entries", []):
            if not entry or not entry.get("id"):
                continue

            # Create SearchResult using base class normalization
            result_data = {
                "video_id": entry["id"],
                "url": f"https://www.youtube.com/watch?v={entry['id']}",
                "title": entry.get("title", ""),
                "channel": entry.get("uploader", ""),
                "channel_id": entry.get("uploader_id", ""),
                "duration": entry.get("duration"),
                "view_count": entry.get("view_count", 0),
                "upload_date": entry.get("upload_date"),
                "search_method": "yt_dlp",
                "relevance_score": self._calculate_relevance_score(
                    entry.get("title", "").lower(), original_query
                ),
            }

            result = self.normalize_result(result_data, original_query)
            results.append(result)

        # Sort by relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results

    def _calculate_relevance_score(self, title: str, query: str) -> float:
        """Calculate relevance score for ranking results."""
        title_lower = title.lower()
        query_lower = query.lower()

        score = 0.0

        # Exact query match
        if query_lower in title_lower:
            score += 1.0

        # Individual term matching
        query_terms = query_lower.split()
        if query_terms:
            matching_terms = sum(1 for term in query_terms if term in title_lower)
            score += (matching_terms / len(query_terms)) * 0.5

        # Quality indicators
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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def extract_channel_info(self, channel_url: str) -> Dict:
        """Extract channel information and metadata."""
        try:
            self.yt_dlp_opts["user_agent"] = random.choice(self.scraping_config.user_agents)
            result = await self._rate_limited_request(self._execute_channel_extraction, channel_url)

            if not result:
                return {}

            channel_data = {
                "channel_id": result.get("id"),
                "channel_url": channel_url,
                "channel_name": result.get("title", ""),
                "description": result.get("description", ""),
                "subscriber_count": result.get("subscriber_count", 0),
                "video_count": result.get("video_count", 0),
                "is_karaoke_focused": self._detect_karaoke_channel(result),
            }

            self.logger.info(f"Extracted channel info: {channel_data['channel_name']}")
            return channel_data

        except Exception as e:
            self.logger.error(f"Failed to extract channel info from {channel_url}: {e}")
            return {}

    def _execute_channel_extraction(self, channel_url: str) -> Dict:
        """Execute yt-dlp channel info extraction."""
        if yt_dlp is None:
            raise RuntimeError("yt-dlp not available")

        opts = self.yt_dlp_opts.copy()
        opts["extract_flat"] = True
        opts["playlistend"] = 0

        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(channel_url, download=False)
            return info or {}

    def _detect_karaoke_channel(self, channel_data: Dict) -> bool:
        """Detect if a channel is focused on karaoke content."""
        channel_name = channel_data.get("title", "").lower()
        description = channel_data.get("description", "").lower()

        karaoke_indicators = [
            "karaoke", "backing track", "instrumental", "sing along",
            "minus one", "playback", "accompaniment", "lyrics",
        ]

        text_to_check = f"{channel_name} {description}"
        indicator_count = sum(1 for indicator in karaoke_indicators if indicator in text_to_check)

        return indicator_count >= 2 or "karaoke" in channel_name