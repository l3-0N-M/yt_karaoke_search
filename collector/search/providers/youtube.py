"""YouTube search provider using yt-dlp."""

import asyncio
import logging
import random
import time
from datetime import datetime
from typing import Dict, List, Optional

try:
    import yt_dlp  # type: ignore
except ImportError:
    yt_dlp = None

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, wait_random  # type: ignore
except ImportError:

    class _DummyWait:
        def __add__(self, other):
            return self

    def retry(*dargs, **dkwargs):
        def _decorator(fn):
            return fn

        return _decorator

    def stop_after_attempt(*args, **kwargs):
        return None

    def wait_exponential(*args, **kwargs):
        return _DummyWait()

    def wait_random(*args, **kwargs):
        return _DummyWait()


from ...config import ScrapingConfig
from .base import SearchProvider, SearchResult

logger = logging.getLogger(__name__)


class YouTubeSearchProvider(SearchProvider):
    """YouTube search provider using yt-dlp with intelligent query expansion."""

    def __init__(self, scraping_config: ScrapingConfig):
        super().__init__()
        self.scraping_config = scraping_config
        self.yt_dlp_opts = self._setup_yt_dlp()

        # Rate limiting to prevent API blocking
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
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

    def _process_search_results(
        self, search_results: Dict, original_query: str
    ) -> List[SearchResult]:
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
            "karaoke",
            "backing track",
            "instrumental",
            "sing along",
            "minus one",
            "playback",
            "accompaniment",
            "lyrics",
        ]

        text_to_check = f"{channel_name} {description}"
        indicator_count = sum(1 for indicator in karaoke_indicators if indicator in text_to_check)

        return indicator_count >= 2 or "karaoke" in channel_name

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def extract_channel_videos(
        self, channel_url: str, max_videos: Optional[int] = None, after_date: Optional[str] = None
    ) -> List[SearchResult]:
        """Extract videos from a channel with basic filtering."""
        try:
            self.yt_dlp_opts["user_agent"] = random.choice(self.scraping_config.user_agents)

            channel_opts = self.yt_dlp_opts.copy()
            channel_opts["extract_flat"] = True
            if max_videos:
                channel_opts["playlistend"] = max_videos

            result = await self._rate_limited_request(
                self._execute_channel_videos_extraction, channel_url, channel_opts
            )

            if not result or "entries" not in result:
                logger.warning(f"No entries found in channel extraction for {channel_url}")
                return []

            results = []
            channel_id = result.get("id", "")
            channel_name = result.get("title", "")
            
            total_entries = len(result["entries"]) if result["entries"] else 0
            logger.info(f"Found {total_entries} total entries in channel {channel_name}")

            filtered_counts = {
                "no_id": 0,
                "duration_filtered": 0,
                "date_filtered": 0,
                "processed": 0
            }

            for entry in result["entries"]:
                if not entry or not entry.get("id"):
                    filtered_counts["no_id"] += 1
                    continue

                title = entry.get("title", "")
                duration = entry.get("duration")
                try:
                    duration = int(duration) if duration is not None else 0
                except (ValueError, TypeError):
                    duration = 0

                # Apply channel-specific duration filters
                duration_limits = self._get_channel_duration_limits(channel_name.lower())
                min_duration, max_duration = duration_limits
                
                if not (min_duration <= duration <= max_duration):
                    filtered_counts["duration_filtered"] += 1
                    # Log detailed info about filtered videos for debugging
                    logger.info(f"Duration filtered: {title[:50]}... - Duration: {duration}s ({duration//60}:{duration%60:02d}) - Channel: {channel_name} - Limits: {min_duration}-{max_duration}s")
                    continue

                upload_date = entry.get("upload_date")
                if after_date and not self._is_video_after_date(upload_date, after_date):
                    filtered_counts["date_filtered"] += 1
                    continue

                result_data = {
                    "video_id": entry["id"],
                    "url": f"https://www.youtube.com/watch?v={entry['id']}",
                    "title": title,
                    "channel": channel_name,
                    "channel_id": channel_id,
                    "duration": duration,
                    "view_count": entry.get("view_count", 0),
                    "upload_date": upload_date,
                    "search_method": "channel_extraction",
                    "relevance_score": self._calculate_channel_video_score(title),
                }
                results.append(self.normalize_result(result_data, channel_url))
                filtered_counts["processed"] += 1

            # Log filtering statistics
            logger.info(f"Channel {channel_name} filtering stats: "
                       f"total={total_entries}, "
                       f"no_id={filtered_counts['no_id']}, "
                       f"duration_filtered={filtered_counts['duration_filtered']}, "
                       f"date_filtered={filtered_counts['date_filtered']}, "
                       f"processed={filtered_counts['processed']}")
            
            self.logger.info(f"Extracted {len(results)} videos from channel: {channel_name}")
            return results
        except Exception as e:
            self.logger.error(f"Failed to extract videos from channel {channel_url}: {e}")
            return []

    def _execute_channel_videos_extraction(self, channel_url: str, opts: Dict) -> Dict:
        """Execute yt-dlp channel videos extraction."""
        if yt_dlp is None:
            raise RuntimeError("yt-dlp not available")

        try:
            # Try the main channel URL first
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(channel_url, download=False)
                if info is None:
                    logger.warning(f"yt-dlp returned None for channel {channel_url}")
                    return {}
                
                # Check if we got playlist sections instead of videos
                entries = info.get('entries', [])
                if entries and len(entries) <= 5:
                    # Check if these are channel sections (playlists/shorts/live)
                    section_indicators = ['videos', 'live', 'shorts', 'playlists', 'home']
                    entries_are_sections = all(
                        any(indicator in str(entry.get('title', '')).lower() for indicator in section_indicators)
                        or entry.get('duration') == 0
                        for entry in entries if entry
                    )
                    
                    if entries_are_sections:
                        logger.info(f"Channel {channel_url} returned sections, trying /videos tab")
                        # Try the videos tab specifically
                        videos_url = channel_url + "/videos"
                        try:
                            videos_info = ydl.extract_info(videos_url, download=False)
                            if videos_info and videos_info.get('entries'):
                                logger.info(f"Successfully extracted from /videos tab: {len(videos_info.get('entries', []))} entries")
                                return videos_info
                        except Exception as e:
                            logger.warning(f"Failed to extract from /videos tab: {e}")
                
                logger.debug(f"yt-dlp extracted info for {channel_url}: {len(entries)} entries")
                return info
        except yt_dlp.DownloadError as e:
            logger.error(f"yt-dlp download error for channel {channel_url}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error in yt-dlp extraction for {channel_url}: {e}")
            return {}

    def _calculate_channel_video_score(self, title: str) -> float:
        """Simpler relevance scoring for channel videos."""
        title_lower = title.lower()
        score = 0.5

        karaoke_indicators = ["karaoke", "backing track", "instrumental", "sing along"]
        for indicator in karaoke_indicators:
            if indicator in title_lower:
                score += 0.3
                break

        quality_indicators = {
            "hd": 0.1,
            "4k": 0.2,
            "high quality": 0.1,
            "with lyrics": 0.2,
            "guide vocals": 0.1,
        }

        for indicator, weight in quality_indicators.items():
            if indicator in title_lower:
                score += weight

        return min(score, 2.0)

    def _get_channel_duration_limits(self, channel_name_lower: str) -> tuple[int, int]:
        """Get duration limits based on channel characteristics."""
        # Channel-specific duration limits (min_seconds, max_seconds)
        channel_patterns = {
            # High-quality karaoke channels often have longer content
            "sing king": (10, 2400),  # More relaxed for Sing King - up to 40 minutes
            "sing karaoke": (10, 2400),  # More relaxed for Sing Karaoke
            "karafun": (30, 1800),  # KaraFun typically 30s-30min
            "zzang": (15, 1500),  # ZZang KARAOKE - standard range
            "let's sing": (15, 1500),  # Let's Sing - standard range
            
            # Default patterns for unrecognized channels
            "karaoke": (15, 1800),  # General karaoke channels
            "backing": (30, 2400),  # Backing track channels
            "instrumental": (30, 2400),  # Instrumental channels
        }
        
        # Check for specific channel matches first
        for pattern, limits in channel_patterns.items():
            if pattern in channel_name_lower:
                return limits
        
        # Default duration limits for unknown channels
        return (15, 1500)  # 15 seconds to 25 minutes

    def _parse_upload_date(self, upload_date: str) -> Optional[datetime]:
        """Parse upload date string from yt-dlp."""
        if not upload_date:
            return None

        try:
            if len(upload_date) == 8 and upload_date.isdigit():
                return datetime.strptime(upload_date, "%Y%m%d")
            if len(upload_date) == 10 and upload_date.count("-") == 2:
                return datetime.strptime(upload_date, "%Y-%m-%d")
            self.logger.warning(f"Unknown upload date format: {upload_date}")
            return None
        except ValueError as e:
            self.logger.warning(f"Failed to parse upload date '{upload_date}': {e}")
            return None

    def _is_video_after_date(self, upload_date: str, after_date: str) -> bool:
        """Check if the video was uploaded after the specified date."""
        if not after_date or not upload_date:
            return True

        video_date = self._parse_upload_date(upload_date)
        cutoff_date = self._parse_upload_date(after_date.replace("-", "")[:8])

        if not video_date or not cutoff_date:
            return True

        return video_date > cutoff_date
