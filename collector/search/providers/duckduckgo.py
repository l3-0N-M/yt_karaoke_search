"""DuckDuckGo search provider for video discovery as additional fallback."""

import logging
import re
import time
from typing import Dict, List

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from .base import SearchProvider, SearchResult

logger = logging.getLogger(__name__)


class DuckDuckGoSearchProvider(SearchProvider):
    """DuckDuckGo search provider for video discovery."""

    def __init__(self, config=None):
        super().__init__(config)

        if not HAS_REQUESTS:
            logger.warning("requests library not available, DuckDuckGoSearchProvider disabled")

        # DuckDuckGo configuration
        self.search_url = "https://duckduckgo.com/"
        self.video_search_url = "https://duckduckgo.com/v.js"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 2.0  # 2 seconds between requests (more conservative)

    async def is_available(self) -> bool:
        """Check if DuckDuckGo search is available."""
        return HAS_REQUESTS

    def get_provider_weight(self) -> float:
        """DuckDuckGo is a secondary fallback provider with lower weight."""
        return 0.4

    async def search_videos(self, query: str, max_results: int = 100) -> List[SearchResult]:
        """Search for videos using DuckDuckGo."""
        if not HAS_REQUESTS:
            return []

        start_time = time.time()

        try:
            # Apply rate limiting
            await self._apply_rate_limiting()

            # Enhance query for karaoke content
            enhanced_query = self._enhance_query_for_karaoke(query)

            # Perform search
            raw_results = await self._perform_ddg_search(enhanced_query, max_results)

            # Process and normalize results
            results = self._process_ddg_results(raw_results, query)

            # Filter for karaoke content
            filtered_results = self.filter_karaoke_content(results)

            response_time = time.time() - start_time
            self.update_statistics(True, len(filtered_results), response_time)

            self.logger.info(
                f"Found {len(filtered_results)} karaoke videos via DuckDuckGo for query: '{query}'"
            )
            return filtered_results

        except Exception as e:
            response_time = time.time() - start_time
            self.update_statistics(False, 0, response_time)
            self.logger.error(f"DuckDuckGo search failed for '{query}': {e}")
            return []

    async def _apply_rate_limiting(self):
        """Apply rate limiting to prevent being blocked."""
        current_time = time.time()
        elapsed = current_time - self._last_request_time

        if elapsed < self._min_request_interval:
            import asyncio

            sleep_time = self._min_request_interval - elapsed
            await asyncio.sleep(sleep_time)

        self._last_request_time = time.time()

    def _enhance_query_for_karaoke(self, query: str) -> str:
        """Enhance search query to find karaoke content."""
        # Add karaoke-specific terms if not already present
        karaoke_terms = ["karaoke", "instrumental", "backing track", "sing along"]

        query_lower = query.lower()
        if not any(term in query_lower for term in karaoke_terms):
            enhanced_query = f"{query} karaoke"
        else:
            enhanced_query = query

        # Focus on video platforms
        enhanced_query += " site:youtube.com"

        return enhanced_query

    async def _perform_ddg_search(self, query: str, max_results: int) -> List[Dict]:
        """Perform the actual DuckDuckGo search."""
        import asyncio

        try:
            # First, get the search token
            session = requests.Session()
            session.headers.update(self.headers)

            # Get initial page to extract tokens
            initial_response = await asyncio.get_running_loop().run_in_executor(
                None, lambda: session.get(self.search_url, timeout=10)
            )

            # Extract vqd token (needed for video search)
            vqd_match = re.search(r'vqd=([^&"]+)', initial_response.text)
            if not vqd_match:
                self.logger.warning("Could not extract DuckDuckGo search token")
                return []

            vqd = vqd_match.group(1)

            # Perform video search
            params = {
                "l": "us-en",
                "o": "json",
                "q": query,
                "vqd": vqd,
                "f": ",,,",
                "p": "1",
            }

            video_response = await asyncio.get_running_loop().run_in_executor(
                None, lambda: session.get(self.video_search_url, params=params, timeout=15)
            )

            video_response.raise_for_status()

            # Parse JSON response
            try:
                data = video_response.json()
                results = data.get("results", [])
                return results[:max_results]
            except Exception:
                # Fallback to HTML parsing if JSON fails
                return self._parse_ddg_html(video_response.text)

        except Exception as e:
            self.logger.error(f"DuckDuckGo API request failed: {e}")
            return []

    def _parse_ddg_html(self, html_content: str) -> List[Dict]:
        """Parse video results from DuckDuckGo HTML."""
        results = []

        try:
            # Look for YouTube video links in the HTML
            youtube_pattern = r'href="(https://www\.youtube\.com/watch\?v=([^"&]+))"[^>]*>([^<]+)<'

            matches = re.findall(youtube_pattern, html_content, re.IGNORECASE)

            for match in matches[:20]:  # Limit to first 20 matches
                url, video_id, title = match

                result = {
                    "video_id": video_id,
                    "url": url,
                    "title": title.strip(),
                    "channel": "Unknown Channel",
                    "channel_id": "",
                    "duration": None,
                    "view_count": 0,
                    "upload_date": None,
                    "search_method": "ddg_search",
                }

                results.append(result)

        except Exception as e:
            self.logger.error(f"Error parsing DuckDuckGo HTML: {e}")

        return results

    def _process_ddg_results(
        self, raw_results: List[Dict], original_query: str
    ) -> List[SearchResult]:
        """Process and normalize DuckDuckGo search results."""
        results = []

        for raw_result in raw_results:
            # Handle different response formats
            if isinstance(raw_result, dict):
                # JSON format from video API
                video_url = raw_result.get("content", "")
                title = raw_result.get("title", "")

                # Extract YouTube video ID if it's a YouTube URL
                youtube_match = re.search(r"youtube\.com/watch\?v=([^&]+)", video_url)
                if youtube_match:
                    video_id = youtube_match.group(1)

                    processed_result = {
                        "video_id": video_id,
                        "url": video_url,
                        "title": title,
                        "channel": raw_result.get("uploader", "Unknown Channel"),
                        "channel_id": "",
                        "duration": self._parse_duration(raw_result.get("duration", "")),
                        "view_count": 0,
                        "upload_date": None,
                        "search_method": "ddg_search",
                        "relevance_score": self._calculate_ddg_relevance_score(
                            title, original_query
                        ),
                    }

                    result = self.normalize_result(processed_result, original_query)
                    results.append(result)

        # Sort by relevance
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results

    def _parse_duration(self, duration_str: str) -> int:
        """Parse duration string to seconds."""
        if not duration_str:  # Changed from "if not duration_str:" to "if duration_str is None:"
            return None

        try:
            # Handle various duration formats
            if ":" in duration_str:
                parts = duration_str.split(":")
                if len(parts) == 2:
                    minutes, seconds = map(int, parts)
                    return minutes * 60 + seconds
                elif len(parts) == 3:
                    hours, minutes, seconds = map(int, parts)
                    return hours * 3600 + minutes * 60 + seconds
            else:
                # Assume seconds
                return int(duration_str)
        except (ValueError, AttributeError):
            pass

        return None

    def _calculate_ddg_relevance_score(self, title: str, query: str) -> float:
        """Calculate relevance score for DuckDuckGo results."""
        if not title or not query:
            return 0.0

        title_lower = title.lower()
        query_lower = query.lower()

        score = 0.0

        # Exact query match
        if query_lower in title_lower:
            score += 0.7  # Slightly lower than primary provider

        # Individual term matching
        query_terms = query_lower.split()
        if query_terms:
            matching_terms = sum(1 for term in query_terms if term in title_lower)
            score += (matching_terms / len(query_terms)) * 0.3

        # Karaoke-specific bonuses
        karaoke_indicators = {
            "karaoke": 0.2,
            "instrumental": 0.15,
            "backing track": 0.15,
            "sing along": 0.15,
            "with lyrics": 0.15,
        }

        for indicator, bonus in karaoke_indicators.items():
            if indicator in title_lower:
                score += bonus

        return min(score, 1.2)  # Lower cap for secondary fallback provider
