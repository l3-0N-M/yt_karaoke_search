"""Bing search provider for video discovery as YouTube fallback."""

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


class BingSearchProvider(SearchProvider):
    """Bing search provider for video discovery."""

    def __init__(self, config=None):
        super().__init__(config)

        if not HAS_REQUESTS:
            logger.warning("requests library not available, BingSearchProvider disabled")

        # Bing search configuration
        self.search_url = "https://www.bing.com/videos/search"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 1.0  # 1 second between requests

    async def is_available(self) -> bool:
        """Check if Bing search is available."""
        return HAS_REQUESTS

    def get_provider_weight(self) -> float:
        """Bing is a fallback provider with medium weight."""
        return 0.6

    async def search_videos(self, query: str, max_results: int = 100) -> List[SearchResult]:
        """Search for videos using Bing Video Search."""
        if not HAS_REQUESTS:
            return []

        start_time = time.time()

        try:
            # Apply rate limiting
            await self._apply_rate_limiting()

            # Enhance query for karaoke content
            enhanced_query = self._enhance_query_for_karaoke(query)

            # Perform search
            raw_results = await self._perform_bing_search(enhanced_query, max_results)

            # Process and normalize results
            results = self._process_bing_results(raw_results, query)

            # Filter for karaoke content
            filtered_results = self.filter_karaoke_content(results)

            response_time = time.time() - start_time
            self.update_statistics(True, len(filtered_results), response_time)

            self.logger.info(
                f"Found {len(filtered_results)} karaoke videos via Bing for query: '{query}'"
            )
            return filtered_results

        except Exception as e:
            response_time = time.time() - start_time
            self.update_statistics(False, 0, response_time)
            self.logger.error(f"Bing search failed for '{query}': {e}")
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
            # Add the most generic karaoke term
            enhanced_query = f"{query} karaoke"
        else:
            enhanced_query = query

        # Add site filter to focus on video platforms
        enhanced_query += " site:youtube.com OR site:dailymotion.com OR site:vimeo.com"

        return enhanced_query

    async def _perform_bing_search(self, query: str, max_results: int) -> List[Dict]:
        """Perform the actual Bing search."""
        import asyncio

        params = {
            "q": query,
            "count": min(max_results, 50),  # Bing's max per request
            "mkt": "en-US",
            "safesearch": "Moderate",
        }

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(  # type: ignore
                    self.search_url, params=params, headers=self.headers, timeout=30
                ),
            )

            response.raise_for_status()

            # Parse results from HTML (simplified approach)
            results = self._parse_bing_html(response.text)
            return results

        except Exception as e:
            self.logger.error(f"Bing API request failed: {e}")
            return []

    def _parse_bing_html(self, html_content: str) -> List[Dict]:
        """Parse video results from Bing search HTML."""
        results = []

        try:
            # Look for video result patterns in the HTML
            # This is a simplified approach - in practice, you'd want more robust parsing

            # Pattern to extract video information from Bing results
            video_pattern = r'href="(https://www\.youtube\.com/watch\?v=([^"]+))"[^>]*>([^<]+)<'

            matches = re.findall(video_pattern, html_content, re.IGNORECASE)

            for match in matches:
                url, video_id, title = match

                # Extract additional information if available
                duration_match = re.search(
                    rf"{re.escape(video_id)}.*?(\d+:\d+)", html_content, re.IGNORECASE
                )
                duration_str = duration_match.group(1) if duration_match else None

                # Convert duration to seconds
                duration = self._parse_duration(duration_str) if duration_str else None

                result = {
                    "video_id": video_id,
                    "url": url,
                    "title": title.strip(),
                    "channel": self._extract_channel_from_html(html_content, video_id),
                    "channel_id": "",
                    "duration": duration,
                    "view_count": 0,  # Not easily extractable from Bing
                    "upload_date": None,  # Not easily extractable from Bing
                    "search_method": "bing_search",
                }

                results.append(result)

        except Exception as e:
            self.logger.error(f"Error parsing Bing HTML: {e}")

        return results[:50]  # Limit results

    def _extract_channel_from_html(self, html_content: str, video_id: str) -> str:
        """Extract channel name from HTML content."""
        try:
            # Look for channel information near the video
            channel_pattern = rf"{re.escape(video_id)}.*?YouTube.*?by\s+([^<\n]+)"
            match = re.search(channel_pattern, html_content, re.IGNORECASE | re.DOTALL)

            if match:
                return match.group(1).strip()

        except Exception:
            pass

        return "Unknown Channel"

    def _parse_duration(self, duration_str: str) -> int:
        """Parse duration string (MM:SS) to seconds."""
        try:
            parts = duration_str.split(":")
            if len(parts) == 2:
                minutes, seconds = map(int, parts)
                return minutes * 60 + seconds
            elif len(parts) == 3:
                hours, minutes, seconds = map(int, parts)
                return hours * 3600 + minutes * 60 + seconds
        except (ValueError, AttributeError):
            pass

        return 0

    def _process_bing_results(
        self, raw_results: List[Dict], original_query: str
    ) -> List[SearchResult]:
        """Process and normalize Bing search results."""
        results = []

        for raw_result in raw_results:
            # Calculate basic relevance score
            relevance_score = self._calculate_bing_relevance_score(
                raw_result.get("title", ""), original_query
            )

            # Add relevance score to raw result
            raw_result["relevance_score"] = relevance_score

            # Normalize to SearchResult
            result = self.normalize_result(raw_result, original_query)
            results.append(result)

        # Sort by relevance
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results

    def _calculate_bing_relevance_score(self, title: str, query: str) -> float:
        """Calculate relevance score for Bing results."""
        if not title or not query:
            return 0.0

        title_lower = title.lower()
        query_lower = query.lower()

        score = 0.0

        # Exact query match
        if query_lower in title_lower:
            score += 0.8

        # Individual term matching
        query_terms = query_lower.split()
        if query_terms:
            matching_terms = sum(1 for term in query_terms if term in title_lower)
            score += (matching_terms / len(query_terms)) * 0.4

        # Karaoke-specific bonuses
        karaoke_indicators = {
            "karaoke": 0.3,
            "instrumental": 0.2,
            "backing track": 0.2,
            "sing along": 0.2,
            "with lyrics": 0.2,
        }

        for indicator, bonus in karaoke_indicators.items():
            if indicator in title_lower:
                score += bonus

        return min(score, 1.5)  # Cap at 1.5 for fallback provider
