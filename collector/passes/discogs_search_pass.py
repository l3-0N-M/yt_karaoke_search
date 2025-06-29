"""Discogs search pass - API lookup for music metadata as MusicBrainz fallback."""

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..advanced_parser import AdvancedTitleParser, ParseResult
from ..discogs_monitor import DiscogsMonitor
from ..utils import DiscogsRateLimiter
from .base import ParsingPass, PassType

logger = logging.getLogger(__name__)

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    aiohttp = None
    HAS_AIOHTTP = False


@dataclass
class DiscogsMatch:
    """A Discogs search result match."""
    
    release_id: str
    master_id: Optional[str]
    artist_name: str
    song_title: str
    year: Optional[int]
    genres: List[str]
    styles: List[str]
    label: Optional[str]
    country: Optional[str]
    format: Optional[str]
    confidence: float
    metadata: Dict = field(default_factory=dict)


class DiscogsClient:
    """Async Discogs API client with rate limiting."""
    
    BASE_URL = "https://api.discogs.com"
    
    def __init__(self, token: str, rate_limiter: DiscogsRateLimiter, user_agent: str, monitor=None):
        self.token = token
        self.rate_limiter = rate_limiter
        self.monitor = monitor
        self.headers = {
            "User-Agent": user_agent,
            "Authorization": f"Discogs token={token}"
        }
        self.logger = logging.getLogger(__name__)
    
    async def search_release(
        self, 
        artist: str, 
        track: str, 
        max_results: int = 10,
        timeout: int = 10
    ) -> List[DiscogsMatch]:
        """Search for releases on Discogs."""
        
        if not HAS_AIOHTTP:
            self.logger.warning("aiohttp not available for Discogs API")
            return []
        
        # Record rate limiting
        wait_start = time.time()
        await self.rate_limiter.wait_for_request()
        wait_time = (time.time() - wait_start) * 1000  # Convert to ms
        
        if self.monitor:
            self.monitor.record_rate_limiting(wait_time)
        
        params = {
            "artist": artist,
            "track": track,
            "type": "release",
            "per_page": max_results
        }
        
        api_start_time = time.time()
        success = False
        timeout_occurred = False
        
        try:
            async with aiohttp.ClientSession(
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as session:
                async with session.get(
                    f"{self.BASE_URL}/database/search",
                    params=params
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    results = []
                    for item in data.get("results", []):
                        match = self._parse_search_result(item, artist, track)
                        if match:
                            results.append(match)
                    
                    success = True
                    return results
                    
        except asyncio.TimeoutError:
            timeout_occurred = True
            self.logger.warning(f"Discogs search timeout for {artist} - {track}")
            return []
        except aiohttp.ClientError as e:
            self.logger.warning(f"Discogs API error: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected Discogs API error: {e}")
            return []
        finally:
            # Record API call metrics
            response_time = (time.time() - api_start_time) * 1000  # Convert to ms
            if self.monitor:
                self.monitor.record_api_call(
                    success=success,
                    response_time_ms=response_time,
                    timeout=timeout_occurred
                )
    
    def _parse_search_result(self, item: Dict, query_artist: str, query_track: str) -> Optional[DiscogsMatch]:
        """Parse a single search result from Discogs API."""
        
        try:
            # Extract basic info
            artist_name = item.get("artist", "Unknown")
            title = item.get("title", "Unknown")
            release_id = str(item.get("id", ""))
            master_id = str(item.get("master_id", "")) if item.get("master_id") else None
            
            # Extract metadata
            year = item.get("year")
            genres = item.get("genre", [])
            styles = item.get("style", [])
            label = None
            if item.get("label"):
                label = item["label"][0] if isinstance(item["label"], list) else item["label"]
            
            country = item.get("country")
            format_info = None
            if item.get("format"):
                format_info = item["format"][0] if isinstance(item["format"], list) else item["format"]
            
            # Calculate confidence based on match quality
            confidence = self._calculate_confidence(
                item, artist_name, title, query_artist, query_track
            )
            
            return DiscogsMatch(
                release_id=release_id,
                master_id=master_id,
                artist_name=artist_name,
                song_title=title,
                year=year,
                genres=genres if isinstance(genres, list) else [genres] if genres else [],
                styles=styles if isinstance(styles, list) else [styles] if styles else [],
                label=label,
                country=country,
                format=format_info,
                confidence=confidence,
                metadata={
                    "discogs_url": f"https://www.discogs.com/release/{release_id}",
                    "community": item.get("community", {}),
                    "barcode": item.get("barcode"),
                    "catno": item.get("catno")
                }
            )
            
        except Exception as e:
            logger.warning(f"Error parsing Discogs result: {e}")
            return None
    
    def _calculate_confidence(
        self, 
        item: Dict, 
        result_artist: str, 
        result_title: str,
        query_artist: str, 
        query_track: str
    ) -> float:
        """Calculate confidence score for a Discogs match."""
        
        confidence = 0.4  # Base confidence
        
        # Artist name similarity
        artist_similarity = self._text_similarity(query_artist.lower(), result_artist.lower())
        confidence += artist_similarity * 0.3
        
        # Track title similarity  
        title_similarity = self._text_similarity(query_track.lower(), result_title.lower())
        confidence += title_similarity * 0.3
        
        # Bonus factors
        if item.get("master_id"):
            confidence += 0.1
        if item.get("year"):
            confidence += 0.05
        if item.get("genre"):
            confidence += 0.05
        if item.get("style"):
            confidence += 0.05
        
        # Community popularity (more owned = more reliable)
        community = item.get("community", {})
        have_count = community.get("have", 0)
        if have_count > 100:
            confidence += 0.1
        elif have_count > 10:
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity calculation."""
        if not text1 or not text2:
            return 0.0
        
        # Remove common variations
        text1 = re.sub(r'\s*\([^)]*\)', '', text1).strip()
        text2 = re.sub(r'\s*\([^)]*\)', '', text2).strip()
        
        # Simple word overlap calculation
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0


class DiscogsSearchPass(ParsingPass):
    """Discogs search pass for metadata enrichment."""
    
    def __init__(self, advanced_parser: AdvancedTitleParser, config, db_manager=None):
        self.advanced_parser = advanced_parser
        self.db_manager = db_manager
        self.config = config.data_sources
        
        # Get Discogs token from environment - try both variable names
        self.token = os.getenv("DISCOGS_TOKEN") or os.getenv("DISCOGS-TOKEN")
        
        # Initialize components
        self.rate_limiter = DiscogsRateLimiter(
            requests_per_minute=self.config.discogs_requests_per_minute
        )
        
        if self.token:
            self.client = DiscogsClient(
                token=self.token,
                rate_limiter=self.rate_limiter,
                user_agent=self.config.discogs_user_agent
            )
        else:
            self.client = None
            logger.warning(
                "Discogs token not found in environment. "
                "Set DISCOGS_TOKEN or DISCOGS-TOKEN to enable Discogs search."
            )
        
        # Initialize monitoring
        self.monitor = DiscogsMonitor(config)
        
        # Update client with monitor if available
        if self.client:
            self.client.monitor = self.monitor
        
        # Statistics
        self.stats = {
            "total_searches": 0,
            "successful_matches": 0,
            "api_errors": 0,
            "high_confidence_matches": 0,
            "fallback_activations": 0,
        }
    
    @property
    def pass_type(self) -> PassType:
        return PassType.DISCOGS_SEARCH
    
    async def parse(
        self,
        title: str,
        description: str = "",
        tags: str = "",
        channel_name: str = "",
        channel_id: str = "",
        metadata: Optional[Dict] = None,
    ) -> Optional[ParseResult]:
        """Execute Discogs search parsing."""
        
        if not self.config.discogs_enabled:
            return None
        
        if not self.client:
            logger.debug("Discogs client not initialized (missing token)")
            return None
        
        # Check if we should use Discogs as fallback
        is_fallback = False
        if self.config.discogs_use_as_fallback and metadata:
            musicbrainz_confidence = metadata.get("musicbrainz_confidence", 0.0)
            if musicbrainz_confidence >= self.config.discogs_min_musicbrainz_confidence:
                logger.debug(
                    f"Skipping Discogs search - MusicBrainz confidence {musicbrainz_confidence} "
                    f">= threshold {self.config.discogs_min_musicbrainz_confidence}"
                )
                return None
            is_fallback = True
            self.stats["fallback_activations"] += 1
        
        start_time = time.time()
        self.stats["total_searches"] += 1
        
        try:
            # Parse title to extract artist and song
            title_candidates = self._extract_search_candidates(title)
            if not title_candidates:
                logger.debug(f"No search candidates found for title: {title}")
                return None
            
            best_match = None
            best_confidence = 0.0
            
            # Try each candidate
            for artist, track in title_candidates:
                logger.debug(f"Searching Discogs for: {artist} - {track}")
                
                matches = await self.client.search_release(
                    artist=artist,
                    track=track,
                    max_results=self.config.discogs_max_results_per_search,
                    timeout=self.config.discogs_timeout
                )
                
                for match in matches:
                    if match.confidence > best_confidence:
                        best_match = match
                        best_confidence = match.confidence
                
                # Stop if we found a high-confidence match
                if best_confidence >= 0.8:
                    break
            
            if not best_match or best_confidence < self.config.discogs_confidence_threshold:
                logger.debug(
                    f"No sufficient Discogs match found. "
                    f"Best confidence: {best_confidence:.2f}, "
                    f"threshold: {self.config.discogs_confidence_threshold}"
                )
                # Record failed search
                self.monitor.record_search_attempt(
                    success=False, 
                    confidence=best_confidence, 
                    fallback=is_fallback
                )
                return None
            
            # Create parse result
            result = ParseResult(
                original_artist=best_match.artist_name,
                song_title=best_match.song_title,
                confidence=best_confidence,
                metadata={
                    "source": "discogs",
                    "discogs_release_id": best_match.release_id,
                    "discogs_master_id": best_match.master_id,
                    "year": best_match.year,
                    "genres": best_match.genres,
                    "styles": best_match.styles,
                    "label": best_match.label,
                    "country": best_match.country,
                    "format": best_match.format,
                    **best_match.metadata
                }
            )
            
            self.stats["successful_matches"] += 1
            if best_confidence >= 0.8:
                self.stats["high_confidence_matches"] += 1
            
            # Record successful search with monitoring
            self.monitor.record_search_attempt(
                success=True,
                confidence=best_confidence,
                fallback=is_fallback
            )
            
            # Record data quality metrics
            self.monitor.record_data_quality(
                has_year=best_match.year is not None,
                has_genres=bool(best_match.genres),
                has_label=best_match.label is not None
            )
            
            duration = time.time() - start_time
            logger.info(
                f"Discogs search successful: {best_match.artist_name} - {best_match.song_title} "
                f"(confidence: {best_confidence:.2f}, duration: {duration:.2f}s)"
            )
            
            return result
            
        except Exception as e:
            self.stats["api_errors"] += 1
            duration = time.time() - start_time
            logger.error(f"Discogs search error after {duration:.2f}s: {e}")
            return None
    
    def _extract_search_candidates(self, title: str) -> List[tuple]:
        """Extract potential artist/track combinations from title."""
        candidates = []
        
        # Use advanced parser first
        if self.advanced_parser:
            try:
                parse_result = self.advanced_parser.parse_title(title)
                if parse_result and parse_result.original_artist and parse_result.song_title:
                    candidates.append((parse_result.original_artist, parse_result.song_title))
            except Exception as e:
                logger.debug(f"Advanced parser failed: {e}")
        
        # Common karaoke patterns
        patterns = [
            r'^(.+?)\s*-\s*(.+?)(?:\s*\(.*\))?$',  # Artist - Song (optional parentheses)
            r'^(.+?)\s*–\s*(.+?)(?:\s*\[.*\])?$',  # Artist – Song (em dash)
            r'^(.+?)\s*:\s*(.+?)$',                # Artist : Song
            r'^(.+?)\s*\|\s*(.+?)$',               # Artist | Song
        ]
        
        for pattern in patterns:
            match = re.match(pattern, title.strip(), re.IGNORECASE)
            if match:
                artist = match.group(1).strip()
                track = match.group(2).strip()
                
                # Clean up common karaoke suffixes
                track = re.sub(r'\s*(karaoke|instrumental|backing track).*$', '', track, flags=re.IGNORECASE)
                artist = re.sub(r'\s*(karaoke|instrumental|backing track).*$', '', artist, flags=re.IGNORECASE)
                
                if artist and track and len(artist) > 1 and len(track) > 1:
                    candidates.append((artist, track))
        
        return candidates
    
    def get_statistics(self) -> Dict:
        """Return pass statistics."""
        return self.stats.copy()