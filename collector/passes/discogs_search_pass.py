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

import functools


def retry_on_rate_limit(max_retries=3):
    """Decorator to retry on rate limit errors with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            retry_count = 0
            last_exception = None
            
            while retry_count <= max_retries:
                try:
                    return await func(self, *args, **kwargs)
                except Exception as e:
                    if HAS_AIOHTTP and aiohttp and hasattr(aiohttp, 'ClientResponseError') and isinstance(e, aiohttp.ClientResponseError):
                        if e.status == 429:
                            retry_count += 1
                            if retry_count > max_retries:
                                raise
                            
                            # Extract retry-after if available
                            retry_after = None
                            if e.headers:
                                retry_after_str = e.headers.get('Retry-After')
                                if retry_after_str:
                                    try:
                                        retry_after = int(retry_after_str)
                                    except ValueError:
                                        pass
                            
                            if hasattr(self, 'rate_limiter'):
                                self.rate_limiter.handle_429_error(retry_after)
                            
                            # Wait for rate limiter before retrying
                            if hasattr(self, 'rate_limiter'):
                                await self.rate_limiter.wait_for_request()
                            else:
                                # Fallback exponential backoff
                                wait_time = min(60, (2 ** (retry_count - 1)) * 1.0)
                                await asyncio.sleep(wait_time)
                            
                            last_exception = e
                            continue
                        else:
                            raise
                    else:
                        # Don't retry on other exceptions
                        raise
            
            # If we exhausted retries, raise the last exception
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


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
    
    def _normalize_search_query(self, text: str) -> str:
        """Normalize search query for better Discogs matching."""
        # Remove common suffixes that interfere with search
        suffixes_to_remove = [
            r'\s*\(karaoke.*?\)',
            r'\s*\[karaoke.*?\]',
            r'\s*-\s*karaoke.*$',
            r'\s*\(instrumental.*?\)',
            r'\s*\[instrumental.*?\]',
            r'\s*-\s*instrumental.*$',
            r'\s*\(backing track.*?\)',
            r'\s*\[backing track.*?\]',
            r'\s*-\s*backing track.*$',
            r'\s*\(official.*?\)',
            r'\s*\[official.*?\]',
            r'\s*\(lyrics.*?\)',
            r'\s*\[lyrics.*?\]',
            r'\s*\(sing.*?\)',
            r'\s*\[sing.*?\]',
            r'\s*\(cover.*?\)',
            r'\s*\[cover.*?\]',
            r'\s*\(remix.*?\)',
            r'\s*\[remix.*?\]',
            r'\s*\(version.*?\)',
            r'\s*\[version.*?\]',
        ]
        
        normalized = text
        for pattern in suffixes_to_remove:
            normalized = re.sub(pattern, '', normalized, flags=re.IGNORECASE)
        
        # Remove multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized.strip()
    
    def _generate_artist_variations(self, artist: str) -> List[str]:
        """Generate variations of artist name for better matching."""
        variations = [artist]
        
        # Handle special characters
        if '$' in artist:
            variations.append(artist.replace('$', 'S'))
            variations.append(artist.replace('$', ''))
        
        # Handle numbered disambiguations (e.g., "The Click (2)" -> "The Click")
        if re.search(r'\s*\(\d+\)\s*$', artist):
            variations.append(re.sub(r'\s*\(\d+\)\s*$', '', artist))
        
        # Handle parenthetical content (e.g., "PLUTO (72)" -> "PLUTO")
        if '(' in artist and ')' in artist:
            base = re.sub(r'\s*\([^)]*\)', '', artist).strip()
            if base and base != artist:
                variations.append(base)
        
        # Handle & vs and
        if '&' in artist:
            variations.append(artist.replace('&', 'and'))
        elif ' and ' in artist:
            variations.append(artist.replace(' and ', ' & '))
        
        # Handle accented characters
        accent_map = {
            'É': 'E', 'È': 'E', 'Ê': 'E', 'Ë': 'E',
            'À': 'A', 'Á': 'A', 'Â': 'A', 'Ä': 'A',
            'Ò': 'O', 'Ó': 'O', 'Ô': 'O', 'Ö': 'O',
            'Ù': 'U', 'Ú': 'U', 'Û': 'U', 'Ü': 'U',
            'Ç': 'C', 'Ñ': 'N'
        }
        
        normalized = artist
        for accented, plain in accent_map.items():
            if accented in artist:
                normalized = artist.replace(accented, plain)
                if normalized != artist:
                    variations.append(normalized)
        
        # Handle & vs and
        if ' & ' in artist:
            variations.append(artist.replace(' & ', ' and '))
        elif ' and ' in artist:
            variations.append(artist.replace(' and ', ' & '))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for v in variations:
            if v not in seen:
                seen.add(v)
                unique_variations.append(v)
        
        return unique_variations
    
    @retry_on_rate_limit(max_retries=3)
    async def search_release(
        self, 
        artist: str, 
        track: str, 
        max_results: int = 10,
        timeout: int = 10
    ) -> List[DiscogsMatch]:
        """Search for releases on Discogs with enhanced query strategies."""
        
        if not HAS_AIOHTTP:
            self.logger.warning("aiohttp not available for Discogs API")
            return []
        
        # Normalize the track title
        normalized_track = self._normalize_search_query(track)
        
        # Generate artist variations
        artist_variations = self._generate_artist_variations(artist)
        
        # Generate search queries to try
        search_queries = []
        
        # Primary strategy: try each artist variation with normalized track
        for artist_var in artist_variations:
            search_queries.append({
                "artist": artist_var,
                "track": normalized_track,
                "type": "release"
            })
        
        # Secondary strategy: if track has parentheses, try without them
        if '(' in normalized_track and ')' in normalized_track:
            track_without_parens = re.sub(r'\s*\([^)]*\)', '', normalized_track).strip()
            if track_without_parens and track_without_parens != normalized_track:
                for artist_var in artist_variations[:2]:  # Only use top 2 artist variations
                    search_queries.append({
                        "artist": artist_var,
                        "track": track_without_parens,
                        "type": "release"
                    })
        
        # Initialize results list
        all_results = []
        
        # Tertiary strategy: Try searching without year restriction
        # (useful for very new releases that might not be categorized yet)
        if len(search_queries) < 6:  # Limit total queries
            search_queries.append({
                "query": f"{artist} {normalized_track}",
                "type": "release"
            })
        
        # Quaternary strategy: Removed artist-only search as it returns too many unrelated results
        # This was causing issues like matching "ROSÉ - Messy" with soundtrack albums
        
        # Try each query until we get good results
        for query_params in search_queries:
            if len(all_results) >= max_results:
                break
                
            # Record rate limiting
            wait_start = time.time()
            await self.rate_limiter.wait_for_request()
            wait_time = (time.time() - wait_start) * 1000  # Convert to ms
            
            if self.monitor:
                self.monitor.record_rate_limiting(wait_time)
            
            query_params["per_page"] = max_results
            
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
                        params=query_params
                    ) as response:
                        # Update rate limit info from headers
                        if response.headers:
                            remaining = response.headers.get('X-Discogs-Ratelimit-Remaining')
                            reset_time = response.headers.get('X-Discogs-Ratelimit-Reset')
                            if remaining:
                                try:
                                    self.rate_limiter.update_rate_limit_info(int(remaining), None)
                                except ValueError:
                                    pass
                            if reset_time:
                                try:
                                    self.rate_limiter.update_rate_limit_info(None, int(reset_time))
                                except ValueError:
                                    pass
                        
                        response.raise_for_status()
                        data = await response.json()
                        
                        # Record successful request
                        self.rate_limiter.handle_success()
                        
                        results = []
                        for item in data.get("results", []):
                            match = self._parse_search_result(item, artist, track)
                            if match:
                                results.append(match)
                        
                        all_results.extend(results)
                        success = True
                        
                        # If we found high confidence matches, stop searching
                        if any(r.confidence >= 0.8 for r in results):
                            break
                            
            except asyncio.TimeoutError:
                timeout_occurred = True
                self.logger.warning(f"Discogs search timeout for {query_params.get('artist')} - {query_params.get('track')}")
                continue  # Try next query
            except Exception as e:
                if HAS_AIOHTTP and aiohttp and hasattr(aiohttp, 'ClientResponseError') and isinstance(e, aiohttp.ClientResponseError):
                    if e.status == 429:
                        # Handle rate limiting
                        retry_after = None
                        if e.headers:
                            retry_after_str = e.headers.get('Retry-After')
                            if retry_after_str:
                                try:
                                    retry_after = int(retry_after_str)
                                except ValueError:
                                    retry_after = 60  # Default to 60 seconds
                        
                        self.rate_limiter.handle_429_error(retry_after)
                        self.logger.warning(f"Discogs API rate limit hit (429) for {query_params}. Retry after: {retry_after}s")
                        
                        # Don't continue to next query, wait for backoff
                        break
                    else:
                        self.logger.warning(f"Discogs API HTTP error {e.status} for query {query_params}: {e}")
                        continue  # Try next query
                elif HAS_AIOHTTP and aiohttp and hasattr(aiohttp, 'ClientError') and isinstance(e, aiohttp.ClientError):
                    self.logger.warning(f"Discogs API error for query {query_params}: {e}")
                    continue  # Try next query
                else:
                    self.logger.error(f"Unexpected error during Discogs search: {e}")
                    continue  # Try next query
            finally:
                # Record API call metrics
                response_time = (time.time() - api_start_time) * 1000  # Convert to ms
                if self.monitor:
                    self.monitor.record_api_call(
                        success=success,
                        response_time_ms=response_time,
                        timeout=timeout_occurred
                    )
        
        # Remove duplicates and sort by confidence
        seen_releases = set()
        unique_results = []
        
        # First pass: Filter out low-quality matches
        filtered_results = []
        for result in all_results:
            # Skip results with very low confidence
            if result.confidence < 0.3:
                continue
                
            # For popular artists like ROSÉ, be more selective
            # Require higher confidence if we have many results
            if len(all_results) > 10 and result.confidence < 0.5:
                continue
                
            filtered_results.append(result)
        
        # Second pass: Remove duplicates and get best matches
        for result in sorted(filtered_results, key=lambda x: x.confidence, reverse=True):
            release_key = (result.artist_name.lower(), result.song_title.lower())
            if release_key not in seen_releases:
                seen_releases.add(release_key)
                unique_results.append(result)
                
                # For high-match scenarios, stop early if we have good matches
                if len(unique_results) >= 5 and result.confidence >= 0.8:
                    break
                    
                if len(unique_results) >= max_results:
                    break
        
        return unique_results
    
    def _parse_search_result(self, item: Dict, query_artist: str, query_track: str) -> Optional[DiscogsMatch]:
        """Parse a single search result from Discogs API."""
        
        try:
            # Extract basic info
            # Discogs API sometimes returns "Artist - Title" in the title field
            raw_title = item.get("title", "Unknown")
            artist_name = item.get("artist", "")
            
            # If no artist field or artist is generic, try to parse from title
            if not artist_name or artist_name.lower() in ["unknown", "various", "various artists"]:
                # Check if title contains artist - song pattern
                if " - " in raw_title:
                    parts = raw_title.split(" - ", 1)
                    if len(parts) == 2:
                        artist_name = parts[0].strip()
                        title = parts[1].strip()
                    else:
                        title = raw_title
                else:
                    title = raw_title
            else:
                # If we have a proper artist, check if it's duplicated in title
                if raw_title.startswith(artist_name + " - "):
                    title = raw_title[len(artist_name + " - "):]
                else:
                    title = raw_title
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
            
            # Early rejection: Check artist similarity before calculating full confidence
            artist_similarity = self._text_similarity(query_artist.lower(), artist_name.lower(), is_artist=True)
            if artist_similarity < 0.5:
                logger.debug(
                    f"Rejecting result due to low artist similarity: "
                    f"'{query_artist}' vs '{artist_name}' (similarity: {artist_similarity:.2f})"
                )
                return None
            
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
        
        confidence = 0.1  # Lower base confidence to require better matches
        
        # Artist name similarity with stricter matching
        artist_similarity = self._text_similarity(query_artist.lower(), result_artist.lower(), is_artist=True)
        # Penalize non-exact artist matches more heavily
        if artist_similarity < 0.9:
            artist_similarity *= 0.5  # Heavily reduce score for non-exact matches
        if artist_similarity < 0.7:
            artist_similarity *= 0.3  # Even more penalty for poor matches
        confidence += artist_similarity * 0.4  # Increased weight for artist
        
        # Track title similarity  
        title_similarity = self._text_similarity(query_track.lower(), result_title.lower())
        confidence += title_similarity * 0.35
        
        # Only apply bonuses if we have decent core matching
        core_match_score = artist_similarity * 0.4 + title_similarity * 0.35
        if core_match_score >= 0.3:  # Require minimum 30% from artist/title match
            # Bonus factors (reduced when core matching is weak)
            bonus_multiplier = min(1.0, core_match_score * 2)  # Scale bonuses based on core match
            
            if item.get("master_id"):
                confidence += 0.05 * bonus_multiplier
            if item.get("year"):
                year = item.get("year")
                # Special handling for recent releases - they might not have much community data
                if year and int(year) >= 2024:
                    confidence += 0.15 * bonus_multiplier  # Higher bonus for very recent releases
                elif year and int(year) >= 2023:
                    confidence += 0.1 * bonus_multiplier  # Bonus for recent releases
                else:
                    confidence += 0.05 * bonus_multiplier
            if item.get("genre"):
                confidence += 0.05 * bonus_multiplier
            if item.get("style"):
                confidence += 0.03 * bonus_multiplier
        
        # Community popularity (more owned = more reliable)
        # But don't penalize new releases too much
        if core_match_score >= 0.3:  # Only apply community bonuses with decent core match
            community = item.get("community", {})
            have_count = community.get("have", 0)
            want_count = community.get("want", 0)
            year = item.get("year", 0)
            
            # Adjust expectations for recent releases
            if year and int(year) >= 2024:
                # Very new releases - any community data is good
                if have_count > 5 or want_count > 10:
                    confidence += 0.15 * bonus_multiplier
                elif have_count > 0 or want_count > 0:
                    confidence += 0.1 * bonus_multiplier
            elif year and int(year) >= 2023:
                if have_count > 10:
                    confidence += 0.1 * bonus_multiplier
                elif have_count > 0:
                    confidence += 0.05 * bonus_multiplier
            else:
                if have_count > 100:
                    confidence += 0.1 * bonus_multiplier
                elif have_count > 10:
                    confidence += 0.05 * bonus_multiplier
        
        # Special boost for exact matches
        if artist_similarity >= 0.95 and title_similarity >= 0.95:
            confidence = min(confidence + 0.2, 1.0)
        
        # Penalty for compilation albums or various artists
        if result_artist.lower() in ['various', 'various artists', 'compilation']:
            confidence *= 0.5
        
        # Penalty for results that look like karaoke versions
        karaoke_indicators = ['karaoke', 'instrumental', 'backing track', 'sing along']
        result_text = f"{result_artist} {result_title}".lower()
        if any(indicator in result_text for indicator in karaoke_indicators):
            confidence *= 0.7  # We want original recordings, not karaoke versions
        
        return min(confidence, 1.0)
    
    def _text_similarity(self, text1: str, text2: str, is_artist: bool = False) -> float:
        """Enhanced text similarity using multiple strategies.
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            is_artist: If True, use stricter matching suitable for artist names
        """
        if not text1 or not text2:
            return 0.0
        
        # Normalize texts
        text1_normalized = text1.lower().strip()
        text2_normalized = text2.lower().strip()
        
        # Strategy 1: Exact match (highest confidence)
        if text1_normalized == text2_normalized:
            return 1.0
        
        # Strategy 2: Levenshtein distance-based similarity
        def levenshtein_distance(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        # Calculate normalized edit distance
        max_len = max(len(text1_normalized), len(text2_normalized))
        if max_len > 0:
            edit_distance = levenshtein_distance(text1_normalized, text2_normalized)
            edit_similarity = 1.0 - (edit_distance / max_len)
        else:
            edit_similarity = 0.0
        
        # Strategy 3: Token-based similarity (for handling word order differences)
        # Remove common variations and parentheses
        text1_clean = re.sub(r'\s*\([^)]*\)', '', text1_normalized).strip()
        text2_clean = re.sub(r'\s*\([^)]*\)', '', text2_normalized).strip()
        
        # Tokenize
        words1 = set(text1_clean.split())
        words2 = set(text2_clean.split())
        
        if words1 and words2:
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            token_similarity = intersection / union if union > 0 else 0.0
        else:
            token_similarity = 0.0
        
        # Strategy 4: Substring matching (for partial matches)
        substring_similarity = 0.0
        if len(text1_clean) >= 3 and len(text2_clean) >= 3:
            if text1_clean in text2_clean or text2_clean in text1_clean:
                substring_similarity = min(len(text1_clean), len(text2_clean)) / max(len(text1_clean), len(text2_clean))
        
        # Combine strategies with weights
        # Edit distance is most reliable for similar strings
        # Token similarity helps with word reordering
        # Substring helps with partial matches
        
        if is_artist:
            # For artist names, be much stricter
            # Require all words from query to be present in result (or very similar)
            query_words = set(text1_clean.split())
            result_words = set(text2_clean.split())
            
            # Check if all query words are present (or very close)
            words_found = 0
            for q_word in query_words:
                # Check for exact match or very close match
                found = False
                for r_word in result_words:
                    word_sim = 1.0 - (levenshtein_distance(q_word, r_word) / max(len(q_word), len(r_word)))
                    if word_sim >= 0.8:  # 80% similarity for individual words
                        found = True
                        break
                if found:
                    words_found += 1
            
            word_coverage = words_found / len(query_words) if query_words else 0
            
            # For artists, heavily weight exact/near-exact matches
            combined_similarity = (
                edit_similarity * 0.5 +      # Overall string similarity
                word_coverage * 0.4 +         # All words must be present
                token_similarity * 0.1        # Minor weight for token overlap
            )
        else:
            # Original weights for non-artist text
            combined_similarity = (
                edit_similarity * 0.6 +
                token_similarity * 0.25 +
                substring_similarity * 0.15
            )
        
        return min(combined_similarity, 1.0)


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
        
        # Failed search cache to avoid repeated API calls for known failures
        self._failed_cache = {}  # key: (artist, track) -> timestamp
        self._cache_duration = 3600  # Cache failures for 1 hour
        
        # Statistics
        self.stats = {
            "total_searches": 0,
            "successful_matches": 0,
            "api_errors": 0,
            "high_confidence_matches": 0,
            "fallback_activations": 0,
            "cache_hits": 0,
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
        
        # Check if we should use Discogs as fallback or for enrichment
        is_fallback = False
        is_enrichment = metadata and metadata.get("enrichment_mode", False)
        
        if is_enrichment:
            # Enrichment mode: always run to supplement MusicBrainz data
            logger.debug("Running Discogs in enrichment mode to supplement MusicBrainz data")
            self.stats["fallback_activations"] += 1  # Reuse this stat for now
        elif self.config.discogs_use_as_fallback and metadata:
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
            # In enrichment mode, prefer existing parsed data for search candidates
            if is_enrichment and metadata.get("musicbrainz_result"):
                mb_result = metadata["musicbrainz_result"]
                if mb_result.artist and mb_result.song_title:
                    title_candidates = [(mb_result.artist, mb_result.song_title)]
                    logger.info(f"Using MusicBrainz data for Discogs search: {mb_result.artist} - {mb_result.song_title}")
                else:
                    title_candidates = self._extract_search_candidates(title)
            elif metadata.get("parsed_artist") and metadata.get("parsed_title"):
                # Use parsed artist/title from Channel Template for more accurate searches
                title_candidates = [(metadata["parsed_artist"], metadata["parsed_title"])]
                logger.info(f"Using parsed data for Discogs search: {metadata['parsed_artist']} - {metadata['parsed_title']}")
            elif metadata.get("channel_template_result"):
                # Fallback to Channel Template result
                ct_result = metadata["channel_template_result"]
                if ct_result.artist and ct_result.song_title:
                    title_candidates = [(ct_result.artist, ct_result.song_title)]
                    logger.info(f"Using Channel Template data for Discogs search: {ct_result.artist} - {ct_result.song_title}")
                else:
                    title_candidates = self._extract_search_candidates(title)
            else:
                # Parse title to extract artist and song
                title_candidates = self._extract_search_candidates(title)
                
            if not title_candidates:
                logger.info(f"No search candidates found for title: {title}")
                return None
            
            best_match = None
            best_confidence = 0.0
            rate_limited = False
            
            # Try each candidate
            for artist, track in title_candidates:
                # Check cache first
                cache_key = (artist.lower(), track.lower())
                if cache_key in self._failed_cache:
                    cache_time = self._failed_cache[cache_key]
                    if time.time() - cache_time < self._cache_duration:
                        logger.debug(f"Skipping cached failed search: {artist} - {track}")
                        self.stats["cache_hits"] += 1
                        continue
                    else:
                        # Cache expired, remove it
                        del self._failed_cache[cache_key]
                
                logger.info(f"Searching Discogs for: {artist} - {track}")
                
                try:
                    matches = await self.client.search_release(
                        artist=artist,
                        track=track,
                        max_results=self.config.discogs_max_results_per_search,
                        timeout=self.config.discogs_timeout
                    )
                except Exception as e:
                    # Check if it was a rate limit error from the client
                    if "rate limit" in str(e).lower() or "429" in str(e):
                        rate_limited = True
                        logger.warning(f"Rate limited during search for {artist} - {track}")
                        break
                    else:
                        logger.error(f"Error searching Discogs: {e}")
                        continue
                
                logger.info(f"Discogs returned {len(matches)} matches for {artist} - {track}")
                
                # Log match details for debugging
                if matches:
                    logger.debug("Top 3 matches:")
                    for i, match in enumerate(matches[:3]):
                        logger.debug(
                            f"  {i+1}. {match.artist_name} - {match.song_title} "
                            f"(confidence: {match.confidence:.2f}, year: {match.year})"
                        )
                
                # Special handling for many matches (e.g., popular artists)
                if len(matches) > 10:
                    logger.info(f"Many matches found ({len(matches)}), applying stricter filtering")
                    # Filter to only high-confidence matches
                    high_conf_matches = [m for m in matches if m.confidence >= 0.6]
                    if high_conf_matches:
                        matches = high_conf_matches
                        logger.info(f"Filtered to {len(matches)} high-confidence matches")
                
                for match in matches:
                    if match.confidence > best_confidence:
                        best_match = match
                        best_confidence = match.confidence
                
                # Stop if we found a high-confidence match
                if best_confidence >= 0.8:
                    break
            
            if not best_match or best_confidence < self.config.discogs_confidence_threshold:
                # Don't cache as failed if we were rate limited
                if rate_limited:
                    logger.info(
                        f"Discogs search incomplete due to rate limiting. "
                        f"Best confidence so far: {best_confidence:.2f}"
                    )
                else:
                    logger.info(
                        f"No sufficient Discogs match found. "
                        f"Best confidence: {best_confidence:.2f}, "
                        f"threshold: {self.config.discogs_confidence_threshold}"
                    )
                    
                    # Cache failed searches to avoid repeated API calls
                    for artist, track in title_candidates:
                        cache_key = (artist.lower(), track.lower())
                        self._failed_cache[cache_key] = time.time()
                
                # Record failed search
                self.monitor.record_search_attempt(
                    success=False, 
                    confidence=best_confidence, 
                    fallback=is_fallback
                )
                return None
            
            # Create parse result
            discogs_metadata = {
                "source": "discogs",
                "discogs_release_id": best_match.release_id,
                "discogs_master_id": best_match.master_id,
                "year": best_match.year,
                "release_year": best_match.year,  # Add both field names
                "genres": best_match.genres,
                "genre": best_match.genres[0] if best_match.genres else None,  # Use first genre as string
                "styles": best_match.styles,
                "label": best_match.label,
                "country": best_match.country,
                "format": best_match.format,
                **best_match.metadata
            }
            
            logger.info(f"Discogs metadata for {best_match.artist_name} - {best_match.song_title}: "
                       f"year={best_match.year}, genres={best_match.genres}, confidence={best_confidence:.2f}")
            
            result = ParseResult(
                artist=best_match.artist_name,
                song_title=best_match.song_title,
                confidence=best_confidence,
                metadata=discogs_metadata
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
                    # Normalize the parsed results
                    normalized_artist = self._normalize_search_query(parse_result.original_artist)
                    normalized_title = self._normalize_search_query(parse_result.song_title)
                    candidates.append((normalized_artist, normalized_title))
            except Exception as e:
                logger.debug(f"Advanced parser failed: {e}")
        
        # Common karaoke patterns
        patterns = [
            r'^(.+?)\s*-\s*(.+?)(?:\s*\(.*\))?$',  # Artist - Song (optional parentheses)
            r'^(.+?)\s*–\s*(.+?)(?:\s*\[.*\])?$',  # Artist – Song (em dash)
            r'^(.+?)\s*—\s*(.+?)(?:\s*\[.*\])?$',  # Artist — Song (long dash)
            r'^(.+?)\s*:\s*(.+?)$',                # Artist : Song
            r'^(.+?)\s*\|\s*(.+?)$',               # Artist | Song
            r'^(.+?)\s*/\s*(.+?)$',                # Artist / Song
        ]
        
        for pattern in patterns:
            match = re.match(pattern, title.strip(), re.IGNORECASE)
            if match:
                artist = match.group(1).strip()
                track = match.group(2).strip()
                
                # Clean up common karaoke suffixes
                track = re.sub(r'\s*(karaoke|instrumental|backing track|official|video|audio|hd|4k|lyrics?).*$', '', track, flags=re.IGNORECASE)
                artist = re.sub(r'\s*(karaoke|instrumental|backing track|official|video|audio|hd|4k).*$', '', artist, flags=re.IGNORECASE)
                
                # Handle "feat." or "ft." in artist name
                if ' feat.' in artist.lower() or ' ft.' in artist.lower():
                    # Try both with and without featured artist
                    main_artist = re.split(r'\s+(?:feat\.|ft\.)', artist, flags=re.IGNORECASE)[0].strip()
                    if main_artist and track and len(main_artist) > 1 and len(track) > 1:
                        candidates.append((main_artist, track))
                
                # Also add the full artist name
                if artist and track and len(artist) > 1 and len(track) > 1:
                    candidates.append((artist, track))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                unique_candidates.append(candidate)
        
        return unique_candidates
    
    def get_statistics(self) -> Dict:
        """Return pass statistics."""
        return self.stats.copy()