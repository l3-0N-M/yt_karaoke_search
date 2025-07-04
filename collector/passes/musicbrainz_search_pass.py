"""Pass 1: Dedicated MusicBrainz search with direct API lookup and fuzzy matching."""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..advanced_parser import AdvancedTitleParser, ParseResult
from .base import ParsingPass, PassType

logger = logging.getLogger(__name__)

try:
    from urllib.parse import urlencode

    HAS_URLLIB = True
except ImportError:
    urlencode = None  # type: ignore
    HAS_URLLIB = False

try:
    import aiohttp

    HAS_AIOHTTP = True
except ImportError:
    aiohttp = None
    HAS_AIOHTTP = False

try:
    import musicbrainzngs as mb

    HAS_MUSICBRAINZ = True
except ImportError:
    mb = None
    HAS_MUSICBRAINZ = False


@dataclass
class MusicBrainzMatch:
    """A MusicBrainz search result match."""

    recording_id: str
    artist_id: str
    artist_name: str
    song_title: str
    score: int  # MusicBrainz search score
    confidence: float  # Our calculated confidence
    metadata: Dict = field(default_factory=dict)


class MusicBrainzSearchPass(ParsingPass):
    """Dedicated MusicBrainz search pass for direct API lookups."""

    def __init__(self, advanced_parser: AdvancedTitleParser, db_manager=None):
        self.advanced_parser = advanced_parser
        self.db_manager = db_manager

        # MusicBrainz API configuration
        self.mb_base_url = "https://musicbrainz.org/ws/2"
        self.user_agent = "KaraokeCollector/2.1 (https://github.com/karaoke/search)"
        self.rate_limit_delay = 1.0  # MusicBrainz rate limit

        # Artist name variants mapping for reducing false positives
        self.artist_variants = {
            # Common stylistic variations
            "pink": ["p!nk", "pink"],
            "p!nk": ["pink", "p!nk"],
            # Band name variations with "The" prefix
            "goo dolls": ["goo goo dolls", "the goo goo dolls"],
            "goo goo dolls": ["goo dolls", "the goo goo dolls"],
            "the goo goo dolls": ["goo dolls", "goo goo dolls"],
            "kentucky headhunters": ["the kentucky headhunters"],
            "the kentucky headhunters": ["kentucky headhunters"],
            "dave matthews band": [
                "dave matthews",
                "dave matthews & friends",
                "dave matthews & tim reynolds",
            ],
            "dave matthews": ["dave matthews band", "dave matthews & friends"],
            "dave matthews & friends": ["dave matthews", "dave matthews band"],
            # Common abbreviations and expansions
            "mary j blige": ["mary j. blige", "mary blige", "mary jane blige"],
            "mary j. blige": ["mary j blige", "mary blige", "mary jane blige"],
            "mary blige": ["mary j. blige", "mary j blige"],
            # And/& variations
            "tom petty heartbreakers": [
                "tom petty & the heartbreakers",
                "tom petty and the heartbreakers",
            ],
            "tom petty & the heartbreakers": [
                "tom petty heartbreakers",
                "tom petty and the heartbreakers",
            ],
            "tom petty and the heartbreakers": [
                "tom petty heartbreakers",
                "tom petty & the heartbreakers",
            ],
            # The/& variations
            "gladys knight pips": [
                "gladys knight & the pips",
                "gladys knight and the pips",
                "gladys knight the pips",
            ],
            "gladys knight & the pips": [
                "gladys knight pips",
                "gladys knight and the pips",
                "gladys knight the pips",
            ],
            "gladys knight and the pips": [
                "gladys knight pips",
                "gladys knight & the pips",
                "gladys knight the pips",
            ],
            "gladys knight the pips": [
                "gladys knight pips",
                "gladys knight & the pips",
                "gladys knight and the pips",
            ],
            # Common misspellings and truncations
            "marvin gaye": [
                "marvin & tammi terrell gaye",
                "marvin and tammi gaye",
                "marvin tammi terrell gaye",
            ],
            "marvin & tammi terrell gaye": ["marvin gaye", "marvin and tammi gaye"],
            "marvin and tammi gaye": ["marvin gaye", "marvin & tammi terrell gaye"],
            "brandy": ["brandy norwood"],
            "brandy norwood": ["brandy"],
            "puddle of mudd": ["puddle"],
            "puddle": ["puddle of mudd"],
            # Country/Folk variations
            "tracy lawrence": ["lawrence tracy"],
            "lawrence tracy": ["tracy lawrence"],
            # Missing word issues from parsing
            "jackson browne load": ["jackson browne"],  # Parsing error creates bad query
            "jackson browne": ["jackson browne load"],
            "pink don't let me get": ["pink"],  # Parsing error
            "shenandoah next to you me": ["shenandoah"],  # Parsing error
            "staind for you": ["staind"],  # Parsing error
            "marvelettes": ["the marvelettes"],
            "the marvelettes": ["marvelettes"],
            # Specific artist variations from log analysis
            "elvis costello": [
                "elvis costello & the attractions",
                "elvis costello & the imposters",
            ],
            "elvis costello & the attractions": ["elvis costello"],
            "elvis costello & the imposters": ["elvis costello"],
            "anne marie": ["anne-marie", "anne marie"],
            "anne-marie": ["anne marie", "anne marie"],
            # Solo/group variants
            "christina perri": ["christina perri & steve kazee"],
            "christina perri & steve kazee": ["christina perri"],
            # Ed Sheeran variants
            "ed sheeran": ["edward sheeran", "eddie sheeran"],
            "edward sheeran": ["ed sheeran"],
            "eddie sheeran": ["ed sheeran"],
            # Amy Winehouse variants
            "amy winehouse": ["amy jade winehouse"],
            "amy jade winehouse": ["amy winehouse"],
            # Kendrick Lamar variants
            "kendrick lamar": ["k. dot", "kendrick lamar duckworth"],
            "k. dot": ["kendrick lamar"],
            "kendrick lamar duckworth": ["kendrick lamar"],
            # Crash Adams variants
            "crash adams": ["crash & adams"],
            "crash & adams": ["crash adams"],
            # Common featuring variations
            "featuring": ["feat", "ft", "with"],
            "feat": ["featuring", "ft", "with"],
            "ft": ["featuring", "feat", "with"],
            "with": ["featuring", "feat", "ft"],
        }

        # Create a normalized lookup for faster searching
        self._normalized_variants = {}
        for canonical, variants in self.artist_variants.items():
            canonical_norm = self._normalize_artist_name(canonical)
            self._normalized_variants[canonical_norm] = [
                self._normalize_artist_name(v) for v in variants + [canonical]
            ]

        # Configuration
        self.max_search_results = 25
        self.min_confidence_threshold = 0.75  # Further increased to filter more bad matches
        self.fuzzy_match_threshold = 0.85  # Further increased for stricter matching

        # Statistics
        self.stats = {
            "total_searches": 0,
            "successful_matches": 0,
            "api_errors": 0,
            "fuzzy_matches": 0,
            "direct_matches": 0,
            "title_mismatches_detected": 0,
            "title_mismatches_strong_penalty": 0,
        }

        # Simple in-memory cache for search results (query -> results)
        self._search_cache = {}
        self._cache_max_size = 1000  # Limit cache size

    @property
    def pass_type(self) -> PassType:
        return PassType.MUSICBRAINZ_SEARCH

    async def parse(
        self,
        title: str,
        description: str = "",
        tags: str = "",
        channel_name: str = "",
        channel_id: str = "",
        metadata: Optional[Dict] = None,
    ) -> Optional[ParseResult]:
        """Execute MusicBrainz search parsing."""

        if not HAS_AIOHTTP:
            logger.warning("aiohttp dependency not available for MusicBrainz search")
            return None

        start_time = time.time()
        self.stats["total_searches"] += 1

        # Check if we have parsed artist/title from Channel Template
        parsed_artist = None
        parsed_title = None
        if metadata and metadata.get("parsed_artist") and metadata.get("parsed_title"):
            # Use the cleaned, parsed version for better results
            parsed_artist = metadata["parsed_artist"]
            parsed_title = metadata["parsed_title"]
            title = f"{parsed_artist} - {parsed_title}"
            logger.debug(f"Using parsed title for MusicBrainz: {title}")

        try:
            # Step 1: Generate clean search queries
            search_queries = self._generate_search_queries(title)

            # If we have parsed data, add a structured query at the beginning
            if parsed_artist and parsed_title:
                structured_query = f'artist:"{parsed_artist}" AND recording:"{parsed_title}"'
                search_queries.insert(0, structured_query)
                logger.debug(f"Added structured query: {structured_query}")
            if not search_queries:
                return None

            # Step 2: Try each query with MusicBrainz
            best_result = None
            best_confidence = 0.0

            for query in search_queries:
                mb_matches = await self._search_musicbrainz(query)

                if mb_matches:
                    # Convert best MusicBrainz match to ParseResult
                    parse_result = self._convert_to_parse_result(mb_matches[0], query)

                    if parse_result:
                        # Validate the result against the original title
                        validation_score = self._validate_result_against_title(
                            parse_result, title, query
                        )

                        # Adjust confidence based on validation
                        adjusted_confidence = parse_result.confidence * validation_score
                        parse_result.confidence = adjusted_confidence

                        if adjusted_confidence > best_confidence:
                            best_confidence = adjusted_confidence
                            best_result = parse_result

                            # Early exit if confidence is high enough
                            if best_confidence > 0.9:
                                break

            if best_result:
                self.stats["successful_matches"] += 1

            return best_result

        except Exception as e:
            logger.error(f"MusicBrainz search pass failed: {e}")
            self.stats["api_errors"] += 1
            return None
        finally:
            processing_time = time.time() - start_time
            if processing_time > 15.0:  # Log slow searches
                logger.warning(f"MusicBrainz search took {processing_time:.2f}s")

    def _generate_search_queries(self, title: str) -> List[str]:
        """Generate search queries optimized for MusicBrainz."""

        queries = []

        # Clean the title for search - simple karaoke term removal
        base_query = self._clean_query_simple(title)

        if not base_query or len(base_query) < 3:
            return queries

        # Pre-filter for query quality
        if not self._is_valid_query(base_query):
            logger.debug(f"Skipping low-quality query: {base_query}")
            return queries

        # Add spaces around dashes that don't have them (but preserve double dashes)
        # This helps with titles like "Artist-Song" → "Artist - Song"
        spaced_query = re.sub(r"(?<=[a-zA-Z0-9])(?<!-)[-](?!-)(?=[a-zA-Z0-9])", " - ", base_query)

        # Strategy 1: Use spaced query first (more likely to be correct)
        if spaced_query != base_query:
            queries.append(spaced_query)
            queries.append(base_query)  # Add unspaced as fallback
        else:
            queries.append(base_query)

        # Strategy 2: Try to split into artist and song if possible
        # Look for common separators (use spaced_query for better splitting)
        for separator in [" - ", " by ", " from ", ":", " | "]:
            if separator in spaced_query:
                parts = spaced_query.split(separator, 1)
                if len(parts) == 2:
                    # Try both orders
                    queries.append(
                        f'artist:"{parts[0].strip()}" AND recording:"{parts[1].strip()}"'
                    )
                    queries.append(
                        f'artist:"{parts[1].strip()}" AND recording:"{parts[0].strip()}"'
                    )
                break

        # Strategy 3: Extract quoted parts (often song titles)
        quoted_parts = re.findall(r'["\u201c\u201d]([^"\u201c\u201d]+)["\u201c\u201d]', title)
        for quoted in quoted_parts:
            if len(quoted.strip()) > 2:
                queries.append(f'recording:"{quoted.strip()}"')

        # Strategy 4: Try minimal query (remove only most obvious noise)
        minimal = re.sub(
            r"\b(?:karaoke|instrumental|backing|track|melody)\b", "", title, flags=re.IGNORECASE
        ).strip()
        # Also add spaces to minimal query
        minimal_spaced = re.sub(r"(?<=[a-zA-Z0-9])(?<!-)[-](?!-)(?=[a-zA-Z0-9])", " - ", minimal)
        if minimal and minimal != base_query and len(minimal) > 3:
            queries.append(minimal)
            if minimal_spaced != minimal:
                queries.append(minimal_spaced)

        # Remove duplicates and limit
        unique_queries = []
        seen = set()
        for query in queries:
            if query not in seen and len(query) > 3:
                seen.add(query)
                unique_queries.append(query)

        return unique_queries[:5]  # Limit to avoid rate limiting

    def _clean_query_simple(self, query: str) -> str:
        """Simple query cleaning to remove karaoke-related terms."""
        if not query:
            return ""

        # Remove common karaoke-related terms
        karaoke_terms = [
            r"\b(?:karaoke|karaokê|karaoké)\b",
            r"\b(?:instrumental|backing track|minus one|minusone|minus-one)\b",
            r"\b(?:sing along|singalong)\b",
            r"\b(?:mr|inst|playback|accompaniment|melody)\b",
            r"\b(?:hd|hq|4k|1080p|720p|480p|high quality|high def|ultra hd|uhd|full hd)\b",
            r"\b(?:official|music video|video|audio|clip|lyrics)\b",
            r"\b(?:version|cover|tribute|style|remake|rendition)\b",
        ]

        cleaned = query
        for pattern in karaoke_terms:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

        # Clean up whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned

    async def _search_musicbrainz(
        self, query: str, timeout: float = 10.0
    ) -> List[MusicBrainzMatch]:
        """Search MusicBrainz API with the given query using aiohttp."""

        # Check cache first
        if query in self._search_cache:
            logger.info(f"Using cached MusicBrainz results for: {query}")
            self.stats["cache_hits"] = self.stats.get("cache_hits", 0) + 1
            return self._search_cache[query]

        search_start_time = time.time()

        # Build MusicBrainz API URL
        params = {"query": query, "limit": self.max_search_results, "fmt": "json"}
        if not HAS_AIOHTTP:
            logger.error("aiohttp is required for MusicBrainz API calls")
            return []

        if not HAS_URLLIB or not urlencode:
            logger.error("urllib.parse is required for MusicBrainz API calls")
            return []

        url = f"{self.mb_base_url}/recording?" + urlencode(params)

        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}

        # Enhanced retry logic for API rate limiting and server errors
        max_retries = 5  # Increased from 3 to 5
        base_delay = 2.0  # Start with 2 seconds

        # Add request throttling to reduce server load
        await asyncio.sleep(self.rate_limit_delay)

        result = None
        for attempt in range(max_retries):
            try:
                # Create timeout configuration
                if not aiohttp:
                    raise RuntimeError("aiohttp is required for MusicBrainz API calls")

                timeout_config = aiohttp.ClientTimeout(total=timeout)

                async with aiohttp.ClientSession(timeout=timeout_config) as session:
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            result = await response.json()
                            break  # Success, exit retry loop
                        elif response.status == 503:  # Service Unavailable
                            if attempt < max_retries - 1:
                                # Longer delays for 503 errors since server is overloaded
                                retry_delay = base_delay * (3**attempt)  # More aggressive backoff
                                logger.warning(
                                    f"MusicBrainz API returned 503 (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay}s..."
                                )
                                await asyncio.sleep(retry_delay)
                                continue
                            else:
                                logger.error(
                                    f"MusicBrainz API returned 503 after {max_retries} attempts for query: {query}"
                                )
                                self.stats["api_503_failures"] = (
                                    self.stats.get("api_503_failures", 0) + 1
                                )
                                return []
                        elif response.status == 429:  # Rate Limited
                            if attempt < max_retries - 1:
                                retry_delay = base_delay * (
                                    4**attempt
                                )  # Even longer delay for rate limits
                                logger.warning(
                                    f"MusicBrainz API rate limited (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay}s..."
                                )
                                await asyncio.sleep(retry_delay)
                                continue
                            else:
                                logger.error(
                                    f"MusicBrainz API rate limited after {max_retries} attempts for query: {query}"
                                )
                                self.stats["api_rate_limit_failures"] = (
                                    self.stats.get("api_rate_limit_failures", 0) + 1
                                )
                                return []
                        else:
                            logger.warning(
                                f"MusicBrainz API returned status {response.status} for query: {query}"
                            )
                            return []
            except (
                (asyncio.TimeoutError, getattr(aiohttp, "ServerTimeoutError", Exception))
                if aiohttp
                else asyncio.TimeoutError
            ):
                if attempt < max_retries - 1:
                    retry_delay = base_delay * (2**attempt)
                    logger.warning(
                        f"MusicBrainz API timeout (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay}s..."
                    )
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    search_time = time.time() - search_start_time
                    logger.warning(
                        f"MusicBrainz API search timed out after {search_time:.2f}s for '{query}' (limit: {timeout}s)"
                    )
                    self.stats["api_timeouts"] = self.stats.get("api_timeouts", 0) + 1
                    return []
            except getattr(aiohttp, "ClientError", Exception) if aiohttp else Exception as e:
                if attempt < max_retries - 1:
                    retry_delay = base_delay * (2**attempt)
                    logger.warning(
                        f"MusicBrainz API connection error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {retry_delay}s..."
                    )
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    search_time = time.time() - search_start_time
                    logger.warning(
                        f"MusicBrainz API connection error for '{query}' after {search_time:.2f}s: {e}"
                    )
                    return []

        # If we get here without a result, all retries failed
        if result is None:
            logger.error(f"All {max_retries} attempts failed for MusicBrainz query: {query}")
            return []

        matches = []
        recordings = result.get("recordings", [])

        for recording in recordings:
            # Extract basic information (JSON API format)
            recording_id = recording.get("id", "")
            title = recording.get("title", "")
            score = recording.get("score", 0)  # JSON API uses 'score' not 'ext:score'

            # Extract artist information from artist-credit
            artist_credits = recording.get("artist-credit", [])
            if not artist_credits:
                continue

            # Handle artist-credit structure in JSON API
            artist_name = ""
            artist_id = ""

            if isinstance(artist_credits, list) and len(artist_credits) > 0:
                first_credit = artist_credits[0]
                if isinstance(first_credit, dict):
                    if "artist" in first_credit:
                        artist_info = first_credit["artist"]
                        artist_name = artist_info.get("name", "")
                        artist_id = artist_info.get("id", "")
                    elif "name" in first_credit:
                        # Sometimes the artist info is directly in the credit
                        artist_name = first_credit.get("name", "")
                        artist_id = first_credit.get("id", "")

            if not artist_name or not title:
                continue

            # Pre-filter obvious mismatches before detailed confidence calculation
            if not self._quick_artist_relevance_check(artist_name, query):
                continue

            # Calculate confidence based on MusicBrainz score and other factors
            confidence = self._calculate_confidence(score, title, artist_name, query)

            if confidence >= self.min_confidence_threshold:
                match = MusicBrainzMatch(
                    recording_id=recording_id,
                    artist_id=artist_id,
                    artist_name=artist_name,
                    song_title=title,
                    score=score,
                    confidence=confidence,
                    metadata={
                        "mb_score": score,
                        "query": query,
                        "releases": recording.get("releases", []),
                        "length": recording.get("length"),
                        "disambiguation": recording.get("disambiguation", ""),
                    },
                )
                matches.append(match)

        # Sort by confidence
        matches.sort(key=lambda m: m.confidence, reverse=True)

        # Cache the results
        if len(self._search_cache) < self._cache_max_size:
            self._search_cache[query] = matches
        else:
            # Simple cache eviction - remove oldest entry
            if self._search_cache:
                self._search_cache.pop(next(iter(self._search_cache)))
            self._search_cache[query] = matches

        return matches

    def _calculate_confidence(self, mb_score: int, title: str, artist: str, query: str) -> float:
        """Calculate confidence score for a MusicBrainz match."""

        # Ensure mb_score is numeric (sometimes it comes as string)
        try:
            if isinstance(mb_score, str):
                mb_score = int(mb_score)
            elif mb_score is None:
                mb_score = 0
            mb_score = int(mb_score)
        except (ValueError, TypeError):
            mb_score = 0

        # Base confidence from MusicBrainz score (0-100)
        base_confidence = min(mb_score / 100.0, 1.0)

        # CRITICAL: Title-based cross-validation
        # Check if the MB artist significantly differs from what's suggested in the title
        title_artist_penalty = self._check_title_artist_mismatch(query, artist)
        if title_artist_penalty > 0:
            base_confidence *= 1.0 - title_artist_penalty
            # Only warn about mismatches for results that might be used (confidence > 0.6 after penalty)
            if base_confidence > 0.6 and title_artist_penalty > 0.3:
                logger.warning(
                    f"Title-artist mismatch detected: query='{query}', mb_artist='{artist}', penalty={title_artist_penalty:.2f}"
                )

        # Boost for exact matches in query
        query_lower = query.lower()
        title_lower = title.lower()
        artist_lower = artist.lower()

        # Check if both artist and title appear in query
        title_in_query = title_lower in query_lower
        artist_in_query = artist_lower in query_lower

        if title_in_query and artist_in_query:
            base_confidence *= 1.3
            self.stats["direct_matches"] += 1
        elif title_in_query or artist_in_query:
            base_confidence *= 1.1

        # Fuzzy matching boost
        from difflib import SequenceMatcher

        title_similarity = SequenceMatcher(None, title_lower, query_lower).ratio()
        combined_similarity = SequenceMatcher(
            None, f"{artist_lower} {title_lower}", query_lower
        ).ratio()

        max_similarity = max(title_similarity, combined_similarity)
        if max_similarity > self.fuzzy_match_threshold:
            base_confidence *= 1.0 + max_similarity * 0.2
            self.stats["fuzzy_matches"] += 1

        # Penalty for very short matches
        if len(title) < 3 or len(artist) < 2:
            base_confidence *= 0.7

        # Cap at reasonable maximum
        return min(base_confidence, 0.95)

    def _normalize_artist_name(self, artist_name: str) -> str:
        """Normalize artist name for variant matching."""
        if not artist_name:
            return ""

        # Convert to lowercase and remove common punctuation/formatting
        normalized = artist_name.lower()
        normalized = re.sub(r"[^\w\s]", "", normalized)  # Remove punctuation
        normalized = re.sub(r"\s+", " ", normalized).strip()  # Normalize whitespace

        # Remove common words that don't affect artist identity
        common_words = ["the", "and", "&", "feat", "featuring", "ft"]
        words = normalized.split()
        words = [w for w in words if w not in common_words]

        return " ".join(words)

    def _quick_artist_relevance_check(self, artist_name: str, query: str) -> bool:
        """Quick pre-filter to skip obviously irrelevant artists before detailed analysis."""
        if not artist_name or not query:
            return False

        # Normalize for comparison
        artist_lower = artist_name.lower()
        query_lower = query.lower()

        # If artist name appears directly in query, it's relevant
        if artist_lower in query_lower:
            return True

        # Check for known variants
        artist_normalized = self._normalize_artist_name(artist_name)
        if self._artist_appears_in_variants(artist_normalized, query_lower):
            return True

        # Quick similarity check - if completely unrelated, skip
        from difflib import SequenceMatcher

        # Extract potential artist from query (first part before separator)
        potential_artist = ""
        for separator in [" - ", " by ", " from ", ":", " | "]:
            if separator in query:
                potential_artist = query.split(separator, 1)[0].strip()
                break

        if potential_artist:
            similarity = SequenceMatcher(None, artist_lower, potential_artist.lower()).ratio()
            # If similarity is very low, skip this artist entirely
            if similarity < 0.25:
                return False

        # Check if artist has common words with query
        artist_words = set(artist_lower.split())
        query_words = set(query_lower.split())

        # Remove common stop words that don't indicate artist relevance
        stop_words = {"the", "and", "or", "of", "in", "on", "at", "to", "for", "a", "an"}
        artist_words -= stop_words
        query_words -= stop_words

        # If no meaningful words overlap, likely irrelevant
        if len(artist_words & query_words) == 0 and len(artist_words) > 1:
            return False

        return True

    def _artist_appears_in_variants(self, normalized_artist: str, query_lower: str) -> bool:
        """Check if artist appears in query considering known variants."""
        # Check direct variants
        for canonical_norm, variants in self._normalized_variants.items():
            if normalized_artist in variants:
                # Check if any variant appears in the query
                for variant in variants:
                    if variant in query_lower:
                        return True
        return False

    def _is_valid_query(self, query: str) -> bool:
        """Check if a query is worth sending to MusicBrainz."""
        if not query or len(query.strip()) < 3:
            return False

        # Check for obvious parsing errors
        words = query.lower().split()

        # Skip queries with too many noise words
        noise_words = {
            "karaoke",
            "lyrics",
            "instrumental",
            "backing",
            "track",
            "version",
            "cover",
            "live",
            "audio",
            "video",
            "clip",
            "music",
            "song",
        }
        noise_count = sum(1 for word in words if word in noise_words)
        if noise_count > len(words) * 0.6:  # More than 60% noise
            return False

        # Skip queries that look like parsing errors (too many short fragments)
        short_words = [w for w in words if len(w) <= 2]
        if len(short_words) > len(words) * 0.5:  # More than 50% very short words
            return False

        # Skip queries with obvious Korean mixed with fragments
        has_korean = any(ord(char) >= 0xAC00 and ord(char) <= 0xD7AF for char in query)
        if has_korean and len(words) > 5:  # Korean with too many english fragments
            return False

        # Skip queries that are just random fragments
        fragment_indicators = ["load", "outstay", "don't let me get", "next to you me", "for you"]
        for indicator in fragment_indicators:
            if indicator in query.lower():
                return False

        # Skip queries that are mostly single characters or very short words
        if len([w for w in words if len(w) >= 3]) < 2:  # Need at least 2 meaningful words
            return False

        # Skip queries with suspicious patterns (likely parsing errors)
        suspicious_patterns = [
            r"\b\w{1,2}\s+\w{1,2}\s+\w{1,2}\b",  # Three or more consecutive very short words
            r"\b\d+\s+\w{1,2}\s+\w{1,2}\b",  # Number followed by short words
            r"\bthe\s+\w{1,2}\s+\w{1,2}\b",  # "the" followed by short words
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, query.lower()):
                return False

        return True

    def _are_artist_variants(self, artist1: str, artist2: str) -> bool:
        """Check if two normalized artist names are known variants of each other."""
        if not artist1 or not artist2:
            return False

        # Check direct match
        if artist1 == artist2:
            return True

        # Check in variants mapping
        for canonical_norm, variants in self._normalized_variants.items():
            if artist1 in variants and artist2 in variants:
                return True

        # Check common patterns that should be considered variants
        # Remove common words and check again
        def remove_common_words(name):
            words = name.split()
            filtered = [w for w in words if w not in ["the", "and", "ft", "feat", "featuring"]]
            return " ".join(filtered)

        artist1_filtered = remove_common_words(artist1)
        artist2_filtered = remove_common_words(artist2)

        # If one is a subset of the other after filtering common words
        if artist1_filtered and artist2_filtered:
            if artist1_filtered in artist2_filtered or artist2_filtered in artist1_filtered:
                return True

        return False

    def _check_title_artist_mismatch(self, query: str, mb_artist: str) -> float:
        """Check if MusicBrainz artist significantly differs from title-suggested artist.

        Returns penalty (0.0 = no penalty, 1.0 = maximum penalty)
        """
        import re
        from difflib import SequenceMatcher

        # Extract potential artist names from the query
        # Look for patterns like "Artist - Song", "Artist, Name - Song", etc.
        potential_artists = []

        # Pattern 1: "Artist - Song" or "Artist: Song"
        dash_match = re.search(r"^([^-:]+?)[-:]", query)
        if dash_match:
            potential_artists.append(dash_match.group(1).strip())

        # Pattern 2: "LastName, FirstName - Song"
        comma_match = re.search(r"^([^,]+),\s*([^-]+)", query)
        if comma_match:
            # Both "LastName, FirstName" and "FirstName LastName" order
            potential_artists.append(
                f"{comma_match.group(1).strip()}, {comma_match.group(2).strip()}"
            )
            potential_artists.append(
                f"{comma_match.group(2).strip()} {comma_match.group(1).strip()}"
            )

        # Pattern 3: Words at the end of query (often artist after song in some formats)
        words = query.split()
        if len(words) >= 2:
            # Last 1-3 words might be artist
            potential_artists.append(" ".join(words[-2:]))
            if len(words) >= 3:
                potential_artists.append(" ".join(words[-3:]))

        # Normalize the MusicBrainz artist name
        mb_artist_normalized = self._normalize_artist_name(mb_artist)

        # Check similarity between MB artist and each potential artist
        mb_artist_lower = mb_artist.lower()
        max_similarity = 0.0

        for potential in potential_artists:
            potential_lower = potential.lower()

            # Skip if potential artist is too short or contains obvious noise
            if len(potential_lower) < 3 or any(
                noise in potential_lower for noise in ["karaoke", "lyrics", "instrumental"]
            ):
                continue

            # Check for known artist variants first
            potential_normalized = self._normalize_artist_name(potential)

            # Look for exact variant matches
            if self._are_artist_variants(mb_artist_normalized, potential_normalized):
                return 0.0  # No penalty for known variants

            # Regular similarity check
            similarity = SequenceMatcher(None, mb_artist_lower, potential_lower).ratio()
            max_similarity = max(max_similarity, similarity)

        # Also check for exact artist name in query (case insensitive)
        if mb_artist_lower in query.lower():
            # Artist name appears in query, likely a good match
            return 0.0

        # Calculate penalty based on similarity
        # Be much stricter to reduce false positives
        if max_similarity < 0.3:
            penalty = 0.9  # Severe penalty for completely unrelated artists
            self.stats["title_mismatches_strong_penalty"] += 1
        elif max_similarity < 0.5:
            penalty = 0.7  # Very strong penalty for likely wrong artist
            self.stats["title_mismatches_strong_penalty"] += 1
        elif max_similarity < 0.7:
            penalty = 0.4  # Strong penalty for questionable match
            self.stats["title_mismatches_detected"] += 1
        elif max_similarity < 0.85:
            penalty = 0.15  # Moderate penalty for imperfect match
        else:
            penalty = 0.0  # No penalty for very good matches

        return penalty

    def _validate_result_against_title(
        self, parse_result: ParseResult, original_title: str, query: str
    ) -> float:
        """Validate MusicBrainz result against original title to catch mismatches.

        Returns a score between 0 and 1 to multiply with confidence.
        """
        if not parse_result.artist or not parse_result.song_title:
            return 0.5  # Penalize incomplete results

        # Clean the original title for comparison
        clean_title = self._clean_query_simple(original_title).lower()

        # Extract the MusicBrainz result
        mb_artist = parse_result.artist.lower()
        mb_song = parse_result.song_title.lower()
        mb_combined = f"{mb_artist} {mb_song}"

        # Calculate similarity scores
        from difflib import SequenceMatcher

        # Check if artist appears in original title
        artist_in_title = mb_artist in clean_title
        artist_similarity = SequenceMatcher(None, mb_artist, clean_title).ratio()

        # Check if song appears in original title
        song_in_title = mb_song in clean_title
        song_similarity = SequenceMatcher(None, mb_song, clean_title).ratio()

        # Check combined similarity
        combined_similarity = SequenceMatcher(None, mb_combined, clean_title).ratio()

        # Calculate validation score
        validation_score = 1.0

        # Strong validation if both artist and song appear in title
        if artist_in_title and song_in_title:
            validation_score = 1.1  # Boost for perfect match
        # Good validation if high combined similarity
        elif combined_similarity > 0.6:
            validation_score = 0.9 + (combined_similarity - 0.6) * 0.5
        # Moderate validation if at least one component matches well
        elif artist_similarity > 0.7 or song_similarity > 0.7:
            validation_score = 0.8
        # Poor validation if very low similarity
        elif combined_similarity < 0.3:
            # This catches cases like "Elton John - Sorry..." returning "( ) - non serviam"
            validation_score = 0.2
            logger.warning(
                f"Low similarity detected - Original: '{original_title}', "
                f"MB Result: '{mb_artist} - {mb_song}' (similarity: {combined_similarity:.2f})"
            )
        else:
            # Default moderate penalty
            validation_score = 0.6

        # Additional checks for specific mismatches
        # Check for suspicious patterns like "( )" as artist
        if mb_artist in ["( )", "unknown", "various", "n/a", ""]:
            validation_score *= 0.3

        # Check if the result seems completely unrelated (no words in common)
        original_words = set(clean_title.split())
        result_words = set(mb_combined.split())
        common_words = original_words & result_words

        # Remove common stop words
        stop_words = {"the", "a", "an", "of", "in", "on", "and", "or", "for", "to", "is"}
        meaningful_common = common_words - stop_words

        if not meaningful_common and len(original_words) > 2:
            validation_score *= 0.4

        return min(validation_score, 1.0)  # Cap at 1.0

    def _convert_to_parse_result(self, match: MusicBrainzMatch, query: str) -> ParseResult:
        """Convert MusicBrainz match to ParseResult."""

        result = ParseResult(
            artist=match.artist_name,
            song_title=match.song_title,
            confidence=match.confidence,
            method="musicbrainz_search",
            pattern_used="api_lookup",
            metadata={
                "musicbrainz_recording_id": match.recording_id,
                "musicbrainz_artist_id": match.artist_id,
                "musicbrainz_score": match.score,
                "search_query": query,
                "api_confidence": match.confidence,
                **match.metadata,
            },
        )

        return result

    def get_statistics(self) -> Dict:
        """Get pass statistics."""

        total_searches = self.stats["total_searches"]
        success_rate = self.stats["successful_matches"] / max(total_searches, 1)

        cache_hits = self.stats.get("cache_hits", 0)
        cache_hit_rate = cache_hits / max(total_searches, 1)

        return {
            "total_searches": total_searches,
            "successful_matches": self.stats["successful_matches"],
            "api_errors": self.stats["api_errors"],
            "success_rate": success_rate,
            "direct_matches": self.stats["direct_matches"],
            "fuzzy_matches": self.stats["fuzzy_matches"],
            "title_mismatches_detected": self.stats["title_mismatches_detected"],
            "title_mismatches_strong_penalty": self.stats["title_mismatches_strong_penalty"],
            "cache_hits": cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self._search_cache),
            "api_503_failures": self.stats.get("api_503_failures", 0),
            "api_rate_limit_failures": self.stats.get("api_rate_limit_failures", 0),
            "api_timeouts": self.stats.get("api_timeouts", 0),
            "dependencies_available": HAS_AIOHTTP,
        }
