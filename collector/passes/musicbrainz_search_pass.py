"""Pass 1: Dedicated MusicBrainz search with direct API lookup and fuzzy matching."""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..advanced_parser import AdvancedTitleParser, ParseResult
from ..passes.web_search_pass import FillerWordProcessor
from .base import ParsingPass, PassType

logger = logging.getLogger(__name__)

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    requests = None
    HAS_REQUESTS = False

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
        self.filler_processor = FillerWordProcessor()

        # Initialize MusicBrainz client if available
        if HAS_MUSICBRAINZ:
            mb.set_useragent("KaraokeCollector", "2.1", "https://github.com/karaoke/search")
            mb.set_rate_limit(limit_or_interval=1.0, new_requests=1)

        # Configuration
        self.max_search_results = 25
        self.min_confidence_threshold = 0.6
        self.fuzzy_match_threshold = 0.75

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

        if not HAS_MUSICBRAINZ or not HAS_REQUESTS:
            logger.warning("MusicBrainz dependencies not available")
            return None

        start_time = time.time()
        self.stats["total_searches"] += 1

        try:
            # Step 1: Generate clean search queries
            search_queries = self._generate_search_queries(title)
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

                    if parse_result and parse_result.confidence > best_confidence:
                        best_confidence = parse_result.confidence
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

        # Clean the title for search
        cleaned = self.filler_processor.clean_query(title, "english")
        base_query = cleaned.cleaned_query

        if not base_query or len(base_query) < 3:
            return queries

        # Strategy 1: Use cleaned query as-is
        queries.append(base_query)

        # Strategy 2: Try to split into artist and song if possible
        # Look for common separators
        for separator in [" - ", " by ", " from ", ":", " | "]:
            if separator in base_query:
                parts = base_query.split(separator, 1)
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
            r"\b(?:karaoke|instrumental|backing|track)\b", "", title, flags=re.IGNORECASE
        ).strip()
        if minimal and minimal != base_query and len(minimal) > 3:
            queries.append(minimal)

        # Remove duplicates and limit
        unique_queries = []
        seen = set()
        for query in queries:
            if query not in seen and len(query) > 3:
                seen.add(query)
                unique_queries.append(query)

        return unique_queries[:5]  # Limit to avoid rate limiting

    async def _search_musicbrainz(self, query: str) -> List[MusicBrainzMatch]:
        """Search MusicBrainz API with the given query."""

        try:
            # Use asyncio to run the blocking MusicBrainz call
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: mb.search_recordings(
                    query=query, limit=self.max_search_results, strict=False
                ),
            )

            matches = []
            recordings = result.get("recording-list", [])

            for recording in recordings:
                # Extract basic information
                recording_id = recording.get("id", "")
                title = recording.get("title", "")
                score = recording.get("ext:score", 0)

                # Extract artist information
                artist_credits = recording.get("artist-credit", [])
                if not artist_credits:
                    continue

                # Handle different artist-credit structures
                if isinstance(artist_credits[0], dict):
                    artist_info = artist_credits[0].get("artist", {})
                    artist_name = artist_info.get("name", "")
                    artist_id = artist_info.get("id", "")
                else:
                    # Sometimes it's just a string
                    artist_name = str(artist_credits[0])
                    artist_id = ""

                if not artist_name or not title:
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
                            "releases": recording.get("release-list", []),
                            "length": recording.get("length"),
                            "disambiguation": recording.get("disambiguation", ""),
                        },
                    )
                    matches.append(match)

            # Sort by confidence
            matches.sort(key=lambda m: m.confidence, reverse=True)
            return matches

        except Exception as e:
            logger.warning(f"MusicBrainz API search failed for '{query}': {e}")
            return []

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

            similarity = SequenceMatcher(None, mb_artist_lower, potential_lower).ratio()
            max_similarity = max(max_similarity, similarity)

        # Calculate penalty based on similarity
        # If similarity < 0.3, it's probably a wrong match
        if max_similarity < 0.3 and max_similarity > 0:
            penalty = 0.6  # Strong penalty for likely wrong artist
            self.stats["title_mismatches_strong_penalty"] += 1
        elif max_similarity < 0.5:
            penalty = 0.3  # Moderate penalty for questionable match
            self.stats["title_mismatches_detected"] += 1
        else:
            penalty = 0.0  # No penalty for reasonable matches

        return penalty

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

        return {
            "total_searches": total_searches,
            "successful_matches": self.stats["successful_matches"],
            "api_errors": self.stats["api_errors"],
            "success_rate": success_rate,
            "direct_matches": self.stats["direct_matches"],
            "fuzzy_matches": self.stats["fuzzy_matches"],
            "title_mismatches_detected": self.stats["title_mismatches_detected"],
            "title_mismatches_strong_penalty": self.stats["title_mismatches_strong_penalty"],
            "dependencies_available": HAS_MUSICBRAINZ and HAS_REQUESTS,
        }
