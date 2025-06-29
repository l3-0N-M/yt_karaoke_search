"""Pass 3: MusicBrainz validation and enrichment after web search results."""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from ..advanced_parser import AdvancedTitleParser, ParseResult
from .base import ParsingPass, PassType
from .musicbrainz_search_pass import MusicBrainzSearchPass

logger = logging.getLogger(__name__)

try:
    import musicbrainzngs as mb

    HAS_MUSICBRAINZ = True
except ImportError:
    mb = None
    HAS_MUSICBRAINZ = False


@dataclass
class ValidationResult:
    """Result of MusicBrainz validation."""

    validated: bool
    confidence_adjustment: float  # Multiplier for original confidence
    enriched_data: Dict = field(default_factory=dict)
    validation_method: str = ""
    mb_match_score: int = 0


class MusicBrainzValidationPass(ParsingPass):
    """Validates and enriches parsing results using MusicBrainz after web search."""

    def __init__(self, advanced_parser: AdvancedTitleParser, db_manager=None):
        self.advanced_parser = advanced_parser
        self.db_manager = db_manager

        # Reuse the search capabilities from the main MB pass
        self.mb_search_pass = MusicBrainzSearchPass(advanced_parser, db_manager)

        # Configuration
        self.validation_threshold = 0.7  # Minimum confidence for validation
        self.enrichment_enabled = True
        self.strict_validation = False  # If True, requires exact matches

        # Statistics
        self.stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "confidence_boosts": 0,
            "confidence_penalties": 0,
            "enrichments_added": 0,
            "validation_failures": 0,
            "artist_corrections": 0,
            "title_corrections": 0,
            "existing_mb_data_used": 0,
        }

    @property
    def pass_type(self) -> PassType:
        return PassType.MUSICBRAINZ_VALIDATION

    async def parse(
        self,
        title: str,
        description: str = "",
        tags: str = "",
        channel_name: str = "",
        channel_id: str = "",
        metadata: Optional[Dict] = None,
    ) -> Optional[ParseResult]:
        """
        Validate and enrich a ParseResult that should be passed in metadata.

        This pass expects metadata['web_search_result'] containing the ParseResult
        from the previous web search pass.
        """

        if not HAS_MUSICBRAINZ:
            logger.warning("MusicBrainz validation not available - dependencies missing")
            return None

        if not metadata or "web_search_result" not in metadata:
            logger.warning("MusicBrainz validation requires web_search_result in metadata")
            return None

        web_search_result = metadata["web_search_result"]
        if not isinstance(web_search_result, ParseResult):
            logger.warning("Invalid web_search_result format")
            return None

        start_time = time.time()
        self.stats["total_validations"] += 1

        try:
            # Validate the web search result against MusicBrainz
            validation_result = await self._validate_against_musicbrainz(web_search_result, title)

            if validation_result.validated:
                # Apply confidence adjustment and enrichment
                enhanced_result = self._apply_validation_results(
                    web_search_result, validation_result
                )

                self.stats["successful_validations"] += 1

                if validation_result.confidence_adjustment > 1.0:
                    self.stats["confidence_boosts"] += 1
                elif validation_result.confidence_adjustment < 1.0:
                    self.stats["confidence_penalties"] += 1

                if validation_result.enriched_data:
                    self.stats["enrichments_added"] += 1

                return enhanced_result
            else:
                self.stats["validation_failures"] += 1
                # Return original result with slight confidence penalty for failed validation
                web_search_result.confidence *= 0.95
                web_search_result.metadata = web_search_result.metadata or {}
                web_search_result.metadata.update(
                    {
                        "musicbrainz_validation": "failed",
                        "validation_method": validation_result.validation_method,
                    }
                )
                return web_search_result

        except Exception as e:
            logger.error(f"MusicBrainz validation failed: {e}")
            self.stats["validation_failures"] += 1
            return web_search_result  # Return original on error
        finally:
            processing_time = time.time() - start_time
            if processing_time > 10.0:
                logger.warning(f"MusicBrainz validation took {processing_time:.2f}s")

    async def _validate_against_musicbrainz(
        self, parse_result: ParseResult, original_title: str
    ) -> ValidationResult:
        """Validate a parse result against MusicBrainz database."""

        if not parse_result.artist or not parse_result.song_title:
            return ValidationResult(
                validated=False,
                confidence_adjustment=0.9,
                validation_method="incomplete_parse_data",
            )

        # First check if we have existing MusicBrainz data in the database
        existing_mb_data = await self._check_existing_musicbrainz_data(original_title)
        if existing_mb_data:
            logger.info(
                f"Found existing MusicBrainz data for '{original_title}' - using authoritative data"
            )
            self.stats["existing_mb_data_used"] += 1
            return ValidationResult(
                validated=True,
                confidence_adjustment=1.2,  # Boost for existing validated data
                enriched_data=existing_mb_data,
                validation_method="existing_musicbrainz_data",
            )

        # Search MusicBrainz for the artist + song combination
        search_query = f'artist:"{parse_result.artist}" AND recording:"{parse_result.song_title}"'

        try:
            # Use the existing MusicBrainz search functionality
            mb_matches = await self.mb_search_pass._search_musicbrainz(search_query)

            if not mb_matches:
                # Try broader search with just artist or song
                fallback_queries = [
                    f'artist:"{parse_result.song_title}" AND recording:"{parse_result.artist}"',  # Try swapped artist/title
                    f'artist:"{parse_result.artist}"',
                    f'recording:"{parse_result.song_title}"',
                    f"{parse_result.artist} {parse_result.song_title}",
                ]

                for query in fallback_queries:
                    mb_matches = await self.mb_search_pass._search_musicbrainz(query)
                    if mb_matches:
                        break

            if mb_matches:
                best_match = mb_matches[0]  # Already sorted by confidence

                # Calculate validation confidence based on match quality
                validation_confidence = self._calculate_validation_confidence(
                    parse_result, best_match, original_title
                )

                if validation_confidence >= self.validation_threshold:
                    # Extract enrichment data
                    enriched_data = self._extract_enrichment_data(best_match)

                    # Add authoritative artist/title data from MusicBrainz API result
                    enriched_data["authoritative_artist"] = best_match.artist_name
                    enriched_data["authoritative_title"] = best_match.song_title

                    return ValidationResult(
                        validated=True,
                        confidence_adjustment=validation_confidence,
                        enriched_data=enriched_data,
                        validation_method="musicbrainz_match",
                        mb_match_score=best_match.score,
                    )
                else:
                    return ValidationResult(
                        validated=False,
                        confidence_adjustment=0.8,  # Penalty for poor MB match
                        validation_method="low_mb_similarity",
                        mb_match_score=best_match.score,
                    )
            else:
                return ValidationResult(
                    validated=False,
                    confidence_adjustment=0.85,  # Small penalty for no MB match
                    validation_method="no_mb_results",
                )

        except Exception as e:
            logger.warning(f"MusicBrainz validation search failed: {e}")
            return ValidationResult(
                validated=False,
                confidence_adjustment=1.0,  # No penalty for API errors
                validation_method="api_error",
            )

    def _calculate_validation_confidence(
        self, parse_result: ParseResult, mb_match, original_title: str
    ) -> float:
        """Calculate how well the parse result matches the MusicBrainz data."""

        from difflib import SequenceMatcher

        # Compare artist names
        artist_similarity = SequenceMatcher(
            None, parse_result.artist.lower(), mb_match.artist_name.lower()
        ).ratio()

        # Compare song titles
        song_similarity = SequenceMatcher(
            None, parse_result.song_title.lower(), mb_match.song_title.lower()
        ).ratio()

        # Base confidence from similarities
        base_confidence = (artist_similarity + song_similarity) / 2.0

        # Boost for high MusicBrainz search scores
        mb_score_boost = min(mb_match.score / 100.0, 0.2)

        # Boost for exact matches
        exact_match_boost = 0.0
        if artist_similarity > 0.95 and song_similarity > 0.95:
            exact_match_boost = 0.3
        elif artist_similarity > 0.9 or song_similarity > 0.9:
            exact_match_boost = 0.15

        # Penalty for very different names
        difference_penalty = 0.0
        if artist_similarity < 0.5 or song_similarity < 0.5:
            difference_penalty = 0.2

        final_confidence = base_confidence + mb_score_boost + exact_match_boost - difference_penalty

        return max(0.0, min(1.5, final_confidence))  # Cap between 0 and 1.5

    def _extract_enrichment_data(self, mb_match) -> Dict:
        """Extract additional data from MusicBrainz match for enrichment."""

        enrichment = {
            "musicbrainz_recording_id": mb_match.recording_id,
            "musicbrainz_artist_id": mb_match.artist_id,
            "parse_confidence": mb_match.confidence,
            "musicbrainz_search_score": mb_match.score,
        }

        # Add additional metadata if available
        if hasattr(mb_match, "metadata") and mb_match.metadata:
            releases = mb_match.metadata.get("releases", [])
            if releases:
                # Extract release year from earliest release
                years = []
                for release in releases:
                    if "date" in release:
                        try:
                            year = int(release["date"][:4])
                            years.append(year)
                        except (ValueError, TypeError):
                            pass

                if years:
                    enrichment["release_year"] = min(years)

            # Add recording length if available
            if "length" in mb_match.metadata:
                enrichment["recording_length_ms"] = mb_match.metadata["length"]

            # Add disambiguation if available
            if "disambiguation" in mb_match.metadata:
                enrichment["musicbrainz_disambiguation"] = mb_match.metadata["disambiguation"]

        return enrichment

    async def _check_existing_musicbrainz_data(self, original_title: str) -> Optional[Dict]:
        """Check if we already have high-quality MusicBrainz data for this video."""

        if not self.db_manager:
            return None

        try:
            # Query for existing videos with this title that have MusicBrainz data
            query = """
                SELECT artist, song_title, musicbrainz_recording_id,
                       musicbrainz_artist_id, parse_confidence,
                       release_year, recording_length_ms
                FROM videos
                WHERE title = ?
                AND musicbrainz_recording_id IS NOT NULL
                AND musicbrainz_artist_id IS NOT NULL
                AND parse_confidence > 0.7
                ORDER BY parse_confidence DESC
                LIMIT 1
            """

            cursor = await self.db_manager.execute_query(query, (original_title,))
            row = await cursor.fetchone()

            if row:
                return {
                    "authoritative_artist": row[0],
                    "authoritative_title": row[1],
                    "musicbrainz_recording_id": row[2],
                    "musicbrainz_artist_id": row[3],
                    "parse_confidence": row[4],
                    "release_year": row[5],
                    "recording_length_ms": row[6],
                    "data_source": "database",
                }

        except Exception as e:
            logger.warning(f"Failed to check existing MusicBrainz data: {e}")

        return None

    def _apply_validation_results(
        self, parse_result: ParseResult, validation_result: ValidationResult
    ) -> ParseResult:
        """Apply validation results to enhance the parse result."""

        # Check if we should use authoritative MusicBrainz artist/title data
        use_authoritative_data = (
            validation_result.confidence_adjustment >= 0.8
            and validation_result.enriched_data
            and (
                "authoritative_artist" in validation_result.enriched_data
                or "musicbrainz_recording_id" in validation_result.enriched_data
            )
        )

        # Determine final artist and title values
        final_artist = parse_result.artist
        final_title = parse_result.song_title
        artist_corrected = False
        title_corrected = False

        if use_authoritative_data:
            # Use authoritative data from existing database or high-confidence MusicBrainz match
            if "authoritative_artist" in validation_result.enriched_data:
                if validation_result.enriched_data["authoritative_artist"] != parse_result.artist:
                    artist_corrected = True
                    self.stats["artist_corrections"] += 1
                    logger.info(
                        f"Correcting artist from '{parse_result.artist}' to '{validation_result.enriched_data['authoritative_artist']}' using MusicBrainz data"
                    )
                final_artist = validation_result.enriched_data["authoritative_artist"]

            if "authoritative_title" in validation_result.enriched_data:
                if (
                    validation_result.enriched_data["authoritative_title"]
                    != parse_result.song_title
                ):
                    title_corrected = True
                    self.stats["title_corrections"] += 1
                    logger.info(
                        f"Correcting title from '{parse_result.song_title}' to '{validation_result.enriched_data['authoritative_title']}' using MusicBrainz data"
                    )
                final_title = validation_result.enriched_data["authoritative_title"]

        # Create a copy of the parse result with potentially corrected artist/title
        enhanced_result = ParseResult(
            artist=final_artist,
            song_title=final_title,
            featured_artists=parse_result.featured_artists,
            confidence=parse_result.confidence * validation_result.confidence_adjustment,
            method=f"{parse_result.method}_mb_validated",
            pattern_used=parse_result.pattern_used,
            validation_score=validation_result.confidence_adjustment,
            alternative_results=parse_result.alternative_results.copy(),
            metadata=(parse_result.metadata or {}).copy(),
        )

        # Add validation metadata
        enhanced_result.metadata.update(
            {
                "musicbrainz_validation": "success",
                "validation_confidence": validation_result.confidence_adjustment,
                "validation_method": validation_result.validation_method,
                "mb_match_score": validation_result.mb_match_score,
                "artist_corrected": artist_corrected,
                "title_corrected": title_corrected,
            }
        )

        # Add enrichment data
        if validation_result.enriched_data:
            enhanced_result.metadata.update(validation_result.enriched_data)

        # Cap confidence at reasonable maximum
        enhanced_result.confidence = min(enhanced_result.confidence, 0.98)

        return enhanced_result

    def get_statistics(self) -> Dict:
        """Get validation pass statistics."""

        total = self.stats["total_validations"]
        success_rate = self.stats["successful_validations"] / max(total, 1)

        return {
            "total_validations": total,
            "successful_validations": self.stats["successful_validations"],
            "validation_failures": self.stats["validation_failures"],
            "success_rate": success_rate,
            "confidence_boosts": self.stats["confidence_boosts"],
            "confidence_penalties": self.stats["confidence_penalties"],
            "enrichments_added": self.stats["enrichments_added"],
            "artist_corrections": self.stats["artist_corrections"],
            "title_corrections": self.stats["title_corrections"],
            "existing_mb_data_used": self.stats["existing_mb_data_used"],
            "dependencies_available": HAS_MUSICBRAINZ,
        }
