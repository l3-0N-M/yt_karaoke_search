"""Data transformation utilities for schema compatibility."""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class DataTransformer:
    """Transforms data between old and new optimized schema formats."""

    @staticmethod
    def transform_parse_result_to_optimized(result: Dict[str, Any]) -> Dict[str, Any]:
        """Transform ParseResult data to optimized schema format."""
        transformed = result.copy()

        # Field name mapping
        if "original_artist" in transformed:
            transformed["artist"] = transformed.pop("original_artist")

        # Value transformations
        if "parse_confidence" in transformed and transformed["parse_confidence"] is not None:
            transformed["parse_confidence"] = round(transformed["parse_confidence"], 2)

        if (
            "like_dislike_to_views_ratio" in transformed
            and transformed["like_dislike_to_views_ratio"] is not None
        ):
            # Convert to percentage and round to 3 decimals
            transformed["engagement_ratio"] = round(
                transformed["like_dislike_to_views_ratio"] * 100, 3
            )
            # Remove old field
            del transformed["like_dislike_to_views_ratio"]

        # Round other confidence scores
        confidence_fields = ["confidence", "musicbrainz_confidence", "overall_confidence"]
        for field in confidence_fields:
            if field in transformed and transformed[field] is not None:
                transformed[field] = round(transformed[field], 2)

        # Round rating fields to 1 decimal
        rating_fields = ["rating", "quality_rating", "karaoke_rating"]
        for field in rating_fields:
            if field in transformed and transformed[field] is not None:
                transformed[field] = round(transformed[field], 1)

        return transformed

    @staticmethod
    def transform_video_data_to_optimized(video_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform video data to optimized schema format."""
        transformed = video_data.copy()

        # Ensure required fields have default values
        defaults = {
            "view_count": 0,
            "like_count": 0,
            "comment_count": 0,
            "duration_seconds": 0,
            "parse_confidence": None,
            "engagement_ratio": None,
        }

        for key, default_value in defaults.items():
            if key not in transformed:
                transformed[key] = default_value

        # Calculate engagement ratio if like_dislike_to_views_ratio exists
        if (
            "like_dislike_to_views_ratio" in transformed
            and transformed["like_dislike_to_views_ratio"] is not None
        ):
            transformed["engagement_ratio"] = round(
                transformed["like_dislike_to_views_ratio"] * 100, 3
            )
            del transformed["like_dislike_to_views_ratio"]

        # Field name transformations
        field_mappings = {
            "original_artist": "artist",
            "uploader": "channel_name",
            "uploader_id": "channel_id",
            "thumbnail": "thumbnail_url",
        }

        for old_field, new_field in field_mappings.items():
            if old_field in transformed:
                transformed[new_field] = transformed.pop(old_field)

        # Extract artist/song data from features if not already at top level
        features = transformed.get("features", {})
        if features:
            # Promote key feature data to top level if not already there
            if not transformed.get("artist") and features.get("original_artist"):
                transformed["artist"] = features["original_artist"]
            if not transformed.get("song_title") and features.get("song_title"):
                transformed["song_title"] = features["song_title"]
            if not transformed.get("featured_artists") and features.get("featured_artists"):
                transformed["featured_artists"] = features["featured_artists"]
            if not transformed.get("parse_confidence") and features.get("artist_confidence"):
                transformed["parse_confidence"] = features["artist_confidence"]
            if not transformed.get("release_year") and features.get("release_year"):
                transformed["release_year"] = features["release_year"]
            if not transformed.get("genre") and features.get("genre"):
                transformed["genre"] = features["genre"]

        # ALSO extract metadata from parse result metadata (for Discogs/MusicBrainz data)
        parse_metadata = transformed.get("metadata", {})
        if parse_metadata:
            # Check for release year in multiple possible field names
            if not transformed.get("release_year"):
                year_value = (
                    parse_metadata.get("release_year")
                    or parse_metadata.get("year")
                    or parse_metadata.get("release_date")
                )
                if year_value:
                    transformed["release_year"] = year_value
                    logger.info(f"DataTransformer found release_year: {year_value}")

            # Check for genre in multiple possible field names
            if not transformed.get("genre"):
                genre_value = parse_metadata.get("genre")
                if not genre_value and parse_metadata.get("genres"):
                    # If genres is a list, take the first one
                    genres = parse_metadata.get("genres")
                    if isinstance(genres, list) and genres:
                        genre_value = genres[0]
                    elif isinstance(genres, str):
                        genre_value = genres
                if genre_value:
                    transformed["genre"] = genre_value
                    logger.info(f"DataTransformer found genre: {genre_value}")

            # Also populate discogs-specific fields for tracking
            if parse_metadata.get("source") == "discogs":
                transformed["discogs_genre"] = transformed.get("genre")
                transformed["discogs_release_year"] = transformed.get("release_year")

        # Ensure quality scores are properly formatted
        quality_scores = transformed.get("quality_scores", {})
        if quality_scores:
            # Round quality score values
            for score_field in [
                "overall_score",
                "technical_score",
                "engagement_score",
                "metadata_score",
            ]:
                if score_field in quality_scores and quality_scores[score_field] is not None:
                    quality_scores[score_field] = round(quality_scores[score_field], 2)
            transformed["quality_scores"] = quality_scores

        # Ensure RYD data is properly formatted
        ryd_data = transformed.get("ryd_data", {})
        if ryd_data:
            # Round RYD values
            for field in ["ryd_rating", "ryd_confidence"]:
                if field in ryd_data and ryd_data[field] is not None:
                    ryd_data[field] = round(ryd_data[field], 2)
            transformed["ryd_data"] = ryd_data

        # Ensure engagement ratio is properly calculated and formatted
        if not transformed.get("engagement_ratio"):
            view_count = transformed.get("view_count", 0)
            like_count = transformed.get("like_count", 0)
            estimated_dislikes = transformed.get("estimated_dislikes", 0)

            if view_count > 0 and like_count is not None:
                if estimated_dislikes > 0:
                    net_engagement = like_count - estimated_dislikes
                    ratio = net_engagement / view_count
                else:
                    ratio = like_count / view_count
                transformed["engagement_ratio"] = round(ratio * 100, 3)

        # CRITICAL FIX: Preserve the original metadata field for parsing metadata extraction
        # This ensures that metadata from parsing passes (like pattern_used, swap_corrected, etc.)
        # is available when the database manager extracts it for the parsing_metadata column
        if "metadata" in video_data and video_data["metadata"]:
            transformed["metadata"] = video_data["metadata"]
            logger.debug(
                f"Preserved metadata field with keys: {list(video_data['metadata'].keys())}"
            )

        return transformed

    @staticmethod
    def create_backward_compatible_result(optimized_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create backward compatible data from optimized schema."""
        compatible = optimized_data.copy()

        # Add backward compatibility fields
        if "artist" in compatible:
            compatible["original_artist"] = compatible["artist"]

        if "engagement_ratio" in compatible and compatible["engagement_ratio"] is not None:
            # Convert back from percentage to ratio
            compatible["like_dislike_to_views_ratio"] = compatible["engagement_ratio"] / 100

        return compatible

    @staticmethod
    def round_confidence_values(data: Dict[str, Any]) -> Dict[str, Any]:
        """Round confidence values to appropriate precision."""
        rounded = data.copy()

        # Confidence fields (2 decimals)
        confidence_fields = [
            "confidence",
            "parse_confidence",
            "musicbrainz_confidence",
            "overall_confidence",
            "pattern_confidence",
            "fuzzy_confidence",
        ]

        for field in confidence_fields:
            if field in rounded and rounded[field] is not None:
                try:
                    rounded[field] = round(float(rounded[field]), 2)
                except (ValueError, TypeError):
                    logger.warning(f"Could not round confidence field {field}: {rounded[field]}")

        # Rating fields (1 decimal)
        rating_fields = ["rating", "quality_rating", "karaoke_rating"]
        for field in rating_fields:
            if field in rounded and rounded[field] is not None:
                try:
                    rounded[field] = round(float(rounded[field]), 1)
                except (ValueError, TypeError):
                    logger.warning(f"Could not round rating field {field}: {rounded[field]}")

        # Percentage fields (3 decimals)
        percentage_fields = ["engagement_ratio", "success_rate", "completion_rate"]
        for field in percentage_fields:
            if field in rounded and rounded[field] is not None:
                try:
                    rounded[field] = round(float(rounded[field]), 3)
                except (ValueError, TypeError):
                    logger.warning(f"Could not round percentage field {field}: {rounded[field]}")

        return rounded

    @staticmethod
    def merge_metadata_sources(
        musicbrainz_data: Optional[Dict[str, Any]] = None,
        discogs_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Merge metadata from MusicBrainz and Discogs sources intelligently.

        Priority:
        - MusicBrainz takes precedence for core music data (artist, title)
        - Discogs provides additional metadata (genres, year, label)
        - Use highest confidence source for conflicting data
        """
        merged = {}

        # Start with base data from the higher confidence source
        if musicbrainz_data and discogs_data:
            mb_confidence = musicbrainz_data.get("confidence", 0.0)
            discogs_confidence = discogs_data.get("confidence", 0.0)

            if mb_confidence >= discogs_confidence:
                merged.update(musicbrainz_data)
                secondary_data = discogs_data
            else:
                merged.update(discogs_data)
                secondary_data = musicbrainz_data
        elif musicbrainz_data:
            merged.update(musicbrainz_data)
            secondary_data = {}
        elif discogs_data:
            merged.update(discogs_data)
            secondary_data = {}
        else:
            return merged

        # Merge additional fields from secondary source
        if secondary_data:
            # Fields that Discogs typically provides better data for
            discogs_preferred_fields = [
                "genres",
                "styles",
                "year",
                "release_year",
                "label",
                "country",
                "format",
                "discogs_release_id",
                "discogs_master_id",
            ]

            # Fields that MusicBrainz typically provides better data for
            musicbrainz_preferred_fields = [
                "recording_id",
                "artist_id",
                "mbid",
                "musicbrainz_confidence",
            ]

            for field, value in secondary_data.items():
                if field not in merged or not merged[field]:
                    # Add missing fields
                    merged[field] = value
                elif field in discogs_preferred_fields and discogs_data and field in discogs_data:
                    # Prefer Discogs for genre/label/year data
                    merged[field] = discogs_data[field]
                elif (
                    field in musicbrainz_preferred_fields
                    and musicbrainz_data
                    and field in musicbrainz_data
                ):
                    # Prefer MusicBrainz for core music database fields
                    merged[field] = musicbrainz_data[field]

        # Special handling for genres - combine both sources
        if musicbrainz_data and discogs_data:
            mb_genres = musicbrainz_data.get("genres", [])
            discogs_genres = discogs_data.get("genres", [])

            if mb_genres and discogs_genres:
                # Combine and deduplicate genres
                combined_genres = list(set(mb_genres + discogs_genres))
                merged["genres"] = combined_genres
            elif discogs_genres:
                merged["genres"] = discogs_genres
            elif mb_genres:
                merged["genres"] = mb_genres

        # Set metadata source tags
        sources = []
        if musicbrainz_data:
            sources.append("musicbrainz")
        if discogs_data:
            sources.append("discogs")

        merged["metadata_sources"] = sources

        # Calculate combined confidence
        if musicbrainz_data and discogs_data:
            mb_conf = musicbrainz_data.get("confidence", 0.0)
            discogs_conf = discogs_data.get("confidence", 0.0)
            # Use weighted average, giving slight preference to MusicBrainz
            merged["confidence"] = round((mb_conf * 0.6 + discogs_conf * 0.4), 2)

        return merged
