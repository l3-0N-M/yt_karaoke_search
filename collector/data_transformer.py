"""Data transformation utilities for schema compatibility."""

import logging
from typing import Any, Dict

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
        }

        for old_field, new_field in field_mappings.items():
            if old_field in transformed:
                transformed[new_field] = transformed.pop(old_field)

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
