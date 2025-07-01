"""Unit tests for data_transformer.py."""

import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from collector.advanced_parser import ParseResult
from collector.data_transformer import DataTransformer


class TestDataTransformer:
    """Test cases for DataTransformer."""

    def test_transform_parse_result_basic(self):
        """Test basic transformation of ParseResult to optimized format."""
        parse_result = ParseResult(
            artist="Test Artist",
            song_title="Test Song",
            confidence=0.95,
            featured_artists="Featured Artist",
            metadata={"release_year": 2020, "genre": "Pop"},
        )

        video_info: Dict[str, Any] = {
            "video_id": "test123",
            "url": "https://youtube.com/watch?v=test123",
            "title": "Test Artist - Test Song (Acoustic)",
            "description": "Test description",
            "duration_seconds": 240,
            "view_count": 10000,
            "channel_name": "Test Channel",
        }

        # Combine parse result and video info
        combined_data = video_info.copy()
        combined_data["artist"] = parse_result.artist
        combined_data["song_title"] = parse_result.song_title
        combined_data["parse_confidence"] = parse_result.confidence
        combined_data["featured_artists"] = parse_result.featured_artists or ""
        combined_data["metadata"] = parse_result.metadata or {}

        result = DataTransformer.transform_video_data_to_optimized(combined_data)

        assert result["video_id"] == "test123"
        assert result["artist"] == "Test Artist"
        assert result["song_title"] == "Test Song"
        assert result["featured_artists"] == "Featured Artist"
        assert result["release_year"] == 2020
        assert result["genre"] == "Pop"
        assert result["parse_confidence"] == 0.95

    def test_transform_parse_result_with_nulls(self):
        """Test transformation with null/missing values."""
        parse_result = ParseResult(artist="Artist", song_title="Song", confidence=0.7)

        video_info: Dict[str, Any] = {
            "video_id": "test456",
            "url": "https://youtube.com/watch?v=test456",
            "title": "Artist - Song",
        }

        # Combine parse result and video info
        combined_data = video_info.copy()
        combined_data["artist"] = parse_result.artist
        combined_data["song_title"] = parse_result.song_title
        combined_data["parse_confidence"] = parse_result.confidence
        combined_data["featured_artists"] = parse_result.featured_artists or ""
        combined_data["metadata"] = parse_result.metadata or {}

        result = DataTransformer.transform_video_data_to_optimized(combined_data)

        assert result["video_id"] == "test456"
        assert result["featured_artists"] == ""
        assert result.get("release_year") is None
        assert result.get("genre") is None
        assert "description" not in result  # Not provided in input

    def test_transform_video_data_to_optimized(self):
        """Test transformation of raw video data to optimized format."""
        video_data: Dict[str, Any] = {
            "id": "vid123",
            "webpage_url": "https://youtube.com/watch?v=vid123",
            "title": "Full Video Title",
            "description": "Video description",
            "duration": 300,
            "view_count": 50000,
            "like_count": 2000,
            "comment_count": 100,
            "upload_date": "20230615",
            "thumbnail": "https://example.com/thumb.jpg",
            "uploader": "Channel Name",
            "uploader_id": "channel123",
            # Parsed data
            "artist": "Parsed Artist",
            "song_title": "Parsed Song",
            "featured_artists": ["Feat1", "Feat2"],
            "release_year": 2023,
            "genre": "Rock",
            "parse_confidence": 0.85,
        }

        result = DataTransformer.transform_video_data_to_optimized(video_data)

        assert result["video_id"] == "vid123"
        assert result["duration_seconds"] == 300
        assert result["view_count"] == 50000
        assert result["upload_date"] == "20230615"
        assert result["channel_id"] == "channel123"
        assert result["artist"] == "Parsed Artist"
        assert result["featured_artists"] == "Feat1, Feat2"

    def test_transform_featured_artists_handling(self):
        """Test various featured artists formats."""
        test_cases = [
            # List of artists
            (["Artist1", "Artist2"], "Artist1, Artist2"),
            # Single artist in list
            (["Solo"], "Solo"),
            # Empty list
            ([], ""),
            # String input
            ("Artist1 & Artist2", "Artist1 & Artist2"),
            # None
            (None, ""),
            # Complex list
            (["A", "B", "C", "D"], "A, B, C, D"),
        ]

        for input_artists, expected_output in test_cases:
            video_data: Dict[str, Any] = {"id": "test", "featured_artists": input_artists}
            result = DataTransformer.transform_video_data_to_optimized(video_data)
            assert result["featured_artists"] == expected_output

    def test_transform_quality_score_calculation(self):
        """Test quality score calculation during transformation."""
        video_data: Dict[str, Any] = {
            "id": "test",
            "view_count": 100000,
            "like_count": 5000,
            "comment_count": 500,
            "duration": 240,
            "parse_confidence": 0.9,
        }

        result = DataTransformer.transform_video_data_to_optimized(video_data)

        # Quality score should be calculated
        assert "quality_score" in result
        assert isinstance(result["quality_score"], (int, float))
        assert 0 <= result["quality_score"] <= 100

    def test_transform_engagement_ratio_calculation(self):
        """Test engagement ratio calculation."""
        video_data: Dict[str, Any] = {
            "id": "test",
            "view_count": 10000,
            "like_count": 500,
            "comment_count": 50,
        }

        result = DataTransformer.transform_video_data_to_optimized(video_data)

        # Engagement ratio should be calculated
        assert "engagement_ratio" in result
        expected_ratio = ((500 + 50) / 10000) * 100
        assert abs(result["engagement_ratio"] - expected_ratio) < 0.01

    def test_transform_date_handling(self):
        """Test various date format handling."""
        test_cases = [
            # YouTube format
            ("20230615", "20230615"),
            # Already formatted
            ("2023-06-15", "2023-06-15"),
            # Invalid date
            ("invalid", "invalid"),
            # None
            (None, None),
        ]

        for input_date, expected_date in test_cases:
            video_data: Dict[str, Any] = {"id": "test", "upload_date": input_date}
            result = DataTransformer.transform_video_data_to_optimized(video_data)
            assert result["upload_date"] == expected_date

    def test_transform_missing_required_fields(self):
        """Test handling of missing required fields."""
        # Minimal data
        video_data: Dict[str, Any] = {"id": "minimal123"}

        result = DataTransformer.transform_video_data_to_optimized(video_data)

        # Should provide defaults for required fields
        assert result["video_id"] == "minimal123"
        assert result["url"] is not None
        assert result["title"] is None
        assert result["view_count"] == 0
        assert result["like_count"] == 0

    def test_transform_field_mapping(self):
        """Test field name mapping from various sources."""
        video_data: Dict[str, Any] = {
            # YouTube field names
            "id": "yt123",
            "webpage_url": "https://youtube.com/watch?v=yt123",
            "uploader": "YouTube Channel",
            "uploader_id": "UCxxxxxx",
            # Alternative field names
            "channel": "Alt Channel",
            "channel_id": "alt_channel_id",
            "views": 12345,
            "likes": 678,
        }

        result = DataTransformer.transform_video_data_to_optimized(video_data)

        # Should map to standard names
        assert result["video_id"] == "yt123"
        assert result["url"] == "https://youtube.com/watch?v=yt123"
        assert result["channel_name"] == "YouTube Channel"
        assert result["channel_id"] == "UCxxxxxx"

    def test_transform_discogs_data_integration(self):
        """Test integration of Discogs data."""
        video_data: Dict[str, Any] = {
            "id": "test",
            "discogs_data": {
                "artist_id": "12345",
                "artist_name": "Discogs Artist",
                "release_id": "67890",
                "release_title": "Album Title",
                "year": 1985,
                "label": "Test Records",
                "genre": "Rock",
                "style": "Classic Rock",
            },
        }

        result = DataTransformer.transform_video_data_to_optimized(video_data)

        assert result["discogs_artist_id"] == "12345"
        assert result["discogs_artist_name"] == "Discogs Artist"
        assert result["discogs_release_year"] == 1985
        assert result["discogs_checked"] == 1

    def test_transform_musicbrainz_data_integration(self):
        """Test integration of MusicBrainz data."""
        video_data: Dict[str, Any] = {
            "id": "test",
            "musicbrainz_data": {
                "recording_id": "mb-rec-123",
                "artist_id": "mb-art-456",
                "tags": ["rock", "pop"],
                "length": 240000,
            },
        }

        result = DataTransformer.transform_video_data_to_optimized(video_data)

        assert result["musicbrainz_checked"] == 1
        # MusicBrainz data should be in additional metadata
        if "additional_metadata" in result:
            assert "musicbrainz_data" in result["additional_metadata"]

    def test_transform_sanitization(self):
        """Test data sanitization during transformation."""
        video_data: Dict[str, Any] = {
            "id": "test",
            "title": 'Title with <script>alert("xss")</script>',
            "description": "Description with\x00null\x01bytes",
            "artist": "  Artist with spaces  ",
            "featured_artists": ["  Artist1  ", "  Artist2  "],
        }

        result = DataTransformer.transform_video_data_to_optimized(video_data)

        # Should sanitize data
        assert "<script>" not in result["title"]
        assert "\x00" not in result["description"]
        assert result["artist"] == "Artist with spaces"  # Trimmed
        assert "Artist1" in result["featured_artists"]  # Trimmed

    def test_transform_type_conversions(self):
        """Test type conversions during transformation."""
        video_data: Dict[str, Any] = {
            "id": "test",
            "view_count": "12345",  # String number
            "like_count": 67.89,  # Float
            "duration": "240",  # String duration
            "parse_confidence": "0.85",  # String float
        }

        result = DataTransformer.transform_video_data_to_optimized(video_data)

        assert isinstance(result["view_count"], int)
        assert result["view_count"] == 12345
        assert isinstance(result["like_count"], int)
        assert result["like_count"] == 67
        assert isinstance(result["duration_seconds"], int)
        assert isinstance(result["parse_confidence"], float)

    def test_transform_edge_cases(self):
        """Test various edge cases."""
        # Very large numbers
        video_data1 = {"id": "test1", "view_count": 999999999999, "like_count": 999999999}
        result1 = DataTransformer.transform_video_data_to_optimized(video_data1)
        assert isinstance(result1["view_count"], int)

        # Special characters
        video_data2 = {"id": "test2", "title": "Title with émojis 🎵🎤", "artist": "Beyoncé"}
        result2 = DataTransformer.transform_video_data_to_optimized(video_data2)
        assert "🎵" in result2["title"]
        assert result2["artist"] == "Beyoncé"

        # Empty strings vs None
        video_data3 = {"id": "test3", "artist": "", "song_title": None}
        result3 = DataTransformer.transform_video_data_to_optimized(video_data3)
        assert result3["artist"] == ""
        assert result3["song_title"] is None

    def test_transform_with_processing_result(self):
        """Test transformation with ProcessingResult wrapper."""
        from collections import namedtuple

        ProcessingResult = namedtuple("ProcessingResult", ["video_data", "parse_result"])

        parse_result = ParseResult(
            artist="Wrapped Artist", song_title="Wrapped Song", confidence=0.9
        )

        video_data: Dict[str, Any] = {"id": "wrapped123", "title": "Original Title"}

        processing_result = ProcessingResult(video_data=video_data, parse_result=parse_result)

        # Should handle both direct video_data and wrapped ProcessingResult
        result = DataTransformer.transform_video_data_to_optimized(processing_result.__dict__)
        assert result["video_id"] == "wrapped123"
