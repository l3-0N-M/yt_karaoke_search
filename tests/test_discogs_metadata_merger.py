"""Tests for Discogs metadata merging functionality."""

import pytest

from collector.data_transformer import DataTransformer


class TestDiscogsMetadataMerger:
    """Test the metadata merger for MusicBrainz and Discogs data."""

    def test_merge_empty_sources(self):
        """Test merging with empty or None sources."""
        result = DataTransformer.merge_metadata_sources()
        assert result == {}
        
        result = DataTransformer.merge_metadata_sources(None, None)
        assert result == {}

    def test_merge_musicbrainz_only(self):
        """Test merging with only MusicBrainz data."""
        mb_data = {
            "original_artist": "MusicBrainz Artist",
            "song_title": "MusicBrainz Song",
            "confidence": 0.8,
            "recording_id": "mb-123",
            "artist_id": "mb-artist-456"
        }
        
        result = DataTransformer.merge_metadata_sources(musicbrainz_data=mb_data)
        
        assert result["original_artist"] == "MusicBrainz Artist"
        assert result["song_title"] == "MusicBrainz Song"
        assert result["confidence"] == 0.8
        assert result["recording_id"] == "mb-123"
        assert result["metadata_sources"] == ["musicbrainz"]

    def test_merge_discogs_only(self):
        """Test merging with only Discogs data."""
        discogs_data = {
            "original_artist": "Discogs Artist",
            "song_title": "Discogs Song",
            "confidence": 0.9,
            "discogs_release_id": "discogs-789",
            "year": 2020,
            "genres": ["Pop", "Rock"],
            "label": "Test Label"
        }
        
        result = DataTransformer.merge_metadata_sources(discogs_data=discogs_data)
        
        assert result["original_artist"] == "Discogs Artist"
        assert result["song_title"] == "Discogs Song"
        assert result["confidence"] == 0.9
        assert result["discogs_release_id"] == "discogs-789"
        assert result["year"] == 2020
        assert result["genres"] == ["Pop", "Rock"]
        assert result["metadata_sources"] == ["discogs"]

    def test_merge_musicbrainz_higher_confidence(self):
        """Test merging when MusicBrainz has higher confidence."""
        mb_data = {
            "original_artist": "MB Artist",
            "song_title": "MB Song",
            "confidence": 0.9,
            "recording_id": "mb-123"
        }
        
        discogs_data = {
            "original_artist": "Discogs Artist", 
            "song_title": "Discogs Song",
            "confidence": 0.7,
            "discogs_release_id": "discogs-456",
            "year": 2020,
            "genres": ["Pop"]
        }
        
        result = DataTransformer.merge_metadata_sources(mb_data, discogs_data)
        
        # Should use MusicBrainz as primary (higher confidence)
        assert result["original_artist"] == "MB Artist"
        assert result["song_title"] == "MB Song"
        assert result["recording_id"] == "mb-123"
        
        # Should add Discogs metadata that's missing
        assert result["discogs_release_id"] == "discogs-456"
        assert result["year"] == 2020
        assert result["genres"] == ["Pop"]
        
        # Should combine sources
        assert set(result["metadata_sources"]) == {"musicbrainz", "discogs"}

    def test_merge_discogs_higher_confidence(self):
        """Test merging when Discogs has higher confidence."""
        mb_data = {
            "original_artist": "MB Artist",
            "song_title": "MB Song", 
            "confidence": 0.6,
            "recording_id": "mb-123"
        }
        
        discogs_data = {
            "original_artist": "Discogs Artist",
            "song_title": "Discogs Song",
            "confidence": 0.8,
            "discogs_release_id": "discogs-456",
            "year": 2020
        }
        
        result = DataTransformer.merge_metadata_sources(mb_data, discogs_data)
        
        # Should use Discogs as primary (higher confidence)
        assert result["original_artist"] == "Discogs Artist"
        assert result["song_title"] == "Discogs Song"
        assert result["discogs_release_id"] == "discogs-456"
        
        # Should add MusicBrainz metadata that's missing
        assert result["recording_id"] == "mb-123"

    def test_merge_discogs_preferred_fields(self):
        """Test that Discogs-preferred fields take precedence."""
        mb_data = {
            "original_artist": "MB Artist",
            "confidence": 0.8,
            "genres": ["MB Genre"],
            "year": 2018,
            "label": "MB Label"
        }
        
        discogs_data = {
            "original_artist": "Discogs Artist",
            "confidence": 0.7,  # Lower confidence
            "genres": ["Discogs Genre"],
            "year": 2020,
            "label": "Discogs Label",
            "styles": ["Alternative"],
            "country": "US"
        }
        
        result = DataTransformer.merge_metadata_sources(mb_data, discogs_data)
        
        # MusicBrainz should be primary due to higher confidence
        assert result["original_artist"] == "MB Artist"
        
        # But Discogs should win for genre/label/year data
        assert result["genres"] == ["Discogs Genre"]
        assert result["year"] == 2020
        assert result["label"] == "Discogs Label"
        assert result["styles"] == ["Alternative"]
        assert result["country"] == "US"

    def test_merge_musicbrainz_preferred_fields(self):
        """Test that MusicBrainz-preferred fields take precedence."""
        mb_data = {
            "original_artist": "MB Artist",
            "confidence": 0.7,  # Lower confidence
            "recording_id": "mb-123",
            "artist_id": "mb-artist-456",
            "mbid": "mb-song-789"
        }
        
        discogs_data = {
            "original_artist": "Discogs Artist",
            "confidence": 0.8,  # Higher confidence
            "recording_id": "discogs-should-not-override",
            "artist_id": "discogs-artist-should-not-override"
        }
        
        result = DataTransformer.merge_metadata_sources(mb_data, discogs_data)
        
        # Discogs should be primary due to higher confidence
        assert result["original_artist"] == "Discogs Artist"
        
        # But MusicBrainz should win for its preferred fields
        assert result["recording_id"] == "mb-123"
        assert result["artist_id"] == "mb-artist-456"
        assert result["mbid"] == "mb-song-789"

    def test_merge_genres_combination(self):
        """Test that genres from both sources are combined."""
        mb_data = {
            "original_artist": "Artist",
            "confidence": 0.8,
            "genres": ["Rock", "Alternative"]
        }
        
        discogs_data = {
            "original_artist": "Artist",
            "confidence": 0.7,
            "genres": ["Rock", "Pop", "Electronic"]
        }
        
        result = DataTransformer.merge_metadata_sources(mb_data, discogs_data)
        
        # Should combine and deduplicate genres
        combined_genres = result["genres"]
        assert "Rock" in combined_genres
        assert "Alternative" in combined_genres  
        assert "Pop" in combined_genres
        assert "Electronic" in combined_genres
        
        # Should deduplicate
        assert combined_genres.count("Rock") == 1

    def test_merge_combined_confidence_calculation(self):
        """Test combined confidence calculation."""
        mb_data = {
            "original_artist": "Artist",
            "confidence": 0.8
        }
        
        discogs_data = {
            "original_artist": "Artist",
            "confidence": 0.6
        }
        
        result = DataTransformer.merge_metadata_sources(mb_data, discogs_data)
        
        # Should be weighted average: 0.8 * 0.6 + 0.6 * 0.4 = 0.48 + 0.24 = 0.72
        expected_confidence = round(0.8 * 0.6 + 0.6 * 0.4, 2)
        assert result["confidence"] == expected_confidence

    def test_merge_missing_fields_filled(self):
        """Test that missing fields are filled from secondary source."""
        mb_data = {
            "original_artist": "MB Artist",
            "confidence": 0.9,
            "recording_id": "mb-123"
            # Missing: song_title, year, genres
        }
        
        discogs_data = {
            "original_artist": "Discogs Artist",  # Should not override
            "song_title": "Discogs Song",        # Should be added
            "confidence": 0.7,
            "year": 2020,                        # Should be added
            "genres": ["Pop"],                   # Should be added
            "country": "US"                      # Should be added
        }
        
        result = DataTransformer.merge_metadata_sources(mb_data, discogs_data)
        
        # Primary fields from MusicBrainz (higher confidence)
        assert result["original_artist"] == "MB Artist"
        assert result["recording_id"] == "mb-123"
        
        # Missing fields filled from Discogs
        assert result["song_title"] == "Discogs Song"
        assert result["year"] == 2020
        assert result["genres"] == ["Pop"]
        assert result["country"] == "US"

    def test_merge_empty_genres_handling(self):
        """Test handling of empty genre lists."""
        mb_data = {
            "original_artist": "Artist",
            "confidence": 0.8,
            "genres": []
        }
        
        discogs_data = {
            "original_artist": "Artist",
            "confidence": 0.7,
            "genres": ["Pop", "Rock"]
        }
        
        result = DataTransformer.merge_metadata_sources(mb_data, discogs_data)
        
        # Should use Discogs genres since MusicBrainz is empty
        assert result["genres"] == ["Pop", "Rock"]

    def test_merge_no_genres_in_either(self):
        """Test merging when neither source has genres."""
        mb_data = {
            "original_artist": "Artist",
            "confidence": 0.8
        }
        
        discogs_data = {
            "original_artist": "Artist", 
            "confidence": 0.7,
            "year": 2020
        }
        
        result = DataTransformer.merge_metadata_sources(mb_data, discogs_data)
        
        # Should not have genres field or should be empty
        assert "genres" not in result or not result.get("genres")

    def test_merge_metadata_sources_list(self):
        """Test that metadata sources are properly tracked."""
        mb_data = {"original_artist": "Artist", "confidence": 0.8}
        discogs_data = {"original_artist": "Artist", "confidence": 0.7}
        
        result = DataTransformer.merge_metadata_sources(mb_data, discogs_data)
        
        assert "metadata_sources" in result
        assert set(result["metadata_sources"]) == {"musicbrainz", "discogs"}
        
        # Test single source
        result_mb_only = DataTransformer.merge_metadata_sources(musicbrainz_data=mb_data)
        assert result_mb_only["metadata_sources"] == ["musicbrainz"]
        
        result_discogs_only = DataTransformer.merge_metadata_sources(discogs_data=discogs_data)
        assert result_discogs_only["metadata_sources"] == ["discogs"]

    def test_merge_equal_confidence(self):
        """Test merging when confidence scores are equal."""
        mb_data = {
            "original_artist": "MB Artist",
            "confidence": 0.8,
            "recording_id": "mb-123"
        }
        
        discogs_data = {
            "original_artist": "Discogs Artist",
            "confidence": 0.8,  # Same confidence
            "year": 2020
        }
        
        result = DataTransformer.merge_metadata_sources(mb_data, discogs_data)
        
        # When equal, MusicBrainz should win (>= condition)
        assert result["original_artist"] == "MB Artist"
        assert result["recording_id"] == "mb-123"
        assert result["year"] == 2020