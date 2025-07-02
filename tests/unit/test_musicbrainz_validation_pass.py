"""Unit tests for musicbrainz_validation_pass.py."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from collector.advanced_parser import AdvancedTitleParser, ParseResult
from collector.passes.musicbrainz_validation_pass import MusicBrainzValidationPass, ValidationResult


class TestMusicBrainzValidationPass:
    """Test cases for MusicBrainzValidationPass."""

    @pytest.fixture
    def advanced_parser(self):
        """Create a mock advanced parser."""
        parser = Mock(spec=AdvancedTitleParser)
        return parser

    @pytest.fixture
    def db_manager(self):
        """Create a mock database manager."""
        db = AsyncMock()
        db.execute_query = AsyncMock()
        cursor = AsyncMock()
        cursor.fetchone = AsyncMock()
        db.execute_query.return_value = cursor
        return db

    @pytest.fixture
    def validation_pass(self, advanced_parser, db_manager):
        """Create a MusicBrainzValidationPass instance."""
        with patch("collector.passes.musicbrainz_validation_pass.HAS_MUSICBRAINZ", True):
            return MusicBrainzValidationPass(advanced_parser, db_manager)

    @pytest.mark.asyncio
    async def test_parse_without_parse_result(self, validation_pass):
        """Test parsing without parse result returns None."""
        result = await validation_pass.parse(title="Test Song", metadata={})

        assert result is None

    @pytest.mark.asyncio
    async def test_parse_with_parse_result(self, validation_pass):
        """Test parsing with parse result."""
        parse_result = ParseResult(artist="Test Artist", song_title="Test Song", confidence=0.7)

        result = await validation_pass.parse(
            title="Test Artist - Test Song", metadata={"parse_result": parse_result}
        )

        assert result is not None
        assert validation_pass.stats["total_validations"] == 1

    @pytest.mark.asyncio
    async def test_validation_success(self, validation_pass):
        """Test successful validation."""
        web_result = ParseResult(artist="The Beatles", song_title="Yesterday", confidence=0.7)

        # Mock MusicBrainz search result
        mb_match = Mock()
        mb_match.artist_name = "The Beatles"
        mb_match.song_title = "Yesterday"
        mb_match.score = 95
        mb_match.recording_id = "test-recording-id"
        mb_match.artist_id = "test-artist-id"
        mb_match.confidence = 0.9
        mb_match.metadata = {}

        validation_pass.mb_search_pass._search_musicbrainz = AsyncMock(return_value=[mb_match])

        validation_result = await validation_pass._validate_against_musicbrainz(
            web_result, "The Beatles - Yesterday"
        )

        assert validation_result.validated
        assert validation_result.confidence_adjustment >= 0.7

    @pytest.mark.asyncio
    async def test_validation_with_existing_data(self, validation_pass, db_manager):
        """Test validation using existing database data."""
        web_result = ParseResult(artist="Known Artist", song_title="Known Song", confidence=0.6)

        # Mock existing MusicBrainz data in database
        db_manager.execute_query.return_value.fetchone.return_value = (
            "Known Artist",
            "Known Song",
            "mb-recording-id",
            "mb-artist-id",
            0.95,  # High confidence
            1975,  # Release year
            180000,  # Duration in ms
        )

        validation_result = await validation_pass._validate_against_musicbrainz(
            web_result, "Known Artist - Known Song"
        )

        assert validation_result.validated
        assert validation_result.validation_method == "existing_musicbrainz_data"
        assert validation_pass.stats["existing_mb_data_used"] == 1

    @pytest.mark.asyncio
    async def test_validation_failure_no_match(self, validation_pass):
        """Test validation failure when no MusicBrainz match found."""
        web_result = ParseResult(artist="Unknown Artist", song_title="Unknown Song", confidence=0.7)

        # Mock no existing MusicBrainz data in database
        validation_pass._check_existing_musicbrainz_data = AsyncMock(return_value=None)

        # Mock no MusicBrainz search results
        validation_pass.mb_search_pass._search_musicbrainz = AsyncMock(return_value=[])

        validation_result = await validation_pass._validate_against_musicbrainz(
            web_result, "Unknown Artist - Unknown Song"
        )

        assert not validation_result.validated
        assert validation_result.validation_method == "no_mb_results"

    def test_calculate_validation_confidence(self, validation_pass):
        """Test validation confidence calculation."""
        parse_result = ParseResult(artist="Test Artist", song_title="Test Song", confidence=0.7)

        mb_match = Mock()
        mb_match.artist_name = "Test Artist"
        mb_match.song_title = "Test Song"
        mb_match.score = 90

        confidence = validation_pass._calculate_validation_confidence(
            parse_result, mb_match, "Test Artist - Test Song"
        )

        assert confidence > 0.7
        assert confidence <= 1.5

    def test_calculate_validation_confidence_mismatch(self, validation_pass):
        """Test confidence calculation with mismatched names."""
        parse_result = ParseResult(artist="ABC", song_title="XYZ", confidence=0.7)

        mb_match = Mock()
        mb_match.artist_name = "Totally Different Artist Name"
        mb_match.song_title = "Completely Different Song Title"
        mb_match.score = 80

        confidence = validation_pass._calculate_validation_confidence(
            parse_result, mb_match, "Test"
        )

        # With very different names, confidence should be low
        assert confidence < 0.5  # Should be significantly penalized

    def test_extract_enrichment_data(self, validation_pass):
        """Test extraction of enrichment data from MusicBrainz match."""
        mb_match = Mock()
        mb_match.recording_id = "rec-123"
        mb_match.artist_id = "art-456"
        mb_match.confidence = 0.85
        mb_match.score = 92
        mb_match.metadata = {
            "releases": [{"date": "1995-05-15"}, {"date": "1996"}],
            "length": 240000,
            "disambiguation": "live version",
        }

        enrichment = validation_pass._extract_enrichment_data(mb_match)

        assert enrichment["musicbrainz_recording_id"] == "rec-123"
        assert enrichment["musicbrainz_artist_id"] == "art-456"
        assert enrichment["release_year"] == 1995
        assert enrichment["recording_length_ms"] == 240000
        assert enrichment["musicbrainz_disambiguation"] == "live version"

    def test_apply_validation_results_correction(self, validation_pass):
        """Test applying validation results with corrections."""
        parse_result = ParseResult(
            artist="beatles", song_title="yesterday", confidence=0.7  # Lowercase  # Lowercase
        )

        validation_result = ValidationResult(
            validated=True,
            confidence_adjustment=1.2,
            enriched_data={
                "authoritative_artist": "The Beatles",
                "authoritative_title": "Yesterday",
                "musicbrainz_recording_id": "test-id",
            },
            validation_method="musicbrainz_match",
        )

        enhanced = validation_pass._apply_validation_results(parse_result, validation_result)

        assert enhanced.artist == "The Beatles"
        assert enhanced.song_title == "Yesterday"
        assert enhanced.confidence > parse_result.confidence
        assert validation_pass.stats["artist_corrections"] == 1
        assert validation_pass.stats["title_corrections"] == 1

    def test_apply_validation_results_no_correction(self, validation_pass):
        """Test applying validation results without corrections."""
        parse_result = ParseResult(artist="Test Artist", song_title="Test Song", confidence=0.8)

        validation_result = ValidationResult(
            validated=False, confidence_adjustment=0.9, validation_method="low_mb_similarity"
        )

        enhanced = validation_pass._apply_validation_results(parse_result, validation_result)

        assert enhanced.artist == "Test Artist"
        assert enhanced.song_title == "Test Song"
        assert enhanced.confidence == 0.8 * 0.9

    @pytest.mark.asyncio
    async def test_validation_with_swapped_artist_title(self, validation_pass):
        """Test validation with swapped artist/title."""
        web_result = ParseResult(
            artist="Yesterday",  # Actually the song
            song_title="The Beatles",  # Actually the artist
            confidence=0.6,
        )

        # Mock search with swapped query
        mb_match = Mock()
        mb_match.artist_name = "The Beatles"
        mb_match.song_title = "Yesterday"
        mb_match.score = 90
        mb_match.recording_id = "test-id"
        mb_match.artist_id = "test-artist"
        mb_match.confidence = 0.85
        mb_match.metadata = {}

        validation_pass.mb_search_pass._search_musicbrainz = AsyncMock(
            side_effect=[[], [mb_match]]  # No results for original  # Results for swapped
        )

        validation_result = await validation_pass._validate_against_musicbrainz(
            web_result, "Yesterday - The Beatles"
        )

        assert validation_result.validated

    def test_statistics_collection(self, validation_pass):
        """Test statistics reporting."""
        validation_pass.stats = {
            "total_validations": 100,
            "successful_validations": 75,
            "validation_failures": 25,
            "confidence_boosts": 60,
            "confidence_penalties": 15,
            "enrichments_added": 70,
            "artist_corrections": 10,
            "title_corrections": 8,
            "existing_mb_data_used": 20,
        }

        stats = validation_pass.get_statistics()

        assert stats["total_validations"] == 100
        assert stats["success_rate"] == 0.75
        assert stats["artist_corrections"] == 10
        assert stats["dependencies_available"]

    @pytest.mark.asyncio
    async def test_parse_without_musicbrainz(self):
        """Test behavior when MusicBrainz is not available."""
        with patch("collector.passes.musicbrainz_validation_pass.HAS_MUSICBRAINZ", False):
            validation_pass = MusicBrainzValidationPass(Mock(), Mock())

            result = await validation_pass.parse(title="Test", metadata={"parse_result": Mock()})

            assert result is None
