"""Unit tests for validation_corrector.py."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from collector.advanced_parser import ParseResult
from collector.validation_corrector import ValidationCorrector


class TestValidationCorrector:
    """Test cases for ValidationCorrector."""

    @pytest.fixture
    def db_manager(self):
        """Create a mock database manager."""
        db = AsyncMock()
        db.execute = AsyncMock()
        db.fetchone = AsyncMock()
        db.fetchall = AsyncMock()
        return db

    @pytest.fixture
    def corrector(self):
        """Create a ValidationCorrector instance."""
        return ValidationCorrector()

    @pytest.mark.asyncio
    async def test_validate_and_correct_basic(self, corrector):
        """Test basic validation and correction."""
        parse_result = ParseResult(artist="Test Artist", song_title="Test Song", confidence=0.8)

        corrected = await corrector.validate_and_correct(
            parse_result, "Test Artist - Test Song (Karaoke)"
        )

        assert corrected.artist == "Test Artist"
        assert corrected.song_title == "Test Song"
        assert corrected.confidence >= 0.8

    @pytest.mark.asyncio
    async def test_artist_name_correction(self, corrector, db_manager):
        """Test artist name correction from database."""
        # Mock database response with known artist
        db_manager.fetchall.return_value = [("The Beatles", 10), ("Beatles", 5)]

        parse_result = ParseResult(
            artist="beatles", song_title="Yesterday", confidence=0.7  # Lowercase, missing "The"
        )

        corrected = await corrector.validate_and_correct(
            parse_result, "beatles - yesterday karaoke"
        )

        # Should correct to most common form
        assert corrected.artist == "The Beatles"
        assert corrected.confidence > 0.7  # Should boost confidence

    @pytest.mark.asyncio
    async def test_song_title_correction(self, corrector, db_manager):
        """Test song title correction."""
        # Mock database response
        db_manager.fetchall.side_effect = [
            [],  # No artist corrections
            [("Bohemian Rhapsody", 15)],  # Song title match
        ]

        parse_result = ParseResult(
            artist="Queen", song_title="bohemian rapsody", confidence=0.6  # Misspelled
        )

        corrected = await corrector.validate_and_correct(parse_result, "Queen - bohemian rapsody")

        assert corrected.song_title == "Bohemian Rhapsody"

    @pytest.mark.asyncio
    async def test_common_misspelling_correction(self, corrector):
        """Test correction of common misspellings."""
        parse_result = ParseResult(
            artist="Beetles", song_title="Let It Be", confidence=0.7  # Common misspelling
        )

        corrected = await corrector.validate_and_correct(parse_result, "Beetles - Let It Be")

        # Should correct common misspelling
        assert corrected.artist in ["Beatles", "The Beatles"]

    @pytest.mark.asyncio
    async def test_featuring_artist_normalization(self, corrector):
        """Test normalization of featuring artists."""
        parse_result = ParseResult(
            artist="Artist A feat. Artist B", song_title="Collaboration Song", confidence=0.8
        )

        corrected = await corrector.validate_and_correct(
            parse_result, "Artist A feat. Artist B - Collaboration Song"
        )

        # Should normalize featuring format
        assert corrected.artist == "Artist A"
        assert corrected.featured_artists == ["Artist B"]

    @pytest.mark.asyncio
    async def test_year_validation(self, corrector):
        """Test year validation and correction."""
        parse_result = ParseResult(
            artist="Test Artist",
            song_title="Test Song",
            confidence=0.8,
            metadata={"release_year": 2025},  # Future year
        )

        corrected = await corrector.validate_and_correct(
            parse_result, "Test Artist - Test Song (2025)"
        )

        # Should remove invalid future year
        release_year = corrected.metadata.get("release_year")
        assert release_year is None or release_year < 2025

    @pytest.mark.asyncio
    async def test_confidence_adjustment(self, corrector, db_manager):
        """Test confidence score adjustment."""
        # Mock high-confidence existing data
        db_manager.fetchone.return_value = ("Known Artist", "Known Song", 0.95)  # High confidence

        parse_result = ParseResult(
            artist="Known Artist", song_title="Known Song", confidence=0.5  # Low initial confidence
        )

        corrected = await corrector.validate_and_correct(parse_result, "Known Artist - Known Song")

        # Should boost confidence for known good matches
        assert corrected.confidence > 0.5

    @pytest.mark.asyncio
    async def test_special_character_handling(self, corrector):
        """Test handling of special characters."""
        parse_result = ParseResult(artist="Beyoncé", song_title="Déjà Vu", confidence=0.7)

        corrected = await corrector.validate_and_correct(
            parse_result, "Beyonce - Deja Vu"  # Without special chars
        )

        # Should handle special characters properly
        assert corrected.artist in ["Beyoncé", "Beyonce"]
        assert corrected.song_title in ["Déjà Vu", "Deja Vu"]

    @pytest.mark.asyncio
    async def test_remix_version_handling(self, corrector):
        """Test handling of remixes and versions."""
        parse_result = ParseResult(
            artist="Test Artist", song_title="Test Song (Remix)", confidence=0.8
        )

        corrected = await corrector.validate_and_correct(
            parse_result, "Test Artist - Test Song Remix"
        )

        # Should normalize remix notation
        assert "remix" in corrected.song_title.lower()

    @pytest.mark.asyncio
    async def test_null_input_handling(self, corrector):
        """Test handling of null/empty inputs."""
        parse_result = ParseResult(artist=None, song_title="Test Song", confidence=0.5)

        corrected = await corrector.validate_and_correct(parse_result, "Test Song")

        # Should handle gracefully
        assert corrected is not None
        assert corrected.song_title == "Test Song"

    @pytest.mark.asyncio
    async def test_database_error_handling(self, corrector, db_manager):
        """Test handling of database errors."""
        # Make database fail
        db_manager.fetchall.side_effect = Exception("DB Error")

        parse_result = ParseResult(artist="Test Artist", song_title="Test Song", confidence=0.7)

        corrected = await corrector.validate_and_correct(parse_result, "Test Artist - Test Song")

        # Should return original on error
        assert corrected.artist == "Test Artist"
        assert corrected.song_title == "Test Song"

    @pytest.mark.asyncio
    async def test_case_sensitivity(self, corrector, db_manager):
        """Test case-sensitive matching."""
        # Mock case variations in database
        db_manager.fetchall.return_value = [("ACDC", 5), ("AC/DC", 20), ("Ac/Dc", 2)]

        parse_result = ParseResult(artist="acdc", song_title="Thunderstruck", confidence=0.6)

        corrected = await corrector.validate_and_correct(parse_result, "acdc - thunderstruck")

        # Should use most common case format
        assert corrected.artist == "AC/DC"

    @pytest.mark.asyncio
    async def test_multiple_corrections(self, corrector, db_manager):
        """Test multiple simultaneous corrections."""
        # Mock both artist and song corrections
        db_manager.fetchall.side_effect = [
            [("Guns N' Roses", 30)],  # Artist correction
            [("Sweet Child O' Mine", 25)],  # Song correction
        ]

        parse_result = ParseResult(
            artist="guns and roses", song_title="sweet child of mine", confidence=0.5
        )

        corrected = await corrector.validate_and_correct(
            parse_result, "guns and roses - sweet child of mine"
        )

        assert corrected.artist == "Guns N' Roses"
        assert corrected.song_title == "Sweet Child O' Mine"
        assert corrected.confidence > 0.5

    def test_get_statistics(self, corrector):
        """Test statistics collection."""
        stats = corrector.get_statistics()

        assert "total_validations" in stats
        assert "corrections_made" in stats
        assert "confidence_improvements" in stats

    @pytest.mark.asyncio
    async def test_batch_validation(self, corrector):
        """Test batch validation of multiple results."""
        parse_results = [
            ParseResult(artist="Artist1", song_title="Song1", confidence=0.7),
            ParseResult(artist="Artist2", song_title="Song2", confidence=0.8),
            ParseResult(artist="Artist3", song_title="Song3", confidence=0.6),
        ]

        corrected_results = await corrector.validate_batch(parse_results)

        assert len(corrected_results) == 3
        assert all(isinstance(r, ParseResult) for r in corrected_results)
