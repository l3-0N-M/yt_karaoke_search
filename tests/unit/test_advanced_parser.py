"""Unit tests for advanced_parser.py."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from collector.advanced_parser import AdvancedTitleParser, ParseResult


class TestParseResult:
    """Test cases for ParseResult dataclass."""

    def test_parse_result_creation(self):
        """Test creating a ParseResult with all fields."""
        result = ParseResult(
            artist="Test Artist", song_title="Test Song", confidence=0.95, featured_artists="Feat1"
        )

        assert result.artist == "Test Artist"
        assert result.song_title == "Test Song"
        assert result.confidence == 0.95
        assert result.featured_artists == "Feat1"

    def test_parse_result_defaults(self):
        """Test ParseResult with minimal fields."""
        result = ParseResult(artist="Artist", song_title="Song", confidence=0.8)

        assert result.featured_artists is None or result.featured_artists == []
        assert result.metadata.get("version") is None
        assert result.metadata.get("language") is None
        assert result.metadata.get("year") is None
        # No default attributes for is_cover, original_artist, etc.


class TestAdvancedTitleParser:
    """Test cases for AdvancedTitleParser."""

    @pytest.fixture
    def parser(self):
        """Create an AdvancedTitleParser instance."""
        return AdvancedTitleParser()

    def test_parse_basic_title(self, parser):
        """Test parsing a basic karaoke title."""
        result = parser.parse_title(
            title="Adele - Hello (Karaoke Version)", description="Karaoke version of Hello by Adele"
        )

        assert result is not None
        assert result.artist == "Adele"
        assert result.song_title == "Hello"
        assert result.confidence > 0.7

    def test_parse_with_featured_artists(self, parser):
        """Test parsing title with featured artists."""
        test_cases = [
            ("Song Title (feat. Artist2)", ["Artist2"]),
            ("Song Title (ft. Artist2 & Artist3)", ["Artist2", "Artist3"]),
            ("Song Title (featuring Artist2, Artist3)", ["Artist2", "Artist3"]),
            ("Song Title (with Artist2)", ["Artist2"]),
        ]

        for title_suffix, expected_featured in test_cases:
            result = parser.parse_title(title=f"Artist1 - {title_suffix}", description="")

            assert result is not None
            assert result.artist == "Artist1"
            assert "feat" not in result.song_title.lower()
            assert result.featured_artists is not None

    def test_parse_version_info(self, parser):
        """Test extracting version information."""
        test_cases = [
            ("Song (Acoustic Version)", "Acoustic"),
            ("Song (Piano Version)", "Piano"),
            ("Song (Live Version)", "Live"),
            ("Song (Instrumental)", "Instrumental"),
            ("Song (Demo Version)", "Demo"),
        ]

        for title, expected_version in test_cases:
            result = parser.parse_title(title=f"Artist - {title}", description="")

            assert result is not None
            assert result.metadata.get("version") == expected_version

    def test_parse_remix_info(self, parser):
        """Test extracting remix information."""
        test_cases = [
            "Song (DJ Test Remix)",
            "Song (Test Remix)",
            "Song [Remix by DJ Test]",
            "Song - Test's Remix",
        ]

        for title in test_cases:
            result = parser.parse_title(title=f"Artist - {title}", description="")

            assert result is not None
            # Remix info should be in metadata or version
            assert (
                result.metadata.get("version") is not None
                or "remix" in (result.metadata.get("version", "") or "").lower()
            )

    def test_parse_cover_detection(self, parser):
        """Test cover song detection."""
        result = parser.parse_title(
            title="New Artist - Old Song (Originally by Original Artist)",
            description="Cover version",
        )

        assert result is not None
        # Cover info should be in metadata
        assert (
            result.metadata.get("is_cover") is True
            or result.metadata.get("original_artist") == "Original Artist"
        )

    def test_parse_year_extraction(self, parser):
        """Test year extraction from title."""
        test_cases = [
            ("Song (1985)", 1985),
            ("Song [2000]", 2000),
            ("Song from 1999", 1999),
            ("Song - 1975 Version", 1975),
        ]

        for title, expected_year in test_cases:
            result = parser.parse_title(title=f"Artist - {title}", description="")

            assert result is not None
            assert result.metadata.get("year") == expected_year

    def test_parse_language_detection(self, parser):
        """Test language detection from title/tags."""
        test_cases = [
            ("Song (English Version)", "English"),
            ("Song (Spanish Version)", "Spanish"),
            ("Song (日本語版)", "Japanese"),
            ("Song (Version Française)", "French"),
        ]

        for title, expected_language in test_cases:
            result = parser.parse_title(title=f"Artist - {title}", description="")

            assert result is not None
            assert result.metadata.get("language") == expected_language

    def test_parse_with_channel_name_removal(self, parser):
        """Test removal of channel name from title."""
        result = parser.parse_title(
            title="Artist - Song | Karaoke Channel", description="", channel_name="Karaoke Channel"
        )

        assert result is not None
        assert "Karaoke Channel" not in result.song_title
        assert result.artist == "Artist"
        assert result.song_title == "Song"

    def test_parse_complex_title(self, parser):
        """Test parsing a complex title with multiple elements."""
        result = parser.parse_title(
            title="Bruno Mars - Just The Way You Are (feat. Lupe Fiasco) [Acoustic Remix] (2010)",
            description="Acoustic remix version",
        )

        assert result is not None
        assert result.artist == "Bruno Mars"
        assert result.song_title == "Just The Way You Are"
        assert result.featured_artists == "Lupe Fiasco"
        assert result.metadata.get("version") == "Acoustic"
        assert result.metadata.get("year") == 2010

    def test_parse_with_special_characters(self, parser):
        """Test parsing titles with special characters."""
        test_cases = [
            "Beyoncé - Déjà Vu",
            "P!nk - What's Up",
            "Ke$ha - TiK ToK",
            "will.i.am - Scream & Shout",
        ]

        for title in test_cases:
            result = parser.parse_title(title=f"{title} (Karaoke)", description="")

            assert result is not None
            assert result.artist is not None
            assert result.song_title is not None

    def test_parse_confidence_scoring(self, parser):
        """Test confidence scoring for different title formats."""
        # Well-formatted title should have high confidence
        result1 = parser.parse_title(
            title="Artist - Song Title (Karaoke Version)", description="Clear karaoke version"
        )
        assert result1.confidence > 0.8

        # Poorly formatted title should have lower confidence
        result2 = parser.parse_title(title="artist song karaoke", description="")
        assert result2.confidence < result1.confidence

    def test_parse_empty_input(self, parser):
        """Test parsing with empty or None input."""
        assert parser.parse_title(title="", description="") is None
        assert parser.parse_title(title=None, description="") is None
        assert parser.parse_title(title="   ", description="") is None

    def test_parse_genre_from_tags(self, parser):
        """Test genre extraction from tags."""
        result = parser.parse_title(title="Artist - Song", description="", tags="pop, rock, 80s")

        assert result is not None
        assert result.metadata.get("genre") is not None
        assert (
            "pop" in result.metadata.get("genre").lower()
            or "rock" in result.metadata.get("genre").lower()
        )

    def test_parse_metadata_from_description(self, parser):
        """Test extracting metadata from description."""
        result = parser.parse_title(
            title="Artist - Song",
            description="Original artist: Original Band\nYear: 1985\nGenre: Rock",
        )

        assert result is not None
        assert result.metadata.get("year") == 1985
        assert result.metadata.get("genre") == "Rock"
        assert result.metadata.get("original_artist") == "Original Band"

    def test_parse_edge_cases(self, parser):
        """Test various edge cases."""
        # Title with multiple dashes
        result1 = parser.parse_title(title="Jean-Michel Jarre - Oxygène - Part IV", description="")
        assert result1 is not None

        # Title with parentheses in song name
        result2 = parser.parse_title(title="Artist - (I Can't Get No) Satisfaction", description="")
        assert result2 is not None
        assert "(I Can't Get No)" in result2.song_title

        # Non-English characters
        result3 = parser.parse_title(title="アーティスト - 曲名 (カラオケ)", description="")
        assert result3 is not None

    def test_parse_with_fuzzy_matching(self, parser):
        """Test integration with fuzzy matcher if available."""
        with patch.object(parser, "fuzzy_matcher", create=True) as mock_fuzzy:
            mock_fuzzy.find_best_match.return_value = Mock(
                similarity=0.9, matched_text="Corrected Artist"
            )

            result = parser.parse_title(title="Artst - Song (typo in artist name)", description="")

            # Parser might use fuzzy matching for correction
            assert result is not None

    def test_parse_pattern_priority(self, parser):
        """Test that certain patterns have priority."""
        # Official video pattern should be recognized
        result = parser.parse_title(title="Artist - Song (Official Karaoke Video)", description="")

        assert result is not None
        assert result.confidence > 0.85  # Higher confidence for official videos

    def test_parse_with_metadata_integration(self, parser):
        """Test parsing with additional metadata."""
        result = parser.parse_title(
            title="Artist - Song",
            description="Great karaoke version",
            metadata={"duration": 240, "upload_date": "2023-01-01", "view_count": 10000},
        )

        assert result is not None
        assert result.additional_metadata is not None
        assert "duration" in result.additional_metadata
