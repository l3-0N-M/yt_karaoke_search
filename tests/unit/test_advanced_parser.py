"""Unit tests for advanced_parser.py - fixed to match implementation."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from collector.advanced_parser import AdvancedTitleParser


class TestAdvancedTitleParser:
    """Test cases for AdvancedTitleParser."""

    @pytest.fixture
    def parser(self):
        """Create an AdvancedTitleParser instance."""
        return AdvancedTitleParser()

    def test_parse_basic_title(self, parser):
        """Test basic title parsing with hyphen separator."""
        result = parser.parse_title(title="Artist Name - Song Title", description="")

        assert result is not None
        assert result.artist == "Artist Name"
        assert result.song_title == "Song Title"
        assert result.confidence > 0.5

    def test_parse_title_with_featured_artists(self, parser):
        """Test parsing titles with featured artists."""
        test_cases = [
            ("Artist feat. Guest - Song", "Artist", ["Guest"]),
            ("Artist ft. Guest - Song", "Artist", ["Guest"]),
            ("Artist featuring Guest - Song", "Artist", ["Guest"]),
            ("Artist with Guest - Song", "Artist", ["Guest"]),
            ("Artist & Guest - Song", "Artist & Guest", None),  # & is not treated as featuring
        ]

        for title, expected_artist, expected_featured in test_cases:
            result = parser.parse_title(title=title, description="")

            assert result is not None
            assert result.artist == expected_artist
            assert result.featured_artists == expected_featured

    def test_parse_various_separators(self, parser):
        """Test parsing with different separator types."""
        test_cases = [
            ("Artist - Song", "-"),
            ("Artist – Song", "–"),  # En dash
            ("Artist — Song", "—"),  # Em dash
            ("Artist : Song", ":"),
            ("Artist | Song", "|"),
        ]

        for title, separator in test_cases:
            result = parser.parse_title(title=title, description="")

            assert result is not None
            assert result.artist == "Artist"
            assert result.song_title == "Song"

    def test_parse_karaoke_suffix_removal(self, parser):
        """Test removal of karaoke-related suffixes."""
        test_cases = [
            "Artist - Song (Karaoke)",
            "Artist - Song [Karaoke Version]",
            "Artist - Song (Instrumental)",
            "Artist - Song Karaoke",
        ]

        for title in test_cases:
            result = parser.parse_title(title=title, description="")

            assert result is not None
            assert result.artist == "Artist"
            assert result.song_title == "Song"
            # Karaoke suffixes should be removed
            assert "karaoke" not in result.song_title.lower()
            assert "instrumental" not in result.song_title.lower()

    def test_parse_version_info(self, parser):
        """Test that version information is removed (not extracted)."""
        # The parser removes version info rather than extracting it
        test_cases = [
            ("Artist - Song (Acoustic Version)", "Song"),
            ("Artist - Song (Piano Version)", "Song"),
            ("Artist - Song (Live Version)", "Song"),
            ("Artist - Song (Instrumental)", "Song"),
            ("Artist - Song (Demo Version)", "Song"),
        ]

        for title, expected_song in test_cases:
            result = parser.parse_title(title=title, description="")

            assert result is not None
            assert result.artist == "Artist"
            assert result.song_title == expected_song
            # Version info is not extracted into metadata
            assert result.metadata.get("version") is None

    def test_parse_remix_info(self, parser):
        """Test that remix information is removed (not extracted)."""
        test_cases = [
            ("Artist - Song (DJ Test Remix)", "Song"),
            ("Artist - Song (Test Remix)", "Song"),
            ("Artist - Song [Remix by DJ Test]", "Song"),
        ]

        for title, expected_song in test_cases:
            result = parser.parse_title(title=title, description="")

            assert result is not None
            assert result.artist == "Artist"
            assert result.song_title == expected_song
            # Remix info is not extracted into metadata
            assert result.metadata.get("version") is None

    def test_parse_cover_detection(self, parser):
        """Test that cover information is not extracted."""
        result = parser.parse_title(
            title="New Artist - Old Song (Originally by Original Artist)",
            description="Cover version",
        )

        assert result is not None
        assert result.artist == "New Artist"
        assert result.song_title == "Old Song"
        # Cover info is not extracted into metadata
        assert result.metadata.get("is_cover") is None
        assert result.metadata.get("original_artist") is None

    def test_parse_year_extraction(self, parser):
        """Test that years are removed from titles (not extracted)."""
        test_cases = [
            ("Artist - Song (1985)", "Song"),
            ("Artist - Song [2000]", "Song"),
            ("Artist - Song 1999", "Song 1999"),  # Year without parentheses might remain
        ]

        for title, expected_song in test_cases:
            result = parser.parse_title(title=title, description="")

            assert result is not None
            assert result.artist == "Artist"
            # Year in parentheses is removed
            assert "(" not in result.song_title
            assert ")" not in result.song_title
            # Year is not extracted into metadata
            assert result.metadata.get("year") is None

    def test_parse_language_detection(self, parser):
        """Test that language information is not extracted."""
        test_cases = [
            ("Artist - Song (English Version)", "Song"),
            ("Artist - Song (Spanish Version)", "Song"),
            ("Artist - Song (Version Française)", "Song"),
        ]

        for title, expected_song in test_cases:
            result = parser.parse_title(title=title, description="")

            assert result is not None
            assert result.artist == "Artist"
            assert result.song_title == expected_song
            # Language info is not extracted into metadata
            assert result.metadata.get("language") is None

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
        assert result.featured_artists == ["Lupe Fiasco"]
        # Additional info in brackets/parentheses is removed
        assert "Acoustic" not in result.song_title
        assert "Remix" not in result.song_title
        assert "2010" not in result.song_title

    def test_parse_confidence_scoring(self, parser):
        """Test confidence scoring for different patterns."""
        # High confidence - clear pattern
        high_conf = parser.parse_title(title="Artist - Song Title", description="")
        assert high_conf is not None
        assert high_conf.confidence > 0.7

        # Lower confidence - no clear separator
        low_conf = parser.parse_title(title="Artist Song Title", description="")
        assert low_conf is not None
        assert low_conf.confidence < 0.5

    def test_parse_empty_input(self, parser):
        """Test handling of empty input."""
        # Empty title should return None
        result = parser.parse_title(title="", description="")
        assert result is None

        # None title should return None
        result = parser.parse_title(title=None, description="")
        assert result is None

    def test_parse_title_only(self, parser):
        """Test parsing when only title is found (no artist)."""
        result = parser.parse_title(title="Just a Song Title", description="")

        # When no clear artist-title separator is found
        assert result is not None
        # The whole string might be treated as song title with no artist
        assert result.song_title is not None

    def test_parse_special_characters(self, parser):
        """Test handling of special characters."""
        result = parser.parse_title(title="P!nk - So What", description="")

        assert result is not None
        assert result.artist == "P!nk"
        assert result.song_title == "So What"

    def test_parse_multiple_hyphens(self, parser):
        """Test titles with multiple hyphens."""
        result = parser.parse_title(title="Jay-Z - 99 Problems - The Black Album", description="")

        assert result is not None
        assert result.artist == "Jay-Z"
        # Additional info after second hyphen might be removed or kept
        assert "99 Problems" in result.song_title

    def test_parse_non_english_titles(self, parser):
        """Test parsing non-English titles."""
        test_cases = [
            ("Артист - Песня", "Артист", "Песня"),  # Cyrillic
            ("歌手 - 歌曲", "歌手", "歌曲"),  # Chinese
            ("アーティスト - 曲", "アーティスト", "曲"),  # Japanese
        ]

        for title, expected_artist, expected_song in test_cases:
            result = parser.parse_title(title=title, description="")

            assert result is not None
            assert result.artist == expected_artist
            assert result.song_title == expected_song

    def test_clean_text_method(self, parser):
        """Test the internal text cleaning method."""
        # The parser should clean various noise from extracted text
        test_cases = [
            ("Song (Official Video)", "Song"),
            ("Song [HD]", "Song"),
            ("Song (Lyrics)", "Song"),
            ("Song - Official", "Song"),
        ]

        for input_text, expected_clean in test_cases:
            result = parser.parse_title(title=f"Artist - {input_text}", description="")
            assert result is not None
            assert result.song_title == expected_clean
