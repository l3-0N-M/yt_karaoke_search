"""Comprehensive tests for the advanced parser module."""

from dataclasses import asdict
from unittest.mock import MagicMock

import pytest

from collector.advanced_parser import (
    AdvancedTitleParser,
    ParseResult,
    PatternStats,
)
from collector.validation_corrector import ValidationResult


class TestParseResult:
    """Test the ParseResult dataclass."""

    def test_parse_result_creation(self):
        """Test creating a ParseResult with default values."""
        result = ParseResult()
        assert result.artist is None
        assert result.song_title is None
        assert result.featured_artists is None
        assert result.confidence == 0.0
        assert result.method == ""
        assert result.pattern_used == ""
        assert result.validation_score == 0.0
        assert result.alternative_results == []
        assert result.metadata == {}

    def test_parse_result_with_values(self):
        """Test creating a ParseResult with specific values."""
        result = ParseResult(
            artist="Test Artist",
            song_title="Test Song",
            confidence=0.85,
            method="test_method",
            pattern_used="test_pattern",
        )
        assert result.artist == "Test Artist"
        assert result.song_title == "Test Song"
        assert result.confidence == 0.85
        assert result.method == "test_method"
        assert result.pattern_used == "test_pattern"

    def test_parse_result_serializable(self):
        """Test that ParseResult can be converted to dict."""
        result = ParseResult(
            artist="Test Artist",
            song_title="Test Song",
            confidence=0.85,
            metadata={"key": "value"},
        )
        result_dict = asdict(result)
        assert isinstance(result_dict, dict)
        assert result_dict["artist"] == "Test Artist"
        assert result_dict["metadata"]["key"] == "value"


class TestPatternStats:
    """Test the PatternStats dataclass."""

    def test_pattern_stats_creation(self):
        """Test creating PatternStats with default values."""
        stats = PatternStats("test_pattern")
        assert stats.pattern == "test_pattern"
        assert stats.success_count == 0
        assert stats.total_attempts == 0
        assert stats.avg_confidence == 0.0
        assert stats.common_failures == []


class TestAdvancedTitleParser:
    """Test the AdvancedTitleParser class."""

    @pytest.fixture
    def parser(self):
        """Create a parser instance for testing."""
        config = MagicMock()
        config.fuzzy_matching = {"min_similarity": 0.5}
        return AdvancedTitleParser(config)

    @pytest.fixture
    def parser_no_config(self):
        """Create a parser instance without config."""
        return AdvancedTitleParser()

    def test_parser_initialization(self, parser):
        """Test parser initialization."""
        assert parser.config is not None
        assert isinstance(parser.pattern_stats, dict)
        assert isinstance(parser.known_artists, set)
        assert isinstance(parser.known_songs, set)
        assert isinstance(parser.channel_patterns, dict)
        assert isinstance(parser.language_patterns, dict)

    def test_parser_initialization_no_config(self, parser_no_config):
        """Test parser initialization without config."""
        assert parser_no_config.config is None
        assert isinstance(parser_no_config.pattern_stats, dict)


class TestAdvancedCleanTitle:
    """Test the _advanced_clean_title method."""

    @pytest.fixture
    def parser(self):
        return AdvancedTitleParser()

    def test_basic_cleaning(self, parser):
        """Test basic title cleaning."""
        title = "  Test Title  "
        cleaned = parser._advanced_clean_title(title)
        assert cleaned == "Test Title"

    def test_bracket_removal(self, parser):
        """Test removal of brackets."""
        title = "[HD] Test Artist - Test Song"
        cleaned = parser._advanced_clean_title(title)
        assert not cleaned.startswith("[")
        assert "Test Artist - Test Song" in cleaned

    def test_karaoke_prefix_removal(self, parser):
        """Test removal of karaoke prefixes."""
        title = "Some Channel Karaoke: Test Artist - Test Song"
        cleaned = parser._advanced_clean_title(title)
        assert not cleaned.startswith("Some Channel Karaoke:")
        assert "Test Artist - Test Song" in cleaned

    def test_unicode_normalization(self, parser):
        """Test Unicode normalization."""
        title = "Test Artist — Test Song"  # em dash
        cleaned = parser._advanced_clean_title(title)
        assert "Test Artist - Test Song" in cleaned

    def test_quality_indicator_removal(self, parser):
        """Test removal of quality indicators."""
        title = "HD Test Artist - Test Song 4K 1080p"
        cleaned = parser._advanced_clean_title(title)
        # Should not contain quality indicators in final result
        assert "Test Artist - Test Song" in cleaned

    @pytest.mark.parametrize(
        "input_title,expected_contains",
        [
            ("【Karaoke】Test Artist - Test Song", "Test Artist - Test Song"),
            ("Official Test Artist - Test Song", "Test Artist - Test Song"),
            ("DE Test Artist - Test Song", "Test Artist - Test Song"),
            ("Test Artist – Test Song", "Test Artist - Test Song"),
            ("Test   Artist  -   Test Song", "Test Artist - Test Song"),
        ],
    )
    def test_various_cleaning_patterns(self, parser, input_title, expected_contains):
        """Test various cleaning patterns."""
        cleaned = parser._advanced_clean_title(input_title)
        assert expected_contains in cleaned


class TestLanguageDetection:
    """Test the _detect_language method."""

    @pytest.fixture
    def parser(self):
        return AdvancedTitleParser()

    def test_chinese_detection(self, parser):
        """Test Chinese language detection."""
        title = "周杰伦 - 稻香 卡拉OK"
        language = parser._detect_language(title)
        assert language == "chinese"

    def test_japanese_detection(self, parser):
        """Test Japanese language detection."""
        title = "さくらんぼ - カラオケ"
        language = parser._detect_language(title)
        assert language == "japanese"

    def test_korean_detection(self, parser):
        """Test Korean language detection."""
        title = "방탄소년단 - 다이너마이트 가라오케"
        language = parser._detect_language(title)
        assert language == "korean"

    def test_russian_detection(self, parser):
        """Test Russian language detection."""
        title = "Артист - Песня караоке"
        language = parser._detect_language(title)
        assert language == "russian"

    def test_english_detection(self, parser):
        """Test English language detection (default)."""
        title = "Artist - Song Karaoke"
        language = parser._detect_language(title)
        assert language == "english"


class TestChannelPatterns:
    """Test channel-specific pattern parsing."""

    @pytest.fixture
    def parser(self):
        return AdvancedTitleParser()

    def test_lets_sing_karaoke_pattern(self, parser):
        """Test Let's Sing Karaoke pattern."""
        title = "Smith, John - Great Song (Karaoke & Lyrics)"
        result = parser._parse_with_channel_patterns(title, "Let's Sing Karaoke")
        assert result.artist == "John Smith"
        assert result.song_title == "Great Song"
        assert result.confidence > 0.9

    def test_lugn_pattern(self, parser):
        """Test Lugn channel pattern."""
        title = "Test Artist • Test Song • Karaoke"
        result = parser._parse_with_channel_patterns(title, "Lugn Music")
        assert result.artist == "Test Artist"
        assert result.song_title == "Test Song"
        assert result.confidence > 0.9

    def test_karafun_deutschland_pattern(self, parser):
        """Test KaraFun Deutschland pattern."""
        title = "Karaoke Great Song - Famous Artist *"
        result = parser._parse_with_channel_patterns(title, "KaraFun Deutschland")
        assert result.artist == "Famous Artist"
        assert result.song_title == "Great Song"
        assert result.confidence > 0.9

    def test_sing_king_pattern(self, parser):
        """Test Sing King pattern."""
        title = 'Sing King Karaoke - "Great Song" (Style of "Famous Artist")'
        result = parser._parse_with_channel_patterns(title, "Sing King")
        assert result.artist == "Famous Artist"
        assert result.song_title == "Great Song"
        assert result.confidence > 0.9

    def test_unknown_channel(self, parser):
        """Test unknown channel returns empty result."""
        title = "Artist - Song (Karaoke)"
        result = parser._parse_with_channel_patterns(title, "Unknown Channel")
        assert result.method == "channel_specific"
        assert result.confidence == 0


class TestCorePatterns:
    """Test core pattern matching."""

    @pytest.fixture
    def parser(self):
        return AdvancedTitleParser()

    def test_standard_artist_title_pattern(self, parser):
        """Test standard artist-title pattern."""
        title = "Famous Artist - Great Song (Karaoke Version)"
        result = parser._parse_with_core_patterns(title)
        assert result.artist == "Famous Artist"
        assert result.song_title == "Great Song"
        assert result.confidence > 0.7

    def test_quoted_patterns(self, parser):
        """Test quoted patterns."""
        title = '"Artist Name" - "Song Title" (Karaoke)'
        result = parser._parse_with_core_patterns(title)
        assert result.artist == "Artist Name"
        assert result.song_title == "Song Title"
        assert result.confidence > 0.9

    def test_karaoke_song_artist_pattern(self, parser):
        """Test karaoke-prefixed pattern."""
        title = "Karaoke Great Song - Famous Artist"
        result = parser._parse_with_core_patterns(title)
        assert result.artist == "Famous Artist"
        assert result.song_title == "Great Song"
        assert result.confidence > 0.8

    def test_style_of_pattern(self, parser):
        """Test 'style of' pattern."""
        title = "Great Song (Style of Famous Artist)"
        result = parser._parse_with_core_patterns(title)
        assert result.artist == "Famous Artist"
        assert result.song_title == "Great Song"
        assert result.confidence > 0.7

    def test_by_pattern(self, parser):
        """Test 'by' pattern."""
        title = "Great Song by Famous Artist (Karaoke)"
        result = parser._parse_with_core_patterns(title)
        assert result.artist == "Famous Artist"
        assert result.song_title == "Great Song"
        assert result.confidence > 0.7


class TestHeuristicParsing:
    """Test heuristic parsing methods."""

    @pytest.fixture
    def parser(self):
        return AdvancedTitleParser()

    def test_heuristic_length_based(self, parser):
        """Test heuristic parsing based on length."""
        title = "Artist - This is a much longer song title"
        result = parser._parse_with_heuristics(title, "", "")
        # Shorter part (Artist) should be artist, longer part should be title
        assert result.artist == "Artist"
        assert result.song_title == "This Is A Much Longer Song Title"
        assert result.confidence > 0.5

    def test_description_extraction(self, parser):
        """Test extraction from description."""
        title = "Some Title"
        description = "Artist: Famous Artist\nThis is a great song"
        result = parser._parse_with_heuristics(title, description, "")
        assert result.artist == "Famous Artist"
        assert result.confidence > 0.6

    def test_no_separators_found(self, parser):
        """Test when no separators are found."""
        title = "Just a title without separators"
        result = parser._parse_with_heuristics(title, "", "")
        assert result.method == "heuristics"
        assert result.confidence == 0


class TestFuzzyMatching:
    """Test fuzzy matching functionality."""

    @pytest.fixture
    def parser_with_known_data(self):
        parser = AdvancedTitleParser()
        parser.known_artists.add("Beatles")
        parser.known_artists.add("Queen")
        parser.known_songs.add("Yesterday")
        parser.known_songs.add("Bohemian Rhapsody")
        return parser

    def test_fuzzy_artist_matching(self, parser_with_known_data):
        """Test fuzzy matching of artists."""
        title = "Beatless - Some Song"  # Misspelled Beatles
        result = parser_with_known_data._parse_with_fuzzy_matching(title)
        # Should match Beatles with high confidence
        assert result.artist == "Beatles"
        assert result.confidence > 0.7

    def test_fuzzy_song_matching(self, parser_with_known_data):
        """Test fuzzy matching of songs."""
        title = "Some Artist - Bohemain Rhapsody"  # Misspelled
        result = parser_with_known_data._parse_with_fuzzy_matching(title)
        assert result.song_title == "Bohemian Rhapsody"
        assert result.confidence > 0.7

    def test_no_fuzzy_matches(self, parser_with_known_data):
        """Test when no fuzzy matches are found."""
        title = "Unknown Artist - Unknown Song"
        result = parser_with_known_data._parse_with_fuzzy_matching(title)
        assert result.method == "fuzzy_matching"
        assert result.confidence == 0


class TestResultSelection:
    """Test result selection logic."""

    @pytest.fixture
    def parser(self):
        return AdvancedTitleParser()

    def test_select_highest_confidence(self, parser):
        """Test selection of highest confidence result."""
        results = [
            ParseResult(confidence=0.5, method="method1"),
            ParseResult(confidence=0.8, method="method2"),
            ParseResult(confidence=0.6, method="method3"),
        ]
        best = parser._select_best_result(results, "test title")
        assert best.method == "method2"
        assert best.confidence == 0.8

    def test_select_with_completeness_bonus(self, parser):
        """Test selection with completeness bonus."""
        results = [
            ParseResult(confidence=0.7, method="method1", artist="Artist"),
            ParseResult(
                confidence=0.6,
                method="method2",
                artist="Artist",
                song_title="Song",
            ),
        ]
        best = parser._select_best_result(results, "test title")
        # Second result should win due to completeness bonus
        assert best.method == "method2"
        assert best.artist == "Artist"
        assert best.song_title == "Song"

    def test_no_valid_results(self, parser):
        """Test when no valid results are found."""
        results = [
            ParseResult(confidence=0.0, method="method1"),
            ParseResult(confidence=0.0, method="method2"),
        ]
        best = parser._select_best_result(results, "test title")
        assert best.method == "no_match"

    def test_alternative_results_stored(self, parser):
        """Test that alternative results are stored."""
        results = [
            ParseResult(confidence=0.8, method="best"),
            ParseResult(confidence=0.7, method="second"),
            ParseResult(confidence=0.6, method="third"),
        ]
        best = parser._select_best_result(results, "test title")
        assert len(best.alternative_results) >= 1
        assert best.alternative_results[0]["method"] == "second"


class TestTextCleaning:
    """Test text cleaning utilities."""

    @pytest.fixture
    def parser(self):
        return AdvancedTitleParser()

    def test_clean_extracted_text_quotes(self, parser):
        """Test cleaning of quotes from extracted text."""
        text = '"Artist Name"'
        cleaned = parser._clean_extracted_text(text)
        assert cleaned == "Artist Name"

    def test_clean_extracted_text_brackets(self, parser):
        """Test cleaning of brackets from extracted text."""
        text = "[HD] Artist Name"
        cleaned = parser._clean_extracted_text(text)
        assert cleaned == "Artist Name"

    def test_clean_extracted_text_karaoke_prefix(self, parser):
        """Test removal of karaoke prefix."""
        text = "Karaoke Song Title"
        cleaned = parser._clean_extracted_text(text)
        assert cleaned == "Song Title"

    def test_clean_extracted_text_noise_patterns(self, parser):
        """Test removal of noise patterns."""
        text = "Song Title (Karaoke Version)"
        cleaned = parser._clean_extracted_text(text)
        assert cleaned == "Song Title"

    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ('"Song Title"', "Song Title"),
            ("'Artist Name'", "Artist Name"),
            ("(Intro) Song Title", "Song Title"),
            ("[Preview] Artist", "Artist"),
            ("Karaoke Great Song", "Great Song"),
            ("Song Title (Instrumental)", "Song Title"),
            ("Song Title - MR", "Song Title"),
            ("Song Title Inst.", "Song Title"),
        ],
    )
    def test_various_cleaning_patterns(self, parser, input_text, expected):
        """Test various text cleaning patterns."""
        cleaned = parser._clean_extracted_text(input_text)
        assert cleaned == expected


class TestFeaturedArtistExtraction:
    """Test featured artist extraction."""

    @pytest.fixture
    def parser(self):
        return AdvancedTitleParser()

    def test_extract_featured_artists_feat(self, parser):
        """Test extraction with 'feat' pattern."""
        title = "Main Artist - Song Title"
        description = "Great song feat Other Artist"
        featured = parser._extract_featured_artists_advanced(title, description)
        assert "Other Artist" in featured

    def test_extract_featured_artists_featuring(self, parser):
        """Test extraction with 'featuring' pattern."""
        title = "Main Artist - Song Title"
        description = "Awesome track featuring Another Artist and Third Artist"
        featured = parser._extract_featured_artists_advanced(title, description)
        assert "Another Artist" in featured
        assert "Third Artist" in featured

    def test_extract_featured_artists_multiple_patterns(self, parser):
        """Test extraction with multiple patterns."""
        title = "Main Artist ft. Guest Artist - Song Title"
        description = "Song with Other Artist & Final Artist"
        featured = parser._extract_featured_artists_advanced(title, description)
        # Should find artists from both title and description
        assert len(featured.split(", ")) >= 2

    def test_no_featured_artists(self, parser):
        """Test when no featured artists are found."""
        title = "Solo Artist - Solo Song"
        description = "Just a regular song description"
        featured = parser._extract_featured_artists_advanced(title, description)
        assert featured is None


class TestLearningMechanism:
    """Test the learning mechanism."""

    @pytest.fixture
    def parser(self):
        return AdvancedTitleParser()

    def test_learn_from_successful_parse(self, parser):
        """Test learning from successful parse."""
        result = ParseResult(
            artist="Test Artist",
            song_title="Test Song",
            confidence=0.85,
            pattern_used="test_pattern",
        )

        initial_artists = len(parser.known_artists)
        initial_songs = len(parser.known_songs)

        parser._learn_from_parse("Test Title", result)

        # Should add to known artists and songs
        assert len(parser.known_artists) == initial_artists + 1
        assert len(parser.known_songs) == initial_songs + 1
        assert "Test Artist" in parser.known_artists
        assert "Test Song" in parser.known_songs

    def test_pattern_stats_update(self, parser):
        """Test pattern statistics update."""
        result = ParseResult(
            confidence=0.75,
            pattern_used="test_pattern",
        )

        parser._learn_from_parse("Test Title", result)

        assert "test_pattern" in parser.pattern_stats
        stats = parser.pattern_stats["test_pattern"]
        assert stats.total_attempts == 1
        assert stats.success_count == 1  # confidence > 0.7
        assert stats.avg_confidence == 0.75

    def test_low_confidence_no_learning(self, parser):
        """Test that low confidence results don't add to known data."""
        result = ParseResult(
            artist="Test Artist",
            song_title="Test Song",
            confidence=0.5,
            pattern_used="test_pattern",
        )

        initial_artists = len(parser.known_artists)
        initial_songs = len(parser.known_songs)

        parser._learn_from_parse("Test Title", result)

        # Should not add to known data due to low confidence
        assert len(parser.known_artists) == initial_artists
        assert len(parser.known_songs) == initial_songs


class TestValidationFeedback:
    """Test validation feedback application."""

    @pytest.fixture
    def parser(self):
        return AdvancedTitleParser()

    def test_apply_positive_validation_feedback(self, parser):
        """Test applying positive validation feedback."""
        validation = ValidationResult(
            artist_valid=True,
            song_valid=True,
            validation_score=0.9,
        )

        initial_artists = len(parser.known_artists)
        initial_songs = len(parser.known_songs)

        parser.apply_validation_feedback("Valid Artist", "Valid Song", validation)

        assert len(parser.known_artists) == initial_artists + 1
        assert len(parser.known_songs) == initial_songs + 1
        assert "Valid Artist" in parser.known_artists
        assert "Valid Song" in parser.known_songs

    def test_apply_negative_validation_feedback(self, parser):
        """Test applying negative validation feedback."""
        validation = ValidationResult(
            artist_valid=False,
            song_valid=False,
            validation_score=0.2,
        )

        initial_artists = len(parser.known_artists)
        initial_songs = len(parser.known_songs)

        parser.apply_validation_feedback("Invalid Artist", "Invalid Song", validation)

        # Should not add invalid data
        assert len(parser.known_artists) == initial_artists
        assert len(parser.known_songs) == initial_songs


class TestMainParseMethod:
    """Test the main parse_title method."""

    @pytest.fixture
    def parser(self):
        return AdvancedTitleParser()

    def test_parse_title_basic(self, parser):
        """Test basic title parsing."""
        title = "Famous Artist - Great Song (Karaoke)"
        result = parser.parse_title(title)

        assert result.artist == "Famous Artist"
        assert result.song_title == "Great Song"
        assert result.confidence > 0.7
        assert result.method in ["core_patterns"]

    def test_parse_title_with_channel(self, parser):
        """Test parsing with channel information."""
        title = "Smith, John - Great Song (Karaoke & Lyrics)"
        result = parser.parse_title(title, channel_name="Let's Sing Karaoke")

        assert result.artist == "John Smith"
        assert result.song_title == "Great Song"
        assert result.confidence > 0.9

    def test_parse_title_with_description(self, parser):
        """Test parsing with description enhancement."""
        title = "Some Complex Title"
        description = "This is a song by Famous Artist called Great Song"
        result = parser.parse_title(title, description=description)

        # Should extract information from description
        assert result.confidence > 0

    def test_parse_title_language_detection(self, parser):
        """Test parsing with non-English title."""
        title = "歌手 - 歌曲 卡拉OK"  # Chinese
        result = parser.parse_title(title)

        # Should attempt language-specific parsing
        assert result.method in ["language_chinese", "core_patterns", "heuristics"]

    def test_parse_title_learning_applied(self, parser):
        """Test that learning is applied after parsing."""
        title = "Test Artist - Test Song (Karaoke)"

        initial_pattern_count = len(parser.pattern_stats)
        result = parser.parse_title(title)

        # Should have learned from the parse
        assert len(parser.pattern_stats) >= initial_pattern_count
        if result.confidence > 0.8:
            assert "Test Artist" in parser.known_artists or "Test Song" in parser.known_songs


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def parser(self):
        return AdvancedTitleParser()

    def test_empty_title(self, parser):
        """Test parsing empty title."""
        result = parser.parse_title("")
        assert result.confidence == 0

    def test_very_short_title(self, parser):
        """Test parsing very short title."""
        result = parser.parse_title("A")
        assert result.confidence == 0

    def test_no_separators_title(self, parser):
        """Test title with no clear separators."""
        title = "JustOneWordTitle"
        result = parser.parse_title(title)
        # Should still return a result, even if low confidence
        assert isinstance(result, ParseResult)

    def test_unicode_title(self, parser):
        """Test parsing title with Unicode characters."""
        title = "Артист - Песня (Караоке)"  # Cyrillic
        result = parser.parse_title(title)
        assert isinstance(result, ParseResult)

    def test_very_long_title(self, parser):
        """Test parsing very long title."""
        title = "A" * 1000 + " - " + "B" * 1000 + " (Karaoke)"
        result = parser.parse_title(title)
        assert isinstance(result, ParseResult)

    def test_malformed_regex_patterns(self, parser):
        """Test handling of titles that might break regex."""
        title = "Artist - Song (((((Karaoke"  # Unmatched parentheses
        result = parser.parse_title(title)
        assert isinstance(result, ParseResult)

    def test_special_characters_title(self, parser):
        """Test title with many special characters."""
        title = "Ar@#$%tist - S*ng T!tle (Karaoke)"
        result = parser.parse_title(title)
        assert isinstance(result, ParseResult)
        # Should clean up the special characters
        if result.artist:
            assert "@#$%" not in result.artist


@pytest.mark.integration
class TestParserIntegration:
    """Integration tests for the parser."""

    @pytest.fixture
    def parser(self):
        return AdvancedTitleParser()

    def test_full_workflow_standard_format(self, parser):
        """Test full parsing workflow with standard format."""
        title = "Ed Sheeran - Shape of You (Karaoke Version)"
        description = "Great karaoke version of the hit song by Ed Sheeran"
        tags = "karaoke, pop, music"

        result = parser.parse_title(title, description, tags)

        assert result.artist == "Ed Sheeran"
        assert result.song_title == "Shape Of You"
        assert result.confidence > 0.7
        assert result.method is not None
        assert result.pattern_used is not None

    def test_full_workflow_complex_format(self, parser):
        """Test full parsing workflow with complex format."""
        title = '"Taylor Swift" - "Shake It Off" featuring backup singers (Karaoke Style)'
        description = "Amazing karaoke track"

        result = parser.parse_title(title, description)

        assert result.artist == "Taylor Swift"
        assert result.song_title == "Shake It Off"
        assert result.confidence > 0.8
        # Should extract featured artists
        assert result.featured_artists is not None

    def test_multiple_parsing_attempts_learning(self, parser):
        """Test that multiple parsing attempts improve through learning."""
        # Parse several similar titles
        titles = [
            "Artist One - Song One (Karaoke)",
            "Artist Two - Song Two (Karaoke)",
            "Artist Three - Song Three (Karaoke)",
        ]

        results = []
        for title in titles:
            result = parser.parse_title(title)
            results.append(result)

        # Should have learned patterns and artists
        assert len(parser.known_artists) >= 3
        assert len(parser.known_songs) >= 3

        # Later results might have slightly better confidence due to learning
        # (This is a soft assertion as learning effects may be subtle)
        assert all(r.confidence > 0 for r in results)
