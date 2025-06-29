"""Comprehensive tests for the MusicBrainz search pass module."""

from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest

from collector.advanced_parser import AdvancedTitleParser
from collector.passes.base import PassType
from collector.passes.musicbrainz_search_pass import (
    MusicBrainzMatch,
    MusicBrainzSearchPass,
)


class TestMusicBrainzMatch:
    """Test the MusicBrainzMatch dataclass."""

    def test_musicbrainz_match_creation(self):
        """Test creating a MusicBrainzMatch with required values."""
        match = MusicBrainzMatch(
            recording_id="test-recording-id",
            artist_id="test-artist-id",
            artist_name="Test Artist",
            song_title="Test Song",
            score=85,
            confidence=0.85,
        )
        assert match.recording_id == "test-recording-id"
        assert match.artist_id == "test-artist-id"
        assert match.artist_name == "Test Artist"
        assert match.song_title == "Test Song"
        assert match.score == 85
        assert match.confidence == 0.85
        assert match.metadata == {}

    def test_musicbrainz_match_with_metadata(self):
        """Test creating a MusicBrainzMatch with metadata."""
        metadata = {"release_date": "2020", "genre": "pop"}
        match = MusicBrainzMatch(
            recording_id="test-id",
            artist_id="artist-id",
            artist_name="Artist",
            song_title="Song",
            score=90,
            confidence=0.9,
            metadata=metadata,
        )
        assert match.metadata == metadata

    def test_musicbrainz_match_serializable(self):
        """Test that MusicBrainzMatch can be converted to dict."""
        match = MusicBrainzMatch(
            recording_id="test-id",
            artist_id="artist-id",
            artist_name="Artist",
            song_title="Song",
            score=90,
            confidence=0.9,
        )
        match_dict = asdict(match)
        assert isinstance(match_dict, dict)
        assert match_dict["recording_id"] == "test-id"
        assert match_dict["confidence"] == 0.9


class TestMusicBrainzSearchPassInitialization:
    """Test MusicBrainzSearchPass initialization."""

    @pytest.fixture
    def mock_advanced_parser(self):
        """Create a mock AdvancedTitleParser."""
        return MagicMock(spec=AdvancedTitleParser)

    @pytest.fixture
    def mock_db_manager(self):
        """Create a mock database manager."""
        return MagicMock()

    def test_pass_type(self, mock_advanced_parser):
        """Test that the pass type is correctly set."""
        mb_pass = MusicBrainzSearchPass(mock_advanced_parser)
        assert mb_pass.pass_type == PassType.MUSICBRAINZ_SEARCH

    def test_initialization_with_db(self, mock_advanced_parser, mock_db_manager):
        """Test initialization with database manager."""
        mb_pass = MusicBrainzSearchPass(mock_advanced_parser, mock_db_manager)
        assert mb_pass.advanced_parser == mock_advanced_parser
        assert mb_pass.db_manager == mock_db_manager
        assert mb_pass.max_search_results == 25
        assert mb_pass.min_confidence_threshold == 0.6
        assert mb_pass.fuzzy_match_threshold == 0.75
        assert isinstance(mb_pass.stats, dict)

    def test_initialization_without_db(self, mock_advanced_parser):
        """Test initialization without database manager."""
        mb_pass = MusicBrainzSearchPass(mock_advanced_parser)
        assert mb_pass.db_manager is None

    def test_statistics_initialization(self, mock_advanced_parser):
        """Test that statistics are properly initialized."""
        mb_pass = MusicBrainzSearchPass(mock_advanced_parser)
        expected_stats = [
            "total_searches",
            "successful_matches",
            "api_errors",
            "fuzzy_matches",
            "direct_matches",
            "title_mismatches_detected",
            "title_mismatches_strong_penalty",
        ]
        for stat in expected_stats:
            assert stat in mb_pass.stats
            assert mb_pass.stats[stat] == 0


class TestSearchQueryGeneration:
    """Test search query generation methods."""

    @pytest.fixture
    def mb_pass(self):
        """Create a MusicBrainzSearchPass instance."""
        parser = MagicMock(spec=AdvancedTitleParser)
        mb_pass = MusicBrainzSearchPass(parser)

        # Mock the filler processor
        mb_pass.filler_processor = MagicMock()
        mb_pass.filler_processor.clean_query.return_value = MagicMock(
            cleaned_query="Test Artist - Test Song"
        )

        return mb_pass

    def test_generate_search_queries_basic(self, mb_pass):
        """Test basic search query generation."""
        title = "Test Artist - Test Song (Karaoke)"
        queries = mb_pass._generate_search_queries(title)

        assert len(queries) > 0
        assert mb_pass.filler_processor.clean_query.called

    def test_generate_search_queries_with_separators(self, mb_pass):
        """Test query generation with different separators."""
        title = "Artist Name - Song Title"
        mb_pass.filler_processor.clean_query.return_value = MagicMock(cleaned_query=title)

        queries = mb_pass._generate_search_queries(title)

        # Should generate multiple queries including structured ones
        assert len(queries) >= 1
        # Should include the base query
        assert title in queries

    def test_generate_search_queries_quoted_parts(self, mb_pass):
        """Test query generation with quoted parts."""
        title = 'Artist - "Song Title" (Karaoke)'
        mb_pass.filler_processor.clean_query.return_value = MagicMock(cleaned_query=title)

        queries = mb_pass._generate_search_queries(title)

        # Should extract quoted parts
        assert any('recording:"Song Title"' in query for query in queries)

    def test_generate_search_queries_empty_input(self, mb_pass):
        """Test query generation with empty input."""
        title = ""
        mb_pass.filler_processor.clean_query.return_value = MagicMock(cleaned_query="")

        queries = mb_pass._generate_search_queries(title)

        assert len(queries) == 0

    def test_generate_search_queries_short_input(self, mb_pass):
        """Test query generation with very short input."""
        title = "ab"
        mb_pass.filler_processor.clean_query.return_value = MagicMock(cleaned_query="ab")

        queries = mb_pass._generate_search_queries(title)

        assert len(queries) == 0

    @pytest.mark.parametrize(
        "title,separator,expected_parts",
        [
            ("Artist - Song", " - ", ["Artist", "Song"]),
            ("Artist by Song", " by ", ["Artist", "Song"]),
            ("Artist from Album", " from ", ["Artist", "Album"]),
            ("Artist: Song", ":", ["Artist", "Song"]),
            ("Artist | Song", " | ", ["Artist", "Song"]),
        ],
    )
    def test_generate_search_queries_separators(self, mb_pass, title, separator, expected_parts):
        """Test query generation with various separators."""
        mb_pass.filler_processor.clean_query.return_value = MagicMock(cleaned_query=title)

        queries = mb_pass._generate_search_queries(title)

        # Should generate structured queries for artist and recording
        assert any(
            f'artist:"{expected_parts[0]}"' in query and f'recording:"{expected_parts[1]}"' in query
            for query in queries
        )
        assert any(
            f'artist:"{expected_parts[1]}"' in query and f'recording:"{expected_parts[0]}"' in query
            for query in queries
        )


class TestMusicBrainzAPISearch:
    """Test MusicBrainz API search functionality."""

    @pytest.fixture
    def mb_pass(self):
        """Create a MusicBrainzSearchPass instance."""
        parser = MagicMock(spec=AdvancedTitleParser)
        return MusicBrainzSearchPass(parser)

    @pytest.mark.asyncio
    async def test_search_musicbrainz_success(self, mb_pass):
        """Test successful MusicBrainz API search."""
        mock_result = {
            "recording-list": [
                {
                    "id": "test-recording-id",
                    "title": "Test Song",
                    "ext:score": 95,
                    "artist-credit": [{"artist": {"id": "test-artist-id", "name": "Test Artist"}}],
                    "release-list": [],
                    "length": 240000,
                    "disambiguation": "",
                }
            ]
        }

        with patch("collector.passes.musicbrainz_search_pass.HAS_MUSICBRAINZ", True), patch(
            "collector.passes.musicbrainz_search_pass.mb"
        ) as mock_mb:

            mock_mb.search_recordings.return_value = mock_result

            matches = await mb_pass._search_musicbrainz("test query")

            assert len(matches) > 0
            match = matches[0]
            assert match.recording_id == "test-recording-id"
            assert match.artist_name == "Test Artist"
            assert match.song_title == "Test Song"
            assert match.score == 95

    @pytest.mark.asyncio
    async def test_search_musicbrainz_no_results(self, mb_pass):
        """Test MusicBrainz search with no results."""
        mock_result = {"recording-list": []}

        with patch("collector.passes.musicbrainz_search_pass.HAS_MUSICBRAINZ", True), patch(
            "collector.passes.musicbrainz_search_pass.mb"
        ) as mock_mb:

            mock_mb.search_recordings.return_value = mock_result

            matches = await mb_pass._search_musicbrainz("nonexistent query")

            assert len(matches) == 0

    @pytest.mark.asyncio
    async def test_search_musicbrainz_api_error(self, mb_pass):
        """Test MusicBrainz search with API error."""
        with patch("collector.passes.musicbrainz_search_pass.HAS_MUSICBRAINZ", True), patch(
            "collector.passes.musicbrainz_search_pass.mb"
        ) as mock_mb:

            mock_mb.search_recordings.side_effect = Exception("API Error")

            matches = await mb_pass._search_musicbrainz("test query")

            assert len(matches) == 0

    @pytest.mark.asyncio
    async def test_search_musicbrainz_invalid_artist_credit(self, mb_pass):
        """Test handling of invalid artist credit structure."""
        mock_result = {
            "recording-list": [
                {
                    "id": "test-id",
                    "title": "Test Song",
                    "ext:score": 85,
                    "artist-credit": [],  # Empty artist credit
                }
            ]
        }

        with patch("collector.passes.musicbrainz_search_pass.HAS_MUSICBRAINZ", True), patch(
            "collector.passes.musicbrainz_search_pass.mb"
        ) as mock_mb:

            mock_mb.search_recordings.return_value = mock_result

            matches = await mb_pass._search_musicbrainz("test query")

            # Should skip recordings without valid artist credits
            assert len(matches) == 0

    @pytest.mark.asyncio
    async def test_search_musicbrainz_string_artist_credit(self, mb_pass):
        """Test handling of string artist credit."""
        mock_result = {
            "recording-list": [
                {
                    "id": "test-id",
                    "title": "Test Song",
                    "ext:score": 85,
                    "artist-credit": ["Test Artist"],  # String artist credit
                }
            ]
        }

        with patch("collector.passes.musicbrainz_search_pass.HAS_MUSICBRAINZ", True), patch(
            "collector.passes.musicbrainz_search_pass.mb"
        ) as mock_mb:

            mock_mb.search_recordings.return_value = mock_result
            mb_pass.min_confidence_threshold = 0.1  # Lower threshold for test

            matches = await mb_pass._search_musicbrainz("test query")

            assert len(matches) > 0
            match = matches[0]
            assert match.artist_name == "Test Artist"
            assert match.artist_id == ""  # No ID for string credits


class TestConfidenceCalculation:
    """Test confidence calculation methods."""

    @pytest.fixture
    def mb_pass(self):
        """Create a MusicBrainzSearchPass instance."""
        parser = MagicMock(spec=AdvancedTitleParser)
        return MusicBrainzSearchPass(parser)

    def test_calculate_confidence_basic(self, mb_pass):
        """Test basic confidence calculation."""
        confidence = mb_pass._calculate_confidence(
            80, "Test Song", "Test Artist", "Test Artist - Test Song"
        )

        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be reasonably high for good match

    def test_calculate_confidence_string_score(self, mb_pass):
        """Test confidence calculation with string score."""
        confidence = mb_pass._calculate_confidence(
            "85", "Test Song", "Test Artist", "Test Artist - Test Song"
        )

        assert 0.0 <= confidence <= 1.0

    def test_calculate_confidence_none_score(self, mb_pass):
        """Test confidence calculation with None score."""
        confidence = mb_pass._calculate_confidence(
            None, "Test Song", "Test Artist", "Test Artist - Test Song"
        )

        assert confidence >= 0.0

    def test_calculate_confidence_invalid_score(self, mb_pass):
        """Test confidence calculation with invalid score."""
        confidence = mb_pass._calculate_confidence(
            "invalid", "Test Song", "Test Artist", "Test Artist - Test Song"
        )

        assert confidence >= 0.0

    def test_calculate_confidence_direct_match(self, mb_pass):
        """Test confidence boost for direct matches."""
        # Query contains both artist and title
        query = "Test Artist - Test Song"
        confidence = mb_pass._calculate_confidence(80, "Test Song", "Test Artist", query)

        # Should get direct match boost
        assert confidence > 0.8
        assert mb_pass.stats["direct_matches"] > 0

    def test_calculate_confidence_partial_match(self, mb_pass):
        """Test confidence for partial matches."""
        # Query contains only title
        query = "Test Song karaoke"
        confidence = mb_pass._calculate_confidence(80, "Test Song", "Test Artist", query)

        # Should get some boost but less than direct match
        assert 0.5 < confidence < 1.0

    def test_calculate_confidence_fuzzy_match(self, mb_pass):
        """Test confidence boost for fuzzy matches."""
        # Similar but not exact match
        query = "Test Artst - Test Sng"  # Misspelled
        confidence = mb_pass._calculate_confidence(80, "Test Song", "Test Artist", query)

        # Should still get reasonable confidence due to fuzzy matching
        assert confidence > 0.3

    def test_calculate_confidence_short_strings(self, mb_pass):
        """Test confidence penalty for very short strings."""
        confidence = mb_pass._calculate_confidence(90, "A", "B", "A - B")

        # Should be penalized for short strings
        assert confidence < 0.8

    def test_calculate_confidence_with_mismatch_penalty(self, mb_pass):
        """Test confidence with title-artist mismatch penalty."""
        with patch.object(mb_pass, "_check_title_artist_mismatch") as mock_check:
            mock_check.return_value = 0.5  # 50% penalty

            confidence = mb_pass._calculate_confidence(
                90, "Song", "Wrong Artist", "Real Artist - Song"
            )

            # Should be significantly reduced due to penalty
            assert confidence < 0.5


class TestTitleArtistMismatchDetection:
    """Test title-artist mismatch detection."""

    @pytest.fixture
    def mb_pass(self):
        """Create a MusicBrainzSearchPass instance."""
        parser = MagicMock(spec=AdvancedTitleParser)
        return MusicBrainzSearchPass(parser)

    def test_check_title_artist_mismatch_good_match(self, mb_pass):
        """Test mismatch detection with good artist match."""
        query = "Taylor Swift - Shake It Off"
        mb_artist = "Taylor Swift"

        penalty = mb_pass._check_title_artist_mismatch(query, mb_artist)

        assert penalty == 0.0  # No penalty for exact match

    def test_check_title_artist_mismatch_similar_match(self, mb_pass):
        """Test mismatch detection with similar artist."""
        query = "Taylor Swift - Shake It Off"
        mb_artist = "T. Swift"

        penalty = mb_pass._check_title_artist_mismatch(query, mb_artist)

        # Should have low or no penalty for reasonable similarity
        assert penalty <= 0.3

    def test_check_title_artist_mismatch_poor_match(self, mb_pass):
        """Test mismatch detection with poor artist match."""
        query = "Taylor Swift - Shake It Off"
        mb_artist = "Completely Different Artist"

        penalty = mb_pass._check_title_artist_mismatch(query, mb_artist)

        assert penalty > 0.3  # Should have significant penalty

    def test_check_title_artist_mismatch_comma_format(self, mb_pass):
        """Test mismatch detection with comma format."""
        query = "Swift, Taylor - Shake It Off"
        mb_artist = "Taylor Swift"

        penalty = mb_pass._check_title_artist_mismatch(query, mb_artist)

        # Should handle comma format and find match
        assert penalty <= 0.3

    def test_check_title_artist_mismatch_noise_in_query(self, mb_pass):
        """Test mismatch detection with noise in query."""
        query = "Taylor Swift - Shake It Off karaoke lyrics"
        mb_artist = "Taylor Swift"

        penalty = mb_pass._check_title_artist_mismatch(query, mb_artist)

        # Should ignore noise and find good match
        assert penalty == 0.0

    def test_check_title_artist_mismatch_short_potential_artist(self, mb_pass):
        """Test mismatch detection with very short potential artist."""
        query = "TS - Song"
        mb_artist = "Taylor Swift"

        penalty = mb_pass._check_title_artist_mismatch(query, mb_artist)

        # Should not heavily penalize due to short potential artist
        assert penalty >= 0.0

    def test_check_title_artist_mismatch_end_of_query(self, mb_pass):
        """Test mismatch detection with artist at end of query."""
        query = "Shake It Off by Taylor Swift"
        mb_artist = "Taylor Swift"

        penalty = mb_pass._check_title_artist_mismatch(query, mb_artist)

        # Should find artist at end of query
        assert penalty <= 0.3


class TestParseResultConversion:
    """Test conversion to ParseResult."""

    @pytest.fixture
    def mb_pass(self):
        """Create a MusicBrainzSearchPass instance."""
        parser = MagicMock(spec=AdvancedTitleParser)
        return MusicBrainzSearchPass(parser)

    def test_convert_to_parse_result(self, mb_pass):
        """Test conversion of MusicBrainzMatch to ParseResult."""
        match = MusicBrainzMatch(
            recording_id="test-recording-id",
            artist_id="test-artist-id",
            artist_name="Test Artist",
            song_title="Test Song",
            score=85,
            confidence=0.85,
            metadata={"genre": "pop"},
        )

        result = mb_pass._convert_to_parse_result(match, "test query")

        assert result.original_artist == "Test Artist"
        assert result.song_title == "Test Song"
        assert result.confidence == 0.85
        assert result.method == "musicbrainz_search"
        assert result.pattern_used == "api_lookup"
        assert "musicbrainz_recording_id" in result.metadata
        assert result.metadata["musicbrainz_recording_id"] == "test-recording-id"
        assert result.metadata["search_query"] == "test query"
        assert result.metadata["genre"] == "pop"


class TestMainParseMethod:
    """Test the main parse method."""

    @pytest.fixture
    def mb_pass(self):
        """Create a MusicBrainzSearchPass instance."""
        parser = MagicMock(spec=AdvancedTitleParser)
        return MusicBrainzSearchPass(parser)

    @pytest.mark.asyncio
    async def test_parse_dependencies_unavailable(self, mb_pass):
        """Test parse when dependencies are unavailable."""
        with patch("collector.passes.musicbrainz_search_pass.HAS_MUSICBRAINZ", False):
            result = await mb_pass.parse("Test Title")
            assert result is None

    @pytest.mark.asyncio
    async def test_parse_no_queries_generated(self, mb_pass):
        """Test parse when no search queries are generated."""
        with patch("collector.passes.musicbrainz_search_pass.HAS_MUSICBRAINZ", True), patch(
            "collector.passes.musicbrainz_search_pass.HAS_REQUESTS", True
        ), patch.object(mb_pass, "_generate_search_queries") as mock_generate:

            mock_generate.return_value = []

            result = await mb_pass.parse("Test Title")
            assert result is None

    @pytest.mark.asyncio
    async def test_parse_successful_match(self, mb_pass):
        """Test successful parse with MusicBrainz match."""
        mock_match = MusicBrainzMatch(
            recording_id="test-id",
            artist_id="artist-id",
            artist_name="Test Artist",
            song_title="Test Song",
            score=90,
            confidence=0.9,
        )

        with patch("collector.passes.musicbrainz_search_pass.HAS_MUSICBRAINZ", True), patch(
            "collector.passes.musicbrainz_search_pass.HAS_REQUESTS", True
        ), patch.object(mb_pass, "_generate_search_queries") as mock_generate, patch.object(
            mb_pass, "_search_musicbrainz"
        ) as mock_search:

            mock_generate.return_value = ["test query"]
            mock_search.return_value = [mock_match]

            result = await mb_pass.parse("Test Artist - Test Song")

            assert result is not None
            assert result.original_artist == "Test Artist"
            assert result.song_title == "Test Song"
            assert result.confidence == 0.9
            assert mb_pass.stats["total_searches"] == 1
            assert mb_pass.stats["successful_matches"] == 1

    @pytest.mark.asyncio
    async def test_parse_multiple_queries_best_result(self, mb_pass):
        """Test parse with multiple queries returning best result."""
        mock_match1 = MusicBrainzMatch(
            recording_id="id1",
            artist_id="aid1",
            artist_name="Artist1",
            song_title="Song1",
            score=70,
            confidence=0.7,
        )
        mock_match2 = MusicBrainzMatch(
            recording_id="id2",
            artist_id="aid2",
            artist_name="Artist2",
            song_title="Song2",
            score=90,
            confidence=0.9,
        )

        with patch("collector.passes.musicbrainz_search_pass.HAS_MUSICBRAINZ", True), patch(
            "collector.passes.musicbrainz_search_pass.HAS_REQUESTS", True
        ), patch.object(mb_pass, "_generate_search_queries") as mock_generate, patch.object(
            mb_pass, "_search_musicbrainz"
        ) as mock_search:

            mock_generate.return_value = ["query1", "query2"]
            mock_search.side_effect = [[mock_match1], [mock_match2]]

            result = await mb_pass.parse("Test Title")

            assert result is not None
            assert result.confidence == 0.9  # Should pick better result
            assert result.original_artist == "Artist2"

    @pytest.mark.asyncio
    async def test_parse_early_exit_high_confidence(self, mb_pass):
        """Test early exit with high confidence match."""
        mock_match = MusicBrainzMatch(
            recording_id="id",
            artist_id="aid",
            artist_name="Artist",
            song_title="Song",
            score=95,
            confidence=0.95,
        )

        with patch("collector.passes.musicbrainz_search_pass.HAS_MUSICBRAINZ", True), patch(
            "collector.passes.musicbrainz_search_pass.HAS_REQUESTS", True
        ), patch.object(mb_pass, "_generate_search_queries") as mock_generate, patch.object(
            mb_pass, "_search_musicbrainz"
        ) as mock_search:

            mock_generate.return_value = ["query1", "query2"]
            mock_search.return_value = [mock_match]

            result = await mb_pass.parse("Test Title")

            assert result is not None
            assert result.confidence == 0.95
            # Should only call search once due to early exit
            assert mock_search.call_count == 1

    @pytest.mark.asyncio
    async def test_parse_exception_handling(self, mb_pass):
        """Test exception handling in parse method."""
        with patch("collector.passes.musicbrainz_search_pass.HAS_MUSICBRAINZ", True), patch(
            "collector.passes.musicbrainz_search_pass.HAS_REQUESTS", True
        ), patch.object(mb_pass, "_generate_search_queries") as mock_generate:

            mock_generate.side_effect = Exception("Test error")

            result = await mb_pass.parse("Test Title")

            assert result is None
            assert mb_pass.stats["api_errors"] == 1

    @pytest.mark.asyncio
    async def test_parse_slow_search_logging(self, mb_pass):
        """Test logging of slow searches."""
        with patch("collector.passes.musicbrainz_search_pass.HAS_MUSICBRAINZ", True), patch(
            "collector.passes.musicbrainz_search_pass.HAS_REQUESTS", True
        ), patch.object(mb_pass, "_generate_search_queries") as mock_generate, patch(
            "time.time"
        ) as mock_time, patch(
            "collector.passes.musicbrainz_search_pass.logger"
        ) as mock_logger:

            mock_generate.return_value = []
            mock_time.side_effect = [0, 20]  # 20 second processing time

            await mb_pass.parse("Test Title")

            # Should log warning for slow search
            mock_logger.warning.assert_called()


class TestStatistics:
    """Test statistics functionality."""

    @pytest.fixture
    def mb_pass(self):
        """Create a MusicBrainzSearchPass instance."""
        parser = MagicMock(spec=AdvancedTitleParser)
        return MusicBrainzSearchPass(parser)

    def test_get_statistics_initial(self, mb_pass):
        """Test initial statistics."""
        stats = mb_pass.get_statistics()

        assert stats["total_searches"] == 0
        assert stats["successful_matches"] == 0
        assert stats["api_errors"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["direct_matches"] == 0
        assert stats["fuzzy_matches"] == 0
        assert stats["title_mismatches_detected"] == 0
        assert stats["title_mismatches_strong_penalty"] == 0
        assert "dependencies_available" in stats

    def test_get_statistics_with_data(self, mb_pass):
        """Test statistics with data."""
        # Simulate some activity
        mb_pass.stats["total_searches"] = 10
        mb_pass.stats["successful_matches"] = 7
        mb_pass.stats["api_errors"] = 1
        mb_pass.stats["direct_matches"] = 3
        mb_pass.stats["fuzzy_matches"] = 2

        stats = mb_pass.get_statistics()

        assert stats["total_searches"] == 10
        assert stats["successful_matches"] == 7
        assert stats["success_rate"] == 0.7
        assert stats["api_errors"] == 1
        assert stats["direct_matches"] == 3
        assert stats["fuzzy_matches"] == 2

    def test_get_statistics_zero_division(self, mb_pass):
        """Test statistics calculation with zero searches."""
        stats = mb_pass.get_statistics()

        # Should handle zero division gracefully
        assert stats["success_rate"] == 0.0


@pytest.mark.integration
class TestMusicBrainzSearchPassIntegration:
    """Integration tests for the MusicBrainz search pass."""

    @pytest.fixture
    def full_mb_pass(self):
        """Create a fully functional MusicBrainz search pass."""
        parser = AdvancedTitleParser()
        return MusicBrainzSearchPass(parser)

    @pytest.mark.asyncio
    async def test_full_search_workflow_mocked(self, full_mb_pass):
        """Test full search workflow with mocked API."""
        mock_api_result = {
            "recording-list": [
                {
                    "id": "test-recording-id",
                    "title": "Shape of You",
                    "ext:score": 95,
                    "artist-credit": [{"artist": {"id": "test-artist-id", "name": "Ed Sheeran"}}],
                    "release-list": [],
                    "length": 240000,
                }
            ]
        }

        with patch("collector.passes.musicbrainz_search_pass.HAS_MUSICBRAINZ", True), patch(
            "collector.passes.musicbrainz_search_pass.HAS_REQUESTS", True
        ), patch("collector.passes.musicbrainz_search_pass.mb") as mock_mb:

            mock_mb.search_recordings.return_value = mock_api_result

            result = await full_mb_pass.parse("Ed Sheeran - Shape of You (Karaoke)")

            assert result is not None
            assert result.original_artist == "Ed Sheeran"
            assert result.song_title == "Shape of You"
            assert result.confidence > 0.8
            assert result.method == "musicbrainz_search"
            assert "musicbrainz_recording_id" in result.metadata

    @pytest.mark.asyncio
    async def test_query_generation_and_processing(self, full_mb_pass):
        """Test query generation and processing workflow."""
        # Test various title formats
        titles = [
            "Artist - Song (Karaoke)",
            '"Artist Name" - "Song Title"',
            "Artist by Song",
            "Song from Artist",
            "Artist: Song",
        ]

        for title in titles:
            queries = full_mb_pass._generate_search_queries(title)
            assert len(queries) > 0
            assert all(len(q) > 3 for q in queries)
            assert len(set(queries)) == len(queries)  # No duplicates

    def test_confidence_calculation_scenarios(self, full_mb_pass):
        """Test confidence calculation in various scenarios."""
        test_cases = [
            # (mb_score, title, artist, query, expected_min_confidence)
            (95, "Perfect Song", "Perfect Artist", "Perfect Artist - Perfect Song", 0.8),
            (50, "Okay Song", "Okay Artist", "Different - Query", 0.3),
            (90, "A", "B", "A - B", 0.5),  # Short strings penalty
            (80, "Song", "Artist", "Song by Artist", 0.6),  # Partial match
        ]

        for mb_score, title, artist, query, expected_min in test_cases:
            confidence = full_mb_pass._calculate_confidence(mb_score, title, artist, query)
            assert (
                confidence >= expected_min
            ), f"Failed for {query}: got {confidence}, expected >= {expected_min}"
            assert 0.0 <= confidence <= 1.0

    def test_mismatch_detection_scenarios(self, full_mb_pass):
        """Test title-artist mismatch detection scenarios."""
        test_cases = [
            # (query, mb_artist, expected_max_penalty)
            ("Taylor Swift - Shake It Off", "Taylor Swift", 0.1),
            ("Swift, Taylor - Shake It Off", "Taylor Swift", 0.1),
            ("Taylor Swift - Song", "Completely Different", 0.8),
            ("T Swift - Song", "Taylor Swift", 0.4),
            ("TS - Song", "Taylor Swift", 0.6),
        ]

        for query, mb_artist, expected_max in test_cases:
            penalty = full_mb_pass._check_title_artist_mismatch(query, mb_artist)
            assert 0.0 <= penalty <= 1.0
            assert (
                penalty <= expected_max
            ), f"Failed for '{query}' vs '{mb_artist}': got {penalty}, expected <= {expected_max}"
