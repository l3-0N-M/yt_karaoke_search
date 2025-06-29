"""Comprehensive tests for the MusicBrainz validation pass module."""

from dataclasses import asdict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from collector.advanced_parser import AdvancedTitleParser, ParseResult
from collector.passes.base import PassType
from collector.passes.musicbrainz_validation_pass import (
    MusicBrainzValidationPass,
    ValidationResult,
)


class TestValidationResult:
    """Test the ValidationResult dataclass."""

    def test_validation_result_creation(self):
        """Test creating a ValidationResult with default values."""
        result = ValidationResult(
            validated=True,
            confidence_adjustment=1.2,
        )
        assert result.validated is True
        assert result.confidence_adjustment == 1.2
        assert result.enriched_data == {}
        assert result.validation_method == ""
        assert result.mb_match_score == 0

    def test_validation_result_with_all_values(self):
        """Test creating a ValidationResult with all values."""
        enriched_data = {"artist": "Test Artist", "recording_id": "123"}
        result = ValidationResult(
            validated=True,
            confidence_adjustment=1.1,
            enriched_data=enriched_data,
            validation_method="musicbrainz_match",
            mb_match_score=85,
        )
        assert result.validated is True
        assert result.confidence_adjustment == 1.1
        assert result.enriched_data == enriched_data
        assert result.validation_method == "musicbrainz_match"
        assert result.mb_match_score == 85

    def test_validation_result_failed(self):
        """Test creating a failed ValidationResult."""
        result = ValidationResult(
            validated=False,
            confidence_adjustment=0.8,
            validation_method="no_mb_results",
        )
        assert result.validated is False
        assert result.confidence_adjustment == 0.8
        assert result.validation_method == "no_mb_results"

    def test_validation_result_serializable(self):
        """Test that ValidationResult can be converted to dict."""
        result = ValidationResult(
            validated=True,
            confidence_adjustment=1.2,
            validation_method="test_method",
        )
        result_dict = asdict(result)
        assert isinstance(result_dict, dict)
        assert result_dict["validated"] is True
        assert result_dict["confidence_adjustment"] == 1.2


class TestMusicBrainzValidationPassInitialization:
    """Test MusicBrainzValidationPass initialization."""

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
        validation_pass = MusicBrainzValidationPass(mock_advanced_parser)
        assert validation_pass.pass_type == PassType.MUSICBRAINZ_VALIDATION

    def test_initialization_with_db(self, mock_advanced_parser, mock_db_manager):
        """Test initialization with database manager."""
        validation_pass = MusicBrainzValidationPass(mock_advanced_parser, mock_db_manager)
        assert validation_pass.advanced_parser == mock_advanced_parser
        assert validation_pass.db_manager == mock_db_manager
        assert validation_pass.validation_threshold == 0.7
        assert validation_pass.enrichment_enabled is True
        assert validation_pass.strict_validation is False
        assert isinstance(validation_pass.stats, dict)

    def test_initialization_without_db(self, mock_advanced_parser):
        """Test initialization without database manager."""
        validation_pass = MusicBrainzValidationPass(mock_advanced_parser)
        assert validation_pass.db_manager is None

    def test_mb_search_pass_created(self, mock_advanced_parser):
        """Test that MusicBrainzSearchPass is created."""
        validation_pass = MusicBrainzValidationPass(mock_advanced_parser)
        assert validation_pass.mb_search_pass is not None

    def test_statistics_initialization(self, mock_advanced_parser):
        """Test that statistics are properly initialized."""
        validation_pass = MusicBrainzValidationPass(mock_advanced_parser)
        expected_stats = [
            "total_validations",
            "successful_validations",
            "confidence_boosts",
            "confidence_penalties",
            "enrichments_added",
            "validation_failures",
            "artist_corrections",
            "title_corrections",
            "existing_mb_data_used",
        ]
        for stat in expected_stats:
            assert stat in validation_pass.stats
            assert validation_pass.stats[stat] == 0


class TestMainParseMethod:
    """Test the main parse method."""

    @pytest.fixture
    def validation_pass(self):
        """Create a MusicBrainzValidationPass instance."""
        parser = MagicMock(spec=AdvancedTitleParser)
        return MusicBrainzValidationPass(parser)

    @pytest.fixture
    def sample_web_search_result(self):
        """Create a sample web search result."""
        return ParseResult(
            artist="Test Artist",
            song_title="Test Song",
            confidence=0.8,
            method="web_search",
            pattern_used="web_pattern",
        )

    @pytest.mark.asyncio
    async def test_parse_dependencies_unavailable(self, validation_pass):
        """Test parse when MusicBrainz dependencies are unavailable."""
        with patch("collector.passes.musicbrainz_validation_pass.HAS_MUSICBRAINZ", False):
            result = await validation_pass.parse("Test Title")
            assert result is None

    @pytest.mark.asyncio
    async def test_parse_no_metadata(self, validation_pass):
        """Test parse with no metadata."""
        with patch("collector.passes.musicbrainz_validation_pass.HAS_MUSICBRAINZ", True):
            result = await validation_pass.parse("Test Title")
            assert result is None

    @pytest.mark.asyncio
    async def test_parse_no_web_search_result(self, validation_pass):
        """Test parse with metadata but no web_search_result."""
        metadata = {"other_data": "value"}
        with patch("collector.passes.musicbrainz_validation_pass.HAS_MUSICBRAINZ", True):
            result = await validation_pass.parse("Test Title", metadata=metadata)
            assert result is None

    @pytest.mark.asyncio
    async def test_parse_invalid_web_search_result(self, validation_pass):
        """Test parse with invalid web_search_result format."""
        metadata = {"web_search_result": "invalid_format"}
        with patch("collector.passes.musicbrainz_validation_pass.HAS_MUSICBRAINZ", True):
            result = await validation_pass.parse("Test Title", metadata=metadata)
            assert result is None

    @pytest.mark.asyncio
    async def test_parse_successful_validation(self, validation_pass, sample_web_search_result):
        """Test successful validation and enhancement."""
        metadata = {"web_search_result": sample_web_search_result}

        # Mock validation result
        mock_validation_result = ValidationResult(
            validated=True,
            confidence_adjustment=1.2,
            enriched_data={"musicbrainz_recording_id": "test-id"},
            validation_method="musicbrainz_match",
        )

        with patch(
            "collector.passes.musicbrainz_validation_pass.HAS_MUSICBRAINZ", True
        ), patch.object(validation_pass, "_validate_against_musicbrainz") as mock_validate:

            mock_validate.return_value = mock_validation_result

            result = await validation_pass.parse("Test Title", metadata=metadata)

            assert result is not None
            assert result.confidence > sample_web_search_result.confidence  # Should be boosted
            assert "mb_validated" in result.method
            assert validation_pass.stats["total_validations"] == 1
            assert validation_pass.stats["successful_validations"] == 1
            assert validation_pass.stats["confidence_boosts"] == 1

    @pytest.mark.asyncio
    async def test_parse_failed_validation(self, validation_pass, sample_web_search_result):
        """Test failed validation handling."""
        metadata = {"web_search_result": sample_web_search_result}

        # Mock failed validation result
        mock_validation_result = ValidationResult(
            validated=False,
            confidence_adjustment=0.8,
            validation_method="no_mb_results",
        )

        with patch(
            "collector.passes.musicbrainz_validation_pass.HAS_MUSICBRAINZ", True
        ), patch.object(validation_pass, "_validate_against_musicbrainz") as mock_validate:

            mock_validate.return_value = mock_validation_result

            result = await validation_pass.parse("Test Title", metadata=metadata)

            assert result is not None
            assert result.confidence < sample_web_search_result.confidence  # Should be penalized
            assert result.metadata["musicbrainz_validation"] == "failed"
            assert validation_pass.stats["validation_failures"] == 1

    @pytest.mark.asyncio
    async def test_parse_exception_handling(self, validation_pass, sample_web_search_result):
        """Test exception handling in parse method."""
        metadata = {"web_search_result": sample_web_search_result}

        with patch(
            "collector.passes.musicbrainz_validation_pass.HAS_MUSICBRAINZ", True
        ), patch.object(validation_pass, "_validate_against_musicbrainz") as mock_validate:

            mock_validate.side_effect = Exception("Test error")

            result = await validation_pass.parse("Test Title", metadata=metadata)

            # Should return original result on error
            assert result == sample_web_search_result
            assert validation_pass.stats["validation_failures"] == 1

    @pytest.mark.asyncio
    async def test_parse_slow_validation_logging(self, validation_pass, sample_web_search_result):
        """Test logging of slow validations."""
        metadata = {"web_search_result": sample_web_search_result}

        with patch(
            "collector.passes.musicbrainz_validation_pass.HAS_MUSICBRAINZ", True
        ), patch.object(validation_pass, "_validate_against_musicbrainz") as mock_validate, patch(
            "time.time"
        ) as mock_time, patch(
            "collector.passes.musicbrainz_validation_pass.logger"
        ) as mock_logger:

            mock_validate.return_value = ValidationResult(
                validated=False, confidence_adjustment=1.0
            )
            mock_time.side_effect = [0, 15]  # 15 second processing time

            await validation_pass.parse("Test Title", metadata=metadata)

            # Should log warning for slow validation
            mock_logger.warning.assert_called()


class TestValidationAgainstMusicBrainz:
    """Test MusicBrainz validation logic."""

    @pytest.fixture
    def validation_pass(self):
        """Create a MusicBrainzValidationPass instance."""
        parser = MagicMock(spec=AdvancedTitleParser)
        return MusicBrainzValidationPass(parser)

    @pytest.fixture
    def sample_parse_result(self):
        """Create a sample parse result."""
        return ParseResult(
            artist="Test Artist",
            song_title="Test Song",
            confidence=0.8,
            method="web_search",
        )

    @pytest.mark.asyncio
    async def test_validate_incomplete_parse_data(self, validation_pass):
        """Test validation with incomplete parse data."""
        parse_result = ParseResult(
            artist="Test Artist",
            song_title=None,  # Missing title
            confidence=0.8,
        )

        result = await validation_pass._validate_against_musicbrainz(parse_result, "Test Title")

        assert result.validated is False
        assert result.confidence_adjustment == 0.9
        assert result.validation_method == "incomplete_parse_data"

    @pytest.mark.asyncio
    async def test_validate_with_existing_data(self, validation_pass, sample_parse_result):
        """Test validation with existing MusicBrainz data."""
        existing_data = {
            "authoritative_artist": "Test Artist",
            "authoritative_title": "Test Song",
            "musicbrainz_recording_id": "existing-id",
        }

        with patch.object(validation_pass, "_check_existing_musicbrainz_data") as mock_check:
            mock_check.return_value = existing_data

            result = await validation_pass._validate_against_musicbrainz(
                sample_parse_result, "Test Title"
            )

            assert result.validated is True
            assert result.confidence_adjustment == 1.2
            assert result.enriched_data == existing_data
            assert result.validation_method == "existing_musicbrainz_data"

    @pytest.mark.asyncio
    async def test_validate_successful_mb_match(self, validation_pass, sample_parse_result):
        """Test successful MusicBrainz match validation."""
        mock_mb_match = MagicMock()
        mock_mb_match.artist_name = "Test Artist"
        mock_mb_match.song_title = "Test Song"
        mock_mb_match.score = 90
        mock_mb_match.recording_id = "test-recording-id"
        mock_mb_match.artist_id = "test-artist-id"
        mock_mb_match.metadata = {}

        with patch.object(
            validation_pass, "_check_existing_musicbrainz_data"
        ) as mock_check, patch.object(
            validation_pass.mb_search_pass, "_search_musicbrainz"
        ) as mock_search, patch.object(
            validation_pass, "_calculate_validation_confidence"
        ) as mock_calc_conf, patch.object(
            validation_pass, "_extract_enrichment_data"
        ) as mock_extract:

            mock_check.return_value = None
            mock_search.return_value = [mock_mb_match]
            mock_calc_conf.return_value = 0.8  # Above threshold
            mock_extract.return_value = {"musicbrainz_recording_id": "test-recording-id"}

            result = await validation_pass._validate_against_musicbrainz(
                sample_parse_result, "Test Title"
            )

            assert result.validated is True
            assert result.confidence_adjustment == 0.8
            assert result.validation_method == "musicbrainz_match"
            assert result.mb_match_score == 90

    @pytest.mark.asyncio
    async def test_validate_low_mb_similarity(self, validation_pass, sample_parse_result):
        """Test validation with low MusicBrainz similarity."""
        mock_mb_match = MagicMock()
        mock_mb_match.score = 50

        with patch.object(
            validation_pass, "_check_existing_musicbrainz_data"
        ) as mock_check, patch.object(
            validation_pass.mb_search_pass, "_search_musicbrainz"
        ) as mock_search, patch.object(
            validation_pass, "_calculate_validation_confidence"
        ) as mock_calc_conf:

            mock_check.return_value = None
            mock_search.return_value = [mock_mb_match]
            mock_calc_conf.return_value = 0.5  # Below threshold

            result = await validation_pass._validate_against_musicbrainz(
                sample_parse_result, "Test Title"
            )

            assert result.validated is False
            assert result.confidence_adjustment == 0.8
            assert result.validation_method == "low_mb_similarity"

    @pytest.mark.asyncio
    async def test_validate_no_mb_results(self, validation_pass, sample_parse_result):
        """Test validation with no MusicBrainz results."""
        with patch.object(
            validation_pass, "_check_existing_musicbrainz_data"
        ) as mock_check, patch.object(
            validation_pass.mb_search_pass, "_search_musicbrainz"
        ) as mock_search:

            mock_check.return_value = None
            mock_search.return_value = []  # No matches

            result = await validation_pass._validate_against_musicbrainz(
                sample_parse_result, "Test Title"
            )

            assert result.validated is False
            assert result.confidence_adjustment == 0.85
            assert result.validation_method == "no_mb_results"

    @pytest.mark.asyncio
    async def test_validate_fallback_queries(self, validation_pass, sample_parse_result):
        """Test validation with fallback queries."""
        mock_mb_match = MagicMock()
        mock_mb_match.score = 80

        with patch.object(
            validation_pass, "_check_existing_musicbrainz_data"
        ) as mock_check, patch.object(
            validation_pass.mb_search_pass, "_search_musicbrainz"
        ) as mock_search, patch.object(
            validation_pass, "_calculate_validation_confidence"
        ) as mock_calc_conf:

            mock_check.return_value = None
            # First search returns empty, second returns results
            mock_search.side_effect = [[], [mock_mb_match]]
            mock_calc_conf.return_value = 0.8

            result = await validation_pass._validate_against_musicbrainz(
                sample_parse_result, "Test Title"
            )

            assert result.validated is True
            # Should have tried multiple queries
            assert mock_search.call_count > 1

    @pytest.mark.asyncio
    async def test_validate_api_error(self, validation_pass, sample_parse_result):
        """Test validation with API error."""
        with patch.object(
            validation_pass, "_check_existing_musicbrainz_data"
        ) as mock_check, patch.object(
            validation_pass.mb_search_pass, "_search_musicbrainz"
        ) as mock_search:

            mock_check.return_value = None
            mock_search.side_effect = Exception("API Error")

            result = await validation_pass._validate_against_musicbrainz(
                sample_parse_result, "Test Title"
            )

            assert result.validated is False
            assert result.confidence_adjustment == 1.0  # No penalty for API errors
            assert result.validation_method == "api_error"


class TestValidationConfidenceCalculation:
    """Test validation confidence calculation."""

    @pytest.fixture
    def validation_pass(self):
        """Create a MusicBrainzValidationPass instance."""
        parser = MagicMock(spec=AdvancedTitleParser)
        return MusicBrainzValidationPass(parser)

    @pytest.fixture
    def sample_parse_result(self):
        """Create a sample parse result."""
        return ParseResult(
            artist="Test Artist",
            song_title="Test Song",
            confidence=0.8,
        )

    def test_calculate_validation_confidence_exact_match(
        self, validation_pass, sample_parse_result
    ):
        """Test confidence calculation for exact match."""
        mock_mb_match = MagicMock()
        mock_mb_match.artist_name = "Test Artist"
        mock_mb_match.song_title = "Test Song"
        mock_mb_match.score = 95

        confidence = validation_pass._calculate_validation_confidence(
            sample_parse_result, mock_mb_match, "Test Artist - Test Song"
        )

        # Should be high confidence for exact match
        assert confidence > 1.0

    def test_calculate_validation_confidence_similar_match(
        self, validation_pass, sample_parse_result
    ):
        """Test confidence calculation for similar match."""
        mock_mb_match = MagicMock()
        mock_mb_match.artist_name = "Test Artst"  # Slight typo
        mock_mb_match.song_title = "Test Sng"  # Slight typo
        mock_mb_match.score = 80

        confidence = validation_pass._calculate_validation_confidence(
            sample_parse_result, mock_mb_match, "Test Artist - Test Song"
        )

        # Should be moderate confidence for similar match
        assert 0.5 < confidence < 1.0

    def test_calculate_validation_confidence_poor_match(self, validation_pass, sample_parse_result):
        """Test confidence calculation for poor match."""
        mock_mb_match = MagicMock()
        mock_mb_match.artist_name = "Completely Different Artist"
        mock_mb_match.song_title = "Totally Different Song"
        mock_mb_match.score = 30

        confidence = validation_pass._calculate_validation_confidence(
            sample_parse_result, mock_mb_match, "Test Artist - Test Song"
        )

        # Should be low confidence for poor match
        assert confidence < 0.5

    def test_calculate_validation_confidence_high_mb_score(
        self, validation_pass, sample_parse_result
    ):
        """Test confidence boost for high MusicBrainz score."""
        mock_mb_match = MagicMock()
        mock_mb_match.artist_name = "Test Artist"
        mock_mb_match.song_title = "Test Song"
        mock_mb_match.score = 100  # Perfect MB score

        confidence = validation_pass._calculate_validation_confidence(
            sample_parse_result, mock_mb_match, "Test Artist - Test Song"
        )

        # Should get boost from high MB score
        assert confidence > 1.2

    def test_calculate_validation_confidence_capped(self, validation_pass, sample_parse_result):
        """Test that confidence is properly capped."""
        mock_mb_match = MagicMock()
        mock_mb_match.artist_name = "Test Artist"
        mock_mb_match.song_title = "Test Song"
        mock_mb_match.score = 100

        confidence = validation_pass._calculate_validation_confidence(
            sample_parse_result, mock_mb_match, "Test Artist - Test Song"
        )

        # Should be capped at 1.5
        assert confidence <= 1.5


class TestEnrichmentDataExtraction:
    """Test enrichment data extraction."""

    @pytest.fixture
    def validation_pass(self):
        """Create a MusicBrainzValidationPass instance."""
        parser = MagicMock(spec=AdvancedTitleParser)
        return MusicBrainzValidationPass(parser)

    def test_extract_enrichment_data_basic(self, validation_pass):
        """Test basic enrichment data extraction."""
        mock_mb_match = MagicMock()
        mock_mb_match.recording_id = "test-recording-id"
        mock_mb_match.artist_id = "test-artist-id"
        mock_mb_match.confidence = 0.9
        mock_mb_match.score = 85
        mock_mb_match.metadata = {}

        enrichment = validation_pass._extract_enrichment_data(mock_mb_match)

        assert enrichment["musicbrainz_recording_id"] == "test-recording-id"
        assert enrichment["musicbrainz_artist_id"] == "test-artist-id"
        assert enrichment["musicbrainz_confidence"] == 0.9
        assert enrichment["musicbrainz_search_score"] == 85

    def test_extract_enrichment_data_with_releases(self, validation_pass):
        """Test enrichment extraction with release data."""
        mock_mb_match = MagicMock()
        mock_mb_match.recording_id = "test-id"
        mock_mb_match.artist_id = "artist-id"
        mock_mb_match.confidence = 0.9
        mock_mb_match.score = 85
        mock_mb_match.metadata = {
            "releases": [
                {"date": "2020-01-15"},
                {"date": "2019-12-01"},
                {"date": "2021-03-10"},
            ]
        }

        enrichment = validation_pass._extract_enrichment_data(mock_mb_match)

        # Should extract earliest release year
        assert enrichment["estimated_release_year"] == 2019

    def test_extract_enrichment_data_with_length(self, validation_pass):
        """Test enrichment extraction with recording length."""
        mock_mb_match = MagicMock()
        mock_mb_match.recording_id = "test-id"
        mock_mb_match.artist_id = "artist-id"
        mock_mb_match.confidence = 0.9
        mock_mb_match.score = 85
        mock_mb_match.metadata = {
            "length": 240000,  # 4 minutes in milliseconds
        }

        enrichment = validation_pass._extract_enrichment_data(mock_mb_match)

        assert enrichment["recording_length_ms"] == 240000

    def test_extract_enrichment_data_with_disambiguation(self, validation_pass):
        """Test enrichment extraction with disambiguation."""
        mock_mb_match = MagicMock()
        mock_mb_match.recording_id = "test-id"
        mock_mb_match.artist_id = "artist-id"
        mock_mb_match.confidence = 0.9
        mock_mb_match.score = 85
        mock_mb_match.metadata = {
            "disambiguation": "live version",
        }

        enrichment = validation_pass._extract_enrichment_data(mock_mb_match)

        assert enrichment["musicbrainz_disambiguation"] == "live version"

    def test_extract_enrichment_data_invalid_dates(self, validation_pass):
        """Test enrichment extraction with invalid release dates."""
        mock_mb_match = MagicMock()
        mock_mb_match.recording_id = "test-id"
        mock_mb_match.artist_id = "artist-id"
        mock_mb_match.confidence = 0.9
        mock_mb_match.score = 85
        mock_mb_match.metadata = {
            "releases": [
                {"date": "invalid-date"},
                {"date": "202"},  # Too short
                {"date": None},
            ]
        }

        enrichment = validation_pass._extract_enrichment_data(mock_mb_match)

        # Should not include estimated_release_year for invalid dates
        assert "estimated_release_year" not in enrichment


class TestExistingMusicBrainzDataCheck:
    """Test checking for existing MusicBrainz data."""

    @pytest.fixture
    def validation_pass_with_db(self):
        """Create a MusicBrainzValidationPass with mock database."""
        parser = MagicMock(spec=AdvancedTitleParser)
        db_manager = MagicMock()
        return MusicBrainzValidationPass(parser, db_manager)

    @pytest.fixture
    def validation_pass_no_db(self):
        """Create a MusicBrainzValidationPass without database."""
        parser = MagicMock(spec=AdvancedTitleParser)
        return MusicBrainzValidationPass(parser, None)

    @pytest.mark.asyncio
    async def test_check_existing_data_no_db(self, validation_pass_no_db):
        """Test checking existing data without database."""
        result = await validation_pass_no_db._check_existing_musicbrainz_data("Test Title")
        assert result is None

    @pytest.mark.asyncio
    async def test_check_existing_data_found(self, validation_pass_with_db):
        """Test finding existing MusicBrainz data."""
        mock_row = ("Test Artist", "Test Song", "recording-id", "artist-id", 0.85, 2020, 240000)

        mock_cursor = MagicMock()
        mock_cursor.fetchone = AsyncMock(return_value=mock_row)
        validation_pass_with_db.db_manager.execute_query = AsyncMock(return_value=mock_cursor)

        result = await validation_pass_with_db._check_existing_musicbrainz_data("Test Title")

        assert result is not None
        assert result["authoritative_artist"] == "Test Artist"
        assert result["authoritative_title"] == "Test Song"
        assert result["musicbrainz_recording_id"] == "recording-id"
        assert result["data_source"] == "database"

    @pytest.mark.asyncio
    async def test_check_existing_data_not_found(self, validation_pass_with_db):
        """Test when no existing data is found."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone = AsyncMock(return_value=None)
        validation_pass_with_db.db_manager.execute_query = AsyncMock(return_value=mock_cursor)

        result = await validation_pass_with_db._check_existing_musicbrainz_data("Test Title")

        assert result is None

    @pytest.mark.asyncio
    async def test_check_existing_data_db_error(self, validation_pass_with_db):
        """Test database error handling."""
        validation_pass_with_db.db_manager.execute_query = AsyncMock(
            side_effect=Exception("Database error")
        )

        result = await validation_pass_with_db._check_existing_musicbrainz_data("Test Title")

        assert result is None


class TestApplyValidationResults:
    """Test applying validation results to parse results."""

    @pytest.fixture
    def validation_pass(self):
        """Create a MusicBrainzValidationPass instance."""
        parser = MagicMock(spec=AdvancedTitleParser)
        return MusicBrainzValidationPass(parser)

    @pytest.fixture
    def sample_parse_result(self):
        """Create a sample parse result."""
        return ParseResult(
            artist="Original Artist",
            song_title="Original Song",
            confidence=0.8,
            method="web_search",
            pattern_used="web_pattern",
            metadata={"original_data": "value"},
        )

    def test_apply_validation_results_no_corrections(self, validation_pass, sample_parse_result):
        """Test applying validation results without corrections."""
        validation_result = ValidationResult(
            validated=True,
            confidence_adjustment=1.1,
            enriched_data={"musicbrainz_recording_id": "test-id"},
            validation_method="musicbrainz_match",
        )

        result = validation_pass._apply_validation_results(sample_parse_result, validation_result)

        assert result.original_artist == "Original Artist"
        assert result.song_title == "Original Song"
        assert result.confidence == 0.8 * 1.1
        assert "mb_validated" in result.method
        assert result.metadata["musicbrainz_validation"] == "success"
        assert result.metadata["musicbrainz_recording_id"] == "test-id"

    def test_apply_validation_results_with_corrections(self, validation_pass, sample_parse_result):
        """Test applying validation results with artist/title corrections."""
        validation_result = ValidationResult(
            validated=True,
            confidence_adjustment=1.2,
            enriched_data={
                "authoritative_artist": "Corrected Artist",
                "authoritative_title": "Corrected Song",
                "musicbrainz_recording_id": "test-id",
            },
            validation_method="existing_musicbrainz_data",
        )

        result = validation_pass._apply_validation_results(sample_parse_result, validation_result)

        assert result.original_artist == "Corrected Artist"
        assert result.song_title == "Corrected Song"
        assert result.metadata["artist_corrected"] is True
        assert result.metadata["title_corrected"] is True
        assert validation_pass.stats["artist_corrections"] == 1
        assert validation_pass.stats["title_corrections"] == 1

    def test_apply_validation_results_partial_corrections(
        self, validation_pass, sample_parse_result
    ):
        """Test applying validation results with only artist correction."""
        validation_result = ValidationResult(
            validated=True,
            confidence_adjustment=1.1,
            enriched_data={
                "authoritative_artist": "Corrected Artist",
                # No authoritative_title
                "musicbrainz_recording_id": "test-id",
            },
            validation_method="musicbrainz_match",
        )

        result = validation_pass._apply_validation_results(sample_parse_result, validation_result)

        assert result.original_artist == "Corrected Artist"
        assert result.song_title == "Original Song"  # Unchanged
        assert result.metadata["artist_corrected"] is True
        assert result.metadata["title_corrected"] is False

    def test_apply_validation_results_confidence_capping(
        self, validation_pass, sample_parse_result
    ):
        """Test that confidence is properly capped."""
        validation_result = ValidationResult(
            validated=True,
            confidence_adjustment=2.0,  # Very high adjustment
            enriched_data={},
            validation_method="musicbrainz_match",
        )

        result = validation_pass._apply_validation_results(sample_parse_result, validation_result)

        # Should be capped at 0.98
        assert result.confidence <= 0.98

    def test_apply_validation_results_metadata_preservation(
        self, validation_pass, sample_parse_result
    ):
        """Test that original metadata is preserved."""
        validation_result = ValidationResult(
            validated=True,
            confidence_adjustment=1.1,
            enriched_data={"new_data": "new_value"},
            validation_method="musicbrainz_match",
        )

        result = validation_pass._apply_validation_results(sample_parse_result, validation_result)

        # Original metadata should be preserved
        assert result.metadata["original_data"] == "value"
        # New data should be added
        assert result.metadata["new_data"] == "new_value"


class TestStatistics:
    """Test statistics functionality."""

    @pytest.fixture
    def validation_pass(self):
        """Create a MusicBrainzValidationPass instance."""
        parser = MagicMock(spec=AdvancedTitleParser)
        return MusicBrainzValidationPass(parser)

    def test_get_statistics_initial(self, validation_pass):
        """Test initial statistics."""
        stats = validation_pass.get_statistics()

        assert stats["total_validations"] == 0
        assert stats["successful_validations"] == 0
        assert stats["validation_failures"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["confidence_boosts"] == 0
        assert stats["confidence_penalties"] == 0
        assert stats["enrichments_added"] == 0
        assert stats["artist_corrections"] == 0
        assert stats["title_corrections"] == 0
        assert stats["existing_mb_data_used"] == 0
        assert "dependencies_available" in stats

    def test_get_statistics_with_data(self, validation_pass):
        """Test statistics with data."""
        # Simulate some activity
        validation_pass.stats["total_validations"] = 10
        validation_pass.stats["successful_validations"] = 7
        validation_pass.stats["validation_failures"] = 3
        validation_pass.stats["confidence_boosts"] = 4
        validation_pass.stats["confidence_penalties"] = 2
        validation_pass.stats["artist_corrections"] = 2
        validation_pass.stats["title_corrections"] = 1

        stats = validation_pass.get_statistics()

        assert stats["total_validations"] == 10
        assert stats["successful_validations"] == 7
        assert stats["validation_failures"] == 3
        assert stats["success_rate"] == 0.7
        assert stats["confidence_boosts"] == 4
        assert stats["confidence_penalties"] == 2
        assert stats["artist_corrections"] == 2
        assert stats["title_corrections"] == 1

    def test_get_statistics_zero_division(self, validation_pass):
        """Test statistics calculation with zero validations."""
        stats = validation_pass.get_statistics()

        # Should handle zero division gracefully
        assert stats["success_rate"] == 0.0


@pytest.mark.integration
class TestMusicBrainzValidationPassIntegration:
    """Integration tests for the MusicBrainz validation pass."""

    @pytest.fixture
    def full_validation_pass(self):
        """Create a fully functional MusicBrainz validation pass."""
        parser = AdvancedTitleParser()
        return MusicBrainzValidationPass(parser)

    @pytest.mark.asyncio
    async def test_full_validation_workflow_mocked(self, full_validation_pass):
        """Test full validation workflow with mocked components."""
        # Sample web search result to validate
        web_search_result = ParseResult(
            artist="Ed Sheeran",
            song_title="Shape of You",
            confidence=0.75,
            method="web_search",
        )

        metadata = {"web_search_result": web_search_result}

        # Mock MusicBrainz match
        mock_mb_match = MagicMock()
        mock_mb_match.artist_name = "Ed Sheeran"
        mock_mb_match.song_title = "Shape of You"
        mock_mb_match.score = 95
        mock_mb_match.recording_id = "test-recording-id"
        mock_mb_match.artist_id = "test-artist-id"
        mock_mb_match.confidence = 0.95
        mock_mb_match.metadata = {"length": 240000}

        with patch(
            "collector.passes.musicbrainz_validation_pass.HAS_MUSICBRAINZ", True
        ), patch.object(full_validation_pass.mb_search_pass, "_search_musicbrainz") as mock_search:

            mock_search.return_value = [mock_mb_match]

            result = await full_validation_pass.parse(
                "Ed Sheeran - Shape of You", metadata=metadata
            )

            assert result is not None
            assert result.original_artist == "Ed Sheeran"
            assert result.song_title == "Shape of You"
            assert result.confidence > web_search_result.confidence
            assert "mb_validated" in result.method
            assert "musicbrainz_recording_id" in result.metadata

    def test_confidence_calculation_scenarios(self, full_validation_pass):
        """Test confidence calculation in various scenarios."""
        parse_result = ParseResult(
            artist="Test Artist",
            song_title="Test Song",
            confidence=0.8,
        )

        test_cases = [
            # (mb_artist, mb_title, mb_score, expected_min_confidence)
            ("Test Artist", "Test Song", 95, 1.0),  # Exact match
            ("Test Artst", "Test Sng", 80, 0.6),  # Similar match
            ("Different Artist", "Different Song", 50, 0.3),  # Poor match
        ]

        for mb_artist, mb_title, mb_score, expected_min in test_cases:
            mock_mb_match = MagicMock()
            mock_mb_match.artist_name = mb_artist
            mock_mb_match.song_title = mb_title
            mock_mb_match.score = mb_score

            confidence = full_validation_pass._calculate_validation_confidence(
                parse_result, mock_mb_match, "Test Title"
            )

            assert (
                confidence >= expected_min
            ), f"Failed for {mb_artist}/{mb_title}: got {confidence}, expected >= {expected_min}"
            assert 0.0 <= confidence <= 1.5

    def test_enrichment_data_extraction_scenarios(self, full_validation_pass):
        """Test enrichment data extraction scenarios."""
        test_cases = [
            # Basic data
            {
                "recording_id": "id1",
                "artist_id": "aid1",
                "confidence": 0.9,
                "score": 85,
                "metadata": {},
            },
            # With releases
            {
                "recording_id": "id2",
                "artist_id": "aid2",
                "confidence": 0.8,
                "score": 75,
                "metadata": {
                    "releases": [{"date": "2020-01-01"}, {"date": "2019-06-15"}],
                    "length": 180000,
                    "disambiguation": "acoustic version",
                },
            },
        ]

        for case in test_cases:
            mock_mb_match = MagicMock()
            mock_mb_match.recording_id = case["recording_id"]
            mock_mb_match.artist_id = case["artist_id"]
            mock_mb_match.confidence = case["confidence"]
            mock_mb_match.score = case["score"]
            mock_mb_match.metadata = case["metadata"]

            enrichment = full_validation_pass._extract_enrichment_data(mock_mb_match)

            assert enrichment["musicbrainz_recording_id"] == case["recording_id"]
            assert enrichment["musicbrainz_artist_id"] == case["artist_id"]
            assert enrichment["musicbrainz_confidence"] == case["confidence"]
            assert enrichment["musicbrainz_search_score"] == case["score"]

            # Check optional fields
            if case["metadata"].get("releases"):
                assert "estimated_release_year" in enrichment
            if case["metadata"].get("length"):
                assert enrichment["recording_length_ms"] == case["metadata"]["length"]
            if case["metadata"].get("disambiguation"):
                assert (
                    enrichment["musicbrainz_disambiguation"] == case["metadata"]["disambiguation"]
                )
