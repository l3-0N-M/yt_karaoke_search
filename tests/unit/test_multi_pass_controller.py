"""Unit tests for multi_pass_controller.py."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from collector.advanced_parser import ParseResult
from collector.config import MultiPassConfig
from collector.multi_pass_controller import MultiPassParsingController, MultiPassResult, PassResult
from collector.passes.base import ParsingPass, PassType


class TestPassResult:
    """Test cases for PassResult dataclass."""

    def test_pass_result_creation(self):
        """Test creating a PassResult."""
        parse_result = ParseResult(
            artist="Test Artist", song_title="Test Song", confidence=0.9, method="test"
        )

        pass_result = PassResult(
            pass_type=PassType.CHANNEL_TEMPLATE,
            parse_result=parse_result,
            processing_time=150.5,
            metadata={"key": "value"},
        )

        assert pass_result.pass_type == PassType.CHANNEL_TEMPLATE
        assert pass_result.parse_result is not None
        assert pass_result.parse_result.artist == "Test Artist"
        assert pass_result.processing_time == 150.5
        assert pass_result.metadata["key"] == "value"


class TestMultiPassResult:
    """Test cases for MultiPassResult dataclass."""

    def test_multi_pass_result_creation(self):
        """Test creating a MultiPassResult."""
        parse_result = ParseResult(
            artist="Final Artist", song_title="Final Song", confidence=0.95, method="multi_pass"
        )

        pass_results = [
            PassResult(PassType.CHANNEL_TEMPLATE, None, processing_time=100.0),
            PassResult(PassType.MUSICBRAINZ_SEARCH, parse_result, processing_time=200.0),
        ]

        multi_result = MultiPassResult(
            video_id="test_video_123",
            original_title="Test Artist - Test Song",
            final_result=parse_result,
            passes_attempted=pass_results,
            total_processing_time=300,
            final_confidence=0.95,
        )

        assert multi_result.final_result is not None
        assert multi_result.final_result.artist == "Final Artist"
        assert len(multi_result.passes_attempted) == 2
        assert multi_result.total_processing_time == 300
        assert multi_result.final_confidence == 0.95


class TestMultiPassParsingController:
    """Test cases for MultiPassParsingController."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        config = Mock(spec=MultiPassConfig)
        config.channel_template = Mock()
        config.musicbrainz_search = Mock()
        config.discogs_search = Mock()
        config.web_search = Mock()
        config.musicbrainz_validation = Mock()
        config.ml_embedding = Mock()
        config.auto_retemplate = Mock()
        config.confidence_thresholds = {
            "skip_remaining": 0.95,
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4,
        }
        config.max_passes = 4
        config.parallel_execution = False
        config.pass_timeout_seconds = 30
        config.enable_caching = True
        config.retry_on_failure = True
        config.max_retries = 2
        config.retry_delay_seconds = 1
        config.budget_limits = {
            "discogs": {"daily": 10000, "per_video": 3},
            "musicbrainz": {"daily": 50000, "per_video": 5},
        }
        return config

    @pytest.fixture
    def mock_passes(self):
        """Create mock parsing passes."""
        passes = []
        for pass_type in PassType:
            mock_pass = Mock(spec=ParsingPass)
            mock_pass.pass_type = pass_type
            mock_pass.parse = AsyncMock()
            passes.append(mock_pass)
        return passes

    @pytest.fixture
    def controller(self, config, mock_passes):
        """Create a controller instance with mocked passes."""
        with patch("collector.multi_pass_controller.logger"):
            controller = MultiPassParsingController(
                config=config, passes=mock_passes, advanced_parser=Mock()
            )
            return controller

    @pytest.mark.asyncio
    async def test_parse_video_basic_success(self, controller, mock_passes):
        """Test successful parsing with basic pass."""
        video_info = {
            "video_id": "test123",
            "title": "Artist - Song (Karaoke)",
            "description": "Karaoke version",
        }

        # Mock basic pass returns high confidence result
        mock_pass = next(p for p in mock_passes if p.pass_type == PassType.CHANNEL_TEMPLATE)
        mock_pass.parse.return_value = ParseResult(
            artist="Artist", song_title="Song", confidence=0.96, method="channel_template"
        )

        result = await controller.parse_video(**video_info)

        assert result is not None
        assert result.final_result.artist == "Artist"
        assert result.final_result.confidence == 0.96
        assert len(result.passes_attempted) > 0
        mock_pass.parse.assert_called_once()

    @pytest.mark.asyncio
    async def test_parse_video_multi_pass_progression(self, controller, mock_passes):
        """Test progression through multiple passes."""
        video_info = {"video_id": "test456", "title": "Unclear Title Karaoke", "description": ""}

        # Mock progressive improvement
        mock_pass_channel = next(p for p in mock_passes if p.pass_type == PassType.CHANNEL_TEMPLATE)
        mock_pass_channel.parse.return_value = ParseResult(
            artist="Unknown", song_title="Title", confidence=0.5, method="channel_template"
        )

        mock_pass_mb = next(p for p in mock_passes if p.pass_type == PassType.MUSICBRAINZ_SEARCH)
        mock_pass_mb.parse.return_value = ParseResult(
            artist="Artist", song_title="Song Title", confidence=0.85, method="musicbrainz_search"
        )

        result = await controller.parse_video(**video_info)

        assert result.final_result.confidence == 0.85
        assert len(result.passes_attempted) > 1

    @pytest.mark.asyncio
    async def test_parse_video_with_failure(self, controller, mock_passes):
        """Test handling of pass failures."""
        video_info = {"video_id": "test_fail", "title": "Test"}

        # Mock basic pass fails
        mock_pass_channel = next(p for p in mock_passes if p.pass_type == PassType.CHANNEL_TEMPLATE)
        mock_pass_channel.parse.side_effect = Exception("Parse error")

        # Other passes succeed
        mock_pass_mb = next(p for p in mock_passes if p.pass_type == PassType.MUSICBRAINZ_SEARCH)
        mock_pass_mb.parse.return_value = ParseResult(
            artist="Recovered Artist",
            song_title="Recovered Song",
            confidence=0.8,
            method="musicbrainz_search",
        )

        result = await controller.parse_video(**video_info)

        assert result is not None
        assert result.final_result.artist == "Recovered Artist"
        assert any(
            pr.pass_type == PassType.CHANNEL_TEMPLATE and not pr.success
            for pr in result.passes_attempted
        )
