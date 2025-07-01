"""Unit tests for multi_pass_controller.py."""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from collector.advanced_parser import ParseResult
from collector.config import MultiPassConfig
from collector.multi_pass_controller import MultiPassParsingController, MultiPassResult, PassResult
from collector.passes.base import PassType


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
        config.pass_progression = [
            PassType.CHANNEL_TEMPLATE,
            PassType.MUSICBRAINZ_SEARCH,
            PassType.DISCOGS_SEARCH,
            PassType.WEB_SEARCH,
        ]
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
        passes = {}
        for pass_type in [
            PassType.CHANNEL_TEMPLATE,
            PassType.MUSICBRAINZ_SEARCH,
            PassType.DISCOGS_SEARCH,
            PassType.WEB_SEARCH,
        ]:
            mock_pass = AsyncMock()
            mock_pass.pass_type = pass_type
            mock_pass.parse = AsyncMock()
            passes[pass_type] = mock_pass
        return passes

    @pytest.fixture
    def controller(self, config, mock_passes):
        """Create a controller instance with mocked passes."""
        with patch("collector.multi_pass_controller.logger"):
            controller = MultiPassParsingController(
                passes=mock_passes, advanced_parser=Mock(), config=config
            )
            controller.passes = mock_passes
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
        mock_passes[PassType.CHANNEL_TEMPLATE].parse.return_value = ParseResult(
            artist="Artist", song_title="Song", confidence=0.96, method="channel_template"
        )

        result = await controller.parse_video(video_info)

        assert result is not None
        assert result.final_result.artist == "Artist"
        assert result.final_result.confidence == 0.96
        assert len(result.passes_attempted) == 1  # Should stop after basic due to high confidence
        assert mock_passes[PassType.CHANNEL_TEMPLATE].parse.called
        assert not mock_passes[PassType.MUSICBRAINZ_SEARCH].parse.called  # Should skip remaining

    @pytest.mark.asyncio
    async def test_parse_video_multi_pass_progression(self, controller, mock_passes):
        """Test progression through multiple passes."""
        video_info = {"video_id": "test456", "title": "Unclear Title Karaoke", "description": ""}

        # Mock progressive improvement
        mock_passes[PassType.CHANNEL_TEMPLATE].parse.return_value = ParseResult(
            artist="Unknown", song_title="Title", confidence=0.5, method="channel_template"
        )

        mock_passes[PassType.MUSICBRAINZ_SEARCH].parse.return_value = ParseResult(
            artist="Artist", song_title="Song Title", confidence=0.85, method="musicbrainz_search"
        )

        result = await controller.parse_video(video_info)

        assert result.final_result.confidence == 0.85
        assert len(result.passes_attempted) == 3
        assert len(result.passes_attempted) == 3
        assert mock_passes[PassType.WEB_SEARCH].parse.called  # Should continue to web search

    @pytest.mark.asyncio
    async def test_parse_video_with_failure(self, controller, mock_passes):
        """Test handling of pass failures."""
        video_info = {"video_id": "test_fail", "title": "Test"}

        # Mock basic pass fails
        mock_passes[PassType.CHANNEL_TEMPLATE].parse.side_effect = Exception("Parse error")

        # Other passes succeed
        mock_passes[PassType.MUSICBRAINZ_SEARCH].parse.return_value = ParseResult(
            artist="Recovered Artist",
            song_title="Recovered Song",
            confidence=0.8,
            method="musicbrainz_search",
        )

        result = await controller.parse_video(video_info)

        assert result is not None
        assert result.final_result.artist == "Recovered Artist"
        assert any(
            pr.pass_type == PassType.CHANNEL_TEMPLATE and pr.parse_result is None
            for pr in result.passes_attempted
        )

    @pytest.mark.asyncio
    async def test_parse_video_with_timeout(self, controller, mock_passes):
        """Test pass timeout handling."""
        video_info = {"video_id": "test_timeout", "title": "Test"}

        # Mock basic pass times out
        async def slow_parse(*args, **kwargs):
            await asyncio.sleep(35)  # Longer than timeout
            return ParseResult(
                artist="Artist", song_title="Song", confidence=0.9, method="channel_template"
            )

        mock_passes[PassType.CHANNEL_TEMPLATE].parse = slow_parse

        # Backup pass succeeds
        mock_passes[PassType.MUSICBRAINZ_SEARCH].parse.return_value = ParseResult(
            artist="Backup Artist",
            song_title="Backup Song",
            confidence=0.75,
            method="musicbrainz_search",
        )

        with patch(
            "collector.multi_pass_controller.asyncio.wait_for",
            side_effect=[asyncio.TimeoutError, None],
        ):
            result = await controller.parse_video(video_info)

        assert result is not None

    @pytest.mark.asyncio
    async def test_parse_video_budget_enforcement(self, controller, mock_passes):
        """Test budget limit enforcement."""
        video_info = {"video_id": "test_budget", "title": "Test"}

        # Mock budget tracker
        controller.budget_tracker = AsyncMock()
        controller.budget_tracker.can_use_pass.side_effect = [
            True,  # basic - allowed
            True,  # channel_template - allowed
            False,  # enhanced - budget exceeded
            True,  # web_search - allowed
        ]

        # Set up mock results
        for pass_name in mock_passes:
            mock_passes[pass_name].parse.return_value = ParseResult(
                artist=f"{pass_name} Artist", song_title="Song", confidence=0.6, method=pass_name
            )

        result = await controller.parse_video(video_info)

        # Should skip enhanced due to budget
        assert not any(
            pr.pass_type == PassType.MUSICBRAINZ_SEARCH for pr in result.passes_attempted
        )
        assert controller.budget_tracker.can_use_pass.call_count >= 3

    @pytest.mark.asyncio
    async def test_parse_video_parallel_execution(self, controller, mock_passes):
        """Test parallel pass execution when enabled."""
        controller.config.parallel_execution = True
        controller.config.parallel_groups = [
            [PassType.CHANNEL_TEMPLATE],
            [PassType.MUSICBRAINZ_SEARCH, PassType.WEB_SEARCH],
        ]

        video_info = {"video_id": "test_parallel", "title": "Test"}

        # Mock all passes with different timings
        async def create_result(pass_name, delay):
            await asyncio.sleep(delay)
            return ParseResult(
                artist=f"{pass_name} Artist", song_title="Song", confidence=0.7, method=pass_name
            )

        mock_passes[PassType.CHANNEL_TEMPLATE].parse.side_effect = lambda *a, **k: create_result(
            "channel_template", 0.1
        )
        mock_passes[PassType.MUSICBRAINZ_SEARCH].parse.side_effect = lambda *a, **k: create_result(
            "musicbrainz", 0.2
        )
        mock_passes[PassType.DISCOGS_SEARCH].parse.side_effect = lambda *a, **k: create_result(
            "discogs", 0.1
        )
        mock_passes[PassType.WEB_SEARCH].parse.side_effect = lambda *a, **k: create_result(
            "web_search", 0.15
        )

        start_time = asyncio.get_event_loop().time()
        result = await controller.parse_video(video_info)
        duration = asyncio.get_event_loop().time() - start_time

        # Should complete faster than sequential (0.55s)
        assert duration < 0.4  # Max should be ~0.35s with overhead
        assert result is not None

    @pytest.mark.asyncio
    async def test_parse_video_retry_mechanism(self, controller, mock_passes):
        """Test retry mechanism on failure."""
        video_info = {"video_id": "test_retry", "title": "Test"}

        # Mock basic pass fails twice then succeeds
        call_count = 0

        async def flaky_parse(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return ParseResult(
                artist="Artist", song_title="Song", confidence=0.9, method="channel_template"
            )

        mock_passes[PassType.CHANNEL_TEMPLATE].parse = flaky_parse

        with patch("collector.multi_pass_controller.asyncio.sleep"):  # Skip delays
            result = await controller.parse_video(video_info)

        assert result is not None
        assert result.final_result.artist == "Artist"
        assert call_count == 3  # Initial + 2 retries

    @pytest.mark.asyncio
    async def test_parse_video_caching(self, controller, mock_passes):
        """Test result caching."""
        video_info = {"video_id": "test_cache", "title": "Test"}

        mock_passes[PassType.CHANNEL_TEMPLATE].parse.return_value = ParseResult(
            artist="Cached Artist",
            song_title="Cached Song",
            confidence=0.9,
            method="channel_template",
        )

        # First parse
        result1 = await controller.parse_video(video_info)

        # Second parse should use cache
        result2 = await controller.parse_video(video_info)

        assert result1.final_result.artist == result2.final_result.artist
        assert mock_passes[PassType.CHANNEL_TEMPLATE].parse.call_count == 1  # Only called once

    @pytest.mark.asyncio
    async def test_merge_results(self, controller):
        """Test result merging logic."""
        base_result = ParseResult(
            artist="Artist", song_title="Song", confidence=0.7, method="channel_template"
        )

        enhanced_result = ParseResult(
            artist="Artist",
            song_title="Song",
            confidence=0.8,
            method="musicbrainz_search",
            metadata={"genre": "Pop", "year": 2020},
            featured_artists="Featured",
        )

        merged = controller._merge_results(base_result, enhanced_result)

        assert merged.artist == "Artist"
        assert merged.confidence >= 0.8
        assert merged.genre == "Pop"
        assert merged.year == 2020
        assert "Featured" in merged.featured_artists

    def test_should_continue_parsing(self, controller):
        """Test logic for continuing parsing."""
        # High confidence should stop
        result1 = ParseResult(artist="Artist", song_title="Song", confidence=0.96, method="test")
        assert not controller._should_continue_parsing(result1, 1, 4)

        # Low confidence should continue
        result2 = ParseResult(artist="Artist", song_title="Song", confidence=0.5, method="test")
        assert controller._should_continue_parsing(result2, 1, 4)

        # Max passes reached should stop
        result3 = ParseResult(artist="Artist", song_title="Song", confidence=0.7, method="test")
        assert not controller._should_continue_parsing(result3, 4, 4)

    @pytest.mark.asyncio
    async def test_get_statistics(self, controller):
        """Test statistics collection."""
        # Mock some parsing history
        controller.stats = {
            "total_videos": 10,
            "total_passes": 25,
            "pass_success_counts": {"basic": 10, "enhanced": 8},
            "pass_failure_counts": {"web_search": 2},
            "average_confidence": 0.82,
            "cache_hits": 3,
        }

        stats = await controller.get_statistics()

        assert stats["total_videos"] == 10
        assert stats["total_passes"] == 25
        assert stats["cache_hit_rate"] == 0.3  # 3/10
        assert "pass_success_counts" in stats

    @pytest.mark.asyncio
    async def test_parse_video_empty_input(self, controller):
        """Test handling of empty video info."""
        result = await controller.parse_video({})
        assert result is not None
        assert result.final_result is None or result.final_result.confidence < 0.5

        result2 = await controller.parse_video({"title": ""})
        assert result2 is not None
        assert result2.final_result is None or result2.final_result.confidence < 0.5

    @pytest.mark.asyncio
    async def test_improvements_tracking(self, controller, mock_passes):
        """Test tracking of improvements between passes."""
        video_info = {"video_id": "test_improve", "title": "Test"}

        # Progressive improvements
        mock_passes[PassType.CHANNEL_TEMPLATE].parse.return_value = ParseResult(
            artist="Artist", song_title="Song", confidence=0.6, method="channel_template"
        )

        mock_passes[PassType.MUSICBRAINZ_SEARCH].parse.return_value = ParseResult(
            artist="Artist",
            song_title="Song",
            confidence=0.8,
            method="musicbrainz_search",
            metadata={"genre": "Rock", "year": 1985},
        )

        result = await controller.parse_video(video_info)

        assert result.final_confidence > 0
        assert result.final_confidence > 0
