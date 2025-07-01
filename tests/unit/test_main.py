"""Unit tests for main.py (KaraokeCollector)."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from collector.advanced_parser import ParseResult
from collector.config import CollectorConfig
from collector.main import KaraokeCollector


class TestKaraokeCollector:
    """Test cases for KaraokeCollector."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        config = CollectorConfig()
        return config

    @pytest.fixture
    def mock_search_engine(self):
        """Create a mock search engine."""
        engine = AsyncMock()
        engine.search = AsyncMock(return_value=[])
        return engine

    @pytest.fixture
    def mock_db_manager(self):
        """Create a mock database manager."""
        db = Mock()
        db.save_video_data = AsyncMock()
        db.get_video_by_id = Mock(return_value=None)
        db.get_statistics = Mock(return_value={})
        return db

    @pytest.fixture
    def mock_processor(self):
        """Create a mock video processor."""
        processor = Mock()
        processor.process_video = AsyncMock()
        return processor

    @pytest.fixture
    def mock_controller(self):
        """Create a mock multi-pass controller."""
        controller = AsyncMock()
        controller.parse_video = AsyncMock()
        controller.get_statistics = AsyncMock(return_value={})
        return controller

    @pytest.fixture
    def collector(
        self, config, mock_search_engine, mock_db_manager, mock_processor, mock_controller
    ):
        """Create a KaraokeCollector instance with mocked dependencies."""
        with patch("collector.main.MultiStrategySearchEngine"), patch(
            "collector.main.OptimizedDatabaseManager"
        ), patch("collector.main.VideoProcessor"), patch(
            "collector.main.MultiPassParsingController",
        ):
            collector = KaraokeCollector(config=config)
            collector.search_engine = mock_search_engine
            collector.db_manager = mock_db_manager
            collector.video_processor = mock_processor
            collector.multi_pass_controller = mock_controller
            return collector

    @pytest.mark.asyncio
    async def test_collect_videos_basic(self, collector, mock_search_engine):
        """Test basic video collection."""
        mock_search_engine.search_videos.return_value = [
            {
                "id": "video1",
                "title": "Artist - Song (Karaoke)",
                "url": "https://youtube.com/watch?v=video1",
                "duration": 240,
            },
            {
                "id": "video2",
                "title": "Artist2 - Song2 (Karaoke)",
                "url": "https://youtube.com/watch?v=video2",
                "duration": 180,
            },
        ]
        collector.multi_pass_controller.parse_video.return_value = Mock(
            final_result=ParseResult(
                artist="Parsed Artist", song_title="Parsed Song", confidence=0.9, method="test"
            )
        )
        stats = await collector.collect_videos(queries=["karaoke songs"], max_videos_per_query=2)
        assert stats == 0
        assert mock_search_engine.search_videos.called

    @pytest.mark.asyncio
    async def test_collect_videos_with_channel(self, collector, mock_search_engine):
        """Test collecting videos from a specific channel."""
        channel_videos = [{"id": f"ch_video{i}", "title": f"Song {i}"} for i in range(5)]
        mock_search_engine.extract_channel_videos.return_value = channel_videos
        mock_search_engine.extract_channel_info.return_value = {"channel_id": "channel123"}
        stats = await collector.collect_from_channel(channel_url="channel123", max_videos=5)
        assert stats is not None
        search_call = mock_search_engine.extract_channel_videos.call_args
        assert "channel123" in search_call[0]

    @pytest.mark.asyncio
    async def test_collect_videos_duplicate_handling(self, collector, mock_db_manager):
        """Test handling of duplicate videos."""
        video_rows = [
            {"video_id": "dup1", "title": "Duplicate Song"},
            {"video_id": "unique1", "title": "Unique Song"},
        ]
        mock_db_manager.get_existing_video_ids_batch.return_value = ["dup1"]
        processed_count = await collector._process_video_batch(video_rows)
        assert processed_count == 0

    @pytest.mark.asyncio
    async def test_collect_videos_error_handling(self, collector, mock_search_engine):
        """Test error handling during collection."""
        mock_search_engine.search_videos.side_effect = Exception("Search failed")
        stats = await collector.collect_videos(queries=["test"], max_videos_per_query=10)
        assert stats == 0

    @pytest.mark.asyncio
    async def test_process_video_success(self, collector, mock_processor, mock_db_manager):
        """Test successful video processing."""
        video_info = {
            "video_id": "test123",
            "title": "Test Song",
            "url": "https://youtube.com/watch?v=test123",
        }
        mock_processor.process_video.return_value = Mock(
            is_success=True, confidence_score=0.95, video_data=video_info
        )
        mock_db_manager.save_video_data.return_value = True
        with patch.object(collector, "_is_video_processed", return_value=False):
            await collector._process_video_batch([video_info])
        assert mock_db_manager.save_video_data.called

    def test_collector_initialization(self):
        """Test collector initialization with various configs."""
        with patch("collector.main.OptimizedDatabaseManager"), patch(
            "collector.main.VideoProcessor"
        ), patch("collector.main.MultiStrategySearchEngine"), patch(
            "collector.main.MultiPassParsingController"
        ):
            # Default config
            collector1 = KaraokeCollector(config=CollectorConfig())
            assert collector1.config is not None
            assert collector1.config.database.path == "karaoke_videos.db"

            # Custom config
            custom_config = CollectorConfig()
            custom_config.database.path = "custom.db"
            collector2 = KaraokeCollector(custom_config)
            assert collector2.config.database.path == "custom.db"
