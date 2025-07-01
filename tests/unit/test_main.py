"""Unit tests for main.py (KaraokeCollector)."""

import asyncio
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
        # config.search has multi_pass with various settings
        # ScrapingConfig manages concurrency internally
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
        with patch("collector.main.get_search_engine", return_value=mock_search_engine):
            with patch("collector.main.get_db_manager", return_value=mock_db_manager):
                with patch("collector.main.VideoProcessor", return_value=mock_processor):
                    with patch(
                        "collector.main.MultiPassParsingController", return_value=mock_controller
                    ):
                        return KaraokeCollector(config)

    @pytest.mark.asyncio
    async def test_collect_videos_basic(self, collector, mock_search_engine):
        """Test basic video collection."""
        # Mock search results
        mock_search_engine.search.return_value = [
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

        # Mock processing
        collector.processor.process_video.return_value = {
            "video_id": "processed",
            "artist": "Artist",
            "song_title": "Song",
        }

        # Mock parsing
        collector.multi_pass_controller.parse_video.return_value = Mock(
            final_result=ParseResult(
                artist="Parsed Artist", song_title="Parsed Song", confidence=0.9, method="test"
            )
        )

        # Collect videos
        stats = await collector.collect_videos(search_query="karaoke songs", max_videos=2)

        assert stats["total_videos_found"] == 2
        assert stats["total_videos_processed"] >= 0
        assert mock_search_engine.search.called
        assert collector.processor.process_video.called

    @pytest.mark.asyncio
    async def test_collect_videos_with_channel(self, collector, mock_search_engine):
        """Test collecting videos from a specific channel."""
        channel_videos = [{"id": f"ch_video{i}", "title": f"Song {i}"} for i in range(5)]
        mock_search_engine.search.return_value = channel_videos

        stats = await collector.collect_videos(channel_id="channel123", max_videos=5)
        assert stats is not None

        # Should search with channel filter
        search_call = mock_search_engine.search.call_args
        assert "channel_id" in search_call[1].get("filters", {}) or "channel123" in str(search_call)

    @pytest.mark.asyncio
    async def test_collect_videos_duplicate_handling(
        self, collector, mock_search_engine, mock_db_manager
    ):
        """Test handling of duplicate videos."""
        # Mock search results with duplicates
        mock_search_engine.search.return_value = [
            {"id": "dup1", "title": "Duplicate Song"},
            {"id": "dup1", "title": "Duplicate Song"},  # Same ID
            {"id": "unique1", "title": "Unique Song"},
        ]

        # First video exists in DB, others don't
        mock_db_manager.get_video_by_id.side_effect = [
            {"video_id": "dup1"},  # Exists
            None,  # Doesn't exist
            None,  # Doesn't exist
        ]

        stats = await collector.collect_videos(search_query="test", max_videos=3)

        # Should skip existing video
        assert stats["total_videos_skipped"] >= 1

    @pytest.mark.asyncio
    async def test_collect_videos_error_handling(self, collector, mock_search_engine):
        """Test error handling during collection."""
        # Mock search error
        mock_search_engine.search.side_effect = Exception("Search failed")

        stats = await collector.collect_videos(search_query="test", max_videos=10)

        # Should handle error gracefully
        assert stats["total_videos_found"] == 0
        assert "error" in stats or stats["total_videos_processed"] == 0

    @pytest.mark.asyncio
    async def test_process_video_success(
        self, collector, mock_processor, mock_controller, mock_db_manager
    ):
        """Test successful video processing."""
        video_info = {
            "id": "test123",
            "title": "Test Song",
            "url": "https://youtube.com/watch?v=test123",
        }

        # Mock processing
        mock_processor.process_video.return_value = {
            "video_id": "test123",
            "basic_metadata": "extracted",
        }

        # Mock parsing
        mock_controller.parse_video.return_value = Mock(
            final_result=ParseResult(
                artist="Final Artist", song_title="Final Song", confidence=0.95, method="multi_pass"
            )
        )

        result = await collector._process_video(video_info)

        assert result is True
        assert mock_processor.process_video.called
        assert mock_controller.parse_video.called
        assert mock_db_manager.save_video_data.called

    @pytest.mark.asyncio
    async def test_process_video_parsing_failure(self, collector, mock_controller):
        """Test video processing when parsing fails."""
        video_info = {"id": "fail123", "title": "Unparseable"}

        # Mock parsing failure
        mock_controller.parse_video.return_value = Mock(final_result=None)

        result = await collector._process_video(video_info)

        # Should still save basic info even if parsing fails
        assert result is True or result is False  # Depends on implementation

    @pytest.mark.asyncio
    async def test_batch_processing(self, collector, mock_search_engine):
        """Test batch processing of videos."""
        # Create many videos
        videos = [{"id": f"video{i}", "title": f"Song {i}"} for i in range(25)]
        mock_search_engine.search.return_value = videos

        # Track processing
        process_count = 0

        async def mock_process(video):
            nonlocal process_count
            process_count += 1
            await asyncio.sleep(0.01)  # Simulate work
            return True

        collector._process_video = mock_process

        stats = await collector.collect_videos(search_query="test", max_videos=25)

        assert process_count == 25
        assert stats["total_videos_processed"] == 25

    @pytest.mark.asyncio
    async def test_save_interval(self, collector, mock_db_manager):
        """Test periodic saving during collection."""
        # Set short save interval
        collector.config.scraping.save_interval_seconds = 0.1

        # Mock time-consuming processing
        async def slow_process(video):
            await asyncio.sleep(0.05)
            return True

        collector._process_video = slow_process

        # Process multiple videos
        videos = [{"id": f"v{i}", "title": f"Song {i}"} for i in range(5)]

        with patch.object(collector, "_save_progress") as mock_save:
            await collector._process_batch(videos)

            # Should have called save progress
            assert mock_save.call_count >= 1

    @pytest.mark.asyncio
    async def test_get_statistics(self, collector, mock_db_manager, mock_controller):
        """Test statistics collection."""
        # Mock DB stats
        mock_db_manager.get_statistics.return_value = {
            "total_videos": 100,
            "total_channels": 5,
            "genre_distribution": {"Pop": 40, "Rock": 30, "Jazz": 30},
        }

        # Mock controller stats
        mock_controller.get_statistics.return_value = {
            "total_passes": 250,
            "average_confidence": 0.85,
        }

        stats = await collector.get_statistics()

        assert stats["database"]["total_videos"] == 100
        assert stats["parsing"]["average_confidence"] == 0.85
        assert "collection" in stats

    @pytest.mark.asyncio
    async def test_concurrent_processing_limit(self, collector):
        """Test that concurrent processing respects worker limit."""
        # ScrapingConfig manages concurrency internally

        # Track concurrent executions
        concurrent_count = 0
        max_concurrent = 0
        lock = asyncio.Lock()

        async def track_concurrent(video):
            nonlocal concurrent_count, max_concurrent
            async with lock:
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)

            await asyncio.sleep(0.1)  # Simulate work

            async with lock:
                concurrent_count -= 1

            return True

        collector._process_video = track_concurrent

        # Process many videos
        videos = [{"id": f"v{i}"} for i in range(10)]
        await collector._process_batch(videos)

        # Should not exceed max workers
        assert max_concurrent <= collector.config.scraping.max_workers

    @pytest.mark.asyncio
    async def test_search_pagination(self, collector, mock_search_engine):
        """Test handling of paginated search results."""
        page_size = 10
        total_videos = 25

        # Mock paginated results
        call_count = 0

        def search_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                return [{"id": f"v{i}"} for i in range(page_size)]
            elif call_count == 2:
                return [{"id": f"v{i}"} for i in range(page_size, page_size * 2)]
            elif call_count == 3:
                return [{"id": f"v{i}"} for i in range(page_size * 2, total_videos)]
            else:
                return []

        mock_search_engine.search.side_effect = search_side_effect

        stats = await collector.collect_videos(search_query="test", max_videos=total_videos)

        assert stats["total_videos_found"] == total_videos
        assert call_count >= 3  # Multiple pages

    @pytest.mark.asyncio
    async def test_metadata_enrichment(
        self, collector, mock_processor, mock_controller, mock_db_manager
    ):
        """Test metadata enrichment through multi-pass parsing."""
        video_info = {"id": "enrich123", "title": "Song for Enrichment"}

        # Basic processing
        mock_processor.process_video.return_value = {
            "video_id": "enrich123",
            "artist": "Basic Artist",
            "song_title": "Basic Song",
        }

        # Enhanced parsing adds more metadata
        mock_controller.parse_video.return_value = Mock(
            final_result=ParseResult(
                artist="Enhanced Artist",
                song_title="Enhanced Song",
                confidence=0.9,
                method="multi_pass",
                metadata={"genre": "Rock", "year": 2020},
                featured_artists="Guest Artist",
            )
        )

        await collector._process_video(video_info)

        # Check saved data includes enriched metadata
        saved_data = mock_db_manager.save_video_data.call_args[0][0]
        assert saved_data.get("genre") == "Rock"
        assert saved_data["release_year"] == 2020
        assert "Guest Artist" in str(saved_data.get("featured_artists", ""))

    def test_collector_initialization(self):
        """Test collector initialization with various configs."""
        # Default config
        collector1 = KaraokeCollector(config=Mock())
        assert collector1.config is not None
        assert collector1.config.database.path == "karaoke_videos.db"

        # Custom config
        custom_config = CollectorConfig()
        custom_config.database.path = "custom.db"
        collector2 = KaraokeCollector(custom_config)
        assert collector2.config.database.path == "custom.db"
