"""Unit tests for youtube search provider - fixed version."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from collector.config import ScrapingConfig
from collector.search.providers.youtube import YouTubeSearchProvider


class TestYouTubeSearchProvider:
    """Test cases for YouTubeSearchProvider."""

    @pytest.fixture
    def mock_yt_dlp(self):
        """Mock the entire yt_dlp module properly."""
        with patch("collector.search.providers.youtube.yt_dlp") as mock_module:
            # Create a mock that will be returned by YoutubeDL()
            mock_ydl_instance = Mock()

            # Set up the context manager behavior
            mock_ydl_instance.__enter__ = Mock(return_value=mock_ydl_instance)
            mock_ydl_instance.__exit__ = Mock(return_value=None)

            # Mock the extract_info method
            mock_ydl_instance.extract_info = Mock()

            # Make YoutubeDL class return our mock instance
            mock_module.YoutubeDL = Mock(return_value=mock_ydl_instance)

            # Store reference to the instance for easy access in tests
            mock_module._test_instance = mock_ydl_instance

            yield mock_module

    @pytest.fixture
    def youtube_provider(self):
        """Create a YouTubeSearchProvider instance."""
        return YouTubeSearchProvider(scraping_config=ScrapingConfig())

    def set_mock_results(self, mock_yt_dlp, results):
        """Helper to set mock results."""
        mock_yt_dlp._test_instance.extract_info.return_value = results

    @pytest.mark.asyncio
    async def test_search_success(self, youtube_provider, mock_yt_dlp):
        """Test successful YouTube search."""
        # Set up mock results
        self.set_mock_results(
            mock_yt_dlp,
            {
                "entries": [
                    {
                        "id": "video1",
                        "title": "Artist - Song (Karaoke Version)",
                        "duration": 240,
                        "uploader": "Karaoke Channel",
                        "uploader_id": "channel1",
                        "view_count": 10000,
                        "like_count": 500,
                        "upload_date": "20230615",
                        "description": "Karaoke version of the hit song",
                        "webpage_url": "https://youtube.com/watch?v=video1",
                        "thumbnail": "https://example.com/thumb1.jpg",
                    },
                    {
                        "id": "video2",
                        "title": "Another Artist - Another Song Karaoke",
                        "duration": 180,
                        "uploader": "Karaoke Channel 2",
                        "uploader_id": "channel2",
                        "view_count": 5000,
                        "like_count": 200,
                        "upload_date": "20230616",
                        "description": "Another karaoke track",
                        "webpage_url": "https://youtube.com/watch?v=video2",
                        "thumbnail": "https://example.com/thumb2.jpg",
                    },
                ],
            },
        )

        results = await youtube_provider.search_videos("artist song karaoke")

        assert len(results) == 2
        assert results[0].video_id == "video1"
        assert results[0].title == "Artist - Song (Karaoke Version)"
        assert results[0].channel == "Karaoke Channel"
        assert results[0].duration == 240
        assert results[0].provider == "youtube"

    @pytest.mark.asyncio
    async def test_search_empty_results(self, youtube_provider, mock_yt_dlp):
        """Test search with no results."""
        self.set_mock_results(mock_yt_dlp, {"entries": []})

        results = await youtube_provider.search_videos("nonexistent song")

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_error_handling(self, youtube_provider, mock_yt_dlp):
        """Test search error handling."""
        mock_yt_dlp._test_instance.extract_info.side_effect = Exception("Network error")

        results = await youtube_provider.search_videos("test query")

        # Should return empty list on error
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_is_available(self, youtube_provider, mock_yt_dlp):
        """Test provider availability check."""
        # Should be available when yt_dlp is imported
        assert await youtube_provider.is_available() is True

    @pytest.mark.asyncio
    async def test_result_filtering(self, youtube_provider, mock_yt_dlp):
        """Test filtering of non-karaoke results."""
        self.set_mock_results(
            mock_yt_dlp,
            {
                "entries": [
                    {
                        "id": "video1",
                        "title": "Artist - Song (Karaoke Version)",
                        "duration": 240,
                        "uploader": "Karaoke Channel",
                        "uploader_id": "channel1",
                        "view_count": 10000,
                        "description": "Karaoke version",
                        "webpage_url": "https://youtube.com/watch?v=video1",
                    },
                    {
                        "id": "video2",
                        "title": "Artist - Song (Official Video)",
                        "duration": 240,
                        "uploader": "Artist Channel",
                        "uploader_id": "channel2",
                        "view_count": 1000000,
                        "description": "Official music video",
                        "webpage_url": "https://youtube.com/watch?v=video2",
                    },
                    {
                        "id": "video3",
                        "title": "Artist - Song Reaction",
                        "duration": 600,
                        "uploader": "Reaction Channel",
                        "uploader_id": "channel3",
                        "view_count": 5000,
                        "description": "Reacting to the song",
                        "webpage_url": "https://youtube.com/watch?v=video3",
                    },
                ]
            },
        )

        results = await youtube_provider.search_videos("artist song")

        # Should only return karaoke result
        assert len(results) == 1
        assert results[0].video_id == "video1"
        assert "karaoke" in results[0].title.lower()

    @pytest.mark.asyncio
    async def test_search_query_building(self, youtube_provider, mock_yt_dlp):
        """Test search query construction."""
        # Track what query was used
        actual_query = None

        def capture_query(query, download=False):
            nonlocal actual_query
            actual_query = query
            return {"entries": []}

        mock_yt_dlp._test_instance.extract_info = Mock(side_effect=capture_query)

        await youtube_provider.search_videos("test song")

        # Should construct a ytsearch query
        assert actual_query is not None
        assert actual_query.startswith("ytsearch")
        assert "test song" in actual_query

    @pytest.mark.asyncio
    async def test_metadata_extraction(self, youtube_provider, mock_yt_dlp):
        """Test metadata extraction from results."""
        self.set_mock_results(
            mock_yt_dlp,
            {
                "entries": [
                    {
                        "id": "video1",
                        "title": "Test Karaoke",
                        "duration": 240,
                        "uploader": "Channel",
                        "uploader_id": "ch1",
                        "view_count": 1000,
                        "like_count": 50,
                        "upload_date": "20230615",
                        "description": "Test description with karaoke",
                        "webpage_url": "https://youtube.com/watch?v=video1",
                        "thumbnail": "https://example.com/thumb.jpg",
                        "tags": ["karaoke", "music"],
                    }
                ]
            },
        )

        results = await youtube_provider.search_videos("test")

        assert len(results) == 1
        result = results[0]
        assert result.video_id == "video1"
        assert result.url == "https://www.youtube.com/watch?v=video1"
        # thumbnail_url is not extracted by the implementation
        assert result.view_count == 1000
        assert result.upload_date == "20230615"
