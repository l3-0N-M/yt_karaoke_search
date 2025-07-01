"""Unit tests for youtube search provider."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from collector.config import ScrapingConfig
from collector.search.providers.youtube import YouTubeSearchProvider


class TestYouTubeSearchProvider:
    """Test cases for YouTubeSearchProvider."""

    @pytest.fixture
    def mock_ytdlp(self):
        """Mock yt-dlp."""
        with patch("collector.search.providers.youtube.yt_dlp") as mock_yt:
            mock_yt.YoutubeDL = AsyncMock()
            yield mock_yt

    @pytest.fixture
    def youtube_provider(self, mock_ytdlp):
        """Create a YouTubeSearchProvider instance."""
        provider = YouTubeSearchProvider(scraping_config=ScrapingConfig())
        # Mock the YoutubeDL instance
        mock_ydl = Mock()
        mock_ydl.extract_info = AsyncMock()
        # YouTubeSearchProvider uses yt_dlp.YoutubeDL directly in methods
        return provider

    @pytest.mark.asyncio
    async def test_search_success(self, youtube_provider):
        """Test successful YouTube search."""
        # Mock search results
        youtube_provider.ydl.extract_info.return_value = {
            "entries": [
                {
                    "id": "video1",
                    "title": "Artist - Song (Karaoke Version)",
                    "duration": 240,
                    "uploader": "Karaoke Channel",
                    "view_count": 10000,
                    "like_count": 500,
                    "upload_date": "20230615",
                    "description": "Karaoke version of the hit song",
                },
                {
                    "id": "video2",
                    "title": "Artist - Song (Instrumental)",
                    "duration": 235,
                    "uploader": "Music Channel",
                    "view_count": 5000,
                    "like_count": 200,
                    "upload_date": "20230610",
                    "description": "Instrumental karaoke track",
                },
            ]
        }

        results = await youtube_provider.search_videos(query="Artist Song karaoke", max_results=10)

        assert len(results) == 2
        assert results[0]["id"] == "video1"
        assert results[0]["title"] == "Artist - Song (Karaoke Version)"
        assert results[0]["duration"] == 240
        assert results[0]["source"] == "youtube"

    @pytest.mark.asyncio
    async def test_search_with_filters(self, youtube_provider):
        """Test search with various filters."""
        youtube_provider.ydl.extract_info.return_value = {"entries": []}

        # Test with karaoke filter
        await youtube_provider.search_videos(
            query="test song", max_results=5, filters={"karaoke_only": True}
        )

        # Verify search query was modified
        call_args = youtube_provider.ydl.extract_info.call_args
        assert "karaoke" in call_args[0][0].lower()

    @pytest.mark.asyncio
    async def test_search_empty_results(self, youtube_provider):
        """Test search with no results."""
        youtube_provider.ydl.extract_info.return_value = {"entries": []}

        results = await youtube_provider.search_videos(
            query="nonexistent song xyz123", max_results=10
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_search_error_handling(self, youtube_provider):
        """Test error handling during search."""
        youtube_provider.ydl.extract_info.side_effect = Exception("Network error")

        results = await youtube_provider.search_videos(query="test query", max_results=10)

        assert results == []  # Should return empty list on error

    @pytest.mark.asyncio
    async def test_result_filtering(self, youtube_provider):
        """Test filtering of search results."""
        youtube_provider.ydl.extract_info.return_value = {
            "entries": [
                {"id": "karaoke1", "title": "Song (Karaoke Version)", "duration": 240},
                {"id": "notkaraoke", "title": "Song (Music Video)", "duration": 240},
                {"id": "karaoke2", "title": "Song (Instrumental)", "duration": 240},
            ]
        }

        results = await youtube_provider.search_videos(
            query="song", max_results=10, filters={"karaoke_only": True}
        )

        # Should filter out non-karaoke results
        assert len(results) <= 3
        for result in results:
            title_lower = result["title"].lower()
            assert any(term in title_lower for term in ["karaoke", "instrumental"])

    @pytest.mark.asyncio
    async def test_search_pagination(self, youtube_provider):
        """Test search with pagination/offset."""
        # Create 20 mock results
        all_entries = [
            {"id": f"video{i}", "title": f"Song {i} Karaoke", "duration": 240} for i in range(20)
        ]

        youtube_provider.ydl.extract_info.return_value = {"entries": all_entries}

        # First page
        results1 = await youtube_provider.search_videos(query="song karaoke", max_results=10)

        assert len(results1) == 10
        assert results1[0]["id"] == "video0"

    @pytest.mark.asyncio
    async def test_metadata_extraction(self, youtube_provider):
        """Test proper metadata extraction from results."""
        youtube_provider.ydl.extract_info.return_value = {
            "entries": [
                {
                    "id": "test123",
                    "title": "Full Metadata Video",
                    "duration": 180,
                    "uploader": "Test Channel",
                    "uploader_id": "channel123",
                    "view_count": 50000,
                    "like_count": 2000,
                    "dislike_count": 50,
                    "comment_count": 100,
                    "upload_date": "20230101",
                    "description": "Full description",
                    "thumbnail": "https://example.com/thumb.jpg",
                    "webpage_url": "https://youtube.com/watch?v=test123",
                    "tags": ["karaoke", "music", "instrumental"],
                    "categories": ["Music"],
                    "age_limit": 0,
                }
            ]
        }

        results = await youtube_provider.search_videos("test", max_results=1)

        assert len(results) == 1
        result = results[0]

        # Check all metadata is properly extracted
        assert result["id"] == "test123"
        assert result["uploader"] == "Test Channel"
        assert result["view_count"] == 50000
        assert result["upload_date"] == "20230101"
        assert "tags" in result
        assert "karaoke" in result["tags"]

    @pytest.mark.asyncio
    async def test_search_timeout(self, youtube_provider):
        """Test search timeout handling."""
        import asyncio

        async def slow_extract(*args, **kwargs):
            await asyncio.sleep(30)  # Simulate slow response
            return {"entries": []}

        youtube_provider.ydl.extract_info = Mock(side_effect=slow_extract)

        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
            results = await youtube_provider.search_videos(query="test", max_results=10)

            assert results == []  # Should handle timeout gracefully

    @pytest.mark.asyncio
    async def test_search_query_building(self, youtube_provider):
        """Test search query building with various parameters."""
        test_cases = [
            # Basic query
            {"query": "artist song", "expected_contains": ["artist", "song"]},
            # Query with special characters
            {"query": "artist & song (remix)", "expected_contains": ["artist", "song", "remix"]},
            # Query with filters
            {"query": "test", "filters": {"duration": "short"}, "expected_contains": ["test"]},
        ]

        for test_case in test_cases:
            youtube_provider.ydl.extract_info.return_value = {"entries": []}

            await youtube_provider.search_videos(
                query=test_case["query"], max_results=5, filters=test_case.get("filters", {})
            )

            call_args = youtube_provider.ydl.extract_info.call_args[0][0]
            for expected in test_case["expected_contains"]:
                assert expected in call_args

    @pytest.mark.asyncio
    async def test_result_deduplication(self, youtube_provider):
        """Test deduplication of search results."""
        youtube_provider.ydl.extract_info.return_value = {
            "entries": [
                {"id": "video1", "title": "Song Karaoke", "duration": 240},
                {"id": "video1", "title": "Song Karaoke", "duration": 240},  # Duplicate
                {"id": "video2", "title": "Song 2 Karaoke", "duration": 180},
            ]
        }

        results = await youtube_provider.search_videos("song", max_results=10)

        # Should remove duplicates
        assert len(results) == 2
        video_ids = [r["id"] for r in results]
        assert len(set(video_ids)) == len(video_ids)  # All unique

    @pytest.mark.asyncio
    async def test_invalid_result_handling(self, youtube_provider):
        """Test handling of invalid/incomplete results."""
        youtube_provider.ydl.extract_info.return_value = {
            "entries": [
                {"id": "valid", "title": "Valid Video", "duration": 240},
                {"id": None, "title": "Invalid Video"},  # Missing ID
                {"id": "notitle", "duration": 180},  # Missing title
                None,  # Null entry
                {"id": "valid2", "title": "Another Valid", "duration": 200},
            ]
        }

        results = await youtube_provider.search_videos("test", max_results=10)

        # Should only include valid results
        assert len(results) == 2
        assert all(r.get("id") and r.get("title") for r in results)

    def test_is_available(self, youtube_provider):
        """Test provider availability check."""
        assert youtube_provider.is_available() is True

        # Test when yt-dlp is not available
        with patch("collector.search.providers.youtube.yt_dlp", None):
            provider = YouTubeSearchProvider(scraping_config=ScrapingConfig())
            assert provider.is_available() is False

    @pytest.mark.asyncio
    async def test_search_with_channel_filter(self, youtube_provider):
        """Test search filtered by channel."""
        youtube_provider.ydl.extract_info.return_value = {
            "entries": [
                {
                    "id": "video1",
                    "title": "Song 1",
                    "uploader": "Target Channel",
                    "uploader_id": "channel123",
                },
                {
                    "id": "video2",
                    "title": "Song 2",
                    "uploader": "Other Channel",
                    "uploader_id": "channel456",
                },
            ]
        }

        results = await youtube_provider.search_videos(
            query="song", max_results=10, filters={"channel_id": "channel123"}
        )

        # Should filter by channel
        assert all(r.get("uploader_id") == "channel123" for r in results)
