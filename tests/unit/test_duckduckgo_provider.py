"""Unit tests for DuckDuckGo search provider."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from collector.search.providers.duckduckgo import DuckDuckGoSearchProvider


class TestDuckDuckGoSearchProvider:
    """Test cases for DuckDuckGoSearchProvider."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock aiohttp session."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        return session

    @pytest.fixture
    async def ddg_provider(self, mock_session):
        """Create a DuckDuckGoSearchProvider instance."""
        provider = DuckDuckGoSearchProvider()
        return provider

    @pytest.mark.asyncio
    async def test_search_success(self, ddg_provider, mock_session):
        """Test successful DuckDuckGo search."""
        # Mock API response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(
            return_value="""
            <html>
            <div class="result">
                <h2 class="result__title">
                    <a href="https://youtube.com/watch?v=test123">
                        Artist - Song (Karaoke Version)
                    </a>
                </h2>
                <div class="result__snippet">
                    Karaoke version with lyrics on screen. Duration: 3:45
                </div>
            </div>
            <div class="result">
                <h2 class="result__title">
                    <a href="https://youtube.com/watch?v=test456">
                        Artist - Song (Instrumental)
                    </a>
                </h2>
                <div class="result__snippet">
                    Instrumental backing track. Duration: 3:40
                </div>
            </div>
            </html>
        """
        )

        mock_session.get.return_value.__aenter__.return_value = mock_response

        results = await ddg_provider.search(
            query="Artist Song karaoke site:youtube.com", max_results=10
        )

        assert len(results) == 2
        assert results[0]["title"] == "Artist - Song (Karaoke Version)"
        assert "youtube.com/watch?v=test123" in results[0]["url"]
        assert results[0]["source"] == "duckduckgo"

    @pytest.mark.asyncio
    async def test_search_no_results(self, ddg_provider, mock_session):
        """Test search with no results."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="<html><body>No results found</body></html>")

        mock_session.get.return_value.__aenter__.return_value = mock_response

        results = await ddg_provider.search(query="nonexistent xyz123 karaoke", max_results=10)

        assert results == []

    @pytest.mark.asyncio
    async def test_search_error_handling(self, ddg_provider, mock_session):
        """Test error handling during search."""
        mock_session.get.side_effect = aiohttp.ClientError("Network error")

        results = await ddg_provider.search(query="test query", max_results=10)

        assert results == []  # Should return empty list on error

    @pytest.mark.asyncio
    async def test_search_rate_limiting(self, ddg_provider, mock_session):
        """Test handling of rate limiting (503 errors)."""
        mock_response = AsyncMock()
        mock_response.status = 503
        mock_response.text = AsyncMock(return_value="Rate limited")

        mock_session.get.return_value.__aenter__.return_value = mock_response

        results = await ddg_provider.search(query="test", max_results=10)

        assert results == []

    @pytest.mark.asyncio
    async def test_youtube_url_extraction(self, ddg_provider, mock_session):
        """Test extraction of YouTube URLs from results."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(
            return_value="""
            <html>
            <div class="result">
                <a href="https://youtube.com/watch?v=abc123">Video 1</a>
            </div>
            <div class="result">
                <a href="https://youtu.be/def456">Video 2</a>
            </div>
            <div class="result">
                <a href="https://vimeo.com/789">Not YouTube</a>
            </div>
            </html>
        """
        )

        mock_session.get.return_value.__aenter__.return_value = mock_response

        results = await ddg_provider.search(query="karaoke", max_results=10)

        # Should only include YouTube results
        youtube_results = [
            r for r in results if "youtube.com" in r["url"] or "youtu.be" in r["url"]
        ]
        assert len(youtube_results) >= 2

    @pytest.mark.asyncio
    async def test_search_query_formatting(self, ddg_provider, mock_session):
        """Test proper query formatting."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="<html></html>")

        mock_session.get.return_value.__aenter__.return_value = mock_response

        await ddg_provider.search(query="test query with spaces", max_results=5)

        # Check URL encoding
        call_args = mock_session.get.call_args
        url = call_args[0][0]
        assert "test+query+with+spaces" in url or "test%20query%20with%20spaces" in url

    @pytest.mark.asyncio
    async def test_parse_duration(self, ddg_provider):
        """Test duration parsing from various formats."""
        test_cases = [
            ("Duration: 3:45", 225),
            ("4:20 minutes", 260),
            ("Length: 2:30", 150),
            ("5 minutes 15 seconds", 315),
            ("No duration info", None),
        ]

        for text, expected_duration in test_cases:
            duration = ddg_provider._parse_duration(text)
            assert duration == expected_duration

    @pytest.mark.asyncio
    async def test_search_with_filters(self, ddg_provider, mock_session):
        """Test search with various filters."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="<html></html>")

        mock_session.get.return_value.__aenter__.return_value = mock_response

        # Test with site filter
        await ddg_provider.search(query="karaoke", max_results=10, filters={"site": "youtube.com"})

        call_args = mock_session.get.call_args[0][0]
        assert "site:youtube.com" in call_args

    @pytest.mark.asyncio
    async def test_result_deduplication(self, ddg_provider, mock_session):
        """Test deduplication of search results."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(
            return_value="""
            <html>
            <div class="result">
                <a href="https://youtube.com/watch?v=test123">Duplicate Video</a>
            </div>
            <div class="result">
                <a href="https://youtube.com/watch?v=test123">Duplicate Video</a>
            </div>
            <div class="result">
                <a href="https://youtube.com/watch?v=test456">Different Video</a>
            </div>
            </html>
        """
        )

        mock_session.get.return_value.__aenter__.return_value = mock_response

        results = await ddg_provider.search("test", max_results=10)

        # Should remove duplicates based on URL
        unique_urls = set(r["url"] for r in results)
        assert len(unique_urls) == len(results)

    @pytest.mark.asyncio
    async def test_snippet_extraction(self, ddg_provider, mock_session):
        """Test extraction of snippet/description."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(
            return_value="""
            <html>
            <div class="result">
                <h2><a href="https://youtube.com/watch?v=test">Title</a></h2>
                <div class="result__snippet">
                    This is a karaoke version with on-screen lyrics.
                    Perfect for singing along!
                </div>
            </div>
            </html>
        """
        )

        mock_session.get.return_value.__aenter__.return_value = mock_response

        results = await ddg_provider.search("test", max_results=1)

        assert len(results) == 1
        assert "description" in results[0]
        assert "karaoke version" in results[0]["description"]

    @pytest.mark.asyncio
    async def test_search_timeout(self, ddg_provider, mock_session):
        """Test search timeout handling."""
        import asyncio

        async def slow_response(*args, **kwargs):
            await asyncio.sleep(30)
            return AsyncMock()

        mock_session.get.side_effect = slow_response

        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
            results = await ddg_provider.search(query="test", max_results=10)

        assert results == []

    @pytest.mark.asyncio
    async def test_html_parsing_errors(self, ddg_provider, mock_session):
        """Test handling of malformed HTML."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="<html><invalid>Malformed HTML</broken>")

        mock_session.get.return_value.__aenter__.return_value = mock_response

        results = await ddg_provider.search("test", max_results=10)

        # Should handle parsing errors gracefully
        assert isinstance(results, list)

    def test_is_available(self, ddg_provider):
        """Test provider availability check."""
        assert ddg_provider.is_available() is True

    @pytest.mark.asyncio
    async def test_extract_video_id(self, ddg_provider):
        """Test YouTube video ID extraction."""
        test_cases = [
            ("https://youtube.com/watch?v=abc123", "abc123"),
            ("https://youtu.be/def456", "def456"),
            ("https://www.youtube.com/watch?v=ghi789&list=xyz", "ghi789"),
            ("https://m.youtube.com/watch?v=jkl012", "jkl012"),
            ("https://not-youtube.com/video", None),
        ]

        for url, expected_id in test_cases:
            video_id = ddg_provider._extract_video_id(url)
            assert video_id == expected_id

    @pytest.mark.asyncio
    async def test_max_results_limit(self, ddg_provider, mock_session):
        """Test that max_results is respected."""
        # Create many results
        results_html = ""
        for i in range(20):
            results_html += f"""
                <div class="result">
                    <a href="https://youtube.com/watch?v=test{i}">Video {i}</a>
                </div>
            """

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=f"<html>{results_html}</html>")

        mock_session.get.return_value.__aenter__.return_value = mock_response

        results = await ddg_provider.search("test", max_results=5)

        assert len(results) <= 5

    @pytest.mark.asyncio
    async def test_clean_title(self, ddg_provider):
        """Test title cleaning functionality."""
        test_cases = [
            ("Video Title - YouTube", "Video Title"),
            ("Title | Channel Name", "Title"),
            ("Title (Official Video)", "Title (Official Video)"),
            ("  Title with spaces  ", "Title with spaces"),
        ]

        for input_title, expected_title in test_cases:
            cleaned = ddg_provider._clean_title(input_title)
            assert cleaned == expected_title
