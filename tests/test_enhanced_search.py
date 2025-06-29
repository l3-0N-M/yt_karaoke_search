"""Tests for the enhanced search engine."""

import unittest.mock
from unittest.mock import MagicMock

import pytest

from collector.config import ScrapingConfig, SearchConfig
from collector.db import DatabaseManager
from collector.enhanced_search import MultiStrategySearchEngine
from collector.search.providers.base import SearchResult


@pytest.fixture
def search_config():
    return SearchConfig()


@pytest.fixture
def scraping_config():
    return ScrapingConfig()


@pytest.fixture
def db_manager():
    return MagicMock(spec=DatabaseManager)


@pytest.fixture
def search_engine(search_config, scraping_config, db_manager):
    # Patch the YouTubeSearchProvider class directly
    with unittest.mock.patch(
        "collector.search.providers.youtube.YouTubeSearchProvider.search_videos",
        new_callable=unittest.mock.AsyncMock,
    ) as mock_youtube_search_videos, unittest.mock.patch(
        "collector.search.cache_manager.CacheManager.get_search_results",
        new_callable=unittest.mock.AsyncMock,
    ) as mock_get_cache, unittest.mock.patch(
        "collector.search.cache_manager.CacheManager.cache_search_results",
        new_callable=unittest.mock.AsyncMock,
    ) as mock_set_cache:

        # Patch the BingSearchProvider class directly
        with unittest.mock.patch(
            "collector.search.providers.bing.BingSearchProvider.search_videos",
            new_callable=unittest.mock.AsyncMock,
        ) as mock_bing_search_videos:
            # Patch the DuckDuckGoSearchProvider class directly
            with unittest.mock.patch(
                "collector.search.providers.duckduckgo.DuckDuckGoSearchProvider.search_videos",
                new_callable=unittest.mock.AsyncMock,
            ) as mock_duckduckgo_search_videos:
                engine = MultiStrategySearchEngine(search_config, scraping_config, db_manager)
                engine.mock_youtube_search_videos = mock_youtube_search_videos
                engine.mock_bing_search_videos = mock_bing_search_videos
                engine.mock_duckduckgo_search_videos = mock_duckduckgo_search_videos
                engine.mock_get_cache = mock_get_cache
                engine.mock_set_cache = mock_set_cache
                yield engine


@pytest.mark.asyncio
async def test_search_videos_primary_provider(search_engine):
    """Test that the primary provider is used for search."""
    search_engine.mock_youtube_search_videos.return_value = [
        SearchResult(
            video_id="test_video",
            url="http://test.com",
            title="Test Video",
            channel="Test Channel",
            channel_id="test_channel_id",
        )
    ]
    results = await search_engine.search_videos("test query")
    assert len(results) == 1
    assert results[0].video_id == "test_video"
    search_engine.mock_youtube_search_videos.assert_called_once_with("test query", 100)


@pytest.mark.asyncio
async def test_search_videos_fallback(search_engine):
    """Test that fallback providers are used when the primary provider returns few results."""
    search_engine.mock_youtube_search_videos.return_value = []
    search_engine.mock_bing_search_videos.return_value = [
        SearchResult(
            video_id="bing_video",
            url="http://bing.com",
            title="Bing Video",
            channel="Bing Channel",
            channel_id="bing_channel_id",
        )
    ]
    results = await search_engine.search_videos("test query")
    assert len(results) == 1
    assert results[0].video_id == "bing_video"
    search_engine.mock_youtube_search_videos.assert_called_once()
    search_engine.mock_bing_search_videos.assert_called_once()


@pytest.mark.asyncio
async def test_search_videos_caching(search_engine):
    """Test that search results are cached."""
    search_engine.mock_get_cache.return_value = None
    search_engine.mock_youtube_search_videos.return_value = [
        SearchResult(
            video_id="test_video",
            url="http://test.com",
            title="Test Video",
            channel="Test Channel",
            channel_id="test_channel_id",
        )
    ]
    await search_engine.search_videos("test query")
    search_engine.mock_get_cache.assert_called_once()
    search_engine.mock_set_cache.assert_called_once()

    search_engine.mock_get_cache.return_value = [
        SearchResult(
            video_id="cached_video",
            url="http://cached.com",
            title="Cached Video",
            channel="Cached Channel",
            channel_id="cached_channel_id",
        )
    ]
    results = await search_engine.search_videos("test query")
    assert len(results) == 1
    assert results[0].video_id == "cached_video"
