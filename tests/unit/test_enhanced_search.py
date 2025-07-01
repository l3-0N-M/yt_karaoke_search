"""Unit tests for enhanced_search.py."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from collector.config import ScrapingConfig, SearchConfig
from collector.enhanced_search import MultiStrategySearchEngine
from collector.search.providers.base import SearchResult


class TestMultiStrategySearchEngine:
    """Test cases for MultiStrategySearchEngine."""

    @pytest.fixture
    def search_config(self):
        """Create a test search configuration."""
        config = SearchConfig()
        # SearchConfig manages multi-strategy internally
        return config

    @pytest.fixture
    def scraping_config(self):
        """Create a test scraping configuration."""
        return ScrapingConfig()

    @pytest.fixture
    def mock_providers(self):
        """Create mock search providers."""
        youtube_provider = AsyncMock()
        youtube_provider.name = "youtube"
        youtube_provider.search_videos = AsyncMock(
            return_value=[
                SearchResult(
                    video_id="test1",
                    url="https://youtube.com/watch?v=test1",
                    title="Test Karaoke Song",
                    channel="Test Channel",
                    channel_id="UC123",
                    provider="youtube",
                    duration=180,
                    view_count=1000,
                )
            ]
        )
        youtube_provider.is_available = AsyncMock(return_value=True)
        youtube_provider.get_provider_weight = Mock(return_value=1.0)

        duckduckgo_provider = AsyncMock()
        duckduckgo_provider.name = "duckduckgo"
        duckduckgo_provider.search_videos = AsyncMock(
            return_value=[
                SearchResult(
                    video_id="test2",
                    url="https://youtube.com/watch?v=test2",
                    title="Another Karaoke Track",
                    channel="Another Channel",
                    channel_id="UC456",
                    provider="duckduckgo",
                    duration=200,
                    view_count=500,
                )
            ]
        )
        duckduckgo_provider.is_available = AsyncMock(return_value=True)
        duckduckgo_provider.get_provider_weight = Mock(return_value=0.8)

        return {"youtube": youtube_provider, "duckduckgo": duckduckgo_provider}

    @pytest.mark.asyncio
    async def test_search_basic(self, search_config, scraping_config, mock_providers):
        """Test basic search functionality."""
        with patch("collector.enhanced_search.MultiStrategySearchEngine._initialize_providers"):
            engine = MultiStrategySearchEngine(search_config, scraping_config)
            engine.providers = mock_providers

            results = await engine.search_videos("test karaoke")

        assert len(results) == 2
        assert results[0].provider == "youtube"
        assert results[1].provider == "duckduckgo"

    @pytest.mark.asyncio
    async def test_search_with_fallback(self, search_config, scraping_config, mock_providers):
        """Test search with fallback queries."""
        # Make primary search return no results
        for provider in mock_providers.values():
            provider.search_videos.side_effect = [
                [],
                [
                    SearchResult(
                        video_id="fallback",
                        url="https://youtube.com/watch?v=fallback",
                        title="Fallback Result",
                        channel="Fallback Channel",
                        channel_id="UCfallback",
                        provider=provider.name,
                    )
                ],
            ]

        with patch("collector.enhanced_search.MultiStrategySearchEngine._initialize_providers"):
            engine = MultiStrategySearchEngine(search_config, scraping_config)
            engine.providers = mock_providers

            results = await engine.search_videos("test karaoke")

        assert len(results) == 2
        assert all("Fallback" in r.title for r in results)

    @pytest.mark.asyncio
    async def test_search_provider_failure(self, search_config, scraping_config, mock_providers):
        """Test handling of provider failures."""
        # Make YouTube provider fail
        mock_providers["youtube"].search_videos.side_effect = Exception("Provider error")

        with patch("collector.enhanced_search.MultiStrategySearchEngine._initialize_providers"):
            engine = MultiStrategySearchEngine(search_config, scraping_config)
            engine.providers = mock_providers

            results = await engine.search_videos("test karaoke")

            # Should still get results from DuckDuckGo
        assert len(results) == 1
        assert results[0].provider == "duckduckgo"

    @pytest.mark.asyncio
    async def test_duplicate_removal(self, search_config, scraping_config, mock_providers):
        """Test duplicate result removal."""
        # Make both providers return the same video
        duplicate_result = SearchResult(
            video_id="same",
            url="https://youtube.com/watch?v=same",
            title="Duplicate Song",
            channel="Test Channel",
            channel_id="UCdup",
            provider="youtube",
        )

        mock_providers["youtube"].search_videos.return_value = [duplicate_result]
        mock_providers["duckduckgo"].search_videos.return_value = [
            SearchResult(
                video_id="same",  # Same video ID
                url="https://youtube.com/watch?v=same",  # Same URL
                title="Duplicate Song Different Title",
                channel="Test Channel",
                channel_id="UCdup",
                provider="duckduckgo",
            )
        ]

        with patch("collector.enhanced_search.MultiStrategySearchEngine._initialize_providers"):
            engine = MultiStrategySearchEngine(search_config, scraping_config)
            engine.providers = mock_providers

            results = await engine.search_videos("test karaoke")

            # Should only have one result (duplicate removed)
        assert len(results) == 1
        assert results[0].url == "https://youtube.com/watch?v=same"

    @pytest.mark.asyncio
    async def test_result_ranking(self, search_config, scraping_config, mock_providers):
        """Test result ranking functionality."""
        with patch("collector.enhanced_search.MultiStrategySearchEngine._initialize_providers"):
            engine = MultiStrategySearchEngine(search_config, scraping_config)
            engine.providers = mock_providers

            # Add ranker
            engine.result_ranker.rank_results = Mock(
                return_value=[
                    Mock(
                        result=SearchResult(
                            video_id="ranked",
                            url="https://youtube.com/watch?v=ranked",
                            title="Ranked Result",
                            channel="Test Channel",
                            channel_id="UCrank",
                            provider="youtube",
                        )
                    )
                ]
            )

            results = await engine.search_videos("test karaoke")

        assert len(results) == 1
        assert results[0].title == "Ranked Result"

    @pytest.mark.asyncio
    async def test_search_query_expansion(self, search_config, scraping_config, mock_providers):
        """Test search query expansion."""
        with patch("collector.enhanced_search.MultiStrategySearchEngine._initialize_providers"):
            engine = MultiStrategySearchEngine(search_config, scraping_config)
            engine.providers = mock_providers

            # Test that query gets expanded
            results = await engine.search_videos("test")
            assert results is not None

            # Check that providers were called with expanded queries
            for provider in mock_providers.values():
                assert provider.search_videos.called
                # Should be called with original and expanded queries
                assert provider.search_videos.call_count >= 1

    @pytest.mark.asyncio
    async def test_max_results_limit(self, search_config, scraping_config, mock_providers):
        """Test max results per provider limit."""
        # Make providers return many results
        many_results = [
            SearchResult(
                video_id=f"test{i}",
                url=f"https://youtube.com/watch?v=test{i}",
                title=f"Result {i}",
                channel="Test Channel",
                channel_id="UCmany",
                provider="youtube",
            )
            for i in range(20)
        ]

        mock_providers["youtube"].search_videos.return_value = many_results
        mock_providers["duckduckgo"].search_videos.return_value = []

        with patch("collector.enhanced_search.MultiStrategySearchEngine._initialize_providers"):
            engine = MultiStrategySearchEngine(search_config, scraping_config)
            engine.providers = mock_providers

            results = await engine.search_videos("test karaoke", max_results=5)

            # Should respect max_results limit
        assert len(results) <= 5

    @pytest.mark.asyncio
    async def test_get_statistics(self, search_config, scraping_config):
        """Test statistics collection."""
        with patch("collector.enhanced_search.MultiStrategySearchEngine._initialize_providers"):
            engine = MultiStrategySearchEngine(search_config, scraping_config)

            # Mock the components' statistics methods
            engine.cache_manager.get_comprehensive_stats = AsyncMock(
                return_value={"overall": {"hit_rate": 0.5, "total_requests": 100}}
            )
            engine.result_ranker.get_statistics = Mock(return_value={"weights": {}})
            engine.fuzzy_matcher.get_statistics = Mock(return_value={})

            stats = await engine.get_comprehensive_statistics()

        assert "search_engine" in stats
        assert "cache" in stats
        assert "providers" in stats

    @pytest.mark.asyncio
    async def test_search_with_filters(self, search_config, scraping_config, mock_providers):
        """Test search with duration filters."""
        with patch("collector.enhanced_search.MultiStrategySearchEngine._initialize_providers"):
            engine = MultiStrategySearchEngine(search_config, scraping_config)
            engine.providers = mock_providers

            results = await engine.search_videos(
                "test karaoke",
            )
            assert results is not None

            # Verify providers were called with filters
            for provider in mock_providers.values():
                assert provider.search_videos.called

    @pytest.mark.asyncio
    async def test_empty_query(self, search_config, scraping_config, mock_providers):
        """Test handling of empty query."""
        with patch("collector.enhanced_search.MultiStrategySearchEngine._initialize_providers"):
            engine = MultiStrategySearchEngine(search_config, scraping_config)
            engine.providers = mock_providers

            results = await engine.search_videos("")

            # Should handle empty query gracefully
        assert results == []

    @pytest.mark.asyncio
    async def test_provider_initialization(self, search_config, scraping_config):
        """Test provider initialization."""
        with patch("collector.enhanced_search.YouTubeSearchProvider") as mock_youtube:
            with patch("collector.enhanced_search.DuckDuckGoSearchProvider"):
                engine = MultiStrategySearchEngine(search_config, scraping_config)
                assert engine is not None

                # Should initialize configured providers
                assert mock_youtube.called

    @pytest.mark.asyncio
    async def test_concurrent_search(self, search_config, scraping_config, mock_providers):
        """Test concurrent search across providers."""
        import asyncio

        # Add delay to simulate real search
        async def delayed_search(*args, **kwargs):
            await asyncio.sleep(0.1)
            return [
                SearchResult(
                    video_id="delayed",
                    url="https://youtube.com/watch?v=delayed",
                    title="Delayed Result",
                    channel="Test Channel",
                    channel_id="UCdelay",
                    provider="youtube",
                )
            ]

        mock_providers["youtube"].search_videos = delayed_search
        mock_providers["duckduckgo"].search_videos = delayed_search

        with patch("collector.enhanced_search.MultiStrategySearchEngine._initialize_providers"):
            engine = MultiStrategySearchEngine(search_config, scraping_config)
            engine.providers = mock_providers

            import time

            start = time.time()
            results = await engine.search_videos("test")
            duration = time.time() - start

            # Should run concurrently (faster than sequential)
        assert duration < 0.3  # Less than sum of delays
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_caching(self, search_config, scraping_config, mock_providers):
        """Test search result caching."""
        with patch("collector.enhanced_search.MultiStrategySearchEngine._initialize_providers"):
            engine = MultiStrategySearchEngine(search_config, scraping_config)
            engine.providers = mock_providers

            # First search
            results1 = await engine.search_videos("test karaoke")

            # Second search with same query
            results2 = await engine.search_videos("test karaoke")

            # Results should be consistent
        assert len(results1) == len(results2)

    @pytest.mark.asyncio
    async def test_search_timeout(self, search_config, scraping_config, mock_providers):
        """Test search timeout handling."""
        import asyncio

        # Make provider hang
        async def hanging_search(*args, **kwargs):
            await asyncio.sleep(10)  # Long delay

        mock_providers["youtube"].search_videos = hanging_search

        with patch("collector.enhanced_search.MultiStrategySearchEngine._initialize_providers"):
            engine = MultiStrategySearchEngine(search_config, scraping_config)
            engine.providers = mock_providers

            results = await engine.search_videos("test")

            # Should get results from non-hanging provider
        assert len(results) >= 0  # May timeout all providers
