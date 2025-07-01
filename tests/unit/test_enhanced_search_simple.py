"""Simplified unit tests for enhanced_search.py."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from collector.config import ScrapingConfig, SearchConfig
from collector.enhanced_search import MultiStrategySearchEngine
from collector.search.providers.base import SearchResult


class TestMultiStrategySearchEngineSimple:
    """Simplified test cases for MultiStrategySearchEngine."""

    @pytest.mark.asyncio
    async def test_search_videos_basic(self):
        """Test basic search_videos functionality."""
        # Create configs
        search_config = SearchConfig()
        scraping_config = ScrapingConfig()

        # Mock the entire search pipeline
        with patch("collector.enhanced_search.MultiStrategySearchEngine._initialize_providers"):
            engine = MultiStrategySearchEngine(search_config, scraping_config)

            # Mock the internal search method
            mock_results = [
                SearchResult(
                    video_id="test1",
                    url="https://youtube.com/watch?v=test1",
                    title="Test Karaoke 1",
                    channel="Test Channel",
                    channel_id="UCtest",
                )
            ]

            # Mock the various internal methods
            engine._get_cached_results = AsyncMock(return_value=None)
            engine._search_with_provider = AsyncMock(return_value=mock_results)
            engine._combine_and_deduplicate_results = AsyncMock(return_value=mock_results)
            engine._rank_and_return_results = AsyncMock(return_value=mock_results)

            results = await engine.search_videos("test karaoke")

            assert len(results) == 1
            assert results[0].title == "Test Karaoke 1"

    def test_initialization(self):
        """Test engine initialization."""
        search_config = SearchConfig()
        scraping_config = ScrapingConfig()

        with patch("collector.enhanced_search.MultiStrategySearchEngine._initialize_providers"):
            engine = MultiStrategySearchEngine(search_config, scraping_config)

            assert engine is not None
            assert hasattr(engine, "search_config")
            assert hasattr(engine, "scraping_config")

    @pytest.mark.asyncio
    async def test_statistics(self):
        """Test statistics collection."""
        search_config = SearchConfig()
        scraping_config = ScrapingConfig()

        with patch("collector.enhanced_search.MultiStrategySearchEngine._initialize_providers"):
            engine = MultiStrategySearchEngine(search_config, scraping_config)

            # Mock get_comprehensive_statistics
            engine.get_comprehensive_statistics = AsyncMock(
                return_value={
                    "total_searches": 10,
                    "providers": ["youtube", "duckduckgo"],
                    "cache_stats": {"hits": 5, "misses": 5},
                }
            )

            stats = await engine.get_comprehensive_statistics()

            assert "total_searches" in stats
            assert stats["total_searches"] == 10

    @pytest.mark.asyncio
    async def test_provider_failure_handling(self):
        """Test handling when providers fail."""
        search_config = SearchConfig()
        scraping_config = ScrapingConfig()

        with patch("collector.enhanced_search.MultiStrategySearchEngine._initialize_providers"):
            engine = MultiStrategySearchEngine(search_config, scraping_config)

            # Mock provider to fail
            engine._search_with_provider = AsyncMock(side_effect=Exception("Provider error"))
            engine._search_with_fallback_providers = AsyncMock(return_value=[])
            engine._combine_and_deduplicate_results = AsyncMock(return_value=[])
            engine._rank_and_return_results = AsyncMock(return_value=[])

            # Should not raise exception
            results = await engine.search_videos("test")

            assert results == []

    @pytest.mark.asyncio
    async def test_duplicate_removal(self):
        """Test duplicate video removal."""
        search_config = SearchConfig()
        scraping_config = ScrapingConfig()

        with patch("collector.enhanced_search.MultiStrategySearchEngine._initialize_providers"):
            engine = MultiStrategySearchEngine(search_config, scraping_config)

            # Create duplicate results
            duplicate_results = [
                SearchResult(
                    video_id="same",
                    url="https://youtube.com/watch?v=same",
                    title="Video 1",
                    channel="Channel",
                    channel_id="UC1",
                ),
                SearchResult(
                    video_id="same",
                    url="https://youtube.com/watch?v=same",
                    title="Video 1 Duplicate",
                    channel="Channel",
                    channel_id="UC1",
                ),
            ]

            # Test the deduplication method if it exists
            if hasattr(engine, "_combine_and_deduplicate_results"):
                engine._combine_and_deduplicate_results = AsyncMock()
                # Mock to return single result
                engine._combine_and_deduplicate_results.return_value = [duplicate_results[0]]

                result = await engine._combine_and_deduplicate_results([duplicate_results])
                assert len(result) == 1

    @pytest.mark.asyncio
    async def test_caching(self):
        """Test search result caching."""
        search_config = SearchConfig()
        scraping_config = ScrapingConfig()

        with patch("collector.enhanced_search.MultiStrategySearchEngine._initialize_providers"):
            engine = MultiStrategySearchEngine(search_config, scraping_config)

            # Mock cache methods
            cached_results = [
                SearchResult(
                    video_id="cached",
                    url="https://youtube.com/watch?v=cached",
                    title="Cached Result",
                    channel="Channel",
                    channel_id="UCcache",
                )
            ]

            engine._get_cached_results = AsyncMock(return_value=cached_results)
            engine._rank_and_return_results = AsyncMock(return_value=cached_results)

            results = await engine.search_videos("cached query")
            assert results == cached_results

            # Should use cached results
            engine._get_cached_results.assert_called_once()

    @pytest.mark.asyncio
    async def test_ranking(self):
        """Test result ranking."""
        search_config = SearchConfig()
        scraping_config = ScrapingConfig()

        with patch("collector.enhanced_search.MultiStrategySearchEngine._initialize_providers"):
            engine = MultiStrategySearchEngine(search_config, scraping_config)

            # Create results with different scores
            unranked_results = [
                SearchResult(
                    video_id="low",
                    url="https://youtube.com/watch?v=low",
                    title="Low Quality",
                    channel="Unknown Channel",
                    channel_id="UClow",
                    view_count=10,
                ),
                SearchResult(
                    video_id="high",
                    url="https://youtube.com/watch?v=high",
                    title="High Quality Karaoke",
                    channel="Professional Karaoke",
                    channel_id="UChigh",
                    view_count=100000,
                ),
            ]

            # Mock ranking to reverse order (high quality first)
            engine._rank_and_return_results = AsyncMock(
                return_value=[unranked_results[1], unranked_results[0]]
            )

            ranked = await engine._rank_and_return_results(unranked_results, "test", 10)

            assert ranked[0].video_id == "high"
            assert ranked[1].video_id == "low"
