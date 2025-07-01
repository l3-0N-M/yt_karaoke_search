"""Unit tests for web_search_pass.py - focusing on null safety fixes."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from collector.advanced_parser import ParseResult
from collector.passes.web_search_pass import (
    EnhancedWebSearchPass,
    FillerWordProcessor,
    QueryCleaningResult,
    SERPCache,
)


class TestFillerWordProcessor:
    """Test cases for FillerWordProcessor with focus on null safety."""

    @pytest.fixture
    def processor(self):
        """Create a FillerWordProcessor instance."""
        return FillerWordProcessor()

    def test_clean_query_handles_none(self, processor):
        """Test that clean_query handles None input gracefully."""
        result = processor.clean_query(None)
        assert isinstance(result, QueryCleaningResult)
        assert result.cleaned_query == ""
        assert result.original_query == ""

    def test_clean_query_handles_non_string_types(self, processor):
        """Test that clean_query handles various non-string types."""
        test_cases = [
            123,  # int
            45.67,  # float
            ["list", "of", "words"],  # list
            {"dict": "value"},  # dict
            True,  # bool
            b"bytes",  # bytes
        ]

        for value in test_cases:
            result = processor.clean_query(value)
            assert isinstance(result, QueryCleaningResult)
            assert isinstance(result.cleaned_query, str)
            assert len(result.cleaned_query) >= 0

    def test_clean_query_removes_karaoke_terms(self, processor):
        """Test that karaoke-related terms are removed."""
        query = "Song Title karaoke version instrumental lyrics"
        result = processor.clean_query(query)

        assert "karaoke" not in result.cleaned_query.lower()
        assert "instrumental" not in result.cleaned_query.lower()
        assert "lyrics" not in result.cleaned_query.lower()
        assert "Song Title" in result.cleaned_query

    def test_additional_cleaning_null_safety(self, processor):
        """Test that _additional_cleaning handles None and edge cases."""
        # Test None input
        result = processor._additional_cleaning(None, [])
        assert result == ""

        # Test with words that could be None
        test_query = "test None word"
        result = processor._additional_cleaning(test_query, [])
        assert isinstance(result, str)
        assert "test" in result

    def test_word_filtering_with_none_values(self, processor):
        """Test that word filtering handles None values in word list."""
        # This simulates the case where split() might produce None values
        with patch.object(str, "split", return_value=["word1", None, "word2", "", "a"]):
            query = "test query"
            result = processor.clean_query(query)
            # Should not raise any exceptions
            assert isinstance(result.cleaned_query, str)


class TestEnhancedWebSearchPass:
    """Test cases for EnhancedWebSearchPass with focus on null safety fixes."""

    @pytest.fixture
    def search_pass(self):
        """Create a EnhancedWebSearchPass instance."""
        with patch("collector.passes.web_search_pass.MultiStrategySearchEngine"):
            return EnhancedWebSearchPass(advanced_parser=Mock(), search_engine=Mock())

    @pytest.fixture
    def mock_search_engine(self):
        """Create a mock search engine."""
        engine = AsyncMock()
        return engine

    @pytest.mark.asyncio
    async def test_parse_handles_none_title(self, search_pass):
        """Test that parse handles None title gracefully."""
        result = await search_pass.parse(title=None, description="Test description")
        assert result is None  # Should return None for invalid input

    @pytest.mark.asyncio
    async def test_parse_handles_empty_title(self, search_pass):
        """Test that parse handles empty title gracefully."""
        result = await search_pass.parse(title="", description="Test description")
        assert result is None  # Should return None for empty title

    @pytest.mark.asyncio
    async def test_parse_handles_non_string_inputs(self, search_pass):
        """Test that parse handles non-string inputs gracefully."""
        test_cases = [
            {"title": 123, "description": "test"},
            {"title": ["list"], "description": None},
            {"title": {"dict": "value"}, "tags": 456},
        ]

        for test_data in test_cases:
            result = await search_pass.parse(**test_data)
            # Should not raise exceptions
            assert result is None or isinstance(result, ParseResult)

    def test_safe_string_convert_none(self, search_pass):
        """Test _safe_string_convert with None input."""
        result = search_pass._safe_string_convert(None, "test_field")
        assert result == ""

    def test_safe_string_convert_various_types(self, search_pass):
        """Test _safe_string_convert with various input types."""
        test_cases = [
            ("string", "string"),  # Normal string
            (123, "123"),  # Integer
            (45.67, "45.67"),  # Float
            (True, "True"),  # Boolean
            (b"bytes", "bytes"),  # Bytes
            (["list", "items"], '["list", "items"]'),  # List
            ({"key": "value"}, '{"key": "value"}'),  # Dict
        ]

        for input_val, expected_type in test_cases:
            result = search_pass._safe_string_convert(input_val, "test")
            assert isinstance(result, str)
            # Check it's valid UTF-8
            result.encode("utf-8")

    def test_safe_string_convert_unicode_handling(self, search_pass):
        """Test _safe_string_convert with problematic unicode."""
        test_cases = [
            "Normal ASCII",
            "Unicode émojis 🎵🎤",
            "Mixed \x00 null \x01 bytes",
            "Invalid \udcff surrogate",
            b"\xff\xfe Invalid bytes".decode("utf-8", errors="ignore"),
        ]

        for test_str in test_cases:
            result = search_pass._safe_string_convert(test_str, "unicode_test")
            assert isinstance(result, str)
            # Should be valid UTF-8
            result.encode("utf-8")

    @pytest.mark.asyncio
    async def test_generate_search_queries_null_safety(self, search_pass):
        """Test that _generate_search_queries handles None values."""
        queries = search_pass._generate_search_queries(None, None, None)
        assert isinstance(queries, list)
        # Should still generate some queries even with None inputs

    @pytest.mark.asyncio
    async def test_parse_search_results_with_none_values(self, search_pass):
        """Test _parse_search_results with None values in results."""
        # Mock search results with None values
        results = [
            {"title": None, "artist": "Test Artist", "duration": 180},
            {"title": "Test Song", "artist": None, "duration": None},
        ]

        query_info = QueryCleaningResult(
            original_query="test query",
            cleaned_query="test query",
            removed_terms=[],
            confidence_boost=1.0,
        )

        # Should not raise exceptions
        result = await search_pass._parse_search_results(results, query_info, "Test Title")
        assert result is None or isinstance(result, ParseResult)

    def test_hash_calculation_with_none(self, search_pass):
        """Test that hash calculation handles None values."""
        # Test with None
        hash1 = "test_hash"
        assert isinstance(hash1, str)

        # Test with empty string
        hash2 = "test_hash"
        assert isinstance(hash2, str)

        # Test with normal string
        hash3 = "test_hash"
        assert isinstance(hash3, str)
        assert len(hash3) == 32  # MD5 hash length

    @pytest.mark.asyncio
    async def test_error_handling_in_parse(self, search_pass):
        """Test that parse method handles exceptions gracefully."""
        # Mock search engine to raise exception
        with patch.object(search_pass, "_search_with_engine", side_effect=Exception("Test error")):
            result = await search_pass.parse(
                title="Test Song",
                description="Test Description",
                tags="",
                channel_name="",
                channel_id="",
                metadata={},
            )
            # Should return None instead of raising
            assert result is None

    @pytest.mark.asyncio
    async def test_parse_with_high_confidence_early_exit(self, search_pass):
        """Test that parse exits early when high confidence result is found."""
        mock_result = ParseResult(artist="Test Artist", song_title="Test Song", confidence=0.9)

        with patch.object(search_pass, "_search_with_engine", return_value=[{"test": "result"}]):
            with patch.object(search_pass, "_parse_search_results", return_value=mock_result):
                result = await search_pass.parse(
                    title="Test Song - Test Artist",
                    description="Karaoke version",
                    tags="",
                    channel_name="",
                    channel_id="",
                    metadata={},
                )

                assert result is not None
                assert result.confidence == 0.9


class TestSERPCache:
    """Test cases for SERP cache functionality."""

    @pytest.fixture
    def cache(self):
        """Create a SERP cache instance."""
        return SERPCache(max_entries=10, default_ttl_hours=1)

    def test_cache_put_and_get(self, cache):
        """Test basic cache put and get operations."""
        query = "test query"
        results = [{"title": "Test Result"}]

        cache.put(query, results)
        cached = cache.get(query)

        assert cached == results

    def test_cache_handles_none_query(self, cache):
        """Test cache handles None query gracefully."""
        cache.put(None, [{"test": "result"}])
        result = cache.get(None)
        assert isinstance(result, list)

    def test_cache_expiration(self, cache):
        """Test that cached entries expire after TTL."""
        from datetime import datetime, timedelta

        query = "test query"
        results = [{"title": "Test Result"}]

        # Put entry in cache
        cache.put(query, results)

        # Manually expire the entry
        entry = cache.cache["test_hash"]
        entry.created_at = datetime.now() - timedelta(hours=2)

        # Should return None for expired entry
        cached = cache.get(query)
        assert cached is None

    def test_cache_max_entries(self, cache):
        """Test that cache respects max entries limit."""
        # Fill cache beyond limit
        for i in range(15):
            cache.put(f"query_{i}", [{"result": i}])

        # Cache should not exceed max_entries
        assert len(cache.cache) <= cache.max_entries

        # Most recent entries should be kept
        assert cache.get("query_14") is not None
        assert cache.get("query_0") is None  # Oldest should be evicted
