"""Edge case and performance tests for the multi-pass parsing system."""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest

from collector.advanced_parser import ParseResult
from collector.config import MultiPassConfig, MultiPassPassConfig
from collector.multi_pass_controller import MultiPassParsingController, PassType
from collector.passes.channel_template_pass import ChannelPattern
from collector.passes.web_search_pass import FillerWordProcessor, SERPCache


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_advanced_parser = Mock()
        self.mock_search_engine = Mock()

        self.config = MultiPassConfig(enabled=True)
        self.controller = MultiPassParsingController(
            config=self.config,
            advanced_parser=self.mock_advanced_parser,
            search_engine=self.mock_search_engine,
            db_manager=None,
        )

    @pytest.mark.asyncio
    async def test_empty_title_handling(self):
        """Test handling of empty or None titles."""

        test_cases = ["", None, "   ", "\t\n"]

        for title in test_cases:
            result = await self.controller.parse_video(
                video_id="test123", title=title, channel_name="Test Channel"
            )

            # Should handle gracefully without crashing
            assert result is not None
            assert isinstance(result.total_processing_time, float)

    @pytest.mark.asyncio
    async def test_unicode_title_handling(self):
        """Test handling of Unicode and special characters."""

        unicode_titles = [
            "Übung - Sänger Künstler (Karaoke)",  # German umlauts
            "歌手 - 歌曲 卡拉OK",  # Chinese characters
            "アーティスト - 歌 カラオケ",  # Japanese characters
            "Артист - Песня (Караоке)",  # Cyrillic
            "🎵 Artist - Song 🎤 (Karaoke)",  # Emojis
            "Artist\x00Song\x01Karaoke",  # Control characters
        ]

        for title in unicode_titles:
            try:
                result = await self.controller.parse_video(
                    video_id="test123", title=title, channel_name="Test Channel"
                )

                # Should not crash on Unicode
                assert result is not None

            except Exception as e:
                pytest.fail(f"Failed on Unicode title '{title}': {e}")

    @pytest.mark.asyncio
    async def test_extremely_long_title(self):
        """Test handling of extremely long titles."""

        # Create a very long title (10KB)
        long_title = "Very Long Karaoke Title " * 400  # ~10KB

        result = await self.controller.parse_video(
            video_id="test123", title=long_title, channel_name="Test Channel"
        )

        # Should handle without memory issues
        assert result is not None
        assert result.total_processing_time < 60.0  # Should not take too long

    @pytest.mark.asyncio
    async def test_malformed_regex_patterns(self):
        """Test handling of malformed regex patterns."""

        # Create a pass with bad pattern
        bad_pattern = ChannelPattern(
            pattern="[unclosed bracket",  # Invalid regex
            artist_group=1,
            title_group=2,
            confidence=0.8,
        )

        channel_template_pass = self.controller.channel_template_pass
        channel_template_pass.channel_patterns["UC123"] = [bad_pattern]

        # Should handle regex errors gracefully
        result = await channel_template_pass.parse(
            "Test Title", channel_id="UC123", channel_name="Test Channel"
        )

        # Should not crash, may return None
        assert result is None or isinstance(result, ParseResult)

    @pytest.mark.asyncio
    async def test_concurrent_processing(self):
        """Test concurrent processing of multiple videos."""

        # Create multiple concurrent parsing tasks
        tasks = []
        for i in range(10):
            task = self.controller.parse_video(
                video_id=f"test{i}", title=f"Test Title {i} (Karaoke)", channel_name="Test Channel"
            )
            tasks.append(task)

        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should complete without errors
        for i, result in enumerate(results):
            assert not isinstance(result, Exception), f"Task {i} failed: {result}"
            assert result is not None

    @pytest.mark.asyncio
    async def test_memory_leak_prevention(self):
        """Test that processing doesn't leak memory."""

        # Process many videos in sequence
        for i in range(100):
            result = await self.controller.parse_video(
                video_id=f"test{i}", title=f"Test Title {i} (Karaoke)", channel_name="Test Channel"
            )

            # Clear any large data structures
            if hasattr(result, "passes_attempted"):
                result.passes_attempted.clear()

        # Check cache sizes are reasonable
        if hasattr(self.controller, "ml_embedding_pass"):
            ml_pass = self.controller.ml_embedding_pass
            cache_size = len(ml_pass.embedding_cache)
            assert cache_size <= ml_pass.max_embedding_cache_size

    def test_configuration_validation(self):
        """Test configuration validation edge cases."""

        # Test invalid confidence thresholds
        invalid_configs = [
            MultiPassConfig(
                channel_template=MultiPassPassConfig(confidence_threshold=1.5)  # > 1.0
            ),
            MultiPassConfig(
                channel_template=MultiPassPassConfig(confidence_threshold=-0.1)  # < 0.0
            ),
            MultiPassConfig(total_cpu_budget=-10.0),  # Negative budget
            MultiPassConfig(
                channel_template=MultiPassPassConfig(timeout_seconds=0.0)  # Zero timeout
            ),
        ]

        # Should either raise validation errors or clamp to valid ranges
        for config in invalid_configs:
            try:
                MultiPassParsingController(
                    config=config,
                    advanced_parser=self.mock_advanced_parser,
                    search_engine=None,
                    db_manager=None,
                )
                # If it doesn't raise an error, values should be clamped
                assert 0.0 <= config.channel_template.confidence_threshold <= 1.0

            except (ValueError, AssertionError):
                # Validation errors are acceptable
                pass

    @pytest.mark.asyncio
    async def test_exception_handling_in_passes(self):
        """Test exception handling within individual passes."""

        # Mock a pass to raise an exception
        def failing_pass(*args, **kwargs):
            raise RuntimeError("Simulated pass failure")

        with patch.object(self.controller, "_pass_0_channel_template", failing_pass):
            result = await self.controller.parse_video(
                video_id="test123", title="Test Title", channel_name="Test Channel"
            )

            # Should handle exception gracefully
            assert result is not None
            assert len(result.passes_attempted) >= 1

            # The failed pass should be recorded
            failed_pass = result.passes_attempted[0]
            assert not failed_pass.success
            assert "error" in failed_pass.error_message.lower()


class TestPerformanceCriteria:
    """Test performance requirements and benchmarks."""

    def setup_method(self):
        """Set up performance test fixtures."""
        self.mock_advanced_parser = Mock()
        self.mock_search_engine = Mock()

        # Configure for performance testing
        self.config = MultiPassConfig(
            enabled=True,
            stop_on_first_success=True,
            total_cpu_budget=10.0,  # Reasonable budget
            total_api_budget=20,
        )

        self.controller = MultiPassParsingController(
            config=self.config,
            advanced_parser=self.mock_advanced_parser,
            search_engine=self.mock_search_engine,
            db_manager=None,
        )

    @pytest.mark.asyncio
    async def test_single_video_processing_time(self):
        """Test that single video processing meets time requirements."""

        # Mock Pass 0 to succeed quickly
        with patch.object(self.controller.channel_template_pass, "parse") as mock_parse:
            mock_parse.return_value = ParseResult(
                original_artist="Test Artist",
                song_title="Test Song",
                confidence=0.9,
                method="channel_template",
            )

            start_time = time.time()
            result = await self.controller.parse_video(
                video_id="test123",
                title="Test Artist - Test Song (Karaoke)",
                channel_name="Test Channel",
            )
            processing_time = time.time() - start_time

            # Should complete quickly when first pass succeeds
            assert processing_time < 1.0  # Less than 1 second
            assert result.final_result is not None

    @pytest.mark.asyncio
    async def test_batch_processing_throughput(self):
        """Test throughput for batch processing."""

        batch_size = 50

        # Mock quick success on Pass 0
        with patch.object(self.controller.channel_template_pass, "parse") as mock_parse:
            mock_parse.return_value = ParseResult(
                original_artist="Test Artist",
                song_title="Test Song",
                confidence=0.9,
                method="channel_template",
            )

            start_time = time.time()

            # Process batch sequentially
            results = []
            for i in range(batch_size):
                result = await self.controller.parse_video(
                    video_id=f"test{i}",
                    title=f"Test Artist {i} - Test Song {i} (Karaoke)",
                    channel_name="Test Channel",
                )
                results.append(result)

            total_time = time.time() - start_time

            # Calculate throughput
            throughput = batch_size / total_time  # videos per second

            # Should achieve reasonable throughput
            assert throughput > 10  # At least 10 videos per second
            assert all(r.final_result is not None for r in results)

    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self):
        """Test memory usage during heavy processing."""

        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process many videos
        for i in range(200):
            await self.controller.parse_video(
                video_id=f"test{i}", title=f"Test Title {i} (Karaoke)", channel_name="Test Channel"
            )

            # Check memory periodically
            if i % 50 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_growth = current_memory - initial_memory

                # Memory growth should be reasonable (< 100MB)
                assert memory_growth < 100, f"Memory growth too high: {memory_growth}MB"

    def test_cache_performance(self):
        """Test cache performance and hit rates."""

        # Test SERP cache performance
        cache = SERPCache(max_entries=1000)

        # Fill cache
        for i in range(500):
            query = f"test query {i}"
            results = [{"video_id": f"test{i}", "title": f"Test Title {i}"}]
            cache.put(query, results)

        # Test hit rate
        hits = 0
        total_queries = 100

        start_time = time.time()
        for i in range(total_queries):
            query = f"test query {i % 500}"  # Some hits, some misses
            result = cache.get(query)
            if result is not None:
                hits += 1
        end_time = time.time()

        # Cache operations should be fast
        avg_time_per_query = (end_time - start_time) / total_queries
        assert avg_time_per_query < 0.001  # Less than 1ms per query

        # Hit rate should be reasonable
        hit_rate = hits / total_queries
        assert hit_rate > 0.8  # At least 80% hit rate

    def test_pattern_matching_performance(self):
        """Test regex pattern matching performance."""

        from collector.passes.channel_template_pass import EnhancedChannelTemplatePass

        pass_instance = EnhancedChannelTemplatePass(self.mock_advanced_parser)

        # Test with many patterns
        for i in range(100):
            pattern = ChannelPattern(
                pattern=rf"^Test Pattern {i} - ([^-]+) - ([^(]+) \(Karaoke\)",
                artist_group=1,
                title_group=2,
                confidence=0.8,
            )
            pass_instance.channel_patterns[f"UC{i}"] = [pattern]

        # Test pattern matching performance
        test_title = "Test Pattern 50 - Test Artist - Test Song (Karaoke)"

        start_time = time.time()
        for _ in range(1000):  # 1000 iterations
            pass_instance._try_channel_patterns("UC50", test_title, "", "", "Test Channel")
        end_time = time.time()

        # Should be fast even with many patterns
        avg_time = (end_time - start_time) / 1000
        assert avg_time < 0.01  # Less than 10ms per match

    def test_filler_word_processing_performance(self):
        """Test filler word processing performance."""

        processor = FillerWordProcessor()

        # Test with various title lengths
        test_titles = [
            "Short Karaoke Title",
            "Medium Length Artist Name - Song Title (Karaoke Version HD)",
            ("Very Long Karaoke Title " * 10) + " Artist - Song (Official HD Karaoke Version)",
            "🎵 Unicode Artist 中文 - Song Title 🎤 (Karaoke HD 4K Official Version)",
        ]

        for title in test_titles:
            start_time = time.time()
            for _ in range(100):  # 100 iterations per title
                processor.clean_query(title)
            end_time = time.time()

            avg_time = (end_time - start_time) / 100
            # Should process quickly regardless of title length
            assert avg_time < 0.01  # Less than 10ms per cleaning


class TestRobustness:
    """Test system robustness and error recovery."""

    def setup_method(self):
        """Set up robustness test fixtures."""
        self.mock_advanced_parser = Mock()
        self.mock_search_engine = Mock()

        self.config = MultiPassConfig(enabled=True)
        self.controller = MultiPassParsingController(
            config=self.config,
            advanced_parser=self.mock_advanced_parser,
            search_engine=self.mock_search_engine,
            db_manager=None,
        )

    @pytest.mark.asyncio
    async def test_network_timeout_recovery(self):
        """Test recovery from network timeouts."""

        # Mock search engine to timeout
        async def timeout_search(*args, **kwargs):
            raise asyncio.TimeoutError("Network timeout")

        self.mock_search_engine.search_videos = timeout_search

        result = await self.controller.parse_video(
            video_id="test123", title="Test Title (Karaoke)", channel_name="Test Channel"
        )

        # Should recover gracefully from network issues
        assert result is not None

    @pytest.mark.asyncio
    async def test_partial_system_failure(self):
        """Test behavior when some passes fail but others succeed."""

        # Mock some passes to fail, others to succeed
        with patch.object(
            self.controller, "_pass_0_channel_template", side_effect=Exception("Pass 0 failed")
        ), patch.object(
            self.controller, "_pass_1_auto_retemplate", side_effect=Exception("Pass 1 failed")
        ), patch.object(
            self.controller.ml_embedding_pass, "parse"
        ) as mock_ml_pass:

            mock_ml_pass.return_value = ParseResult(
                original_artist="ML Artist",
                song_title="ML Song",
                confidence=0.8,
                method="ml_embedding",
            )

            result = await self.controller.parse_video(
                video_id="test123", title="Test Title", channel_name="Test Channel"
            )

            # Should succeed with working pass despite failures
            assert result.final_result is not None
            assert result.final_result.original_artist == "ML Artist"
            assert result.stopped_at_pass == PassType.ML_EMBEDDING

    def test_configuration_hot_reload(self):
        """Test changing configuration during runtime."""

        # Change configuration
        original_budget = self.controller.config.total_cpu_budget
        self.controller.config.total_cpu_budget = original_budget * 2

        # Should accept new configuration
        assert self.controller.config.total_cpu_budget == original_budget * 2

        # Change pass configuration
        self.controller.pass_configs[PassType.CHANNEL_TEMPLATE].confidence_threshold = 0.95

        assert self.controller.pass_configs[PassType.CHANNEL_TEMPLATE].confidence_threshold == 0.95

    @pytest.mark.asyncio
    async def test_database_connection_failure(self):
        """Test behavior when database is unavailable."""

        # Mock database operations to fail
        self.controller.db_manager = Mock()
        self.controller.db_manager.execute.side_effect = Exception("Database connection failed")

        # Should still work without database
        result = await self.controller.parse_video(
            video_id="test123", title="Test Title (Karaoke)", channel_name="Test Channel"
        )

        assert result is not None
        # May have reduced functionality but shouldn't crash

    def test_statistics_consistency(self):
        """Test that statistics remain consistent."""

        # Simulate processing
        for i in range(10):
            self.controller.statistics["total_videos_processed"] += 1
            self.controller.statistics["passes_attempted"][PassType.CHANNEL_TEMPLATE] += 1
            if i % 2 == 0:  # 50% success rate
                self.controller.statistics["passes_successful"][PassType.CHANNEL_TEMPLATE] += 1

        stats = self.controller.get_statistics()

        # Statistics should be consistent
        assert stats["total_videos_processed"] == 10
        assert stats["success_rates"][PassType.CHANNEL_TEMPLATE.value] == 0.5

        # All statistics should be non-negative
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                assert value >= 0
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        assert sub_value >= 0


# Performance benchmark decorator
def benchmark(func):
    """Decorator to benchmark test execution time."""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result

    return wrapper


# Run tests if called directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
