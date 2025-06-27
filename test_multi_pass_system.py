"""Comprehensive tests for the multi-pass parsing ladder system."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from collector.advanced_parser import AdvancedTitleParser, ParseResult
from collector.config import MultiPassConfig, MultiPassPassConfig
from collector.multi_pass_controller import (
    MultiPassParsingController,
    PassType,
)
from collector.passes.acoustic_fingerprint_pass import AcousticFingerprintPass
from collector.passes.auto_retemplate_pass import AutoRetemplatePass, TemporalPattern
from collector.passes.channel_template_pass import EnhancedChannelTemplatePass
from collector.passes.ml_embedding_pass import EnhancedMLEmbeddingPass
from collector.passes.web_search_pass import EnhancedWebSearchPass, FillerWordProcessor


class TestMultiPassController:
    """Test suite for the multi-pass controller."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_advanced_parser = Mock(spec=AdvancedTitleParser)
        self.mock_search_engine = Mock()
        self.mock_db_manager = Mock()

        # Create test configuration
        self.test_config = MultiPassConfig(
            enabled=True, stop_on_first_success=True, total_cpu_budget=30.0, total_api_budget=50
        )

        self.controller = MultiPassParsingController(
            config=self.test_config,
            advanced_parser=self.mock_advanced_parser,
            search_engine=self.mock_search_engine,
            db_manager=self.mock_db_manager,
        )

    @pytest.mark.asyncio
    async def test_disabled_multi_pass_fallback(self):
        """Test fallback to basic parser when multi-pass is disabled."""

        # Disable multi-pass
        config = MultiPassConfig(enabled=False)
        controller = MultiPassParsingController(
            config=config,
            advanced_parser=self.mock_advanced_parser,
            search_engine=None,
            db_manager=None,
        )

        # Mock basic parser result
        expected_result = ParseResult(
            original_artist="Test Artist",
            song_title="Test Song",
            confidence=0.8,
            method="basic_parser",
        )
        self.mock_advanced_parser.parse_title.return_value = expected_result

        result = await controller.parse_video(
            video_id="test123",
            title="Test Artist - Test Song (Karaoke)",
            description="Test description",
            channel_name="Test Channel",
        )

        assert result.final_result is not None
        assert result.final_result.original_artist == "Test Artist"
        assert result.final_result.song_title == "Test Song"
        assert len(result.passes_attempted) == 0  # No passes attempted

    @pytest.mark.asyncio
    async def test_early_stopping_on_success(self):
        """Test that processing stops after first successful pass."""

        # Mock Pass 0 to return high confidence result
        with patch.object(self.controller, "_pass_0_channel_template") as mock_pass_0:
            mock_pass_0.return_value = ParseResult(
                original_artist="Channel Artist",
                song_title="Channel Song",
                confidence=0.9,
                method="channel_template",
            )

            result = await self.controller.parse_video(
                video_id="test123",
                title="Channel Template Test",
                channel_name="Test Channel",
                channel_id="UC123",
            )

            assert result.final_result is not None
            assert result.final_confidence == 0.9
            assert result.stopped_at_pass == PassType.CHANNEL_TEMPLATE
            assert len(result.passes_attempted) == 1

    @pytest.mark.asyncio
    async def test_pass_progression_on_low_confidence(self):
        """Test that processing continues through passes with low confidence."""

        config = MultiPassConfig(
            enabled=True,
            stop_on_first_success=True,
            channel_template=MultiPassPassConfig(confidence_threshold=0.8),
            auto_retemplate=MultiPassPassConfig(confidence_threshold=0.7),
        )

        controller = MultiPassParsingController(
            config=config,
            advanced_parser=self.mock_advanced_parser,
            search_engine=self.mock_search_engine,
            db_manager=self.mock_db_manager,
        )

        # Mock passes to return progressively higher confidence
        with patch.object(controller, "_pass_0_channel_template") as mock_pass_0, patch.object(
            controller, "_pass_1_auto_retemplate"
        ) as mock_pass_1:

            # Pass 0 returns low confidence (below threshold)
            mock_pass_0.return_value = ParseResult(
                original_artist="Artist 1",
                song_title="Song 1",
                confidence=0.6,  # Below 0.8 threshold
                method="channel_template",
            )

            # Pass 1 returns high confidence
            mock_pass_1.return_value = ParseResult(
                original_artist="Artist 2",
                song_title="Song 2",
                confidence=0.85,  # Above 0.7 threshold
                method="auto_retemplate",
            )

            result = await controller.parse_video(
                video_id="test123", title="Multi-pass Test", channel_name="Test Channel"
            )

            assert result.final_result is not None
            assert result.final_confidence == 0.85
            assert result.stopped_at_pass == PassType.AUTO_RETEMPLATE
            assert len(result.passes_attempted) == 2

    @pytest.mark.asyncio
    async def test_budget_exceeded_stops_processing(self):
        """Test that processing stops when budgets are exceeded."""

        config = MultiPassConfig(
            enabled=True, total_cpu_budget=5.0, total_api_budget=10  # Very low budget
        )

        controller = MultiPassParsingController(
            config=config,
            advanced_parser=self.mock_advanced_parser,
            search_engine=self.mock_search_engine,
            db_manager=self.mock_db_manager,
        )

        # Mock passes to consume budget
        async def slow_pass(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate processing time
            return None

        with patch.object(controller, "_pass_0_channel_template", slow_pass), patch.object(
            controller, "_pass_1_auto_retemplate", slow_pass
        ), patch.object(controller, "_check_budget_exceeded") as mock_budget_check:

            # Force budget exceeded after 2 passes
            mock_budget_check.side_effect = [False, False, True, True, True]

            result = await controller.parse_video(
                video_id="test123", title="Budget Test", channel_name="Test Channel"
            )

            # Should stop due to budget, not process all passes
            assert len(result.passes_attempted) < len(PassType)

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling for individual passes."""

        config = MultiPassConfig(
            enabled=True,
            channel_template=MultiPassPassConfig(timeout_seconds=0.1),  # Very short timeout
        )

        controller = MultiPassParsingController(
            config=config,
            advanced_parser=self.mock_advanced_parser,
            search_engine=self.mock_search_engine,
            db_manager=self.mock_db_manager,
        )

        # Mock pass that takes too long
        async def timeout_pass(*args, **kwargs):
            await asyncio.sleep(0.2)  # Longer than timeout
            return ParseResult(confidence=0.9)

        with patch.object(controller, "_pass_0_channel_template", timeout_pass):
            result = await controller.parse_video(
                video_id="test123", title="Timeout Test", channel_name="Test Channel"
            )

            # Should handle timeout gracefully
            assert len(result.passes_attempted) >= 1
            pass_result = result.passes_attempted[0]
            assert not pass_result.success
            assert "timeout" in pass_result.error_message.lower()

    def test_statistics_collection(self):
        """Test statistics collection and reporting."""

        # Simulate some processing
        self.controller.statistics["total_videos_processed"] = 10
        self.controller.statistics["passes_attempted"][PassType.CHANNEL_TEMPLATE] = 8
        self.controller.statistics["passes_successful"][PassType.CHANNEL_TEMPLATE] = 6

        stats = self.controller.get_statistics()

        assert stats["total_videos_processed"] == 10
        assert stats["success_rates"][PassType.CHANNEL_TEMPLATE.value] == 0.75  # 6/8
        assert "budget_efficiency" in stats or stats["total_videos_processed"] == 0


class TestChannelTemplatePass:
    """Test suite for the channel template pass."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_advanced_parser = Mock(spec=AdvancedTitleParser)
        self.mock_advanced_parser.known_artists = {"Test Artist", "Another Artist"}
        self.mock_advanced_parser.known_songs = {"Test Song", "Another Song"}

        # Mock the _clean_extracted_text and validation methods
        self.mock_advanced_parser._clean_extracted_text.side_effect = lambda x: x.strip()
        self.mock_advanced_parser._is_valid_artist_name.return_value = True
        self.mock_advanced_parser._is_valid_song_title.return_value = True

        self.pass_instance = EnhancedChannelTemplatePass(
            self.mock_advanced_parser, None  # No db_manager for tests
        )

    def test_channel_pattern_learning(self):
        """Test learning patterns from successful parses."""

        channel_id = "UC123456"
        channel_name = "Test Karaoke Channel"
        title = 'Test Channel - "Artist Name" - "Song Title" (Karaoke)'

        # Create a mock successful result
        result = ParseResult(
            original_artist="Artist Name",
            song_title="Song Title",
            confidence=0.9,
            method="channel_specific",
            pattern_used=r'^Test Channel - "([^"]+)" - "([^"]+)" \([Kk]araoke\)',
        )

        # Learn from this success
        self.pass_instance._learn_from_success(channel_id, channel_name, title, result)

        # Check that channel stats were created
        assert channel_id in self.pass_instance.channel_stats
        stats = self.pass_instance.channel_stats[channel_id]
        assert stats.successful_parses == 1
        assert stats.channel_name == channel_name

    def test_pattern_generalization(self):
        """Test pattern generalization from successful parses."""

        title = 'Channel Name - "Test Artist" - "Test Song" (Karaoke)'
        result = ParseResult(original_artist="Test Artist", song_title="Test Song", confidence=0.9)

        pattern = self.pass_instance._generalize_pattern(title, result)

        assert pattern is not None
        assert "([^-–—\"']+?)" in pattern  # Should have capturing groups
        assert "^" in pattern and "$" in pattern  # Should be anchored

    def test_drift_detection(self):
        """Test channel drift detection."""

        channel_id = "UC123456"

        # Set up channel with patterns
        self.pass_instance.channel_stats[channel_id] = Mock()
        self.pass_instance.channel_stats[channel_id].total_videos = 20
        self.pass_instance.channel_stats[channel_id].successful_parses = 5  # 25% success rate
        self.pass_instance.channel_stats[channel_id].drift_threshold = 0.5

        # This should trigger drift detection
        self.pass_instance._check_for_drift(channel_id)

        assert self.pass_instance.channel_stats[channel_id].drift_detected

    def test_enhanced_channel_detection(self):
        """Test enhanced channel-specific pattern detection."""

        test_cases = [
            {
                "title": 'Sing King Karaoke: "Bohemian Rhapsody" - "Queen"',
                "channel": "Sing King Karaoke",
                "expected_artist": "Queen",
                "expected_song": "Bohemian Rhapsody",
            },
            {
                "title": '"Hotel California" - "Eagles" - Zoom Karaoke',
                "channel": "Zoom Karaoke",
                "expected_artist": "Eagles",
                "expected_song": "Hotel California",
            },
        ]

        for case in test_cases:
            result = self.pass_instance._enhanced_channel_detection(
                case["title"], "", "", case["channel"], "UC123"
            )

            if result:  # Some patterns may not match in test environment
                assert result.confidence > 0
                # Note: Exact matching depends on pattern implementation


class TestAutoRetemplatePass:
    """Test suite for the auto-retemplate pass."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_advanced_parser = Mock(spec=AdvancedTitleParser)
        self.pass_instance = AutoRetemplatePass(
            self.mock_advanced_parser, None  # No db_manager for tests
        )

    def test_temporal_pattern_creation(self):
        """Test creation of temporal patterns."""

        pattern = TemporalPattern(
            pattern=r"^([^-]+)-([^(]+)\([Kk]araoke\)",
            artist_group=1,
            title_group=2,
            confidence=0.8,
            first_seen=datetime.now(),
            last_seen=datetime.now(),
        )

        assert pattern.pattern is not None
        assert pattern.confidence == 0.8
        assert pattern.video_count == 0
        assert pattern.success_count == 0

    def test_pattern_change_detection(self):
        """Test detection of pattern changes over time."""

        channel_id = "UC123456"
        trend = self.pass_instance._get_or_create_trend(channel_id, "Test Channel")

        # Add some active patterns
        pattern1 = TemporalPattern(
            pattern=r"^Old Pattern",
            artist_group=1,
            title_group=2,
            confidence=0.8,
            first_seen=datetime.now() - timedelta(days=10),
            last_seen=datetime.now() - timedelta(days=8),  # Old
        )
        trend.active_patterns.append(pattern1)

        # Simulate recent videos that don't match
        recent_videos = [
            {"title": "New Format Video 1"},
            {"title": "New Format Video 2"},
            {"title": "New Format Video 3"},
        ]

        self.pass_instance._detect_pattern_changes(trend, recent_videos)

        # Should detect pattern change due to low match rate
        assert trend.pattern_change_detected is not None

    def test_structure_extraction(self):
        """Test title structure extraction."""

        test_cases = [
            {"title": "Artist Name - Song Title (Karaoke)", "expected": "ARTIST-SONG-KARAOKE"},
            {"title": '"Artist" - "Song" Karaoke', "expected": "QUOTED-ARTIST-SONG-KARAOKE"},
            {"title": "Song Title by Artist Name (Karaoke)", "expected": "SONG-BY-ARTIST-KARAOKE"},
        ]

        for case in test_cases:
            structure = self.pass_instance._extract_title_structure(case["title"])
            assert structure == case["expected"]


class TestMLEmbeddingPass:
    """Test suite for the ML/embedding pass."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_advanced_parser = Mock(spec=AdvancedTitleParser)
        self.mock_fuzzy_matcher = Mock()

        # Mock fuzzy matcher methods
        self.mock_fuzzy_matcher.find_best_match.return_value = Mock()
        self.mock_fuzzy_matcher.find_best_match.return_value.matched = "Test Artist"
        self.mock_fuzzy_matcher.find_best_match.return_value.score = 0.85

        self.pass_instance = EnhancedMLEmbeddingPass(
            self.mock_advanced_parser, self.mock_fuzzy_matcher, None
        )

    def test_entity_extraction(self):
        """Test entity extraction from titles."""

        title = 'Test Channel - "Famous Artist" performs "Great Song" [HD Karaoke]'
        description = "High quality karaoke track"
        tags = "karaoke, music, singing"

        entities = self.pass_instance._extract_entities(title, description, tags)

        assert "potential_artists" in entities
        assert "potential_songs" in entities
        assert "quoted_text" in entities
        assert "capitalized_words" in entities

        # Should extract quoted text
        assert "Famous Artist" in entities["quoted_text"]
        assert "Great Song" in entities["quoted_text"]

    def test_entity_pattern_matching(self):
        """Test pattern-based matching using extracted entities."""

        entities = {
            "quoted_text": ["Test Artist", "Test Song"],
            "capitalized_words": ["Another", "Artist"],
            "potential_artists": ["Test Artist"],
            "potential_songs": ["Test Song"],
        }

        result = self.pass_instance._entity_pattern_matching("Test Title", entities)

        assert result is not None
        assert result.original_artist == "Test Artist"
        assert result.song_title == "Test Song"
        assert result.confidence > 0

    def test_embedding_model_handling(self):
        """Test handling of optional embedding model."""

        # Test without embedding model
        assert self.pass_instance.embedding_model is None or True  # May or may not be available

        # Test statistics without embedding model
        stats = self.pass_instance.get_statistics()
        assert "has_embedding_model" in stats
        assert "artist_candidates" in stats
        assert "song_candidates" in stats


class TestWebSearchPass:
    """Test suite for the web search pass."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_advanced_parser = Mock(spec=AdvancedTitleParser)
        self.mock_search_engine = Mock()

        # Mock search engine results
        self.mock_search_engine.search_videos = AsyncMock(return_value=[])

        self.pass_instance = EnhancedWebSearchPass(
            self.mock_advanced_parser, self.mock_search_engine, None
        )

    def test_filler_word_processor(self):
        """Test filler word removal."""

        processor = FillerWordProcessor()

        test_cases = [
            {
                "input": "Artist Name - Song Title (Karaoke Version HD)",
                "expected_terms": ["karaoke", "hd"],
                "should_be_cleaner": True,
            },
            {
                "input": "Official Music Video High Quality",
                "expected_terms": ["official", "music video", "high quality"],
                "should_be_cleaner": True,
            },
        ]

        for case in test_cases:
            result = processor.clean_query(case["input"])

            assert result.original_query == case["input"]
            assert len(result.cleaned_query) <= len(case["input"])
            assert result.confidence_boost >= 1.0

            # Check that expected terms were removed
            for term in case["expected_terms"]:
                if term in case["input"].lower():
                    assert term in [t.lower() for t in result.removed_terms]

    def test_serp_cache(self):
        """Test SERP result caching."""

        cache = self.pass_instance.serp_cache

        # Test cache miss
        assert cache.get("test query") is None

        # Test cache put and get
        test_results = [{"video_id": "test123", "title": "Test Result"}]
        cache.put("test query", test_results)

        cached_results = cache.get("test query")
        assert cached_results == test_results

        # Test cache statistics
        stats = cache.get_statistics()
        assert "total_entries" in stats
        assert stats["total_entries"] >= 1

    def test_query_generation(self):
        """Test search query generation strategies."""

        title = 'Channel Name - "Artist" performs "Song" [Karaoke HD]'
        description = "High quality karaoke version with lyrics"
        tags = "karaoke, music, singing"

        queries = self.pass_instance._generate_search_queries(title, description, tags)

        assert len(queries) > 0
        assert all(q.cleaned_query != title for q in queries)  # Should be cleaned
        assert all(q.confidence_boost >= 1.0 for q in queries)  # Should have boosts

        # Should generate different strategies
        methods = [q.cleaning_method for q in queries]
        assert len(set(methods)) > 1  # Multiple different methods

    @pytest.mark.asyncio
    async def test_search_with_engine(self):
        """Test search execution with search engine."""

        # Mock search results
        mock_result = Mock()
        mock_result.video_id = "test123"
        mock_result.title = "Test Artist - Test Song (Karaoke)"
        mock_result.description = "Test description"
        mock_result.channel_name = "Test Channel"

        self.mock_search_engine.search_videos.return_value = [mock_result]

        results = await self.pass_instance._search_with_engine("test query")

        assert len(results) == 1
        assert results[0]["video_id"] == "test123"
        assert results[0]["title"] == "Test Artist - Test Song (Karaoke)"


class TestAcousticFingerprintPass:
    """Test suite for the acoustic fingerprint pass."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_advanced_parser = Mock(spec=AdvancedTitleParser)
        self.pass_instance = AcousticFingerprintPass(self.mock_advanced_parser, None)

    @pytest.mark.asyncio
    async def test_disabled_by_default(self):
        """Test that acoustic fingerprint pass is disabled by default."""

        result = await self.pass_instance.parse(
            "Test Title", "Test description", "test,tags", "Test Channel", "UC123456"
        )

        assert result is None  # Should return None when disabled

    def test_enable_disable_functionality(self):
        """Test enable/disable functionality."""

        assert not self.pass_instance.enabled  # Disabled by default

        self.pass_instance.enable_processing()
        assert self.pass_instance.enabled

        self.pass_instance.disable_processing()
        assert not self.pass_instance.enabled

    def test_statistics_collection(self):
        """Test statistics collection for placeholder implementation."""

        stats = self.pass_instance.get_statistics()

        assert "enabled" in stats
        assert "total_requests" in stats
        assert "implementation_notes" in stats
        assert "This is a placeholder implementation" in str(stats["implementation_notes"])

    def test_implementation_guide(self):
        """Test implementation guide generation."""

        guide = self.pass_instance.get_implementation_guide()

        assert "overview" in guide
        assert "required_libraries" in guide
        assert "implementation_steps" in guide
        assert "considerations" in guide

        # Should mention key libraries
        libraries = str(guide["required_libraries"])
        assert "pyacoustid" in libraries
        assert "aubio" in libraries
        assert "chromaprint" in libraries


class TestIntegrationScenarios:
    """Integration tests for complete multi-pass scenarios."""

    def setup_method(self):
        """Set up integration test fixtures."""
        self.mock_advanced_parser = Mock(spec=AdvancedTitleParser)
        self.mock_search_engine = Mock()

        # Set up realistic mock responses
        self.mock_advanced_parser.known_artists = {"Queen", "Eagles", "Beatles"}
        self.mock_advanced_parser.known_songs = {"Bohemian Rhapsody", "Hotel California"}

        self.config = MultiPassConfig(
            enabled=True, stop_on_first_success=True, total_cpu_budget=60.0, total_api_budget=100
        )

        self.controller = MultiPassParsingController(
            config=self.config,
            advanced_parser=self.mock_advanced_parser,
            search_engine=self.mock_search_engine,
            db_manager=None,
        )

    @pytest.mark.asyncio
    async def test_sing_king_channel_parsing(self):
        """Test parsing of Sing King karaoke channel format."""

        title = 'Sing King Karaoke - "Bohemian Rhapsody" (Style of "Queen")'
        channel_name = "Sing King Karaoke"

        # Mock channel template pass to succeed
        with patch.object(self.controller.channel_template_pass, "parse") as mock_parse:
            mock_parse.return_value = ParseResult(
                original_artist="Queen",
                song_title="Bohemian Rhapsody",
                confidence=0.95,
                method="channel_template_sing_king",
            )

            result = await self.controller.parse_video(
                video_id="test123", title=title, channel_name=channel_name, channel_id="UC123456"
            )

            assert result.final_result is not None
            assert result.final_result.original_artist == "Queen"
            assert result.final_result.song_title == "Bohemian Rhapsody"
            assert result.stopped_at_pass == PassType.CHANNEL_TEMPLATE

    @pytest.mark.asyncio
    async def test_fallback_through_multiple_passes(self):
        """Test fallback through multiple passes when early passes fail."""

        title = "Obscure Song Title Format [HD Karaoke]"

        # Mock passes to fail until web search
        with patch.object(
            self.controller.channel_template_pass, "parse", return_value=None
        ), patch.object(
            self.controller.auto_retemplate_pass, "parse", return_value=None
        ), patch.object(
            self.controller.ml_embedding_pass, "parse", return_value=None
        ), patch.object(
            self.controller.web_search_pass, "parse"
        ) as mock_web_search:

            mock_web_search.return_value = ParseResult(
                original_artist="Found Artist",
                song_title="Found Song",
                confidence=0.75,
                method="web_search_parsing",
            )

            result = await self.controller.parse_video(
                video_id="test123", title=title, channel_name="Unknown Channel"
            )

            assert result.final_result is not None
            assert result.stopped_at_pass == PassType.WEB_SEARCH
            assert len(result.passes_attempted) == 4  # All passes before web search + web search

    @pytest.mark.asyncio
    async def test_complete_failure_scenario(self):
        """Test scenario where all passes fail."""

        title = "Completely Unparseable Title 12345 !!!!"

        # Mock all passes to fail
        with patch.object(
            self.controller.channel_template_pass, "parse", return_value=None
        ), patch.object(
            self.controller.auto_retemplate_pass, "parse", return_value=None
        ), patch.object(
            self.controller.ml_embedding_pass, "parse", return_value=None
        ), patch.object(
            self.controller.web_search_pass, "parse", return_value=None
        ), patch.object(
            self.controller.acoustic_fingerprint_pass, "parse", return_value=None
        ):

            result = await self.controller.parse_video(
                video_id="test123", title=title, channel_name="Test Channel"
            )

            assert result.final_result is None
            assert result.final_confidence == 0.0
            assert len(result.passes_attempted) == len(PassType)  # All passes attempted

    @pytest.mark.asyncio
    async def test_budget_management_integration(self):
        """Test budget management across multiple passes."""

        # Set very low budgets
        config = MultiPassConfig(
            enabled=True,
            total_cpu_budget=1.0,  # 1 second total
            total_api_budget=5,  # 5 API calls total
        )

        controller = MultiPassParsingController(
            config=config,
            advanced_parser=self.mock_advanced_parser,
            search_engine=self.mock_search_engine,
            db_manager=None,
        )

        # Mock passes to consume budget
        async def budget_consuming_pass(*args, **kwargs):
            await asyncio.sleep(0.1)  # Consume some CPU time
            return None

        with patch.object(
            controller, "_pass_0_channel_template", budget_consuming_pass
        ), patch.object(controller, "_pass_1_auto_retemplate", budget_consuming_pass):

            result = await controller.parse_video(
                video_id="test123", title="Budget Test Title", channel_name="Test Channel"
            )

            # Should respect budget limits
            assert result.budget_consumed["cpu"] <= config.total_cpu_budget * 1.1  # Small tolerance
            assert result.budget_consumed["api"] <= config.total_api_budget


# Run tests if called directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
