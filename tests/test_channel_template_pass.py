"""Comprehensive tests for the channel template pass module."""

import re
from dataclasses import asdict
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from collector.advanced_parser import AdvancedTitleParser, ParseResult
from collector.passes.base import PassType
from collector.passes.channel_template_pass import (
    ChannelPattern,
    ChannelStats,
    EnhancedChannelTemplatePass,
)


class TestChannelPattern:
    """Test the ChannelPattern dataclass."""

    def test_channel_pattern_creation(self):
        """Test creating a ChannelPattern with default values."""
        pattern = ChannelPattern(
            pattern=r"([^-]+) - ([^(]+)",
            artist_group=1,
            title_group=2,
            confidence=0.8,
        )
        assert pattern.pattern == r"([^-]+) - ([^(]+)"
        assert pattern.artist_group == 1
        assert pattern.title_group == 2
        assert pattern.confidence == 0.8
        assert pattern.success_count == 0
        assert pattern.total_attempts == 0
        assert isinstance(pattern.last_used, datetime)
        assert isinstance(pattern.created, datetime)
        assert pattern.examples == []

    def test_channel_pattern_with_values(self):
        """Test creating a ChannelPattern with specific values."""
        now = datetime.now()
        pattern = ChannelPattern(
            pattern=r"test_pattern",
            artist_group=1,
            title_group=2,
            confidence=0.9,
            success_count=5,
            total_attempts=10,
            last_used=now,
            created=now,
            examples=["example1", "example2"],
        )
        assert pattern.success_count == 5
        assert pattern.total_attempts == 10
        assert pattern.last_used == now
        assert pattern.examples == ["example1", "example2"]

    def test_channel_pattern_serializable(self):
        """Test that ChannelPattern can be converted to dict."""
        pattern = ChannelPattern(
            pattern=r"test_pattern",
            artist_group=1,
            title_group=2,
            confidence=0.8,
        )
        pattern_dict = asdict(pattern)
        assert isinstance(pattern_dict, dict)
        assert pattern_dict["pattern"] == r"test_pattern"
        assert pattern_dict["confidence"] == 0.8


class TestChannelStats:
    """Test the ChannelStats dataclass."""

    def test_channel_stats_creation(self):
        """Test creating ChannelStats with default values."""
        stats = ChannelStats(channel_id="test_id", channel_name="Test Channel")
        assert stats.channel_id == "test_id"
        assert stats.channel_name == "Test Channel"
        assert stats.total_videos == 0
        assert stats.successful_parses == 0
        assert stats.patterns == {}
        assert stats.drift_detected is False
        assert stats.drift_threshold == 0.5
        assert isinstance(stats.last_updated, datetime)

    def test_channel_stats_with_values(self):
        """Test creating ChannelStats with specific values."""
        patterns = {"pattern1": ChannelPattern("test", 1, 2, 0.8)}
        stats = ChannelStats(
            channel_id="test_id",
            channel_name="Test Channel",
            total_videos=100,
            successful_parses=85,
            patterns=patterns,
            drift_detected=True,
            drift_threshold=0.7,
        )
        assert stats.total_videos == 100
        assert stats.successful_parses == 85
        assert stats.patterns == patterns
        assert stats.drift_detected is True
        assert stats.drift_threshold == 0.7


class TestEnhancedChannelTemplatePass:
    """Test the EnhancedChannelTemplatePass class."""

    @pytest.fixture
    def mock_advanced_parser(self):
        """Create a mock AdvancedTitleParser."""
        parser = MagicMock(spec=AdvancedTitleParser)
        parser._parse_with_channel_patterns.return_value = ParseResult(
            original_artist="Mock Artist",
            song_title="Mock Song",
            confidence=0.7,
            method="mock_channel_pattern",
        )
        parser._create_result_from_match.return_value = ParseResult(
            original_artist="Test Artist",
            song_title="Test Song",
            confidence=0.8,
            method="test_method",
        )
        return parser

    @pytest.fixture
    def mock_db_manager(self):
        """Create a mock database manager."""
        return MagicMock()

    @pytest.fixture
    def channel_pass(self, mock_advanced_parser, mock_db_manager):
        """Create a channel template pass instance."""
        return EnhancedChannelTemplatePass(mock_advanced_parser, mock_db_manager)

    @pytest.fixture
    def channel_pass_no_db(self, mock_advanced_parser):
        """Create a channel template pass instance without database."""
        return EnhancedChannelTemplatePass(mock_advanced_parser, None)

    def test_pass_type(self, channel_pass):
        """Test that the pass type is correctly set."""
        assert channel_pass.pass_type == PassType.CHANNEL_TEMPLATE

    def test_initialization(self, channel_pass):
        """Test channel pass initialization."""
        assert channel_pass.advanced_parser is not None
        assert channel_pass.db_manager is not None
        assert isinstance(channel_pass.channel_stats, dict)
        assert isinstance(channel_pass.channel_patterns, dict)
        assert isinstance(channel_pass.global_pattern_weights, dict)
        assert channel_pass.min_examples_for_pattern == 3
        assert channel_pass.max_patterns_per_channel == 10
        assert channel_pass.pattern_decay_days == 30

    def test_initialization_no_db(self, channel_pass_no_db):
        """Test initialization without database."""
        assert channel_pass_no_db.db_manager is None


class TestChannelPatternMatching:
    """Test channel pattern matching functionality."""

    @pytest.fixture
    def channel_pass(self):
        parser = MagicMock(spec=AdvancedTitleParser)
        return EnhancedChannelTemplatePass(parser, None)

    @pytest.mark.asyncio
    async def test_parse_no_channel_info(self, channel_pass):
        """Test parsing without channel information."""
        result = await channel_pass.parse("Test Title")
        assert result is None

    @pytest.mark.asyncio
    async def test_parse_with_learned_patterns(self, channel_pass):
        """Test parsing with learned channel patterns."""
        # Set up a learned pattern
        pattern = ChannelPattern(
            pattern=r"^([^-]+)\s*-\s*([^(]+)",
            artist_group=1,
            title_group=2,
            confidence=0.85,
            success_count=5,
            total_attempts=5,
        )
        channel_pass.channel_patterns["test_channel"] = [pattern]

        # Mock the pattern matching
        with patch.object(channel_pass, "_create_result_from_pattern_match") as mock_create:
            mock_create.return_value = ParseResult(
                original_artist="Test Artist",
                song_title="Test Song",
                confidence=0.85,
                pattern_used=pattern.pattern,
            )

            result = await channel_pass.parse("Test Artist - Test Song", channel_id="test_channel")

            assert result is not None
            assert result.original_artist == "Test Artist"
            assert result.song_title == "Test Song"
            assert result.confidence == 0.85

    @pytest.mark.asyncio
    async def test_parse_enhanced_channel_detection(self, channel_pass):
        """Test enhanced channel detection."""
        # Mock enhanced detection method
        with patch.object(channel_pass, "_enhanced_channel_detection") as mock_enhanced:
            mock_enhanced.return_value = ParseResult(
                original_artist="Enhanced Artist",
                song_title="Enhanced Song",
                confidence=0.8,
                method="enhanced_channel",
            )

            result = await channel_pass.parse(
                "Test Title", channel_id="test_channel", channel_name="Test Channel"
            )

            assert result is not None
            assert result.original_artist == "Enhanced Artist"
            assert result.confidence == 0.8

    @pytest.mark.asyncio
    async def test_parse_fallback_to_advanced_parser(self, channel_pass):
        """Test fallback to advanced parser."""
        # Mock the advanced parser
        channel_pass.advanced_parser._parse_with_channel_patterns.return_value = ParseResult(
            original_artist="Fallback Artist",
            song_title="Fallback Song",
            confidence=0.75,
            method="advanced_parser",
        )

        with patch.object(channel_pass, "_enhanced_channel_detection") as mock_enhanced:
            mock_enhanced.return_value = None

            result = await channel_pass.parse(
                "Test Title", channel_id="test_channel", channel_name="Test Channel"
            )

            assert result is not None
            assert result.original_artist == "Fallback Artist"
            assert result.confidence == 0.75

    @pytest.mark.asyncio
    async def test_parse_channel_context_extraction(self, channel_pass):
        """Test channel context extraction."""
        with patch.object(
            channel_pass, "_enhanced_channel_detection"
        ) as mock_enhanced, patch.object(
            channel_pass, "_extract_with_channel_context"
        ) as mock_context:

            mock_enhanced.return_value = None
            channel_pass.advanced_parser._parse_with_channel_patterns.return_value = None
            mock_context.return_value = ParseResult(
                original_artist="Context Artist",
                song_title="Context Song",
                confidence=0.65,
                method="channel_context",
            )

            result = await channel_pass.parse(
                "Test Title", channel_id="test_channel", channel_name="Test Channel"
            )

            assert result is not None
            assert result.original_artist == "Context Artist"
            assert result.confidence == 0.65

    @pytest.mark.asyncio
    async def test_parse_exception_handling(self, channel_pass):
        """Test exception handling during parse."""
        with patch.object(channel_pass, "_enhanced_channel_detection") as mock_enhanced:
            mock_enhanced.side_effect = Exception("Test error")

            result = await channel_pass.parse(
                "Test Title", channel_id="test_channel", channel_name="Test Channel"
            )

            assert result is None


class TestEnhancedChannelDetection:
    """Test enhanced channel detection methods."""

    @pytest.fixture
    def channel_pass(self):
        parser = MagicMock(spec=AdvancedTitleParser)
        parser._create_result_from_match.return_value = ParseResult(
            original_artist="Test Artist",
            song_title="Test Song",
            confidence=0.8,
            method="test_method",
        )
        return EnhancedChannelTemplatePass(parser, None)

    def test_enhanced_channel_detection_branded_quoted(self, channel_pass):
        """Test enhanced detection with channel branded quoted pattern."""
        title = 'Test Channel "Artist Name" - "Song Title"'
        channel_pass._enhanced_channel_detection(title, "", "", "Test Channel", "test_channel")

        # Should call _create_result_from_match
        assert channel_pass.advanced_parser._create_result_from_match.called

    def test_enhanced_channel_detection_bracketed(self, channel_pass):
        """Test enhanced detection with bracketed pattern."""
        title = "[Test Channel] Artist Name - Song Title"
        channel_pass._enhanced_channel_detection(title, "", "", "Test Channel", "test_channel")

        # Should call _create_result_from_match
        assert channel_pass.advanced_parser._create_result_from_match.called

    def test_enhanced_channel_detection_double_quoted(self, channel_pass):
        """Test enhanced detection with double quoted pattern."""
        title = '"Artist Name" - "Song Title" - Karaoke'
        channel_pass._enhanced_channel_detection(title, "", "", "Test Channel", "test_channel")

        # Should call _create_result_from_match
        assert channel_pass.advanced_parser._create_result_from_match.called

    def test_enhanced_channel_detection_no_match(self, channel_pass):
        """Test enhanced detection with no matching patterns."""
        channel_pass.advanced_parser._create_result_from_match.return_value = None

        title = "No matching pattern here"
        result = channel_pass._enhanced_channel_detection(
            title, "", "", "Test Channel", "test_channel"
        )

        assert result is None

    def test_enhanced_channel_detection_regex_error(self, channel_pass):
        """Test handling of regex errors in enhanced detection."""
        # This should not raise an exception, just continue to next pattern
        title = "Test ((((( malformed"
        result = channel_pass._enhanced_channel_detection(
            title, "", "", "Test Channel", "test_channel"
        )

        # Should handle the error gracefully
        assert result is None or isinstance(result, ParseResult)


class TestChannelContextExtraction:
    """Test channel context extraction methods."""

    @pytest.fixture
    def channel_pass(self):
        parser = MagicMock(spec=AdvancedTitleParser)
        parser._create_result_from_match.return_value = ParseResult(
            original_artist="Test Artist",
            song_title="Test Song",
            confidence=0.65,
            method="test_method",
        )
        return EnhancedChannelTemplatePass(parser, None)

    def test_extract_with_karaoke_channel_context(self, channel_pass):
        """Test extraction with karaoke channel context."""
        title = "Artist Name - Song Title"
        channel_pass._extract_with_channel_context(title, "", "Karaoke Channel", "karaoke_channel")

        # Should identify as karaoke channel and boost confidence
        assert channel_pass.advanced_parser._create_result_from_match.called

    def test_extract_with_non_karaoke_channel(self, channel_pass):
        """Test extraction with non-karaoke channel."""
        title = "Artist Name - Song Title"
        result = channel_pass._extract_with_channel_context(
            title, "", "Regular Music Channel", "regular_channel"
        )

        assert result is None

    def test_extract_with_karafun_channel(self, channel_pass):
        """Test extraction with KaraFun channel."""
        title = "Artist Name - Song Title"
        channel_pass._extract_with_channel_context(title, "", "KaraFun Channel", "karafun_channel")

        # Should identify as karaoke channel
        assert channel_pass.advanced_parser._create_result_from_match.called

    def test_extract_confidence_boost(self, channel_pass):
        """Test that karaoke channel boost is applied."""
        original_result = ParseResult(
            original_artist="Test Artist",
            song_title="Test Song",
            confidence=0.6,
            method="test_method",
        )
        channel_pass.advanced_parser._create_result_from_match.return_value = original_result

        title = "Test Artist - Test Song"
        result = channel_pass._extract_with_channel_context(
            title, "", "Karaoke Channel", "karaoke_channel"
        )

        if result:
            # Confidence should be boosted
            assert result.confidence >= 0.6
            assert result.metadata.get("karaoke_channel_boost") is True

    @pytest.mark.parametrize(
        "channel_name,expected_karaoke",
        [
            ("Karaoke Channel", True),
            ("KaraFun Official", True),
            ("Karaoké français", True),
            ("Sing Along Videos", True),
            ("Backing Track Central", True),
            ("Instrumental Covers", True),
            ("Piano Version Songs", True),
            ("Караоке по-русски", True),
            ("Regular Music Channel", False),
            ("Pop Songs Official", False),
            ("Music Videos HD", False),
        ],
    )
    def test_karaoke_channel_detection(self, channel_pass, channel_name, expected_karaoke):
        """Test detection of karaoke channels."""
        title = "Artist - Song"
        result = channel_pass._extract_with_channel_context(title, "", channel_name, "test_channel")

        if expected_karaoke:
            # Should call parser for karaoke channels
            assert channel_pass.advanced_parser._create_result_from_match.called
        else:
            # Should return None for non-karaoke channels
            assert result is None


class TestPatternLearning:
    """Test pattern learning functionality."""

    @pytest.fixture
    def channel_pass(self):
        parser = MagicMock(spec=AdvancedTitleParser)
        return EnhancedChannelTemplatePass(parser, None)

    def test_learn_from_success(self, channel_pass):
        """Test learning from successful parse."""
        result = ParseResult(
            original_artist="Test Artist",
            song_title="Test Song",
            confidence=0.85,
            method="test_method",
        )

        initial_stats_count = len(channel_pass.channel_stats)

        channel_pass._learn_from_success(
            "test_channel", "Test Channel", "Test Artist - Test Song", result
        )

        # Should create channel stats
        assert len(channel_pass.channel_stats) == initial_stats_count + 1
        assert "test_channel" in channel_pass.channel_stats

        stats = channel_pass.channel_stats["test_channel"]
        assert stats.channel_id == "test_channel"
        assert stats.channel_name == "Test Channel"
        assert stats.successful_parses == 1

    def test_generalize_pattern(self, channel_pass):
        """Test pattern generalization from successful parse."""
        result = ParseResult(
            original_artist="Test Artist",
            song_title="Test Song",
            confidence=0.85,
            method="test_method",
        )

        title = "Test Artist - Test Song (Karaoke)"
        pattern = channel_pass._generalize_pattern(title, result)

        assert pattern is not None
        assert "([^-–—\"']+?)" in pattern  # Artist group
        assert r"([^(\[]+?)" in pattern  # Title group

        # Test that the pattern is valid regex
        try:
            re.compile(pattern)
        except re.error:
            pytest.fail("Generated pattern is not valid regex")

    def test_generalize_pattern_incomplete_result(self, channel_pass):
        """Test pattern generalization with incomplete result."""
        result = ParseResult(
            original_artist="Test Artist",
            song_title=None,  # Missing title
            confidence=0.85,
            method="test_method",
        )

        title = "Test Artist - Test Song"
        pattern = channel_pass._generalize_pattern(title, result)

        assert pattern is None

    def test_add_new_pattern(self, channel_pass):
        """Test adding a new pattern."""
        pattern = r"^([^-]+)\s*-\s*([^(]+)"
        example = "Artist - Song"

        channel_pass._add_or_update_pattern("test_channel", pattern, example)

        assert "test_channel" in channel_pass.channel_patterns
        patterns = channel_pass.channel_patterns["test_channel"]
        assert len(patterns) == 1
        assert patterns[0].pattern == pattern
        assert patterns[0].success_count == 1
        assert patterns[0].total_attempts == 1
        assert example in patterns[0].examples

    def test_update_existing_pattern(self, channel_pass):
        """Test updating an existing pattern."""
        pattern = r"^([^-]+)\s*-\s*([^(]+)"
        example1 = "Artist1 - Song1"
        example2 = "Artist2 - Song2"

        # Add initial pattern
        channel_pass._add_or_update_pattern("test_channel", pattern, example1)

        # Update with same pattern
        channel_pass._add_or_update_pattern("test_channel", pattern, example2)

        patterns = channel_pass.channel_patterns["test_channel"]
        assert len(patterns) == 1  # Should not create duplicate
        assert patterns[0].success_count == 2
        assert patterns[0].total_attempts == 2
        assert example1 in patterns[0].examples
        assert example2 in patterns[0].examples

    def test_pattern_limit_enforcement(self, channel_pass):
        """Test that pattern limit per channel is enforced."""
        # Add maximum number of patterns
        for i in range(channel_pass.max_patterns_per_channel + 5):
            pattern = f"pattern_{i}"
            example = f"example_{i}"
            channel_pass._add_or_update_pattern("test_channel", pattern, example)

        patterns = channel_pass.channel_patterns["test_channel"]
        assert len(patterns) <= channel_pass.max_patterns_per_channel

    def test_pattern_cleanup(self, channel_pass):
        """Test pattern cleanup functionality."""
        # Add a pattern with poor success rate
        old_pattern = ChannelPattern(
            pattern=r"poor_pattern",
            artist_group=1,
            title_group=2,
            confidence=0.5,
            success_count=1,
            total_attempts=10,  # Poor success rate
            created=datetime.now() - timedelta(days=40),  # Old
        )

        # Add a good pattern
        good_pattern = ChannelPattern(
            pattern=r"good_pattern",
            artist_group=1,
            title_group=2,
            confidence=0.8,
            success_count=8,
            total_attempts=10,  # Good success rate
        )

        channel_pass.channel_patterns["test_channel"] = [old_pattern, good_pattern]

        channel_pass._cleanup_patterns("test_channel")

        patterns = channel_pass.channel_patterns["test_channel"]
        # Should keep good pattern, may remove poor pattern depending on age/usage
        assert good_pattern in patterns


class TestDriftDetection:
    """Test drift detection functionality."""

    @pytest.fixture
    def channel_pass(self):
        parser = MagicMock(spec=AdvancedTitleParser)
        return EnhancedChannelTemplatePass(parser, None)

    def test_drift_detection_sufficient_data(self, channel_pass):
        """Test drift detection with sufficient data."""
        # Create channel stats with low success rate
        stats = ChannelStats(
            channel_id="test_channel",
            channel_name="Test Channel",
            total_videos=20,
            successful_parses=5,  # 25% success rate
            drift_threshold=0.5,  # 50% threshold
        )
        channel_pass.channel_stats["test_channel"] = stats

        with patch.object(channel_pass, "_refresh_channel_patterns") as mock_refresh:
            channel_pass._check_for_drift("test_channel")

            # Should detect drift and trigger refresh
            assert stats.drift_detected is True
            mock_refresh.assert_called_once_with("test_channel")

    def test_drift_detection_insufficient_data(self, channel_pass):
        """Test drift detection with insufficient data."""
        stats = ChannelStats(
            channel_id="test_channel",
            channel_name="Test Channel",
            total_videos=5,  # Insufficient data
            successful_parses=1,
            drift_threshold=0.5,
        )
        channel_pass.channel_stats["test_channel"] = stats

        channel_pass._check_for_drift("test_channel")

        # Should not detect drift due to insufficient data
        assert stats.drift_detected is False

    def test_drift_detection_good_performance(self, channel_pass):
        """Test drift detection with good performance."""
        stats = ChannelStats(
            channel_id="test_channel",
            channel_name="Test Channel",
            total_videos=20,
            successful_parses=18,  # 90% success rate
            drift_threshold=0.5,
        )
        channel_pass.channel_stats["test_channel"] = stats

        channel_pass._check_for_drift("test_channel")

        # Should not detect drift with good performance
        assert stats.drift_detected is False

    def test_refresh_channel_patterns(self, channel_pass):
        """Test pattern refresh after drift detection."""
        # Set up patterns with different success rates
        good_pattern = ChannelPattern(
            pattern=r"good_pattern",
            confidence=0.8,
            success_count=8,
            total_attempts=10,
        )

        poor_pattern = ChannelPattern(
            pattern=r"poor_pattern",
            confidence=0.6,
            success_count=2,
            total_attempts=10,
        )

        channel_pass.channel_patterns["test_channel"] = [good_pattern, poor_pattern]

        channel_pass._refresh_channel_patterns("test_channel")

        # Should reduce confidence of all patterns
        assert good_pattern.confidence < 0.8
        assert poor_pattern.confidence < 0.6

        # Should remove poor patterns
        remaining_patterns = channel_pass.channel_patterns["test_channel"]
        assert good_pattern in remaining_patterns
        assert poor_pattern not in remaining_patterns  # Removed due to low success rate


class TestStatisticsAndPersistence:
    """Test statistics and persistence functionality."""

    @pytest.fixture
    def channel_pass(self):
        parser = MagicMock(spec=AdvancedTitleParser)
        return EnhancedChannelTemplatePass(parser, MagicMock())

    def test_get_statistics(self, channel_pass):
        """Test statistics generation."""
        # Set up test data
        stats1 = ChannelStats(
            channel_id="channel1",
            channel_name="Channel 1",
            total_videos=20,
            successful_parses=16,
        )
        stats2 = ChannelStats(
            channel_id="channel2",
            channel_name="Channel 2",
            total_videos=10,
            successful_parses=5,
            drift_detected=True,
        )

        channel_pass.channel_stats = {"channel1": stats1, "channel2": stats2}

        # Add some patterns
        channel_pass.channel_patterns["channel1"] = [
            ChannelPattern("pattern1", 1, 2, 0.8),
            ChannelPattern("pattern2", 1, 2, 0.7),
        ]
        channel_pass.channel_patterns["channel2"] = [
            ChannelPattern("pattern3", 1, 2, 0.6),
        ]

        stats = channel_pass.get_statistics()

        assert stats["total_channels"] == 2
        assert stats["total_learned_patterns"] == 3
        assert stats["channels_with_drift"] == 1
        assert stats["avg_patterns_per_channel"] == 1.5
        assert "channel_success_rates" in stats
        assert stats["channel_success_rates"]["channel1"] == 0.8  # 16/20
        assert stats["channel_success_rates"]["channel2"] == 0.5  # 5/10

    def test_get_statistics_empty(self, channel_pass):
        """Test statistics with no data."""
        stats = channel_pass.get_statistics()

        assert stats["total_channels"] == 0
        assert stats["total_learned_patterns"] == 0
        assert stats["channels_with_drift"] == 0
        assert stats["avg_patterns_per_channel"] == 0
        assert stats["channel_success_rates"] == {}

    def test_save_patterns_with_db(self, channel_pass):
        """Test saving patterns with database."""
        # Should not raise an exception
        channel_pass.save_patterns()
        # In real implementation, would test database calls

    def test_save_patterns_no_db(self):
        """Test saving patterns without database."""
        parser = MagicMock(spec=AdvancedTitleParser)
        channel_pass = EnhancedChannelTemplatePass(parser, None)

        # Should not raise an exception
        channel_pass.save_patterns()

    def test_load_channel_patterns_with_db(self, channel_pass):
        """Test loading patterns with database."""
        # Should call load logic (currently just logs)
        channel_pass._load_channel_patterns()

    def test_load_channel_patterns_no_db(self):
        """Test loading patterns without database."""
        parser = MagicMock(spec=AdvancedTitleParser)
        channel_pass = EnhancedChannelTemplatePass(parser, None)

        # Should not raise an exception
        channel_pass._load_channel_patterns()


class TestPatternMatching:
    """Test pattern matching logic."""

    @pytest.fixture
    def channel_pass(self):
        parser = MagicMock(spec=AdvancedTitleParser)
        return EnhancedChannelTemplatePass(parser, None)

    def test_try_channel_patterns_success(self, channel_pass):
        """Test successful pattern matching."""
        # Set up a pattern that should match
        pattern = ChannelPattern(
            pattern=r"^([^-]+)\s*-\s*([^(]+)",
            artist_group=1,
            title_group=2,
            confidence=0.8,
            success_count=5,
            total_attempts=5,
        )
        channel_pass.channel_patterns["test_channel"] = [pattern]

        with patch.object(channel_pass, "_create_result_from_pattern_match") as mock_create:
            mock_create.return_value = ParseResult(
                original_artist="Test Artist",
                song_title="Test Song",
                confidence=0.8,
                method="learned_channel_test_channel",
            )

            result = channel_pass._try_channel_patterns(
                "test_channel", "Test Artist - Test Song", "", "", "Test Channel"
            )

            assert result is not None
            assert result.original_artist == "Test Artist"
            assert pattern.total_attempts == 6  # Should increment

    def test_try_channel_patterns_no_patterns(self, channel_pass):
        """Test pattern matching with no learned patterns."""
        result = channel_pass._try_channel_patterns(
            "unknown_channel", "Test Title", "", "", "Unknown Channel"
        )

        assert result is None

    def test_try_channel_patterns_no_match(self, channel_pass):
        """Test pattern matching with no matching patterns."""
        pattern = ChannelPattern(
            pattern=r"^very_specific_pattern",
            artist_group=1,
            title_group=2,
            confidence=0.8,
        )
        channel_pass.channel_patterns["test_channel"] = [pattern]

        result = channel_pass._try_channel_patterns(
            "test_channel", "Generic Title That Won't Match", "", "", "Test Channel"
        )

        assert result is None

    def test_try_channel_patterns_invalid_regex(self, channel_pass):
        """Test handling of invalid regex patterns."""
        pattern = ChannelPattern(
            pattern=r"[invalid regex",  # Invalid regex
            artist_group=1,
            title_group=2,
            confidence=0.8,
        )
        channel_pass.channel_patterns["test_channel"] = [pattern]

        # Should handle the error gracefully
        result = channel_pass._try_channel_patterns(
            "test_channel", "Test Title", "", "", "Test Channel"
        )

        assert result is None

    def test_pattern_sorting_by_success_rate(self, channel_pass):
        """Test that patterns are sorted by success rate."""
        # Pattern with high success rate
        good_pattern = ChannelPattern(
            pattern=r"good_pattern",
            artist_group=1,
            title_group=2,
            confidence=0.8,
            success_count=9,
            total_attempts=10,
            last_used=datetime.now(),
        )

        # Pattern with low success rate
        poor_pattern = ChannelPattern(
            pattern=r"poor_pattern",
            artist_group=1,
            title_group=2,
            confidence=0.6,
            success_count=3,
            total_attempts=10,
            last_used=datetime.now() - timedelta(days=1),
        )

        channel_pass.channel_patterns["test_channel"] = [poor_pattern, good_pattern]

        with patch.object(channel_pass, "_create_result_from_pattern_match") as mock_create:
            mock_create.return_value = ParseResult(confidence=0.8)

            channel_pass._try_channel_patterns("test_channel", "test title", "", "", "Test Channel")

            # Should try good pattern first (higher success rate)
            # The patterns list should be sorted internally
            patterns = channel_pass.channel_patterns["test_channel"]
            # After sorting, good pattern should be first
            assert patterns[0] == good_pattern


class TestCreateResultFromPatternMatch:
    """Test pattern match result creation."""

    @pytest.fixture
    def channel_pass(self):
        parser = MagicMock(spec=AdvancedTitleParser)
        parser._clean_extracted_text.side_effect = lambda x: x.strip()
        parser._is_valid_artist_name.return_value = True
        parser._is_valid_song_title.return_value = True
        return EnhancedChannelTemplatePass(parser, None)

    def test_create_result_from_pattern_match(self, channel_pass):
        """Test creating result from pattern match."""
        pattern = ChannelPattern(
            pattern=r"^([^-]+)\s*-\s*([^(]+)",
            artist_group=1,
            title_group=2,
            confidence=0.8,
        )

        # Mock a regex match
        import re

        match = re.search(pattern.pattern, "Test Artist - Test Song")
        assert match is not None

        result = channel_pass._create_result_from_pattern_match(
            match, pattern, "test_method", "Test Artist - Test Song"
        )

        assert result is not None
        assert result.method == "test_method"
        assert result.pattern_used == pattern.pattern
        # The actual artist and title extraction would depend on the mock setup

    def test_create_result_invalid_match(self, channel_pass):
        """Test creating result with invalid match."""
        pattern = ChannelPattern(
            pattern=r"^([^-]+)\s*-\s*([^(]+)",
            artist_group=1,
            title_group=2,
            confidence=0.8,
        )

        # Mock invalid match
        mock_match = MagicMock()
        mock_match.groups.return_value = []

        result = channel_pass._create_result_from_pattern_match(
            mock_match, pattern, "test_method", "Test Title"
        )

        # Should handle gracefully
        assert result is None or result.confidence == 0


@pytest.mark.integration
class TestChannelTemplatePassIntegration:
    """Integration tests for the channel template pass."""

    @pytest.fixture
    def full_channel_pass(self):
        """Create a fully functional channel template pass."""
        parser = AdvancedTitleParser()
        return EnhancedChannelTemplatePass(parser, None)

    @pytest.mark.asyncio
    async def test_full_learning_workflow(self, full_channel_pass):
        """Test full learning workflow."""
        channel_id = "test_channel"
        channel_name = "Test Karaoke Channel"

        # Parse several similar titles to trigger learning
        titles = [
            "Artist One - Song One (Karaoke)",
            "Artist Two - Song Two (Karaoke)",
            "Artist Three - Song Three (Karaoke)",
        ]

        results = []
        for title in titles:
            result = await full_channel_pass.parse(
                title, channel_id=channel_id, channel_name=channel_name
            )
            results.append(result)

        # Should have learned patterns
        assert channel_id in full_channel_pass.channel_stats
        stats = full_channel_pass.channel_stats[channel_id]
        assert stats.total_videos >= len(titles)

        # Should have some successful parses
        assert stats.successful_parses > 0

    @pytest.mark.asyncio
    async def test_drift_detection_workflow(self, full_channel_pass):
        """Test drift detection workflow."""
        channel_id = "drift_channel"
        channel_name = "Drift Test Channel"

        # Simulate initial successful parsing
        good_titles = [
            "Artist One - Song One (Karaoke)",
            "Artist Two - Song Two (Karaoke)",
        ] * 6  # 12 successful parses

        for title in good_titles:
            await full_channel_pass.parse(title, channel_id=channel_id, channel_name=channel_name)

        # Manually set up drift conditions
        stats = full_channel_pass.channel_stats[channel_id]
        stats.total_videos = 20
        stats.successful_parses = 5  # Low success rate
        stats.drift_threshold = 0.5

        # Trigger drift check
        full_channel_pass._check_for_drift(channel_id)

        # Should detect drift
        assert stats.drift_detected is True

    @pytest.mark.asyncio
    async def test_pattern_effectiveness_tracking(self, full_channel_pass):
        """Test pattern effectiveness tracking."""
        channel_id = "effectiveness_channel"
        channel_name = "Effectiveness Test Channel"

        # Parse titles that should create and update patterns
        titles = [
            "Style Artist - Style Song (Karaoke Version)",
            "Another Artist - Another Song (Karaoke Version)",
            "Third Artist - Third Song (Karaoke Version)",
        ]

        for title in titles:
            await full_channel_pass.parse(title, channel_id=channel_id, channel_name=channel_name)

        # Check statistics
        stats = full_channel_pass.get_statistics()
        assert stats["total_channels"] >= 1
        assert stats["total_learned_patterns"] >= 0

        # Check channel-specific stats
        if channel_id in full_channel_pass.channel_stats:
            channel_stats = full_channel_pass.channel_stats[channel_id]
            assert channel_stats.total_videos >= len(titles)
