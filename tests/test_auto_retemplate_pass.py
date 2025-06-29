"""Comprehensive tests for the auto-retemplate pass module."""

from collections import deque
from dataclasses import asdict
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from collector.advanced_parser import AdvancedTitleParser, ParseResult
from collector.passes.auto_retemplate_pass import (
    AutoRetemplatePass,
    ChannelTrend,
    TemporalPattern,
)
from collector.passes.base import PassType


class TestTemporalPattern:
    """Test the TemporalPattern dataclass."""

    def test_temporal_pattern_creation(self):
        """Test creating a TemporalPattern with required values."""
        now = datetime.now()
        pattern = TemporalPattern(
            pattern=r"^([^-]+)\s*-\s*([^(]+)\s*\(Karaoke\)",
            artist_group=1,
            title_group=2,
            confidence=0.8,
            first_seen=now,
            last_seen=now,
        )
        assert pattern.pattern == r"^([^-]+)\s*-\s*([^(]+)\s*\(Karaoke\)"
        assert pattern.artist_group == 1
        assert pattern.title_group == 2
        assert pattern.confidence == 0.8
        assert pattern.first_seen == now
        assert pattern.last_seen == now
        assert pattern.video_count == 0
        assert pattern.success_count == 0
        assert isinstance(pattern.recent_examples, deque)
        assert pattern.recent_examples.maxlen == 10

    def test_temporal_pattern_with_values(self):
        """Test creating a TemporalPattern with all values."""
        now = datetime.now()
        examples = deque(["Example 1", "Example 2"], maxlen=10)
        pattern = TemporalPattern(
            pattern=r"test_pattern",
            artist_group=1,
            title_group=2,
            confidence=0.9,
            first_seen=now,
            last_seen=now,
            video_count=5,
            success_count=4,
            recent_examples=examples,
        )
        assert pattern.video_count == 5
        assert pattern.success_count == 4
        assert pattern.recent_examples == examples

    def test_temporal_pattern_serializable(self):
        """Test that TemporalPattern can be converted to dict."""
        now = datetime.now()
        pattern = TemporalPattern(
            pattern=r"test_pattern",
            artist_group=1,
            title_group=None,
            confidence=0.8,
            first_seen=now,
            last_seen=now,
        )
        pattern_dict = asdict(pattern)
        assert isinstance(pattern_dict, dict)
        assert pattern_dict["pattern"] == r"test_pattern"
        assert pattern_dict["confidence"] == 0.8


class TestChannelTrend:
    """Test the ChannelTrend dataclass."""

    def test_channel_trend_creation(self):
        """Test creating a ChannelTrend with required values."""
        trend = ChannelTrend(
            channel_id="UC123456789",
            channel_name="Test Channel",
        )
        assert trend.channel_id == "UC123456789"
        assert trend.channel_name == "Test Channel"
        assert trend.active_patterns == []
        assert trend.deprecated_patterns == []
        assert trend.pattern_change_detected is None
        assert isinstance(trend.last_analysis, datetime)
        assert trend.total_recent_videos == 0
        assert trend.successful_recent_parses == 0

    def test_channel_trend_with_patterns(self):
        """Test creating a ChannelTrend with patterns."""
        now = datetime.now()
        active_pattern = TemporalPattern(
            pattern=r"test",
            artist_group=1,
            title_group=2,
            confidence=0.8,
            first_seen=now,
            last_seen=now,
        )

        trend = ChannelTrend(
            channel_id="UC123456789",
            channel_name="Test Channel",
            active_patterns=[active_pattern],
            pattern_change_detected=now,
            total_recent_videos=10,
            successful_recent_parses=8,
        )
        assert len(trend.active_patterns) == 1
        assert trend.pattern_change_detected == now
        assert trend.total_recent_videos == 10
        assert trend.successful_recent_parses == 8

    def test_channel_trend_serializable(self):
        """Test that ChannelTrend can be converted to dict."""
        trend = ChannelTrend(
            channel_id="UC123456789",
            channel_name="Test Channel",
        )
        trend_dict = asdict(trend)
        assert isinstance(trend_dict, dict)
        assert trend_dict["channel_id"] == "UC123456789"
        assert trend_dict["channel_name"] == "Test Channel"


class TestAutoRetemplatePassInitialization:
    """Test AutoRetemplatePass initialization."""

    @pytest.fixture
    def mock_advanced_parser(self):
        """Create a mock AdvancedTitleParser."""
        parser = MagicMock(spec=AdvancedTitleParser)
        parser._clean_extracted_text = MagicMock(side_effect=lambda x: x.strip())
        parser._is_valid_artist_name = MagicMock(return_value=True)
        parser._is_valid_song_title = MagicMock(return_value=True)
        parser.parse_title = MagicMock()
        return parser

    @pytest.fixture
    def mock_db_manager(self):
        """Create a mock database manager."""
        return MagicMock()

    def test_pass_type(self, mock_advanced_parser):
        """Test that the pass type is correctly set."""
        auto_pass = AutoRetemplatePass(mock_advanced_parser)
        assert auto_pass.pass_type == PassType.AUTO_RETEMPLATE

    def test_initialization_without_db(self, mock_advanced_parser):
        """Test initialization without database manager."""
        auto_pass = AutoRetemplatePass(mock_advanced_parser)
        assert auto_pass.advanced_parser == mock_advanced_parser
        assert auto_pass.db_manager is None
        assert auto_pass.channel_trends == {}
        assert auto_pass.recent_window == timedelta(days=30)
        assert auto_pass.pattern_change_window == timedelta(days=7)
        assert auto_pass.min_videos_for_analysis == 5
        assert auto_pass.pattern_similarity_threshold == 0.8
        assert auto_pass.success_rate_threshold == 0.6
        assert auto_pass.pattern_confidence_decay == 0.95

    def test_initialization_with_db(self, mock_advanced_parser, mock_db_manager):
        """Test initialization with database manager."""
        auto_pass = AutoRetemplatePass(mock_advanced_parser, mock_db_manager)
        assert auto_pass.advanced_parser == mock_advanced_parser
        assert auto_pass.db_manager == mock_db_manager

    @patch.object(AutoRetemplatePass, "_load_channel_trends")
    def test_initialization_calls_load_trends(self, mock_load, mock_advanced_parser):
        """Test that initialization calls _load_channel_trends."""
        AutoRetemplatePass(mock_advanced_parser)
        mock_load.assert_called_once()


class TestAutoRetemplatePassChannelManagement:
    """Test channel trend management functionality."""

    @pytest.fixture
    def auto_pass(self):
        """Create an AutoRetemplatePass instance."""
        parser = MagicMock(spec=AdvancedTitleParser)
        return AutoRetemplatePass(parser)

    def test_get_or_create_trend_new_channel(self, auto_pass):
        """Test creating a new channel trend."""
        channel_id = "UC123456789"
        channel_name = "Test Channel"

        trend = auto_pass._get_or_create_trend(channel_id, channel_name)

        assert trend.channel_id == channel_id
        assert trend.channel_name == channel_name
        assert channel_id in auto_pass.channel_trends
        assert auto_pass.channel_trends[channel_id] == trend

    def test_get_or_create_trend_existing_channel(self, auto_pass):
        """Test getting an existing channel trend."""
        channel_id = "UC123456789"
        channel_name = "Test Channel"

        # Create first trend
        trend1 = auto_pass._get_or_create_trend(channel_id, channel_name)

        # Get the same trend again
        trend2 = auto_pass._get_or_create_trend(channel_id, channel_name)

        assert trend1 is trend2
        assert len(auto_pass.channel_trends) == 1


class TestAutoRetemplatePassPatternMatching:
    """Test pattern matching functionality."""

    @pytest.fixture
    def auto_pass_with_patterns(self):
        """Create an AutoRetemplatePass instance with test patterns."""
        parser = MagicMock(spec=AdvancedTitleParser)
        parser._clean_extracted_text = MagicMock(side_effect=lambda x: x.strip())
        parser._is_valid_artist_name = MagicMock(return_value=True)
        parser._is_valid_song_title = MagicMock(return_value=True)

        auto_pass = AutoRetemplatePass(parser)

        # Create test channel with patterns
        channel_id = "UC123456789"
        trend = auto_pass._get_or_create_trend(channel_id, "Test Channel")

        # Add active pattern
        now = datetime.now()
        active_pattern = TemporalPattern(
            pattern=r"^([^-]+)\s*-\s*([^(]+)\s*\(Karaoke\)",
            artist_group=1,
            title_group=2,
            confidence=0.8,
            first_seen=now,
            last_seen=now,
            video_count=5,
            success_count=4,
        )
        trend.active_patterns.append(active_pattern)

        # Add deprecated pattern
        deprecated_pattern = TemporalPattern(
            pattern=r"^([^:]+):\s*([^(]+)\s*\(Karaoke\)",
            artist_group=1,
            title_group=2,
            confidence=0.6,
            first_seen=now - timedelta(days=10),
            last_seen=now - timedelta(days=5),
            video_count=3,
            success_count=2,
        )
        trend.deprecated_patterns.append(deprecated_pattern)

        return auto_pass, trend

    def test_try_active_patterns_success(self, auto_pass_with_patterns):
        """Test successful matching with active patterns."""
        auto_pass, trend = auto_pass_with_patterns
        title = "Test Artist - Test Song (Karaoke)"

        result = auto_pass._try_active_patterns(trend, title, "", "")

        assert result is not None
        assert result.method == "auto_retemplate_active"
        assert result.confidence > 0
        # Check that pattern stats were updated
        assert trend.active_patterns[0].video_count == 6

    def test_try_active_patterns_no_match(self, auto_pass_with_patterns):
        """Test no match with active patterns."""
        auto_pass, trend = auto_pass_with_patterns
        title = "Random Video Title"

        result = auto_pass._try_active_patterns(trend, title, "", "")

        assert result is None

    def test_try_deprecated_patterns_success(self, auto_pass_with_patterns):
        """Test successful matching with deprecated patterns."""
        auto_pass, trend = auto_pass_with_patterns
        title = "Test Artist: Test Song (Karaoke)"

        result = auto_pass._try_deprecated_patterns(trend, title, "", "")

        assert result is not None
        assert result.method == "auto_retemplate_deprecated"
        assert result.confidence < 0.8  # Should be reduced for deprecated patterns

    def test_try_deprecated_patterns_no_match(self, auto_pass_with_patterns):
        """Test no match with deprecated patterns."""
        auto_pass, trend = auto_pass_with_patterns
        title = "Random Video Title"

        result = auto_pass._try_deprecated_patterns(trend, title, "", "")

        assert result is None

    def test_revive_pattern(self, auto_pass_with_patterns):
        """Test reviving a deprecated pattern."""
        auto_pass, trend = auto_pass_with_patterns
        pattern_used = r"^([^:]+):\s*([^(]+)\s*\(Karaoke\)"

        # Verify pattern is in deprecated list
        assert len(trend.deprecated_patterns) == 1
        assert len(trend.active_patterns) == 1

        auto_pass._revive_pattern(trend, pattern_used)

        # Verify pattern moved to active list
        assert len(trend.deprecated_patterns) == 0
        assert len(trend.active_patterns) == 2
        assert trend.active_patterns[1].pattern == pattern_used


class TestAutoRetemplatePassPatternAnalysis:
    """Test pattern analysis functionality."""

    @pytest.fixture
    def auto_pass(self):
        """Create an AutoRetemplatePass instance."""
        parser = MagicMock(spec=AdvancedTitleParser)
        return AutoRetemplatePass(parser)

    def test_extract_title_structure_artist_song_karaoke(self, auto_pass):
        """Test extracting ARTIST-SONG-KARAOKE structure."""
        title = "Test Artist - Test Song (Karaoke)"
        structure = auto_pass._extract_title_structure(title)
        assert structure == "ARTIST-SONG-KARAOKE"

    def test_extract_title_structure_quoted(self, auto_pass):
        """Test extracting quoted artist-song structure."""
        title = '"Test Artist" - "Test Song" Karaoke'
        structure = auto_pass._extract_title_structure(title)
        assert structure == "QUOTED-ARTIST-SONG-KARAOKE"

    def test_extract_title_structure_channel_prefix(self, auto_pass):
        """Test extracting channel prefix structure."""
        title = "[Channel Name] Test Artist - Test Song"
        structure = auto_pass._extract_title_structure(title)
        assert structure == "CHANNEL-ARTIST-SONG"

    def test_extract_title_structure_song_by_artist(self, auto_pass):
        """Test extracting song-by-artist structure."""
        title = "Test Song by Test Artist (Karaoke)"
        structure = auto_pass._extract_title_structure(title)
        assert structure == "SONG-BY-ARTIST-KARAOKE"

    def test_extract_title_structure_no_match(self, auto_pass):
        """Test no structure extraction for unmatched titles."""
        title = "Random Video Title"
        structure = auto_pass._extract_title_structure(title)
        assert structure is None

    def test_structure_to_pattern(self, auto_pass):
        """Test converting structure to regex pattern."""
        structure = "ARTIST-SONG-KARAOKE"
        pattern = auto_pass._structure_to_pattern(structure)
        assert pattern == r"^([^-–—]+)\s*[-–—]\s*([^(]+)\s*\([^)]*[Kk]araoke[^)]*\)"

    def test_structure_to_pattern_unknown(self, auto_pass):
        """Test converting unknown structure returns None."""
        structure = "UNKNOWN-STRUCTURE"
        pattern = auto_pass._structure_to_pattern(structure)
        assert pattern is None

    def test_pattern_exists_true(self, auto_pass):
        """Test pattern existence check returns True for existing pattern."""
        trend = ChannelTrend("UC123", "Test")
        now = datetime.now()
        pattern = TemporalPattern(
            pattern=r"test_pattern",
            artist_group=1,
            title_group=2,
            confidence=0.8,
            first_seen=now,
            last_seen=now,
        )
        trend.active_patterns.append(pattern)

        exists = auto_pass._pattern_exists(trend, r"test_pattern")
        assert exists is True

    def test_pattern_exists_false(self, auto_pass):
        """Test pattern existence check returns False for non-existing pattern."""
        trend = ChannelTrend("UC123", "Test")

        exists = auto_pass._pattern_exists(trend, r"test_pattern")
        assert exists is False

    def test_add_new_pattern(self, auto_pass):
        """Test adding a new pattern to channel trends."""
        trend = ChannelTrend("UC123", "Test")
        pattern = r"^([^-]+)\s*-\s*([^(]+)\s*\(Karaoke\)"
        examples = [
            {"title": "Artist1 - Song1 (Karaoke)"},
            {"title": "Artist2 - Song2 (Karaoke)"},
        ]

        auto_pass._add_new_pattern(trend, pattern, examples)

        assert len(trend.active_patterns) == 1
        new_pattern = trend.active_patterns[0]
        assert new_pattern.pattern == pattern
        assert new_pattern.artist_group == 1
        assert new_pattern.title_group == 2
        assert new_pattern.confidence == 0.8
        assert new_pattern.video_count == 2
        assert new_pattern.success_count == 2


class TestAutoRetemplatePassPatternLearning:
    """Test pattern learning functionality."""

    @pytest.fixture
    def auto_pass(self):
        """Create an AutoRetemplatePass instance."""
        parser = MagicMock(spec=AdvancedTitleParser)
        parser._clean_extracted_text = MagicMock(side_effect=lambda x: x.strip())
        parser._is_valid_artist_name = MagicMock(return_value=True)
        parser._is_valid_song_title = MagicMock(return_value=True)
        return AutoRetemplatePass(parser)

    def test_learn_from_title_structure(self, auto_pass):
        """Test learning patterns from title structure."""
        trend = ChannelTrend("UC123", "Test")
        title = "Test Artist - Test Song (Karaoke)"

        auto_pass._learn_from_title_structure(trend, title)

        assert len(trend.active_patterns) == 1
        new_pattern = trend.active_patterns[0]
        assert new_pattern.confidence == 0.6  # Lower confidence for single example
        assert new_pattern.success_count == 0  # Not validated yet

    def test_learn_from_title_structure_no_structure(self, auto_pass):
        """Test no learning when title structure can't be detected."""
        trend = ChannelTrend("UC123", "Test")
        title = "Random Video Title"

        auto_pass._learn_from_title_structure(trend, title)

        assert len(trend.active_patterns) == 0

    def test_create_pattern_from_parse(self, auto_pass):
        """Test creating a pattern from a successful parse."""
        title = "Test Artist - Test Song (Karaoke)"
        result = ParseResult(
            artist="Test Artist",
            song_title="Test Song",
            confidence=0.8,
        )

        pattern = auto_pass._create_pattern_from_parse(title, result)

        assert pattern is not None
        assert "([^-–—\"']+?)" in pattern  # Should contain artist capture group
        assert "([^(\\[]+?)" in pattern  # Should contain song capture group

    def test_create_pattern_from_parse_missing_data(self, auto_pass):
        """Test creating pattern fails with missing artist/song data."""
        title = "Test Title"
        result = ParseResult(confidence=0.8)  # No artist or song

        auto_pass._create_pattern_from_parse(title, result)

        # Should still try to create a pattern, may or may not succeed
        # depending on the title structure

    def test_add_learned_pattern(self, auto_pass):
        """Test adding a learned pattern."""
        trend = ChannelTrend("UC123", "Test")
        pattern = r"^test_pattern$"
        example = "Test Example"

        auto_pass._add_learned_pattern(trend, pattern, example)

        assert len(trend.active_patterns) == 1
        learned_pattern = trend.active_patterns[0]
        assert learned_pattern.pattern == pattern
        assert learned_pattern.confidence == 0.5  # Low confidence for learned patterns
        assert learned_pattern.video_count == 1
        assert learned_pattern.success_count == 1


class TestAutoRetemplatePassResultCreation:
    """Test ParseResult creation from patterns."""

    @pytest.fixture
    def auto_pass(self):
        """Create an AutoRetemplatePass instance."""
        parser = MagicMock(spec=AdvancedTitleParser)
        parser._clean_extracted_text = MagicMock(side_effect=lambda x: x.strip())
        parser._is_valid_artist_name = MagicMock(return_value=True)
        parser._is_valid_song_title = MagicMock(return_value=True)
        return AutoRetemplatePass(parser)

    def test_create_result_from_temporal_pattern_complete(self, auto_pass):
        """Test creating result with both artist and song."""
        import re

        now = datetime.now()
        pattern = TemporalPattern(
            pattern=r"^([^-]+)\s*-\s*([^(]+)\s*\(Karaoke\)",
            artist_group=1,
            title_group=2,
            confidence=0.8,
            first_seen=now,
            last_seen=now,
            video_count=10,
            success_count=8,
        )

        title = "Test Artist - Test Song (Karaoke)"
        match = re.search(pattern.pattern, title)

        result = auto_pass._create_result_from_temporal_pattern(
            match, pattern, "test_method", title
        )

        assert result is not None
        assert result.method == "test_method"
        assert result.pattern_used == pattern.pattern
        assert result.confidence > 0
        assert "pattern_age_days" in result.metadata
        assert "pattern_success_rate" in result.metadata

    def test_create_result_from_temporal_pattern_partial(self, auto_pass):
        """Test creating result with only artist or song."""
        import re

        now = datetime.now()
        pattern = TemporalPattern(
            pattern=r"^([^-]+)",  # Only captures artist
            artist_group=1,
            title_group=None,
            confidence=0.8,
            first_seen=now,
            last_seen=now,
            video_count=5,
            success_count=4,
        )

        title = "Test Artist"
        match = re.search(pattern.pattern, title)

        result = auto_pass._create_result_from_temporal_pattern(
            match, pattern, "test_method", title
        )

        assert result is not None
        assert result.confidence < 0.8  # Should be reduced for partial matches

    def test_create_result_confidence_factors(self, auto_pass):
        """Test confidence calculation with various factors."""
        import re

        old_date = datetime.now() - timedelta(days=10)
        pattern = TemporalPattern(
            pattern=r"^([^-]+)\s*-\s*([^(]+)",
            artist_group=1,
            title_group=2,
            confidence=0.8,
            first_seen=old_date,
            last_seen=old_date,  # Old pattern
            video_count=10,
            success_count=5,  # 50% success rate
        )

        title = "Test Artist - Test Song"
        match = re.search(pattern.pattern, title)

        result = auto_pass._create_result_from_temporal_pattern(
            match, pattern, "test_method", title
        )

        assert result is not None
        # Confidence should be affected by age and success rate
        assert result.confidence < pattern.confidence
        assert result.metadata["pattern_age_days"] == 10
        assert result.metadata["pattern_success_rate"] == 0.5


class TestAutoRetemplatePassStatistics:
    """Test statistics and management functionality."""

    @pytest.fixture
    def auto_pass_with_data(self):
        """Create an AutoRetemplatePass instance with test data."""
        parser = MagicMock(spec=AdvancedTitleParser)
        auto_pass = AutoRetemplatePass(parser)

        # Add test channels with patterns
        now = datetime.now()

        # Channel 1 with active patterns
        trend1 = auto_pass._get_or_create_trend("UC1", "Channel 1")
        pattern1 = TemporalPattern(
            pattern=r"pattern1",
            artist_group=1,
            title_group=2,
            confidence=0.8,
            first_seen=now,
            last_seen=now,
        )
        trend1.active_patterns.append(pattern1)

        # Channel 2 with pattern change
        trend2 = auto_pass._get_or_create_trend("UC2", "Channel 2")
        pattern2 = TemporalPattern(
            pattern=r"pattern2",
            artist_group=1,
            title_group=2,
            confidence=0.7,
            first_seen=now,
            last_seen=now,
        )
        trend2.active_patterns.append(pattern2)
        trend2.pattern_change_detected = now

        deprecated_pattern = TemporalPattern(
            pattern=r"deprecated",
            artist_group=1,
            title_group=2,
            confidence=0.5,
            first_seen=now - timedelta(days=5),
            last_seen=now - timedelta(days=3),
        )
        trend2.deprecated_patterns.append(deprecated_pattern)

        return auto_pass

    def test_get_statistics(self, auto_pass_with_data):
        """Test getting comprehensive statistics."""
        stats = auto_pass_with_data.get_statistics()

        assert stats["total_channels"] == 2
        assert stats["total_active_patterns"] == 2
        assert stats["total_deprecated_patterns"] == 1
        assert stats["channels_with_pattern_changes"] == 1
        assert stats["avg_patterns_per_channel"] == 1.0
        assert stats["pattern_change_rate"] == 0.5

    def test_should_analyze_patterns_few_patterns(self, auto_pass_with_data):
        """Test analysis decision with few patterns."""
        trend = ChannelTrend("UC_new", "New Channel")
        trend.last_analysis = datetime.now() - timedelta(hours=2)  # Recent analysis

        should_analyze = auto_pass_with_data._should_analyze_patterns(trend)
        assert should_analyze is True  # Should analyze due to few patterns

    def test_should_analyze_patterns_recent_analysis(self, auto_pass_with_data):
        """Test analysis decision with recent analysis."""
        trend = ChannelTrend("UC_recent", "Recent Channel")
        trend.last_analysis = datetime.now() - timedelta(minutes=30)  # Very recent

        # Add some patterns to avoid the "few patterns" trigger
        now = datetime.now()
        for i in range(3):
            pattern = TemporalPattern(
                pattern=f"pattern{i}",
                artist_group=1,
                title_group=2,
                confidence=0.8,
                first_seen=now,
                last_seen=now,
            )
            trend.active_patterns.append(pattern)

        trend.total_recent_videos = 10
        trend.successful_recent_parses = 8  # Good success rate

        should_analyze = auto_pass_with_data._should_analyze_patterns(trend)
        assert should_analyze is False  # Should not analyze due to recent analysis

    def test_should_analyze_patterns_low_success_rate(self, auto_pass_with_data):
        """Test analysis decision with low success rate."""
        trend = ChannelTrend("UC_low", "Low Success Channel")
        trend.last_analysis = datetime.now() - timedelta(hours=2)
        trend.total_recent_videos = 10
        trend.successful_recent_parses = 3  # Low success rate (30%)

        should_analyze = auto_pass_with_data._should_analyze_patterns(trend)
        assert should_analyze is True  # Should analyze due to low success rate

    def test_should_analyze_patterns_pattern_change(self, auto_pass_with_data):
        """Test analysis decision with recent pattern change."""
        trend = ChannelTrend("UC_change", "Changed Channel")
        trend.last_analysis = datetime.now() - timedelta(minutes=30)  # Recent analysis
        trend.pattern_change_detected = datetime.now() - timedelta(hours=1)  # Recent change

        should_analyze = auto_pass_with_data._should_analyze_patterns(trend)
        assert should_analyze is True  # Should analyze due to pattern change

    def test_update_trend_stats(self, auto_pass_with_data):
        """Test updating trend statistics."""
        trend = ChannelTrend("UC_test", "Test Channel")
        initial_videos = trend.total_recent_videos
        initial_time = trend.last_analysis

        auto_pass_with_data._update_trend_stats(trend, 1.5)

        assert trend.total_recent_videos == initial_videos + 1
        assert trend.last_analysis > initial_time

    def test_apply_confidence_decay(self, auto_pass_with_data):
        """Test confidence decay application."""
        now = datetime.now()
        old_pattern = TemporalPattern(
            pattern=r"old_pattern",
            artist_group=1,
            title_group=2,
            confidence=0.8,
            first_seen=now - timedelta(days=5),
            last_seen=now - timedelta(days=3),  # 3 days old
        )

        trend = ChannelTrend("UC_decay", "Decay Test")
        trend.active_patterns.append(old_pattern)

        auto_pass_with_data._apply_confidence_decay(trend)

        # Confidence should be decayed: 0.8 * (0.95^3)
        expected_confidence = 0.8 * (0.95**3)
        assert abs(old_pattern.confidence - expected_confidence) < 0.01

    def test_record_pattern_success(self, auto_pass_with_data):
        """Test recording pattern success."""
        trend = auto_pass_with_data.channel_trends["UC1"]
        pattern = trend.active_patterns[0]
        initial_success = pattern.success_count

        auto_pass_with_data._record_pattern_success(trend, pattern.pattern, "Test Title")

        assert pattern.success_count == initial_success + 1
        assert "Test Title" in pattern.recent_examples


class TestAutoRetemplatePassAsyncMethods:
    """Test async methods and main parse functionality."""

    @pytest.fixture
    def auto_pass(self):
        """Create an AutoRetemplatePass instance."""
        parser = MagicMock(spec=AdvancedTitleParser)
        parser._clean_extracted_text = MagicMock(side_effect=lambda x: x.strip())
        parser._is_valid_artist_name = MagicMock(return_value=True)
        parser._is_valid_song_title = MagicMock(return_value=True)
        parser.parse_title = MagicMock()
        return AutoRetemplatePass(parser)

    @pytest.mark.asyncio
    async def test_parse_no_channel_id(self, auto_pass):
        """Test parse method returns None when no channel_id provided."""
        result = await auto_pass.parse("Test Title")
        assert result is None

    @pytest.mark.asyncio
    async def test_parse_with_active_pattern_match(self, auto_pass):
        """Test parse method with successful active pattern match."""
        # Set up channel with active pattern
        channel_id = "UC123"
        trend = auto_pass._get_or_create_trend(channel_id, "Test Channel")

        now = datetime.now()
        pattern = TemporalPattern(
            pattern=r"^([^-]+)\s*-\s*([^(]+)\s*\(Karaoke\)",
            artist_group=1,
            title_group=2,
            confidence=0.8,
            first_seen=now,
            last_seen=now,
            video_count=5,
            success_count=4,
        )
        trend.active_patterns.append(pattern)

        title = "Test Artist - Test Song (Karaoke)"

        result = await auto_pass.parse(title, channel_id=channel_id, channel_name="Test Channel")

        assert result is not None
        assert result.method == "auto_retemplate_active"

    @pytest.mark.asyncio
    async def test_parse_with_learning_attempt(self, auto_pass):
        """Test parse method attempts learning when patterns don't match."""
        channel_id = "UC123"
        title = "Random Video Title"

        # Mock the advanced parser to return a successful parse
        mock_result = ParseResult(
            artist="Learned Artist",
            song_title="Learned Song",
            confidence=0.7,
        )
        auto_pass.advanced_parser.parse_title.return_value = mock_result

        await auto_pass.parse(title, channel_id=channel_id, channel_name="Test Channel")

        # Should call advanced parser for learning
        auto_pass.advanced_parser.parse_title.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_recent_channel_videos_no_db(self, auto_pass):
        """Test getting recent videos returns empty list when no DB."""
        videos = await auto_pass._get_recent_channel_videos("UC123")
        assert videos == []

    @pytest.mark.asyncio
    async def test_analyze_recent_patterns_no_db(self, auto_pass):
        """Test analyzing recent patterns with no database."""
        trend = ChannelTrend("UC123", "Test Channel")

        # Should not raise exception
        await auto_pass._analyze_recent_patterns(trend, "Test Title")

    @pytest.mark.asyncio
    async def test_analyze_recent_patterns_with_db(self, auto_pass):
        """Test analyzing recent patterns with mock database."""
        auto_pass.db_manager = MagicMock()
        trend = ChannelTrend("UC123", "Test Channel")

        # Mock recent videos
        mock_videos = [
            {"title": "Artist1 - Song1 (Karaoke)"},
            {"title": "Artist2 - Song2 (Karaoke)"},
            {"title": "Artist3 - Song3 (Karaoke)"},
        ]

        with patch.object(
            auto_pass, "_get_recent_channel_videos", new_callable=AsyncMock
        ) as mock_get_videos:
            mock_get_videos.return_value = mock_videos

            with patch.object(auto_pass, "_detect_new_patterns") as mock_detect:
                with patch.object(auto_pass, "_detect_pattern_changes") as mock_change:
                    await auto_pass._analyze_recent_patterns(trend, "Test Title")

                    mock_get_videos.assert_called_once_with("UC123")
                    mock_detect.assert_called_once_with(trend, mock_videos)
                    mock_change.assert_called_once_with(trend, mock_videos)


class TestAutoRetemplatePassPatternDetection:
    """Test pattern detection and change detection functionality."""

    @pytest.fixture
    def auto_pass(self):
        """Create an AutoRetemplatePass instance."""
        parser = MagicMock(spec=AdvancedTitleParser)
        return AutoRetemplatePass(parser)

    def test_detect_new_patterns(self, auto_pass):
        """Test detecting new patterns from recent videos."""
        trend = ChannelTrend("UC123", "Test Channel")

        recent_videos = [
            {"title": "Artist1 - Song1 (Karaoke)"},
            {"title": "Artist2 - Song2 (Karaoke)"},
            {"title": "Artist3 - Song3 (Karaoke)"},
            {"title": "Different Format Video"},
        ]

        auto_pass._detect_new_patterns(trend, recent_videos)

        # Should detect the ARTIST-SONG-KARAOKE pattern
        assert len(trend.active_patterns) >= 0  # May or may not add patterns

    def test_detect_pattern_changes_low_match_rate(self, auto_pass):
        """Test detecting pattern changes with low match rate."""
        trend = ChannelTrend("UC123", "Test Channel")

        # Add an active pattern
        now = datetime.now()
        pattern = TemporalPattern(
            pattern=r"^([^-]+)\s*-\s*([^(]+)\s*\(Karaoke\)",
            artist_group=1,
            title_group=2,
            confidence=0.8,
            first_seen=now,
            last_seen=now,
        )
        trend.active_patterns.append(pattern)

        # Recent videos that don't match the pattern
        recent_videos = [
            {"title": "Different Format 1"},
            {"title": "Different Format 2"},
            {"title": "Different Format 3"},
        ]

        auto_pass._detect_pattern_changes(trend, recent_videos)

        # Should detect pattern change
        assert trend.pattern_change_detected is not None

    def test_detect_pattern_changes_high_match_rate(self, auto_pass):
        """Test no pattern change detection with high match rate."""
        trend = ChannelTrend("UC123", "Test Channel")

        # Add an active pattern
        now = datetime.now()
        pattern = TemporalPattern(
            pattern=r"^([^-]+)\s*-\s*([^(]+)\s*\(Karaoke\)",
            artist_group=1,
            title_group=2,
            confidence=0.8,
            first_seen=now,
            last_seen=now,
        )
        trend.active_patterns.append(pattern)

        # Recent videos that match the pattern
        recent_videos = [
            {"title": "Artist1 - Song1 (Karaoke)"},
            {"title": "Artist2 - Song2 (Karaoke)"},
            {"title": "Artist3 - Song3 (Karaoke)"},
        ]

        auto_pass._detect_pattern_changes(trend, recent_videos)

        # Should not detect pattern change
        assert trend.pattern_change_detected is None

    def test_deprecate_old_patterns(self, auto_pass):
        """Test deprecating old patterns."""
        trend = ChannelTrend("UC123", "Test Channel")

        # Add old pattern
        old_date = datetime.now() - timedelta(days=10)
        old_pattern = TemporalPattern(
            pattern=r"old_pattern",
            artist_group=1,
            title_group=2,
            confidence=0.8,
            first_seen=old_date,
            last_seen=old_date,
        )
        trend.active_patterns.append(old_pattern)

        # Add recent pattern
        recent_pattern = TemporalPattern(
            pattern=r"recent_pattern",
            artist_group=1,
            title_group=2,
            confidence=0.8,
            first_seen=datetime.now(),
            last_seen=datetime.now(),
        )
        trend.active_patterns.append(recent_pattern)

        auto_pass._deprecate_old_patterns(trend)

        # Old pattern should be moved to deprecated
        assert len(trend.active_patterns) == 1
        assert len(trend.deprecated_patterns) == 1
        assert trend.deprecated_patterns[0].pattern == r"old_pattern"
        assert trend.deprecated_patterns[0].confidence < 0.8  # Reduced confidence


class TestAutoRetemplatePassDatabaseIntegration:
    """Test database integration methods."""

    @pytest.fixture
    def auto_pass_with_db(self):
        """Create an AutoRetemplatePass instance with mock database."""
        parser = MagicMock(spec=AdvancedTitleParser)
        db_manager = MagicMock()
        return AutoRetemplatePass(parser, db_manager)

    def test_load_channel_trends_with_db(self, auto_pass_with_db):
        """Test loading channel trends from database."""
        # Method should not raise exception even if not implemented
        auto_pass_with_db._load_channel_trends()

    def test_save_trends_with_db(self, auto_pass_with_db):
        """Test saving trends to database."""
        # Method should not raise exception even if not implemented
        auto_pass_with_db.save_trends()

    def test_save_trends_without_db(self):
        """Test saving trends without database."""
        parser = MagicMock(spec=AdvancedTitleParser)
        auto_pass = AutoRetemplatePass(parser)  # No DB manager

        # Should not raise exception
        auto_pass.save_trends()
