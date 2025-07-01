"""Unit tests for auto_retemplate_pass.py."""

import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from collector.advanced_parser import AdvancedTitleParser, ParseResult
from collector.passes.auto_retemplate_pass import AutoRetemplatePass, TemporalPattern


class TestAutoRetemplatePass:
    """Test cases for AutoRetemplatePass."""

    @pytest.fixture
    def advanced_parser(self):
        """Create a mock advanced parser."""
        parser = Mock(spec=AdvancedTitleParser)
        parser._clean_extracted_text = Mock(side_effect=lambda x: x.strip() if x else "")
        parser._is_valid_artist_name = Mock(return_value=True)
        parser._is_valid_song_title = Mock(return_value=True)
        parser.parse_title = Mock(
            return_value=ParseResult(artist="Test Artist", song_title="Test Song", confidence=0.6)
        )
        return parser

    @pytest.fixture
    def db_manager(self):
        """Create a mock database manager."""
        db = AsyncMock()
        return db

    @pytest.fixture
    def retemplate_pass(self, advanced_parser, db_manager):
        """Create an AutoRetemplatePass instance."""
        return AutoRetemplatePass(advanced_parser, db_manager)

    @pytest.mark.asyncio
    async def test_parse_with_channel_id(self, retemplate_pass):
        """Test parsing with channel ID."""
        result = await retemplate_pass.parse(
            title="Artist - Song (Karaoke)", channel_id="UC123", channel_name="Test Channel"
        )

        # Should create channel trend
        assert "UC123" in retemplate_pass.channel_trends
        assert result is not None

    @pytest.mark.asyncio
    async def test_parse_without_channel_id(self, retemplate_pass):
        """Test parsing without channel ID returns None."""
        result = await retemplate_pass.parse(title="Artist - Song (Karaoke)")

        assert result is None

    def test_create_channel_trend(self, retemplate_pass):
        """Test channel trend creation."""
        trend = retemplate_pass._get_or_create_trend("UC123", "Test Channel")

        assert trend.channel_id == "UC123"
        assert trend.channel_name == "Test Channel"
        assert len(trend.active_patterns) == 0

    @pytest.mark.asyncio
    async def test_active_pattern_matching(self, retemplate_pass):
        """Test matching against active patterns."""
        # Add a pattern to channel
        trend = retemplate_pass._get_or_create_trend("UC123", "Test Channel")
        pattern = TemporalPattern(
            pattern=r"^([^-]+)\s*-\s*([^(]+)\s*\(Karaoke\)",
            artist_group=1,
            title_group=2,
            confidence=0.8,
            first_seen=datetime.now(),
            last_seen=datetime.now(),
        )
        trend.active_patterns.append(pattern)

        result = await retemplate_pass._try_active_patterns(
            trend, "Test Artist - Test Song (Karaoke)", "", ""
        )

        assert result is not None
        assert result.artist == "Test Artist"
        assert result.song_title == "Test Song"

    def test_extract_title_structure(self, retemplate_pass):
        """Test title structure extraction."""
        structure = retemplate_pass._extract_title_structure("Artist - Song (Karaoke)")
        assert structure == "ARTIST-SONG-KARAOKE"

        structure2 = retemplate_pass._extract_title_structure('"Artist" - "Song" Karaoke')
        assert structure2 == "QUOTED-ARTIST-SONG-KARAOKE"

    def test_structure_to_pattern(self, retemplate_pass):
        """Test converting structure to regex pattern."""
        pattern = retemplate_pass._structure_to_pattern("ARTIST-SONG-KARAOKE")
        assert pattern is not None

        # Test pattern matches expected format
        match = re.search(pattern, "Test Artist - Test Song (Karaoke)", re.IGNORECASE)
        assert match is not None

    def test_add_new_pattern(self, retemplate_pass):
        """Test adding new pattern to channel."""
        trend = retemplate_pass._get_or_create_trend("UC123", "Test Channel")

        retemplate_pass._add_new_pattern(
            trend, r"^([^-]+)\s*-\s*([^(]+)", [{"title": "Example 1"}, {"title": "Example 2"}]
        )

        assert len(trend.active_patterns) == 1
        assert trend.active_patterns[0].video_count == 2

    def test_pattern_deprecation(self, retemplate_pass):
        """Test pattern deprecation based on age."""
        trend = retemplate_pass._get_or_create_trend("UC123", "Test Channel")

        # Add old pattern
        old_pattern = TemporalPattern(
            pattern="old_pattern",
            artist_group=1,
            title_group=2,
            confidence=0.8,
            first_seen=datetime.now() - timedelta(days=10),
            last_seen=datetime.now() - timedelta(days=8),
        )
        trend.active_patterns.append(old_pattern)

        retemplate_pass._deprecate_old_patterns(trend)

        assert len(trend.active_patterns) == 0
        assert len(trend.deprecated_patterns) == 1

    def test_pattern_revival(self, retemplate_pass):
        """Test reviving deprecated patterns."""
        trend = retemplate_pass._get_or_create_trend("UC123", "Test Channel")

        # Add deprecated pattern
        deprecated = TemporalPattern(
            pattern="deprecated_pattern",
            artist_group=1,
            title_group=2,
            confidence=0.5,
            first_seen=datetime.now(),
            last_seen=datetime.now(),
        )
        trend.deprecated_patterns.append(deprecated)

        retemplate_pass._revive_pattern(trend, "deprecated_pattern")

        assert len(trend.active_patterns) == 1
        assert len(trend.deprecated_patterns) == 0
        assert trend.active_patterns[0].confidence > 0.5

    @pytest.mark.asyncio
    async def test_learning_from_successful_parse(self, retemplate_pass, advanced_parser):
        """Test learning patterns from successful parse."""
        trend = retemplate_pass._get_or_create_trend("UC123", "Test Channel")

        result = await retemplate_pass._attempt_learning(trend, "New Artist - New Song", "", "")

        # Should attempt to learn if basic parsing succeeds
        assert advanced_parser.parse_title.called
        assert result is not None

    def test_confidence_decay(self, retemplate_pass):
        """Test confidence decay over time."""
        trend = retemplate_pass._get_or_create_trend("UC123", "Test Channel")

        # Add pattern with old last_seen
        pattern = TemporalPattern(
            pattern="test",
            artist_group=1,
            title_group=2,
            confidence=1.0,
            first_seen=datetime.now(),
            last_seen=datetime.now() - timedelta(days=5),
        )
        trend.active_patterns.append(pattern)

        retemplate_pass._apply_confidence_decay(trend)

        assert pattern.confidence < 1.0

    def test_should_analyze_patterns(self, retemplate_pass):
        """Test pattern analysis trigger logic."""
        trend = retemplate_pass._get_or_create_trend("UC123", "Test Channel")

        # Should analyze if no patterns
        assert retemplate_pass._should_analyze_patterns(trend)

        # Add patterns and recent analysis
        trend.active_patterns.append(Mock())
        trend.active_patterns.append(Mock())
        trend.last_analysis = datetime.now()

        # Should not analyze immediately after
        assert not retemplate_pass._should_analyze_patterns(trend)

    def test_create_result_from_temporal_pattern(self, retemplate_pass):
        """Test creating ParseResult from temporal pattern match."""
        pattern = TemporalPattern(
            pattern=r"^([^-]+)\s*-\s*([^(]+)",
            artist_group=1,
            title_group=2,
            confidence=0.8,
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            video_count=10,
            success_count=8,
        )

        match = re.search(pattern.pattern, "Artist - Song")
        result = retemplate_pass._create_result_from_temporal_pattern(
            match, pattern, "test_method", "Artist - Song"
        )

        assert result is not None
        assert result.artist == "Artist"
        assert result.song_title == "Song"
        assert result.confidence > 0

    def test_pattern_success_recording(self, retemplate_pass):
        """Test recording successful pattern matches."""
        trend = retemplate_pass._get_or_create_trend("UC123", "Test Channel")

        pattern = TemporalPattern(
            pattern="test_pattern",
            artist_group=1,
            title_group=2,
            confidence=0.8,
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            success_count=5,
        )
        trend.active_patterns.append(pattern)

        retemplate_pass._record_pattern_success(trend, "test_pattern", "Example Title")

        assert pattern.success_count == 6
        assert len(pattern.recent_examples) == 1

    def test_statistics_collection(self, retemplate_pass):
        """Test statistics reporting."""
        # Add some test data
        retemplate_pass._get_or_create_trend("UC1", "Channel 1")
        retemplate_pass._get_or_create_trend("UC2", "Channel 2")

        stats = retemplate_pass.get_statistics()

        assert stats["total_channels"] == 2
        assert "total_active_patterns" in stats
        assert "avg_patterns_per_channel" in stats

    @pytest.mark.asyncio
    async def test_pattern_change_detection(self, retemplate_pass):
        """Test detection of pattern changes."""
        trend = retemplate_pass._get_or_create_trend("UC123", "Test Channel")

        # Add active pattern
        pattern = TemporalPattern(
            pattern=r"^Old Pattern",
            artist_group=1,
            title_group=2,
            confidence=0.8,
            first_seen=datetime.now(),
            last_seen=datetime.now(),
        )
        trend.active_patterns.append(pattern)

        # Simulate recent videos not matching
        recent_videos = [
            {"title": "New Format: Artist - Song"},
            {"title": "New Format: Another - Track"},
        ]

        retemplate_pass._detect_pattern_changes(trend, recent_videos)

        # Should detect change
        assert trend.pattern_change_detected is not None

    @pytest.mark.asyncio
    async def test_multi_pattern_priority(self, retemplate_pass):
        """Test pattern priority based on recency and success."""
        trend = retemplate_pass._get_or_create_trend("UC123", "Test Channel")

        # Add multiple patterns with different stats
        recent_pattern = TemporalPattern(
            pattern=r"^Recent",
            artist_group=1,
            title_group=2,
            confidence=0.7,
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            success_count=5,
            video_count=5,
        )

        old_pattern = TemporalPattern(
            pattern=r"^Old",
            artist_group=1,
            title_group=2,
            confidence=0.9,
            first_seen=datetime.now() - timedelta(days=30),
            last_seen=datetime.now() - timedelta(days=10),
            success_count=10,
            video_count=15,
        )

        trend.active_patterns.extend([old_pattern, recent_pattern])

        # Recent pattern should be tried first despite lower confidence
        result = await retemplate_pass._try_active_patterns(trend, "Recent Pattern", "", "")

        # Verify patterns are sorted by recency
        assert trend.active_patterns[0].pattern == r"^Recent"
        assert result is not None
