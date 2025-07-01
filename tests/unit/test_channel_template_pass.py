"""Unit tests for channel_template_pass.py."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from collector.passes.base import PassType
from collector.passes.channel_template_pass import EnhancedChannelTemplatePass


class TestEnhancedChannelTemplatePass:
    """Test cases for EnhancedChannelTemplatePass."""

    @pytest.fixture
    def mock_db_manager(self):
        """Create a mock database manager."""
        mock_db = Mock()
        mock_db.get_channel_videos = Mock(return_value=[])
        mock_db.get_channel_templates = Mock(return_value=None)
        mock_db.save_channel_template = AsyncMock()
        return mock_db

    @pytest.fixture
    def channel_pass(self, mock_db_manager):
        """Create a EnhancedChannelTemplatePass instance."""
        with patch(
            "collector.passes.channel_template_pass.get_db_manager", return_value=mock_db_manager
        ):
            return EnhancedChannelTemplatePass(advanced_parser=Mock())

    @pytest.mark.asyncio
    async def test_parse_with_learned_template(self, channel_pass, mock_db_manager):
        """Test parsing when channel has a learned template."""
        # Mock channel history
        channel_videos = [
            {
                "title": "Artist1 - Song1 | Karaoke Lyrics",
                "artist": "Artist1",
                "song_title": "Song1",
                "parse_confidence": 0.9,
            },
            {
                "title": "Artist2 - Song2 | Karaoke Lyrics",
                "artist": "Artist2",
                "song_title": "Song2",
                "parse_confidence": 0.85,
            },
            {
                "title": "Artist3 - Song3 | Karaoke Lyrics",
                "artist": "Artist3",
                "song_title": "Song3",
                "parse_confidence": 0.88,
            },
        ]
        mock_db_manager.get_channel_videos.return_value = channel_videos

        # Test with matching template
        result = await channel_pass.parse(
            title="NewArtist - NewSong | Karaoke Lyrics",
            description="",
            tags="",
            channel_name="",
            channel_id="channel123",
            metadata={},
        )

        assert result is not None
        assert result.artist == "NewArtist"
        assert result.song_title == "NewSong"
        assert result.confidence > 0.7
        assert result.method == PassType.CHANNEL_TEMPLATE

    @pytest.mark.asyncio
    async def test_parse_without_channel_id(self, channel_pass):
        """Test parsing without channel ID returns None."""
        result = await channel_pass.parse(
            title="Artist - Song",
            description="",
            tags="",
            channel_name="",
            channel_id=None,
            metadata={},
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_parse_with_no_channel_history(self, channel_pass, mock_db_manager):
        """Test parsing when channel has no history."""
        mock_db_manager.get_channel_videos.return_value = []

        result = await channel_pass.parse(
            title="Artist - Song",
            description="",
            tags="",
            channel_name="",
            channel_id="new_channel",
            metadata={},
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_learn_channel_patterns(self, channel_pass, mock_db_manager):
        """Test learning patterns from channel history."""
        channel_videos = [
            {
                "title": "[KARAOKE] Artist1 - Song1 (MR)",
                "artist": "Artist1",
                "song_title": "Song1",
                "parse_confidence": 0.9,
            },
            {
                "title": "[KARAOKE] Artist2 - Song2 (MR)",
                "artist": "Artist2",
                "song_title": "Song2",
                "parse_confidence": 0.92,
            },
            {
                "title": "[KARAOKE] Artist3 - Song3 (MR)",
                "artist": "Artist3",
                "song_title": "Song3",
                "parse_confidence": 0.88,
            },
        ]
        mock_db_manager.get_channel_videos.return_value = channel_videos

        patterns = channel_pass._learn_channel_patterns(channel_videos)

        assert len(patterns) > 0
        assert any("[KARAOKE]" in pattern["pattern"] for pattern in patterns)
        assert any("(MR)" in pattern["pattern"] for pattern in patterns)

    @pytest.mark.asyncio
    async def test_extract_with_template(self, channel_pass):
        """Test extraction using a template pattern."""
        template = {
            "pattern": "{artist} - {song} | Official Karaoke",
            "confidence": 0.9,
            "sample_count": 10,
        }

        title = "Queen - Bohemian Rhapsody | Official Karaoke"
        result = channel_pass._extract_with_template(title, template)

        assert result is not None
        assert result["artist"] == "Queen"
        assert result["song"] == "Bohemian Rhapsody"

    @pytest.mark.asyncio
    async def test_extract_with_complex_template(self, channel_pass):
        """Test extraction with complex template patterns."""
        template = {
            "pattern": "🎤 {song} - {artist} [Karaoke Version]",
            "confidence": 0.85,
            "sample_count": 5,
        }

        title = "🎤 Yesterday - The Beatles [Karaoke Version]"
        result = channel_pass._extract_with_template(title, template)

        assert result is not None
        assert result["artist"] == "The Beatles"
        assert result["song"] == "Yesterday"

    @pytest.mark.asyncio
    async def test_confidence_calculation(self, channel_pass):
        """Test confidence score calculation."""
        # High confidence template with many samples
        template1 = {"pattern": "{artist} - {song}", "confidence": 0.95, "sample_count": 50}
        confidence1 = channel_pass._calculate_confidence(template1, exact_match=True)
        assert confidence1 > 0.9

        # Lower confidence template with few samples
        template2 = {"pattern": "{artist} {song}", "confidence": 0.7, "sample_count": 3}
        confidence2 = channel_pass._calculate_confidence(template2, exact_match=False)
        assert confidence2 < confidence1
        assert confidence2 < 0.8

    @pytest.mark.asyncio
    async def test_parse_with_saved_template(self, channel_pass, mock_db_manager):
        """Test using previously saved channel template."""
        saved_templates = [
            {"pattern": "Karaoke - {artist} - {song} (HD)", "confidence": 0.92, "sample_count": 25}
        ]
        mock_db_manager.get_channel_templates.return_value = saved_templates
        mock_db_manager.get_channel_videos.return_value = []  # No recent videos

        result = await channel_pass.parse(
            title="Karaoke - Adele - Hello (HD)",
            description="",
            tags="",
            channel_name="",
            channel_id="channel123",
            metadata={},
        )

        assert result is not None
        assert result.artist == "Adele"
        assert result.song_title == "Hello"

    @pytest.mark.asyncio
    async def test_template_pattern_variations(self, channel_pass):
        """Test handling various template pattern variations."""
        test_cases = [
            # Pattern with optional parts
            {
                "pattern": "{artist} - {song} (Karaoke)",
                "title": "Artist - Song (Karaoke)",
                "expected": {"artist": "Artist", "song": "Song"},
            },
            # Pattern with special characters
            {
                "pattern": "♪ {artist} ♪ {song} ♪",
                "title": "♪ Test Artist ♪ Test Song ♪",
                "expected": {"artist": "Test Artist", "song": "Test Song"},
            },
            # Pattern with brackets
            {
                "pattern": "[{artist}] - [{song}]",
                "title": "[Queen] - [We Will Rock You]",
                "expected": {"artist": "Queen", "song": "We Will Rock You"},
            },
        ]

        for test_case in test_cases:
            template = {"pattern": test_case["pattern"], "confidence": 0.9, "sample_count": 10}
            result = channel_pass._extract_with_template(test_case["title"], template)

            assert result is not None
            for key, expected_value in test_case["expected"].items():
                assert result[key] == expected_value

    @pytest.mark.asyncio
    async def test_minimum_history_requirement(self, channel_pass, mock_db_manager):
        """Test that minimum history is required for pattern learning."""
        # Only 2 videos - below threshold
        channel_videos = [
            {
                "title": "Artist1 - Song1",
                "artist": "Artist1",
                "song_title": "Song1",
                "parse_confidence": 0.9,
            },
            {
                "title": "Artist2 - Song2",
                "artist": "Artist2",
                "song_title": "Song2",
                "parse_confidence": 0.9,
            },
        ]
        mock_db_manager.get_channel_videos.return_value = channel_videos

        result = await channel_pass.parse(
            title="Artist3 - Song3",
            description="",
            tags="",
            channel_name="",
            channel_id="channel123",
            metadata={},
        )

        # Should not learn pattern with too few samples
        assert result is None or result.confidence < 0.7

    @pytest.mark.asyncio
    async def test_inconsistent_channel_patterns(self, channel_pass, mock_db_manager):
        """Test handling channels with inconsistent patterns."""
        channel_videos = [
            {
                "title": "Artist1 - Song1",
                "artist": "Artist1",
                "song_title": "Song1",
                "parse_confidence": 0.9,
            },
            {
                "title": "Song2 by Artist2",
                "artist": "Artist2",
                "song_title": "Song2",
                "parse_confidence": 0.9,
            },
            {
                "title": 'Karaoke: Artist3 "Song3"',
                "artist": "Artist3",
                "song_title": "Song3",
                "parse_confidence": 0.9,
            },
            {
                "title": "Artist4 | Song4 | Karaoke",
                "artist": "Artist4",
                "song_title": "Song4",
                "parse_confidence": 0.9,
            },
        ]
        mock_db_manager.get_channel_videos.return_value = channel_videos

        patterns = channel_pass._learn_channel_patterns(channel_videos)

        # Should not find strong patterns
        assert len(patterns) == 0 or all(p["confidence"] < 0.8 for p in patterns)

    @pytest.mark.asyncio
    async def test_featured_artists_in_template(self, channel_pass, mock_db_manager):
        """Test handling featured artists in channel templates."""
        channel_videos = [
            {
                "title": "Artist1 feat. Artist2 - Song1 [Karaoke]",
                "artist": "Artist1",
                "song_title": "Song1",
                "featured_artists": "Artist2",
                "parse_confidence": 0.9,
            },
            {
                "title": "Artist3 feat. Artist4 - Song2 [Karaoke]",
                "artist": "Artist3",
                "song_title": "Song2",
                "featured_artists": "Artist4",
                "parse_confidence": 0.9,
            },
        ]
        mock_db_manager.get_channel_videos.return_value = channel_videos

        result = await channel_pass.parse(
            title="NewArtist feat. Guest - NewSong [Karaoke]",
            description="",
            tags="",
            channel_name="",
            channel_id="channel123",
            metadata={},
        )

        assert result is not None
        assert result.artist == "NewArtist"
        assert result.song_title == "NewSong"
        assert result.featured_artists and "Guest" in result.featured_artists

    @pytest.mark.asyncio
    async def test_template_caching(self, channel_pass, mock_db_manager):
        """Test that learned templates are cached."""
        channel_videos = [
            {
                "title": f"Artist{i} - Song{i} | KaraokeFun",
                "artist": f"Artist{i}",
                "song_title": f"Song{i}",
                "parse_confidence": 0.9,
            }
            for i in range(5)
        ]
        mock_db_manager.get_channel_videos.return_value = channel_videos

        # First parse
        result1 = await channel_pass.parse(
            title="TestArtist - TestSong | KaraokeFun",
            description="",
            tags="",
            channel_name="",
            channel_id="channel123",
            metadata={},
        )

        # Second parse with same channel
        result2 = await channel_pass.parse(
            title="TestArtist2 - TestSong2 | KaraokeFun",
            description="",
            tags="",
            channel_name="",
            channel_id="channel123",
            metadata={},
        )

        # Should only query channel videos once (cached)
        assert mock_db_manager.get_channel_videos.call_count == 1
        assert result1 is not None
        assert result2 is not None
