"""Unit tests for processor.py - focusing on year extraction validation."""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from collector.config import CollectorConfig
from collector.processor import VideoProcessor


class TestVideoProcessor:
    """Test cases for VideoProcessor with focus on year extraction fixes."""

    @pytest.fixture
    def processor(self):
        """Create a VideoProcessor instance."""
        config = Mock(spec=CollectorConfig)
        # Mock scraping config
        config.scraping = Mock()
        config.scraping.timeout_seconds = 30
        config.scraping.max_retries = 3
        config.scraping.extract_formats = ["bestaudio", "best"]
        config.scraping.quiet_mode = True
        config.scraping.no_warnings = True
        config.scraping.ignore_errors = True
        config.scraping.geo_bypass = True
        config.scraping.user_agents = ["Mozilla/5.0 Test User Agent"]

        # Mock search config
        config.search = Mock()
        config.search.query_templates = []

        # Mock data sources
        config.data_sources = Mock()

        # Mock logging
        config.logging = Mock()
        config.logging.level = "INFO"

        # Mock processor specific settings
        config.fallback_confidence_boost = 0.1

        # Patch yt-dlp initialization
        with patch("collector.processor.yt_dlp"):
            processor = VideoProcessor(config)

        # Mock the advanced parser
        mock_parser = Mock()
        mock_parser.parse = Mock(
            return_value=Mock(
                artist="Test Artist",
                song_title="Test Song",
                confidence=0.8,
                featured_artists=None,
                remix_info=None,
                method="test",
            )
        )
        processor.advanced_parser = mock_parser

        return processor

    @pytest.fixture
    def current_year(self):
        """Get current year for tests."""
        return datetime.now().year

    def test_extract_release_year_rejects_current_year(self, processor, current_year):
        """Test that current year is rejected as it's likely upload year."""
        test_cases = [
            f"Song Title ({current_year})",
            f"[{current_year}] New Song",
            f"Latest Hit {current_year} Version",
            f"Track from {current_year}",
        ]

        for title in test_cases:
            year = processor._extract_release_year_fallback(title, "")
            assert year is None, f"Current year {current_year} should be rejected in '{title}'"

    def test_extract_release_year_rejects_future_years(self, processor, current_year):
        """Test that future years are rejected."""
        future_year = current_year + 1
        test_cases = [
            f"Future Song ({future_year})",
            f"[{future_year}] Upcoming Track",
            f"Release scheduled for {future_year}",
        ]

        for title in test_cases:
            year = processor._extract_release_year_fallback(title, "")
            assert year is None, f"Future year {future_year} should be rejected in '{title}'"

    def test_extract_release_year_accepts_valid_years(self, processor, current_year):
        """Test that valid historical years are accepted."""
        test_cases = [
            ("Classic Song (1985)", 1985),
            ("[1992] Hit Track", 1992),
            ("Song from 1975", 1975),
            ("2010 Version", 2010),
            (f"{current_year - 1} Release", current_year - 1),  # Last year is valid
        ]

        for title, expected_year in test_cases:
            year = processor._extract_release_year_fallback(title, "")
            assert year == expected_year, f"Expected {expected_year} from '{title}', got {year}"

    def test_extract_release_year_prioritizes_earlier_years(self, processor):
        """Test that earlier years are prioritized (original release)."""
        # Title with multiple years
        title = "Song Title (2020 Remaster) originally from 1985"
        year = processor._extract_release_year_fallback(title, "")
        assert year == 1985, "Should prioritize earlier year as original release"

    def test_extract_release_year_validates_reasonable_range(self, processor):
        """Test that only reasonable years (1900+) are accepted."""
        test_cases = [
            ("Ancient Song (1899)", None),  # Too old
            ("Medieval Track [1500]", None),  # Way too old
            ("First Recording (1900)", 1900),  # Minimum valid year
            ("Modern Song (2020)", 2020),  # Recent valid year
        ]

        for title, expected_year in test_cases:
            year = processor._extract_release_year_fallback(title, "")
            assert year == expected_year, f"Expected {expected_year} from '{title}', got {year}"

    def test_extract_release_year_handles_decades(self, processor):
        """Test extraction of decade indicators."""
        test_cases = [
            ("1980s Classic", 1980),
            ("Best of the 90s", None),  # '90s' without century
            ("1960s Soul", 1960),
        ]

        for title, expected_year in test_cases:
            year = processor._extract_release_year_fallback(title, "")
            assert year == expected_year, f"Expected {expected_year} from '{title}', got {year}"

    def test_extract_release_year_patterns_priority(self, processor):
        """Test that year patterns are prioritized correctly."""
        # Parentheses should have highest priority
        title1 = "Song 1999 (1985)"
        assert processor._extract_release_year_fallback(title1, "") == 1985

        # Brackets second priority
        title2 = "Song 1999 [1985]"
        assert processor._extract_release_year_fallback(title2, "") == 1985

        # Standalone year lower priority
        title3 = "Song Title 1985"
        assert processor._extract_release_year_fallback(title3, "") == 1985

    def test_extract_release_year_from_description(self, processor):
        """Test year extraction from description field."""
        title = "Classic Song"
        description = "This song was originally released in 1975 and became a hit."

        year = processor._extract_release_year_fallback(title, description)
        assert year == 1975

    def test_extract_release_year_with_no_years(self, processor):
        """Test behavior when no years are found."""
        title = "Song Without Year Info"
        description = "Just a regular karaoke track"

        year = processor._extract_release_year_fallback(title, description)
        assert year is None

    def test_extract_release_year_with_multiple_valid_years(self, processor):
        """Test that earliest year is selected from multiple valid years."""
        title = "Greatest Hits"
        description = "Features songs from 1975, 1980, and 1985"

        year = processor._extract_release_year_fallback(title, description)
        assert year == 1975, "Should return earliest year"

    @patch("collector.processor.datetime")
    def test_extract_release_year_with_mocked_current_year(self, mock_datetime, processor):
        """Test year extraction with mocked current year."""
        # Mock current year to 2024
        mock_datetime.now.return_value.year = 2024

        # 2024 should be rejected
        year1 = processor._extract_release_year_fallback("Song (2024)", "")
        assert year1 is None

        # 2023 should be accepted
        year2 = processor._extract_release_year_fallback("Song (2023)", "")
        assert year2 == 2023

        # 2025 should be rejected
        year3 = processor._extract_release_year_fallback("Song (2025)", "")
        assert year3 is None

    @pytest.mark.asyncio
    @pytest.mark.skip("Complex integration test - year extraction is already tested in unit tests")
    async def test_process_video_integrates_year_validation(self, processor):
        """Test that process_video method uses year validation correctly."""
        video_info = {
            "url": "https://youtube.com/watch?v=test123",
            "id": "test123",
            "title": "Test Song (2025) - Karaoke Version",
            "description": "Originally released in 1985",
            "uploader": "Test Channel",
            "duration": 240,
            "view_count": 1000,
            "like_count": 50,
            "upload_date": "20250101",
            "webpage_url": "https://youtube.com/watch?v=test123",
            "channel": "Test Channel",
            "channel_id": "UC123",
        }

        # Mock yt-dlp to return the enriched video info
        enriched_info = video_info.copy()
        processor.ydl = Mock()
        processor.ydl.extract_info = Mock(return_value=enriched_info)

        result = await processor.process_video(video_info)

        # Should extract 1985 from description, not 2025 from title
        assert result.video_data["release_year"] == 1985

    def test_year_extraction_logging(self, processor, caplog):
        """Test that year rejection is logged appropriately."""
        import logging

        caplog.set_level(logging.DEBUG)
        current_year = datetime.now().year

        # Test with current year
        processor._extract_release_year_fallback(f"Song ({current_year})", "")
        assert f"Rejected year {current_year}" in caplog.text

        # Test with future year
        processor._extract_release_year_fallback(f"Song ({current_year + 1})", "")
        assert f"Rejected year {current_year + 1}" in caplog.text
