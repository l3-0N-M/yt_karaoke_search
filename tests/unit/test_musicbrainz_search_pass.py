"""Unit tests for musicbrainz_search_pass.py."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from collector.passes.musicbrainz_search_pass import MusicBrainzSearchPass


class TestMusicBrainzSearchPass:
    """Test cases for MusicBrainzSearchPass."""

    @pytest.fixture
    def mock_musicbrainzngs(self):
        """Mock the musicbrainzngs library."""
        with patch("collector.passes.musicbrainz_search_pass.musicbrainzngs") as mock_mb:
            mock_mb.search_recordings = AsyncMock()
            mock_mb.search_artists = AsyncMock()
            mock_mb.get_recording_by_id = AsyncMock()
            yield mock_mb

    @pytest.fixture
    def mb_pass(self, mock_musicbrainzngs):
        """Create a MusicBrainzSearchPass instance."""
        return MusicBrainzSearchPass(advanced_parser=Mock())

    @pytest.mark.asyncio
    async def test_parse_successful_search(self, mb_pass, mock_musicbrainzngs):
        """Test successful MusicBrainz search."""
        # Mock search results
        mock_musicbrainzngs.search_recordings.return_value = {
            "recording-list": [
                {
                    "id": "recording-123",
                    "title": "Test Song",
                    "artist-credit": [
                        {
                            "artist": {
                                "id": "artist-456",
                                "name": "Test Artist",
                                "sort-name": "Artist, Test",
                            }
                        }
                    ],
                    "release-list": [
                        {
                            "title": "Test Album",
                            "date": "1985-06-15",
                            "country": "US",
                            "medium-list": [
                                {"format": "CD", "track-list": [{"position": "5", "number": "5"}]}
                            ],
                        }
                    ],
                    "tag-list": [{"name": "rock", "count": 10}, {"name": "pop", "count": 5}],
                    "length": 240000,  # milliseconds
                }
            ]
        }

        result = await mb_pass.parse(
            title="Test Artist - Test Song (Karaoke)",
            description="Karaoke version",
            tags="",
            channel_name="",
            channel_id="",
            metadata={},
        )

        assert result is not None
        assert result.artist == "Test Artist"
        assert result.song_title == "Test Song"
        assert result.metadata.get("year") == 1985
        assert result.metadata.get("genre") == "rock"
        assert result.confidence > 0.7

    @pytest.mark.asyncio
    async def test_parse_no_results(self, mb_pass, mock_musicbrainzngs):
        """Test when no results are found."""
        mock_musicbrainzngs.search_recordings.return_value = {"recording-list": []}

        result = await mb_pass.parse(
            title="Unknown Artist - Unknown Song",
            description="",
            tags="",
            channel_name="",
            channel_id="",
            metadata={},
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_parse_with_featured_artists(self, mb_pass, mock_musicbrainzngs):
        """Test parsing with featured artists."""
        mock_musicbrainzngs.search_recordings.return_value = {
            "recording-list": [
                {
                    "id": "rec-123",
                    "title": "Collaboration",
                    "artist-credit": [
                        {"artist": {"id": "artist-1", "name": "Main Artist"}},
                        {
                            "joinphrase": " feat. ",
                            "artist": {"id": "artist-2", "name": "Featured Artist"},
                        },
                    ],
                    "release-list": [{"date": "2020"}],
                }
            ]
        }

        result = await mb_pass.parse(
            title="Main Artist feat. Featured Artist - Collaboration",
            description="",
            tags="",
            channel_name="",
            channel_id="",
            metadata={},
        )

        assert result is not None
        assert result.artist == "Main Artist"
        assert "Featured Artist" in result.featured_artists

    @pytest.mark.asyncio
    async def test_parse_with_aliases(self, mb_pass, mock_musicbrainzngs):
        """Test handling of artist aliases."""
        mock_musicbrainzngs.search_recordings.return_value = {
            "recording-list": [
                {
                    "id": "rec-123",
                    "title": "Song",
                    "artist-credit": [
                        {
                            "artist": {
                                "id": "artist-123",
                                "name": "Official Name",
                                "alias-list": [
                                    {
                                        "alias": "Stage Name",
                                        "type": "Artist name",
                                        "primary": "primary",
                                    }
                                ],
                            }
                        }
                    ],
                }
            ]
        }

        # Search with stage name
        result = await mb_pass.parse(
            title="Stage Name - Song",
            description="",
            tags="",
            channel_name="",
            channel_id="",
            metadata={},
        )

        assert result is not None
        # Should recognize the artist despite using alias

    @pytest.mark.asyncio
    async def test_parse_various_date_formats(self, mb_pass, mock_musicbrainzngs):
        """Test parsing various date formats."""
        test_cases = [
            ("1985-12-25", 1985),
            ("1990-06", 1990),
            ("2000", 2000),
            ("invalid-date", None),
            ("", None),
        ]

        for date_str, expected_year in test_cases:
            mock_musicbrainzngs.search_recordings.return_value = {
                "recording-list": [
                    {
                        "id": "rec-123",
                        "title": "Song",
                        "artist-credit": [{"artist": {"name": "Artist"}}],
                        "release-list": [{"date": date_str}],
                    }
                ]
            }

            result = await mb_pass.parse(
                title="Artist - Song",
                description="",
                tags="",
                channel_name="",
                channel_id="",
                metadata={},
            )

            if expected_year:
                assert result is not None
                assert result.metadata.get("year") == expected_year
            else:
                assert result is None or result.metadata.get("year") is None

    @pytest.mark.asyncio
    async def test_confidence_calculation(self, mb_pass, mock_musicbrainzngs):
        """Test confidence score calculation."""
        # High confidence - exact match
        mock_musicbrainzngs.search_recordings.return_value = {
            "recording-list": [
                {
                    "id": "rec-123",
                    "title": "Exact Title Match",
                    "artist-credit": [{"artist": {"name": "Exact Artist Match"}}],
                    "ext:score": "100",  # MusicBrainz search score
                }
            ]
        }

        result = await mb_pass.parse(
            title="Exact Artist Match - Exact Title Match",
            description="",
            tags="",
            channel_name="",
            channel_id="",
            metadata={},
        )

        assert result is not None
        assert result.confidence > 0.9

    @pytest.mark.asyncio
    async def test_parse_with_disambiguation(self, mb_pass, mock_musicbrainzngs):
        """Test handling of disambiguation data."""
        mock_musicbrainzngs.search_recordings.return_value = {
            "recording-list": [
                {
                    "id": "rec-123",
                    "title": "Song",
                    "disambiguation": "live version",
                    "artist-credit": [{"artist": {"name": "Artist"}}],
                }
            ]
        }

        result = await mb_pass.parse(
            title="Artist - Song (Live)",
            description="",
            tags="",
            channel_name="",
            channel_id="",
            metadata={},
        )

        assert result is not None
        assert (
            result.metadata.get("version") == "live version"
            or "live" in result.metadata.get("version").lower()
        )

    @pytest.mark.asyncio
    async def test_genre_extraction(self, mb_pass, mock_musicbrainzngs):
        """Test genre extraction from tags."""
        mock_musicbrainzngs.search_recordings.return_value = {
            "recording-list": [
                {
                    "id": "rec-123",
                    "title": "Song",
                    "artist-credit": [{"artist": {"name": "Artist"}}],
                    "tag-list": [
                        {"name": "electronic", "count": 20},
                        {"name": "dance", "count": 15},
                        {"name": "pop", "count": 5},
                    ],
                }
            ]
        }

        result = await mb_pass.parse(
            title="Artist - Song",
            description="",
            tags="",
            channel_name="",
            channel_id="",
            metadata={},
        )

        assert result is not None
        assert result.metadata.get("genre") in ["electronic", "dance", "electronic, dance"]

    @pytest.mark.asyncio
    async def test_isrc_handling(self, mb_pass, mock_musicbrainzngs):
        """Test ISRC code handling."""
        mock_musicbrainzngs.search_recordings.return_value = {
            "recording-list": [
                {
                    "id": "rec-123",
                    "title": "Song",
                    "artist-credit": [{"artist": {"name": "Artist"}}],
                    "isrc-list": ["USRC17607839"],
                }
            ]
        }

        result = await mb_pass.parse(
            title="Artist - Song",
            description="",
            tags="",
            channel_name="",
            channel_id="",
            metadata={},
        )

        assert result is not None
        assert result.additional_metadata is not None
        assert "isrc" in result.additional_metadata

    @pytest.mark.asyncio
    async def test_error_handling(self, mb_pass, mock_musicbrainzngs):
        """Test error handling in MusicBrainz API calls."""
        mock_musicbrainzngs.search_recordings.side_effect = Exception("API Error")

        result = await mb_pass.parse(
            title="Artist - Song",
            description="",
            tags="",
            channel_name="",
            channel_id="",
            metadata={},
        )

        assert result is None  # Should handle error gracefully

    @pytest.mark.asyncio
    async def test_search_with_limit(self, mb_pass, mock_musicbrainzngs):
        """Test that search respects result limits."""
        # Return many results
        large_result_list = [
            {
                "id": f"rec-{i}",
                "title": f"Song {i}",
                "artist-credit": [{"artist": {"name": "Artist"}}],
            }
            for i in range(100)
        ]

        mock_musicbrainzngs.search_recordings.return_value = {"recording-list": large_result_list}

        result = await mb_pass.parse(
            title="Artist - Song",
            description="",
            tags="",
            channel_name="",
            channel_id="",
            metadata={},
        )
        assert result is not None

        # Verify search was called with a reasonable limit
        call_args = mock_musicbrainzngs.search_recordings.call_args
        assert "limit" in call_args[1]
        assert call_args[1]["limit"] <= 25

    @pytest.mark.asyncio
    async def test_cover_detection(self, mb_pass, mock_musicbrainzngs):
        """Test detection of cover versions."""
        mock_musicbrainzngs.search_recordings.return_value = {
            "recording-list": [
                {
                    "id": "rec-123",
                    "title": "Song (cover)",
                    "artist-credit": [{"artist": {"name": "Cover Artist"}}],
                    "disambiguation": "cover of Original Artist",
                    "relation-list": [
                        {"type": "cover", "direction": "forward", "target": "original-recording-id"}
                    ],
                }
            ]
        }

        result = await mb_pass.parse(
            title="Cover Artist - Song",
            description="",
            tags="",
            channel_name="",
            channel_id="",
            metadata={},
        )

        assert result is not None
        assert result.is_cover is True

    @pytest.mark.asyncio
    async def test_search_fallback_strategies(self, mb_pass, mock_musicbrainzngs):
        """Test fallback search strategies."""
        call_count = 0

        def search_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First search returns nothing
                return {"recording-list": []}
            else:
                # Fallback search returns result
                return {
                    "recording-list": [
                        {
                            "id": "rec-123",
                            "title": "Song",
                            "artist-credit": [{"artist": {"name": "Artist"}}],
                        }
                    ]
                }

        mock_musicbrainzngs.search_recordings.side_effect = search_side_effect

        result = await mb_pass.parse(
            title="Artist feat. Guest - Song (Remix)",
            description="",
            tags="",
            channel_name="",
            channel_id="",
            metadata={},
        )

        assert result is not None
        assert call_count > 1  # Multiple search attempts

    @pytest.mark.asyncio
    async def test_parse_with_metadata(self, mb_pass, mock_musicbrainzngs):
        """Test parsing with additional metadata."""
        mock_musicbrainzngs.search_recordings.return_value = {
            "recording-list": [
                {
                    "id": "rec-123",
                    "title": "Song",
                    "artist-credit": [{"artist": {"name": "Artist"}}],
                    "length": 180000,  # 3 minutes
                    "video": "true",
                }
            ]
        }

        result = await mb_pass.parse(
            title="Artist - Song", description="", metadata={"duration": 180}  # Duration in seconds
        )

        assert result is not None
        # Duration should match and boost confidence
