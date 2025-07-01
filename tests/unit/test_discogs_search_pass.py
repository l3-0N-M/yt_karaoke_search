"""Unit tests for discogs_search_pass.py."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from collector.advanced_parser import ParseResult
from collector.passes.discogs_search_pass import DiscogsClient, DiscogsMatch, DiscogsSearchPass
from collector.utils import DiscogsRateLimiter


class TestDiscogsMatch:
    """Test cases for DiscogsMatch dataclass."""

    def test_discogs_match_creation(self):
        """Test creating a DiscogsMatch."""
        match = DiscogsMatch(
            artist_name="Test Artist",
            song_title="Test Song",
            year=1985,
            confidence=0.95,
            release_id="67890",
            master_id="11111",
            label="Test Records",
            genres=["Rock", "Pop"],
            styles=["Classic Rock"],
            country="US",
            format="CD",
        )

        assert match.artist_name == "Test Artist"
        assert match.year == 1985
        assert match.confidence == 0.95
        assert "Rock" in match.genres


# # SearchCandidate tests removed as the class no longer exists
# class TestSearchCandidateRemoved:
#     """Test cases for SearchCandidate dataclass."""
#
#     def test_search_candidate_creation(self):
#         """Test creating a SearchCandidate."""
#         candidate = SearchCandidate(
#             artist="Test Artist",
#             track="Test Track",
#             confidence=0.8,
#             search_type="exact",
#             original_data={
#                 'featured_artists': ['Feat1'],
#                 'version': 'Remix'
#             }
#         )
#
#         assert candidate.artist == "Test Artist"
#         assert candidate.track == "Test Track"
#         assert candidate.confidence == 0.8
#         assert candidate.search_type == "exact"
#         assert 'featured_artists' in candidate.original_data
#
#
class TestDiscogsClient:
    """Test cases for DiscogsClient."""

    @pytest.fixture
    def rate_limiter(self):
        """Create a mock rate limiter."""
        limiter = Mock(spec=DiscogsRateLimiter)
        limiter.wait_for_request = AsyncMock()
        limiter.handle_429_error = AsyncMock()
        limiter.handle_success = AsyncMock()
        return limiter

    @pytest.fixture
    def mock_session(self):
        """Create a mock aiohttp session."""
        session = AsyncMock()
        return session

    @pytest.fixture
    def discogs_client(self, rate_limiter):
        """Create a DiscogsClient instance."""
        client = DiscogsClient(token="test_token", rate_limiter=rate_limiter, user_agent="Test/1.0")
        return client

    @pytest.mark.asyncio
    async def test_search_release_success(self, discogs_client, mock_session):
        """Test successful release search."""
        # Mock API response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "results": [
                    {
                        "id": 12345,
                        "type": "release",
                        "title": "Test Artist - Test Album",
                        "year": "1985",
                        "genre": ["Rock"],
                        "style": ["Classic Rock"],
                        "label": ["Test Records"],
                        "country": "US",
                        "format": ["CD"],
                        "catno": "TR-001",
                        "barcode": ["123456789"],
                    }
                ]
            }
        )

        mock_session.get.return_value.__aenter__.return_value = mock_response

        matches = await discogs_client.search_release(artist="Test Artist", track="Test Song")

        assert len(matches) > 0
        assert matches[0].artist_name == "Test Artist"
        assert matches[0].year == 1985
        assert matches[0].label == "Test Records"

    @pytest.mark.asyncio
    async def test_search_release_with_429_error(self, discogs_client, mock_session, rate_limiter):
        """Test handling 429 rate limit errors."""
        # Mock 429 error response
        error = MagicMock()
        error.status = 429
        error.headers = {"Retry-After": "60"}

        mock_session.get.return_value.__aenter__.side_effect = error

        matches = await discogs_client.search_release(artist="Test Artist", track="Test Song")

        assert matches == []
        assert rate_limiter.handle_429_error.called

    @pytest.mark.asyncio
    async def test_search_release_with_year_filter(self, discogs_client, mock_session):
        """Test search with year filtering."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "results": [
                    {"id": 1, "title": "Artist - Album 1", "year": "1985", "genre": ["Rock"]},
                    {"id": 2, "title": "Artist - Album 2", "year": "2020", "genre": ["Pop"]},
                ]
            }
        )

        mock_session.get.return_value.__aenter__.return_value = mock_response

        matches = await discogs_client.search_release(
            artist="Artist", track="Song", year_tolerance=2
        )

        # Should prefer the 1985 release
        assert len(matches) > 0
        assert any(m.year == 1985 for m in matches)

    @pytest.mark.asyncio
    async def test_generate_artist_variations(self, discogs_client):
        """Test artist name variation generation."""
        variations = discogs_client._generate_artist_variations("The Beatles")

        assert "The Beatles" in variations
        assert "Beatles" in variations
        assert "Beatles, The" in variations

    @pytest.mark.asyncio
    async def test_normalize_search_query(self, discogs_client):
        """Test search query normalization."""
        test_cases = [
            ("Song (Karaoke Version)", "Song"),
            ("Song [Official Audio]", "Song"),
            ("Song (feat. Artist2)", "Song"),
            ("Song - Remix", "Song Remix"),
            ("Song (Live)", "Song"),
        ]

        for input_query, expected in test_cases:
            normalized = discogs_client._normalize_search_query(input_query)
            assert normalized == expected


class TestDiscogsSearchPass:
    """Test cases for DiscogsSearchPass."""

    @pytest.fixture
    def mock_monitor(self):
        """Create a mock Discogs monitor."""
        monitor = Mock()
        monitor.record_search = AsyncMock()
        monitor.record_api_call = AsyncMock()
        monitor.can_make_request = Mock(return_value=True)
        return monitor

    @pytest.fixture
    def mock_client(self):
        """Create a mock Discogs client."""
        client = AsyncMock(spec=DiscogsClient)
        client.search_release = AsyncMock(return_value=[])
        return client

    @pytest.fixture
    def discogs_pass(self, mock_monitor, mock_client):
        """Create a DiscogsSearchPass instance."""
        with patch(
            "collector.passes.discogs_search_pass.DiscogsMonitor", return_value=mock_monitor
        ):
            with patch(
                "collector.passes.discogs_search_pass.DiscogsClient", return_value=mock_client
            ):
                pass_instance = DiscogsSearchPass(advanced_parser=Mock(), config={})
                pass_instance.client = mock_client
                pass_instance.monitor = mock_monitor
                return pass_instance

    @pytest.mark.asyncio
    async def test_parse_basic_success(self, discogs_pass, mock_client):
        """Test successful basic parsing."""
        # Mock search results
        mock_client.search_release.return_value = [
            DiscogsMatch(
                artist_name="Queen",
                song_title="Bohemian Rhapsody",
                year=1975,
                confidence=0.95,
                release_id="67890",
                master_id=None,
                label="EMI",
                genres=["Rock"],
                styles=["Prog Rock"],
                country=None,
                format=None,
            )
        ]

        result = await discogs_pass.parse(
            title="Queen - Bohemian Rhapsody (Karaoke)",
            description="Classic rock karaoke",
            tags="",
            channel_name="",
            channel_id="",
            metadata={},
        )

        assert result is not None
        assert result.artist == "Queen"
        assert result.song_title == "Bohemian Rhapsody"
        assert result.confidence > 0.8

    @pytest.mark.asyncio
    async def test_parse_with_no_matches(self, discogs_pass, mock_client):
        """Test parsing when no matches found."""
        mock_client.search_release.return_value = []

        result = await discogs_pass.parse(
            title="Unknown Artist - Unknown Song",
            description="",
            tags="",
            channel_name="",
            channel_id="",
            metadata={},
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_parse_with_monitor_blocking(self, discogs_pass, mock_monitor):
        """Test when monitor blocks requests."""
        mock_monitor.can_make_request.return_value = False

        result = await discogs_pass.parse(
            title="Artist - Song",
            description="",
            tags="",
            channel_name="",
            channel_id="",
            metadata={},
        )

        assert result is None
        assert mock_monitor.can_make_request.called

    @pytest.mark.asyncio
    async def test_generate_search_candidates(self, discogs_pass):
        """Test search candidate generation."""
        parse_result = ParseResult(
            artist="Test Artist",
            song_title="Test Song",
            confidence=0.8,
            featured_artists="Feat1, Feat2",
        )

        candidates = discogs_pass._generate_search_candidates(
            "Test Artist - Test Song (feat. Feat1 & Feat2) [Remix]", "", parse_result
        )

        assert len(candidates) > 0
        assert any(c.artist == "Test Artist" for c in candidates)
        assert any("Feat1" in c.artist or "Feat1" in c.track for c in candidates)

    @pytest.mark.asyncio
    async def test_select_best_match(self, discogs_pass):
        """Test best match selection."""
        matches = [
            DiscogsMatch(
                artist_name="Artist",
                song_title="Song",
                year=1985,
                confidence=0.7,
                release_id="1",
                master_id=None,
                genres=[],
                styles=[],
                label=None,
                country=None,
                format=None,
            ),
            DiscogsMatch(
                artist_name="Artist",
                song_title="Song Title",
                year=1985,
                confidence=0.9,
                release_id="2",
                master_id=None,
                genres=[],
                styles=[],
                label=None,
                country=None,
                format=None,
            ),
            DiscogsMatch(
                artist_name="Different Artist",
                song_title="Song",
                year=1985,
                confidence=0.6,
                release_id="3",
                master_id=None,
                genres=[],
                styles=[],
                label=None,
                country=None,
                format=None,
            ),
        ]

        parse_result = ParseResult(artist="Artist", song_title="Song Title", confidence=0.8)

        best_match = discogs_pass._select_best_match(matches, parse_result)

        assert best_match is not None
        assert best_match.confidence == 0.9
        assert best_match.song_title == "Song Title"

    @pytest.mark.asyncio
    async def test_parse_with_featured_artists(self, discogs_pass, mock_client):
        """Test parsing with featured artists."""
        mock_client.search_release.return_value = [
            DiscogsMatch(
                artist_name="Main Artist feat. Featured Artist",
                song_title="Collaboration Song",
                year=1985,
                confidence=0.85,
                release_id="456",
                master_id=None,
                genres=[],
                styles=[],
                label=None,
                country=None,
                format=None,
            )
        ]

        result = await discogs_pass.parse(
            title="Main Artist feat. Featured Artist - Collaboration Song",
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
    async def test_confidence_adjustment(self, discogs_pass):
        """Test confidence score adjustment based on match quality."""
        # Perfect match should maintain high confidence
        match1 = DiscogsMatch(
            artist_name="Exact Artist",
            song_title="Exact Title",
            year=1985,
            confidence=0.95,
            release_id="1",
            master_id=None,
            genres=[],
            styles=[],
            label=None,
            country=None,
            format=None,
        )

        parse_result1 = ParseResult(artist="Exact Artist", song_title="Exact Title", confidence=0.8)

        adjusted1 = discogs_pass._adjust_confidence(match1, parse_result1)
        assert adjusted1 > 0.9

        # Poor match should have lower confidence
        match2 = DiscogsMatch(
            artist_name="Different Artist",
            song_title="Different Title",
            year=1985,
            confidence=0.95,
            release_id="2",
            master_id=None,
            genres=[],
            styles=[],
            label=None,
            country=None,
            format=None,
        )

        parse_result2 = ParseResult(
            artist="Original Artist", song_title="Original Title", confidence=0.8
        )

        adjusted2 = discogs_pass._adjust_confidence(match2, parse_result2)
        assert adjusted2 < 0.5

    @pytest.mark.asyncio
    async def test_parse_with_year_validation(self, discogs_pass, mock_client):
        """Test that future years are handled correctly."""
        mock_client.search_release.return_value = [
            DiscogsMatch(
                artist_name="Artist",
                song_title="Song",
                year=2020,  # Recent year
                confidence=0.9,
                release_id="456",
                master_id=None,
                genres=[],
                styles=[],
                label=None,
                country=None,
                format=None,
            )
        ]

        result = await discogs_pass.parse(
            title="Artist - Song",
            description="",
            tags="",
            channel_name="",
            channel_id="",
            metadata={},
        )

        # Should still return result but year might be filtered
        assert result is not None

    @pytest.mark.asyncio
    async def test_multiple_search_strategies(self, discogs_pass, mock_client):
        """Test that multiple search strategies are attempted."""
        # First search returns nothing
        # Second search returns a match
        call_count = 0

        async def search_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return []  # First search fails
            else:
                return [
                    DiscogsMatch(
                        artist_name="Artist",
                        song_title="Song",
                        year=1985,
                        confidence=0.8,
                        release_id="456",
                        master_id=None,
                        genres=[],
                        styles=[],
                        label=None,
                        country=None,
                        format=None,
                    )
                ]

        mock_client.search_release.side_effect = search_side_effect

        result = await discogs_pass.parse(
            title="Artist - Song (Remix)",
            description="",
            tags="",
            channel_name="",
            channel_id="",
            metadata={},
        )

        assert result is not None
        assert call_count > 1  # Multiple searches attempted
