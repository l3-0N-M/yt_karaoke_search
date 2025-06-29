"""Comprehensive tests for the Discogs search pass module."""

import asyncio
import json
import os
from dataclasses import asdict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from collector.advanced_parser import AdvancedTitleParser, ParseResult
from collector.config import CollectorConfig
from collector.passes.base import PassType
from collector.passes.discogs_search_pass import (
    DiscogsClient,
    DiscogsMatch,
    DiscogsSearchPass,
)
from collector.utils import DiscogsRateLimiter


class TestDiscogsMatch:
    """Test the DiscogsMatch dataclass."""

    def test_discogs_match_creation(self):
        """Test creating a DiscogsMatch with required values."""
        match = DiscogsMatch(
            release_id="12345",
            master_id="67890",
            artist_name="Test Artist",
            song_title="Test Song",
            year=2020,
            genres=["Pop", "Rock"],
            styles=["Alternative"],
            label="Test Label",
            country="US",
            format="Vinyl",
            confidence=0.85,
        )
        assert match.release_id == "12345"
        assert match.master_id == "67890"
        assert match.artist_name == "Test Artist"
        assert match.song_title == "Test Song"
        assert match.year == 2020
        assert match.genres == ["Pop", "Rock"]
        assert match.styles == ["Alternative"]
        assert match.label == "Test Label"
        assert match.country == "US"
        assert match.format == "Vinyl"
        assert match.confidence == 0.85
        assert match.metadata == {}

    def test_discogs_match_with_metadata(self):
        """Test creating a DiscogsMatch with metadata."""
        metadata = {"community": {"have": 150, "want": 50}, "barcode": "123456789"}
        match = DiscogsMatch(
            release_id="12345",
            master_id=None,
            artist_name="Artist",
            song_title="Song",
            year=None,
            genres=[],
            styles=[],
            label=None,
            country=None,
            format=None,
            confidence=0.9,
            metadata=metadata,
        )
        assert match.metadata == metadata

    def test_discogs_match_serializable(self):
        """Test that DiscogsMatch can be converted to dict."""
        match = DiscogsMatch(
            release_id="12345",
            master_id="67890",
            artist_name="Artist",
            song_title="Song",
            year=2020,
            genres=["Pop"],
            styles=["Alternative"],
            label="Label",
            country="US",
            format="CD",
            confidence=0.9,
        )
        match_dict = asdict(match)
        assert isinstance(match_dict, dict)
        assert match_dict["release_id"] == "12345"
        assert match_dict["confidence"] == 0.9
        assert match_dict["genres"] == ["Pop"]


class TestDiscogsClient:
    """Test the DiscogsClient class."""

    @pytest.fixture
    def rate_limiter(self):
        """Create a mock rate limiter."""
        return MagicMock(spec=DiscogsRateLimiter)

    @pytest.fixture
    def discogs_client(self, rate_limiter):
        """Create a DiscogsClient instance."""
        return DiscogsClient(
            token="test_token",
            rate_limiter=rate_limiter,
            user_agent="TestAgent/1.0"
        )

    def test_client_initialization(self, discogs_client, rate_limiter):
        """Test DiscogsClient initialization."""
        assert discogs_client.token == "test_token"
        assert discogs_client.rate_limiter == rate_limiter
        assert discogs_client.headers["Authorization"] == "Discogs token=test_token"
        assert discogs_client.headers["User-Agent"] == "TestAgent/1.0"

    def test_text_similarity(self, discogs_client):
        """Test text similarity calculation."""
        # Exact match
        assert discogs_client._text_similarity("test", "test") == 1.0
        
        # No match
        assert discogs_client._text_similarity("test", "different") == 0.0
        
        # Partial match
        similarity = discogs_client._text_similarity("test song", "test track")
        assert 0 < similarity < 1
        
        # Empty strings
        assert discogs_client._text_similarity("", "") == 0.0
        assert discogs_client._text_similarity("test", "") == 0.0

    def test_calculate_confidence(self, discogs_client):
        """Test confidence calculation."""
        item = {
            "artist": "Test Artist",
            "title": "Test Song",
            "master_id": "12345",
            "year": 2020,
            "genre": ["Pop"],
            "style": ["Alternative"],
            "community": {"have": 200}
        }
        
        confidence = discogs_client._calculate_confidence(
            item, "Test Artist", "Test Song", "Test Artist", "Test Song"
        )
        
        # Should be high confidence due to exact match and good metadata
        assert confidence > 0.8
        assert confidence <= 1.0

    def test_parse_search_result(self, discogs_client):
        """Test parsing of search result."""
        item = {
            "id": "12345",
            "master_id": "67890",
            "artist": "Test Artist",
            "title": "Test Song",
            "year": 2020,
            "genre": ["Pop", "Rock"],
            "style": ["Alternative"],
            "label": ["Test Label"],
            "country": "US",
            "format": ["Vinyl"],
            "community": {"have": 150, "want": 50},
            "barcode": ["123456789"],
            "catno": "TL001"
        }
        
        result = discogs_client._parse_search_result(item, "Test Artist", "Test Song")
        
        assert result is not None
        assert result.release_id == "12345"
        assert result.master_id == "67890"
        assert result.artist_name == "Test Artist"
        assert result.song_title == "Test Song"
        assert result.year == 2020
        assert result.genres == ["Pop", "Rock"]
        assert result.styles == ["Alternative"]
        assert result.label == "Test Label"
        assert result.country == "US"
        assert result.format == "Vinyl"
        assert result.confidence > 0.0

    @pytest.mark.asyncio
    async def test_search_release_no_aiohttp(self, discogs_client):
        """Test search_release when aiohttp is not available."""
        with patch('collector.passes.discogs_search_pass.HAS_AIOHTTP', False):
            results = await discogs_client.search_release("Artist", "Song")
            assert results == []

    @pytest.mark.asyncio
    async def test_search_release_success(self, discogs_client):
        """Test successful search_release."""
        mock_response_data = {
            "results": [
                {
                    "id": "12345",
                    "artist": "Test Artist",
                    "title": "Test Song",
                    "year": 2020,
                    "genre": ["Pop"],
                    "style": ["Alternative"]
                }
            ]
        }
        
        with patch('collector.passes.discogs_search_pass.aiohttp') as mock_aiohttp:
            # Setup mock response
            mock_response = AsyncMock()
            mock_response.raise_for_status = AsyncMock()
            mock_response.json = AsyncMock(return_value=mock_response_data)
            
            # Setup mock session
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)
            
            mock_aiohttp.ClientSession.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_aiohttp.ClientSession.return_value.__aexit__ = AsyncMock(return_value=None)
            
            results = await discogs_client.search_release("Test Artist", "Test Song")
            
            assert len(results) == 1
            assert results[0].release_id == "12345"
            assert results[0].artist_name == "Test Artist"

    @pytest.mark.asyncio
    async def test_search_release_timeout(self, discogs_client):
        """Test search_release with timeout."""
        with patch('collector.passes.discogs_search_pass.aiohttp') as mock_aiohttp:
            mock_aiohttp.ClientSession.side_effect = asyncio.TimeoutError()
            
            results = await discogs_client.search_release("Artist", "Song")
            assert results == []


class TestDiscogsSearchPass:
    """Test the DiscogsSearchPass class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = MagicMock()
        config.data_sources.discogs_enabled = True
        config.data_sources.discogs_requests_per_minute = 60
        config.data_sources.discogs_user_agent = "TestAgent/1.0"
        config.data_sources.discogs_use_as_fallback = True
        config.data_sources.discogs_min_musicbrainz_confidence = 0.6
        config.data_sources.discogs_max_results_per_search = 10
        config.data_sources.discogs_timeout = 10
        config.data_sources.discogs_confidence_threshold = 0.5
        return config

    @pytest.fixture
    def mock_advanced_parser(self):
        """Create a mock AdvancedTitleParser."""
        return MagicMock(spec=AdvancedTitleParser)

    @pytest.fixture
    def mock_db_manager(self):
        """Create a mock database manager."""
        return MagicMock()

    @pytest.fixture
    def discogs_pass_with_token(self, mock_advanced_parser, mock_config, mock_db_manager):
        """Create a DiscogsSearchPass with token."""
        with patch.dict(os.environ, {'DISCOGS_TOKEN': 'test_token'}):
            return DiscogsSearchPass(mock_advanced_parser, mock_config, mock_db_manager)

    @pytest.fixture
    def discogs_pass_no_token(self, mock_advanced_parser, mock_config, mock_db_manager):
        """Create a DiscogsSearchPass without token."""
        with patch.dict(os.environ, {}, clear=True):
            return DiscogsSearchPass(mock_advanced_parser, mock_config, mock_db_manager)

    def test_pass_type(self, discogs_pass_with_token):
        """Test that the pass type is correctly set."""
        assert discogs_pass_with_token.pass_type == PassType.DISCOGS_SEARCH

    def test_initialization_with_token(self, discogs_pass_with_token):
        """Test initialization with token."""
        assert discogs_pass_with_token.client is not None
        assert discogs_pass_with_token.token == "test_token"
        assert isinstance(discogs_pass_with_token.stats, dict)

    def test_initialization_without_token(self, discogs_pass_no_token):
        """Test initialization without token."""
        assert discogs_pass_no_token.client is None
        assert discogs_pass_no_token.token is None

    def test_initialization_with_discogs_dash_token(self, mock_advanced_parser, mock_config, mock_db_manager):
        """Test initialization with DISCOGS-TOKEN environment variable."""
        with patch.dict(os.environ, {'DISCOGS-TOKEN': 'dash_token'}, clear=True):
            pass_instance = DiscogsSearchPass(mock_advanced_parser, mock_config, mock_db_manager)
            assert pass_instance.token == "dash_token"
            assert pass_instance.client is not None

    def test_extract_search_candidates(self, discogs_pass_with_token):
        """Test extraction of search candidates from title."""
        # Test common patterns
        candidates = discogs_pass_with_token._extract_search_candidates("Artist - Song Title")
        assert len(candidates) > 0
        assert ("Artist", "Song Title") in candidates

        candidates = discogs_pass_with_token._extract_search_candidates("Artist – Song Title")
        assert len(candidates) > 0
        
        candidates = discogs_pass_with_token._extract_search_candidates("Artist : Song Title")
        assert len(candidates) > 0
        
        candidates = discogs_pass_with_token._extract_search_candidates("Artist | Song Title")
        assert len(candidates) > 0

    def test_extract_search_candidates_with_parser(self, discogs_pass_with_token, mock_advanced_parser):
        """Test extraction using advanced parser."""
        mock_parse_result = MagicMock()
        mock_parse_result.original_artist = "Parser Artist"
        mock_parse_result.song_title = "Parser Song"
        mock_advanced_parser.parse_title.return_value = mock_parse_result
        
        candidates = discogs_pass_with_token._extract_search_candidates("Some Complex Title")
        assert ("Parser Artist", "Parser Song") in candidates

    @pytest.mark.asyncio
    async def test_parse_disabled(self, discogs_pass_with_token, mock_config):
        """Test parse when Discogs is disabled."""
        mock_config.data_sources.discogs_enabled = False
        
        result = await discogs_pass_with_token.parse("Artist - Song")
        assert result is None

    @pytest.mark.asyncio
    async def test_parse_no_client(self, discogs_pass_no_token):
        """Test parse when client is not available."""
        result = await discogs_pass_no_token.parse("Artist - Song")
        assert result is None

    @pytest.mark.asyncio
    async def test_parse_fallback_logic(self, discogs_pass_with_token, mock_config):
        """Test fallback logic based on MusicBrainz confidence."""
        mock_config.data_sources.discogs_use_as_fallback = True
        mock_config.data_sources.discogs_min_musicbrainz_confidence = 0.6
        
        # High MusicBrainz confidence - should skip Discogs
        metadata = {"musicbrainz_confidence": 0.8}
        result = await discogs_pass_with_token.parse("Artist - Song", metadata=metadata)
        assert result is None
        
        # Low MusicBrainz confidence - should proceed with Discogs
        metadata = {"musicbrainz_confidence": 0.4}
        with patch.object(discogs_pass_with_token.client, 'search_release', return_value=[]):
            result = await discogs_pass_with_token.parse("Artist - Song", metadata=metadata)
            # Should attempt search but return None due to no results

    @pytest.mark.asyncio
    async def test_parse_successful_match(self, discogs_pass_with_token):
        """Test successful parsing with good match."""
        mock_match = DiscogsMatch(
            release_id="12345",
            master_id="67890",
            artist_name="Test Artist",
            song_title="Test Song",
            year=2020,
            genres=["Pop"],
            styles=["Alternative"],
            label="Test Label",
            country="US",
            format="CD",
            confidence=0.85,
            metadata={"discogs_url": "https://www.discogs.com/release/12345"}
        )
        
        with patch.object(discogs_pass_with_token.client, 'search_release', return_value=[mock_match]):
            result = await discogs_pass_with_token.parse("Test Artist - Test Song")
            
            assert result is not None
            assert isinstance(result, ParseResult)
            assert result.original_artist == "Test Artist"
            assert result.song_title == "Test Song"
            assert result.confidence == 0.85
            assert result.metadata["source"] == "discogs"
            assert result.metadata["discogs_release_id"] == "12345"
            assert result.metadata["year"] == 2020
            assert result.metadata["genres"] == ["Pop"]

    @pytest.mark.asyncio
    async def test_parse_low_confidence_match(self, discogs_pass_with_token, mock_config):
        """Test parsing with low confidence match."""
        mock_config.data_sources.discogs_confidence_threshold = 0.8
        
        mock_match = DiscogsMatch(
            release_id="12345",
            master_id=None,
            artist_name="Test Artist",
            song_title="Test Song",
            year=None,
            genres=[],
            styles=[],
            label=None,
            country=None,
            format=None,
            confidence=0.3,  # Below threshold
        )
        
        with patch.object(discogs_pass_with_token.client, 'search_release', return_value=[mock_match]):
            result = await discogs_pass_with_token.parse("Test Artist - Test Song")
            assert result is None  # Should reject low confidence match

    @pytest.mark.asyncio
    async def test_parse_no_candidates(self, discogs_pass_with_token):
        """Test parsing when no search candidates found."""
        result = await discogs_pass_with_token.parse("Invalid Title Format")
        assert result is None

    @pytest.mark.asyncio
    async def test_parse_api_error(self, discogs_pass_with_token):
        """Test parsing when API error occurs."""
        with patch.object(discogs_pass_with_token.client, 'search_release', side_effect=Exception("API Error")):
            result = await discogs_pass_with_token.parse("Artist - Song")
            assert result is None

    def test_get_statistics(self, discogs_pass_with_token):
        """Test statistics retrieval."""
        stats = discogs_pass_with_token.get_statistics()
        assert isinstance(stats, dict)
        assert "total_searches" in stats
        assert "successful_matches" in stats
        assert "api_errors" in stats
        assert "high_confidence_matches" in stats
        assert "fallback_activations" in stats


class TestDiscogsIntegration:
    """Integration tests for Discogs functionality."""

    @pytest.mark.asyncio
    async def test_end_to_end_flow(self):
        """Test complete end-to-end Discogs search flow."""
        # This test would require actual API calls or comprehensive mocking
        # For now, we'll test the basic flow structure
        
        config = MagicMock()
        config.data_sources.discogs_enabled = True
        config.data_sources.discogs_requests_per_minute = 60
        config.data_sources.discogs_user_agent = "TestAgent/1.0"
        config.data_sources.discogs_use_as_fallback = False
        config.data_sources.discogs_confidence_threshold = 0.5
        config.data_sources.discogs_max_results_per_search = 5
        config.data_sources.discogs_timeout = 10
        
        advanced_parser = MagicMock()
        
        with patch.dict(os.environ, {'DISCOGS_TOKEN': 'test_token'}):
            pass_instance = DiscogsSearchPass(advanced_parser, config)
            
            # Mock a successful search
            mock_match = DiscogsMatch(
                release_id="12345",
                master_id=None,
                artist_name="Adele",
                song_title="Hello",
                year=2015,
                genres=["Pop"],
                styles=["Ballad"],
                label="XL Recordings",
                country="UK",
                format="CD",
                confidence=0.9,
            )
            
            with patch.object(pass_instance.client, 'search_release', return_value=[mock_match]):
                result = await pass_instance.parse("Adele - Hello")
                
                assert result is not None
                assert result.original_artist == "Adele"
                assert result.song_title == "Hello"
                assert result.metadata["year"] == 2015
                assert result.metadata["genres"] == ["Pop"]
                assert result.metadata["label"] == "XL Recordings"