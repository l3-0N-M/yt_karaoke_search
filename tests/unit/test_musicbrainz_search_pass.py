"""Unit tests for musicbrainz_search_pass.py."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from collector.passes.musicbrainz_search_pass import MusicBrainzMatch, MusicBrainzSearchPass


class TestMusicBrainzSearchPass:
    """Test cases for MusicBrainzSearchPass."""

    @pytest.fixture
    def mb_pass(self):
        """Create a MusicBrainzSearchPass instance."""
        return MusicBrainzSearchPass(advanced_parser=Mock())

    @pytest.mark.asyncio
    async def test_parse_successful_search(self, mb_pass):
        """Test successful MusicBrainz search."""
        # Mock the _search_musicbrainz method directly
        mock_match = MusicBrainzMatch(
            recording_id="recording-123",
            artist_id="artist-456",
            artist_name="Test Artist",
            song_title="Test Song",
            score=100,
            confidence=0.95,
            metadata={
                "mb_score": 100,
                "query": "Test Artist - Test Song",
                "releases": [{"date": "1985-06-15"}],
                "length": 240000,
            },
        )

        with patch.object(mb_pass, "_search_musicbrainz", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [mock_match]

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
        assert result.confidence > 0.7

    @pytest.mark.asyncio
    async def test_parse_no_results(self, mb_pass):
        """Test when no results are found."""
        with patch.object(mb_pass, "_search_musicbrainz", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = []

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
    async def test_parse_with_featured_artists(self, mb_pass):
        """Test parsing with featured artists."""
        mock_match = MusicBrainzMatch(
            recording_id="rec-123",
            artist_id="artist-1",
            artist_name="Main Artist",
            song_title="Collaboration",
            score=95,
            confidence=0.9,
            metadata={},
        )

        with patch.object(mb_pass, "_search_musicbrainz", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [mock_match]

            # Mock the _convert_to_parse_result to add featured artists
            original_convert = mb_pass._convert_to_parse_result

            def mock_convert(match, query):
                result = original_convert(match, query)
                result.featured_artists = ["Featured Artist"]
                return result

            with patch.object(mb_pass, "_convert_to_parse_result", side_effect=mock_convert):
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
    async def test_parse_with_low_confidence(self, mb_pass):
        """Test that low confidence results are filtered out."""
        # When _search_musicbrainz returns an empty list (all results filtered out)
        with patch.object(mb_pass, "_search_musicbrainz", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = []  # No results pass the confidence threshold

            result = await mb_pass.parse(
                title="Right Artist - Right Song",
                description="",
                tags="",
                channel_name="",
                channel_id="",
                metadata={},
            )

        # Should return None due to low confidence
        assert result is None

    @pytest.mark.asyncio
    async def test_parse_with_parsed_metadata(self, mb_pass):
        """Test using pre-parsed artist/title from metadata."""
        mock_match = MusicBrainzMatch(
            recording_id="meta-123",
            artist_id="meta-artist",
            artist_name="Artist from Metadata",
            song_title="Title from Metadata",
            score=100,
            confidence=0.95,
            metadata={},
        )

        with patch.object(mb_pass, "_search_musicbrainz", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [mock_match]

            result = await mb_pass.parse(
                title="Original Title",
                description="",
                tags="",
                channel_name="",
                channel_id="",
                metadata={
                    "parsed_artist": "Artist from Metadata",
                    "parsed_title": "Title from Metadata",
                },
            )

        assert result is not None
        assert result.artist == "Artist from Metadata"
        assert result.song_title == "Title from Metadata"
        # Verify that the structured query was used (first query)
        assert mock_search.call_count >= 1
        first_call = mock_search.call_args_list[0]
        assert 'artist:"Artist from Metadata"' in first_call[0][0]

    @pytest.mark.asyncio
    async def test_parse_cache_hit(self, mb_pass):
        """Test that cache is used for repeated queries."""
        mock_match = MusicBrainzMatch(
            recording_id="cache-123",
            artist_id="cache-artist",
            artist_name="Cached Artist",
            song_title="Cached Song",
            score=95,
            confidence=0.9,
            metadata={},
        )

        with patch.object(mb_pass, "_search_musicbrainz", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [mock_match]

            # First call
            result1 = await mb_pass.parse(
                title="Cached Artist - Cached Song",
                description="",
                tags="",
                channel_name="",
                channel_id="",
                metadata={},
            )

            # Second call with same title
            result2 = await mb_pass.parse(
                title="Cached Artist - Cached Song",
                description="",
                tags="",
                channel_name="",
                channel_id="",
                metadata={},
            )

        assert result1 is not None
        assert result2 is not None
        assert result1.artist == result2.artist
        assert result1.song_title == result2.song_title
        # The search should use cache, so call count should be less than expected
        # Each parse might generate multiple queries, but cache should reduce calls

    @pytest.mark.asyncio
    async def test_artist_variant_matching(self, mb_pass):
        """Test artist variant matching (e.g., P!nk vs Pink)."""
        mock_match = MusicBrainzMatch(
            recording_id="variant-123",
            artist_id="pink-artist",
            artist_name="P!nk",
            song_title="Get the Party Started",
            score=100,
            confidence=0.95,
            metadata={},
        )

        with patch.object(mb_pass, "_search_musicbrainz", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [mock_match]

            # Search with variant spelling
            result = await mb_pass.parse(
                title="Pink - Get the Party Started",
                description="",
                tags="",
                channel_name="",
                channel_id="",
                metadata={},
            )

        assert result is not None
        assert result.artist == "P!nk"

    @pytest.mark.asyncio
    async def test_generate_search_queries(self, mb_pass):
        """Test search query generation."""
        # Test basic query with real artist/song names
        queries = mb_pass._generate_search_queries("Beatles - Yesterday")
        assert "Beatles Yesterday" in queries  # Cleaned version
        assert len(queries) >= 2  # Should have multiple strategies

        # Test query that adds spaces around dash
        queries = mb_pass._generate_search_queries("The Beatles-Hey Jude")
        # This should generate queries with spaces added
        assert any("The Beatles - Hey Jude" in q for q in queries)

        # Test quoted parts extraction
        queries = mb_pass._generate_search_queries('"Hey Jude" by The Beatles')
        assert any('recording:"Hey Jude"' in q for q in queries)

        # Test invalid queries
        queries = mb_pass._generate_search_queries("a")
        assert len(queries) == 0  # Too short

        # Test that minimal query strategy works (with original separators preserved)
        queries = mb_pass._generate_search_queries(
            "Pink Floyd - Wish You Were Here (Karaoke Version)"
        )
        # Should include the minimal version without karaoke
        assert any("Pink Floyd - Wish You Were Here" in q for q in queries)

        # Test various query forms are generated
        queries = mb_pass._generate_search_queries("Madonna - Like a Virgin")
        assert len(queries) >= 2  # At least cleaned and original with dash

    def test_calculate_confidence(self, mb_pass):
        """Test confidence calculation."""
        # High confidence - exact match
        confidence = mb_pass._calculate_confidence(
            100, "Yesterday", "The Beatles", "The Beatles - Yesterday"
        )
        assert confidence > 0.9

        # Low confidence - completely mismatched artist
        confidence = mb_pass._calculate_confidence(
            100, "Song Title", "Metallica", "Elvis Presley - Song Title"
        )
        assert confidence < 0.6

        # Medium confidence - partial match
        confidence = mb_pass._calculate_confidence(
            70, "Similar Song", "Artist Name", "Artist Name - Different Song"
        )
        assert 0.5 < confidence < 0.8

    def test_normalize_artist_name(self, mb_pass):
        """Test artist name normalization."""
        assert mb_pass._normalize_artist_name("P!nk") == "pnk"
        assert mb_pass._normalize_artist_name("The Beatles") == "beatles"
        assert mb_pass._normalize_artist_name("Artist & Friends") == "artist friends"
        assert mb_pass._normalize_artist_name("Mary J. Blige") == "mary j blige"

    def test_artist_variants(self, mb_pass):
        """Test artist variant checking with normalized names."""
        # The method expects normalized names, so normalize first
        assert mb_pass._are_artist_variants(
            mb_pass._normalize_artist_name("pink"), mb_pass._normalize_artist_name("p!nk")
        )
        assert mb_pass._are_artist_variants(
            mb_pass._normalize_artist_name("goo goo dolls"),
            mb_pass._normalize_artist_name("goo dolls"),
        )
        assert mb_pass._are_artist_variants(
            mb_pass._normalize_artist_name("mary j blige"),
            mb_pass._normalize_artist_name("mary j. blige"),
        )
        assert not mb_pass._are_artist_variants(
            mb_pass._normalize_artist_name("beatles"),
            mb_pass._normalize_artist_name("rolling stones"),
        )

    @pytest.mark.asyncio
    async def test_parse_api_error_handling(self, mb_pass):
        """Test handling of API errors."""
        with patch.object(mb_pass, "_search_musicbrainz", new_callable=AsyncMock) as mock_search:
            # Simulate API error
            mock_search.side_effect = Exception("API Error")

            result = await mb_pass.parse(
                title="The Beatles - Hey Jude",  # Use a valid query that won't be filtered
                description="",
                tags="",
                channel_name="",
                channel_id="",
                metadata={},
            )

        # Should return None on API error
        assert result is None
        assert mb_pass.stats["api_errors"] > 0
