"""Comprehensive tests for the web search pass module."""

from dataclasses import asdict
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from collector.advanced_parser import AdvancedTitleParser, ParseResult
from collector.enhanced_search import MultiStrategySearchEngine
from collector.passes.base import PassType
from collector.passes.web_search_pass import (
    EnhancedWebSearchPass,
    FillerWordProcessor,
    QueryCleaningResult,
    SERPCache,
    SERPCacheEntry,
)
from collector.search.providers.base import SearchResult


class TestSERPCacheEntry:
    """Test the SERPCacheEntry dataclass."""

    def test_serp_cache_entry_creation(self):
        """Test creating a SERPCacheEntry with default values."""
        now = datetime.now()
        entry = SERPCacheEntry(
            query="test query",
            query_hash="test_hash",
            results=[{"video_id": "123", "title": "Test"}],
            created_at=now,
            last_accessed=now,
        )
        assert entry.query == "test query"
        assert entry.query_hash == "test_hash"
        assert len(entry.results) == 1
        assert entry.access_count == 0
        assert entry.ttl_hours == 168

    def test_serp_cache_entry_with_custom_values(self):
        """Test creating a SERPCacheEntry with custom values."""
        now = datetime.now()
        entry = SERPCacheEntry(
            query="test query",
            query_hash="test_hash",
            results=[],
            created_at=now,
            last_accessed=now,
            access_count=5,
            ttl_hours=24,
        )
        assert entry.access_count == 5
        assert entry.ttl_hours == 24

    def test_serp_cache_entry_serializable(self):
        """Test that SERPCacheEntry can be converted to dict."""
        now = datetime.now()
        entry = SERPCacheEntry(
            query="test query",
            query_hash="test_hash",
            results=[{"test": "data"}],
            created_at=now,
            last_accessed=now,
        )
        entry_dict = asdict(entry)
        assert isinstance(entry_dict, dict)
        assert entry_dict["query"] == "test query"
        assert entry_dict["results"] == [{"test": "data"}]


class TestQueryCleaningResult:
    """Test the QueryCleaningResult dataclass."""

    def test_query_cleaning_result_creation(self):
        """Test creating a QueryCleaningResult with default values."""
        result = QueryCleaningResult(
            original_query="test query",
            cleaned_query="clean query",
            removed_terms=["noise"],
        )
        assert result.original_query == "test query"
        assert result.cleaned_query == "clean query"
        assert result.removed_terms == ["noise"]
        assert result.confidence_boost == 1.0
        assert result.cleaning_method == ""

    def test_query_cleaning_result_with_values(self):
        """Test creating a QueryCleaningResult with all values."""
        result = QueryCleaningResult(
            original_query="test query",
            cleaned_query="clean query",
            removed_terms=["noise", "filler"],
            confidence_boost=1.2,
            cleaning_method="english_filler_removal",
        )
        assert result.confidence_boost == 1.2
        assert result.cleaning_method == "english_filler_removal"
        assert len(result.removed_terms) == 2


class TestFillerWordProcessor:
    """Test the FillerWordProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create a FillerWordProcessor instance."""
        return FillerWordProcessor()

    def test_processor_initialization(self, processor):
        """Test processor initialization."""
        assert "english" in processor.filler_words
        assert "karaoke_terms" in processor.filler_words["english"]
        assert "quality_terms" in processor.filler_words["english"]
        assert "english" in processor.compiled_patterns

    def test_clean_query_basic_english(self, processor):
        """Test basic English query cleaning."""
        query = "Artist Name - Song Title Karaoke HD"
        result = processor.clean_query(query, "english")

        assert "karaoke" not in result.cleaned_query.lower()
        assert "hd" not in result.cleaned_query.lower()
        assert "Artist Name - Song Title" in result.cleaned_query
        assert "karaoke" in [term.lower() for term in result.removed_terms]
        assert result.confidence_boost > 1.0

    def test_clean_query_karaoke_terms(self, processor):
        """Test removal of karaoke-specific terms."""
        test_cases = [
            "Song Title Karaoke",
            "Artist - Song Instrumental",
            "Song Title Backing Track",
            "Artist - Song (Minus One)",
            "Song Title Playback",
        ]

        for query in test_cases:
            result = processor.clean_query(query, "english")
            # Should remove karaoke-related terms
            assert result.confidence_boost >= 1.2  # Strong boost for karaoke terms
            assert len(result.removed_terms) > 0

    def test_clean_query_quality_terms(self, processor):
        """Test removal of quality terms."""
        query = "Artist - Song HD 1080p High Quality"
        result = processor.clean_query(query, "english")

        assert "hd" not in result.cleaned_query.lower()
        assert "1080p" not in result.cleaned_query.lower()
        assert "high quality" not in result.cleaned_query.lower()
        assert result.confidence_boost > 1.0

    def test_clean_query_video_terms(self, processor):
        """Test removal of video-related terms."""
        query = "Artist - Song Official Music Video"
        result = processor.clean_query(query, "english")

        assert "official" not in result.cleaned_query.lower()
        assert "music video" not in result.cleaned_query.lower()
        assert result.confidence_boost > 1.0

    def test_clean_query_spanish(self, processor):
        """Test Spanish query cleaning."""
        query = "Artista - Canción Karaoke Pista"
        result = processor.clean_query(query, "spanish")

        assert "pista" not in result.cleaned_query.lower()
        assert result.confidence_boost > 1.0

    def test_clean_query_fallback_to_english(self, processor):
        """Test fallback to English for unknown languages."""
        query = "Artist - Song Karaoke"
        result = processor.clean_query(query, "unknown_language")

        # Should still remove English karaoke terms
        assert "karaoke" not in result.cleaned_query.lower()
        assert result.confidence_boost > 1.0

    def test_clean_query_no_changes(self, processor):
        """Test query with no filler words."""
        query = "Clean Artist Name Song Title"
        result = processor.clean_query(query, "english")

        assert result.cleaned_query == query
        assert result.removed_terms == []
        assert result.confidence_boost == 1.0

    def test_additional_cleaning_id_removal(self, processor):
        """Test removal of large ID numbers."""
        query = "Artist - Song 123456789"
        result = processor.clean_query(query, "english")

        assert "123456789" not in result.cleaned_query

    def test_additional_cleaning_language_prefixes(self, processor):
        """Test removal of language prefixes."""
        query = "DE Artist - Song Title"
        result = processor.clean_query(query, "english")

        assert not result.cleaned_query.startswith("DE")

    def test_additional_cleaning_timestamps(self, processor):
        """Test removal of timestamps."""
        query = "Artist - Song 3:45 Duration"
        result = processor.clean_query(query, "english")

        assert "3:45" not in result.cleaned_query

    def test_additional_cleaning_short_words(self, processor):
        """Test removal of very short words while preserving important ones."""
        query = "Artist - Song a I am to x"
        result = processor.clean_query(query, "english")

        # Should keep important short words like "a", "I", "am", "to"
        assert "a" in result.cleaned_query.lower()
        assert "am" in result.cleaned_query.lower()
        # Should remove meaningless single letters like "x"
        assert result.cleaned_query.count("x") == 0

    def test_additional_cleaning_duplicate_removal(self, processor):
        """Test removal of duplicate words."""
        query = "Artist Artist - Song Song Title"
        result = processor.clean_query(query, "english")

        # Should only have one instance of each word
        words = result.cleaned_query.lower().split()
        assert words.count("artist") == 1
        assert words.count("song") == 1

    @pytest.mark.parametrize(
        "input_query,language,expected_in_cleaned",
        [
            ("Artist - Song Karaoke", "english", "Artist - Song"),
            ("Artista - Canción Karaoké", "french", "Artista - Canción"),
            ("Artist - Song HD 4K", "english", "Artist - Song"),
            ("Official Artist Song Video", "english", "Artist Song"),
            ("123456 Artist Song", "english", "Artist Song"),
        ],
    )
    def test_clean_query_various_patterns(
        self, processor, input_query, language, expected_in_cleaned
    ):
        """Test various cleaning patterns."""
        result = processor.clean_query(input_query, language)
        assert expected_in_cleaned.lower() in result.cleaned_query.lower()


class TestSERPCache:
    """Test the SERPCache class."""

    @pytest.fixture
    def cache(self):
        """Create a SERPCache instance."""
        return SERPCache(max_entries=100, default_ttl_hours=24)

    def test_cache_initialization(self, cache):
        """Test cache initialization."""
        assert cache.max_entries == 100
        assert cache.default_ttl_hours == 24
        assert len(cache.cache) == 0

    def test_get_query_hash(self, cache):
        """Test query hash generation."""
        query = "test query"
        hash1 = cache._get_query_hash(query)
        hash2 = cache._get_query_hash(query.upper())  # Different case

        assert len(hash1) == 32  # MD5 hash length
        assert hash1 == hash2  # Case insensitive

    def test_put_and_get(self, cache):
        """Test putting and getting cache entries."""
        query = "test query"
        results = [{"video_id": "123", "title": "Test Video"}]

        cache.put(query, results)
        cached_results = cache.get(query)

        assert cached_results == results
        assert len(cache.cache) == 1

    def test_get_nonexistent(self, cache):
        """Test getting non-existent cache entry."""
        result = cache.get("nonexistent query")
        assert result is None

    def test_get_expired_entry(self, cache):
        """Test getting expired cache entry."""
        query = "test query"
        results = [{"video_id": "123", "title": "Test Video"}]

        # Create entry with very short TTL
        cache.put(query, results, ttl_hours=0.001)  # Very short TTL

        # Wait a bit and try to get it
        import time

        time.sleep(0.1)

        cached_results = cache.get(query)
        assert cached_results is None  # Should be expired and removed

    def test_access_count_update(self, cache):
        """Test that access count is updated."""
        query = "test query"
        results = [{"video_id": "123", "title": "Test Video"}]

        cache.put(query, results)

        # Access multiple times
        cache.get(query)
        cache.get(query)
        cache.get(query)

        query_hash = cache._get_query_hash(query)
        entry = cache.cache[query_hash]
        assert entry.access_count == 3

    def test_cleanup_expired(self, cache):
        """Test cleanup of expired entries."""
        # Add some entries with different TTLs
        cache.put("fresh_query", [{"id": "1"}], ttl_hours=24)
        cache.put("expired_query", [{"id": "2"}], ttl_hours=0.001)

        assert len(cache.cache) == 2

        cache.cleanup_expired()

        # Expired entry should be removed
        assert len(cache.cache) == 1
        assert cache.get("fresh_query") is not None
        assert cache.get("expired_query") is None

    def test_evict_lru(self, cache):
        """Test LRU eviction when cache is full."""
        # Fill cache to max capacity
        for i in range(cache.max_entries):
            cache.put(f"query_{i}", [{"id": str(i)}])

        assert len(cache.cache) == cache.max_entries

        # Access first entry to make it recently used
        cache.get("query_0")

        # Add one more entry to trigger eviction
        cache.put("new_query", [{"id": "new"}])

        # Should still be at max capacity
        assert len(cache.cache) == cache.max_entries

        # First entry should still be there (recently accessed)
        assert cache.get("query_0") is not None

        # Some other entry should be evicted
        assert cache.get("new_query") is not None

    def test_get_statistics_empty(self, cache):
        """Test statistics for empty cache."""
        stats = cache.get_statistics()

        assert stats["total_entries"] == 0
        assert stats["expired_entries"] == 0
        assert stats["active_entries"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["avg_age_hours"] == 0.0
        assert stats["most_accessed_queries"] == []

    def test_get_statistics_with_data(self, cache):
        """Test statistics with cache data."""
        # Add some entries
        cache.put("query1", [{"id": "1"}])
        cache.put("query2", [{"id": "2"}])

        # Access them to update statistics
        cache.get("query1")
        cache.get("query1")
        cache.get("query2")

        cache.set_total_requests(5)  # Simulate total requests

        stats = cache.get_statistics()

        assert stats["total_entries"] == 2
        assert stats["active_entries"] == 2
        assert stats["hit_rate"] > 0.0
        assert len(stats["most_accessed_queries"]) == 2


class TestEnhancedWebSearchPass:
    """Test the EnhancedWebSearchPass class."""

    @pytest.fixture
    def mock_advanced_parser(self):
        """Create a mock AdvancedTitleParser."""
        parser = MagicMock(spec=AdvancedTitleParser)
        parser.parse_title.return_value = ParseResult(
            artist="Test Artist",
            song_title="Test Song",
            confidence=0.8,
            method="advanced_parser",
        )
        return parser

    @pytest.fixture
    def mock_search_engine(self):
        """Create a mock MultiStrategySearchEngine."""
        engine = MagicMock(spec=MultiStrategySearchEngine)
        engine.search_videos = AsyncMock(
            return_value=[
                SearchResult(
                    video_id="test_video_1",
                    url="http://test1.com",
                    title="Test Artist - Test Song (Karaoke)",
                    channel="Test Channel",
                    channel_id="test_channel_1",
                ),
                SearchResult(
                    video_id="test_video_2",
                    url="http://test2.com",
                    title="Another Artist - Another Song",
                    channel="Another Channel",
                    channel_id="test_channel_2",
                ),
            ]
        )
        return engine

    @pytest.fixture
    def web_search_pass(self, mock_advanced_parser, mock_search_engine):
        """Create an EnhancedWebSearchPass instance."""
        return EnhancedWebSearchPass(mock_advanced_parser, mock_search_engine)

    def test_pass_type(self, web_search_pass):
        """Test that the pass type is correctly set."""
        assert web_search_pass.pass_type == PassType.WEB_SEARCH

    def test_initialization(self, web_search_pass):
        """Test web search pass initialization."""
        assert web_search_pass.advanced_parser is not None
        assert web_search_pass.search_engine is not None
        assert isinstance(web_search_pass.filler_processor, FillerWordProcessor)
        assert isinstance(web_search_pass.serp_cache, SERPCache)
        assert web_search_pass.max_search_results == 50
        assert web_search_pass.query_expansion_enabled is True

    def test_initialization_with_db(self, mock_advanced_parser, mock_search_engine):
        """Test initialization with database manager."""
        mock_db = MagicMock()
        web_search_pass = EnhancedWebSearchPass(mock_advanced_parser, mock_search_engine, mock_db)
        assert web_search_pass.db_manager == mock_db


class TestSearchQueryGeneration:
    """Test search query generation methods."""

    @pytest.fixture
    def web_search_pass(self, mock_advanced_parser, mock_search_engine):
        return EnhancedWebSearchPass(mock_advanced_parser, mock_search_engine)

    def test_generate_search_queries_basic(self, web_search_pass):
        """Test basic search query generation."""
        title = "Artist Name - Song Title (Karaoke HD)"
        queries = web_search_pass._generate_search_queries(title, "", "")

        assert len(queries) > 0
        # Should have base cleaning
        assert any(q.cleaning_method.endswith("filler_removal") for q in queries)
        # All queries should be meaningful length
        assert all(len(q.cleaned_query) > 3 for q in queries)

    def test_generate_search_queries_with_description(self, web_search_pass):
        """Test query generation with description."""
        title = "Song Title"
        description = "Great song by Famous Artist with amazing vocals"
        queries = web_search_pass._generate_search_queries(title, description, "")

        # Should generate multiple strategies
        assert len(queries) > 1

    def test_generate_search_queries_quoted_extraction(self, web_search_pass):
        """Test extraction of quoted parts."""
        title = 'Artist - "Song Title" (Karaoke Version)'
        queries = web_search_pass._generate_search_queries(title, "", "")

        # Should extract quoted song title
        assert any("quoted_extraction" in q.cleaning_method for q in queries)
        quoted_queries = [q for q in queries if "quoted_extraction" in q.cleaning_method]
        assert len(quoted_queries) > 0
        assert "Song Title" in quoted_queries[0].cleaned_query

    def test_generate_search_queries_non_ascii(self, web_search_pass):
        """Test query generation with non-ASCII characters."""
        title = "Artista - Canción (Karaoke)"  # Spanish
        queries = web_search_pass._generate_search_queries(title, "", "")

        # Should try language-specific cleaning
        language_specific = [q for q in queries if "_specific" in q.cleaning_method]
        assert len(language_specific) > 0

    def test_generate_search_queries_minimal_cleaning(self, web_search_pass):
        """Test minimal cleaning strategy."""
        title = "Artist - Song Karaoke Instrumental HD"
        queries = web_search_pass._generate_search_queries(title, "", "")

        # Should include minimal cleaning strategy
        minimal_queries = [q for q in queries if q.cleaning_method == "minimal"]
        assert len(minimal_queries) > 0

    def test_generate_search_queries_deduplication(self, web_search_pass):
        """Test that duplicate queries are removed."""
        title = "Simple Title"
        queries = web_search_pass._generate_search_queries(title, "", "")

        # All queries should be unique
        cleaned_queries = [q.cleaned_query for q in queries]
        assert len(cleaned_queries) == len(set(cleaned_queries))

    def test_generate_search_queries_sorting(self, web_search_pass):
        """Test that queries are sorted by confidence boost."""
        title = "Artist - Song Karaoke HD Quality"
        queries = web_search_pass._generate_search_queries(title, "", "")

        # Should be sorted by confidence boost (highest first)
        confidence_boosts = [q.confidence_boost for q in queries]
        assert confidence_boosts == sorted(confidence_boosts, reverse=True)

    def test_generate_search_queries_limit(self, web_search_pass):
        """Test that query count is limited."""
        title = "Very Complex Title with Many Terms Karaoke HD Official Video"
        queries = web_search_pass._generate_search_queries(title, "Long description", "tags")

        # Should be limited to 5 queries
        assert len(queries) <= 5


class TestSearchExecution:
    """Test search execution methods."""

    @pytest.fixture
    def web_search_pass(self, mock_advanced_parser, mock_search_engine):
        return EnhancedWebSearchPass(mock_advanced_parser, mock_search_engine)

    @pytest.mark.asyncio
    async def test_search_with_engine_success(self, web_search_pass):
        """Test successful search with engine."""
        query = "test query"
        results = await web_search_pass._search_with_engine(query)

        assert len(results) > 0
        # Should call search engine
        web_search_pass.search_engine.search_videos.assert_called_once()

        # Results should be converted to dictionaries
        for result in results:
            assert isinstance(result, dict)
            assert "video_id" in result
            assert "title" in result

    @pytest.mark.asyncio
    async def test_search_with_engine_failure(self, web_search_pass):
        """Test search engine failure."""
        web_search_pass.search_engine.search_videos.side_effect = Exception("Search failed")

        query = "test query"
        results = await web_search_pass._search_with_engine(query)

        assert results == []

    @pytest.mark.asyncio
    async def test_search_with_engine_empty_results(self, web_search_pass):
        """Test search with empty results."""
        web_search_pass.search_engine.search_videos.return_value = []

        query = "test query"
        results = await web_search_pass._search_with_engine(query)

        assert results == []


class TestResultParsing:
    """Test search result parsing methods."""

    @pytest.fixture
    def web_search_pass(self, mock_advanced_parser, mock_search_engine):
        return EnhancedWebSearchPass(mock_advanced_parser, mock_search_engine)

    @pytest.fixture
    def sample_search_results(self):
        """Create sample search results."""
        return [
            {
                "video_id": "video1",
                "title": "Test Artist - Test Song (Karaoke)",
                "description": "Great karaoke version",
                "channel_name": "Karaoke Channel",
                "relevance_score": 0.9,
            },
            {
                "video_id": "video2",
                "title": "Test Artist - Test Song Live",
                "description": "Live performance",
                "channel_name": "Live Channel",
                "relevance_score": 0.7,
            },
            {
                "video_id": "video3",
                "title": "Different Artist - Different Song",
                "description": "Completely different",
                "channel_name": "Other Channel",
                "relevance_score": 0.5,
            },
        ]

    @pytest.fixture
    def sample_query_info(self):
        """Create sample query cleaning result."""
        return QueryCleaningResult(
            original_query="Test Artist - Test Song Karaoke",
            cleaned_query="Test Artist - Test Song",
            removed_terms=["karaoke"],
            confidence_boost=1.2,
            cleaning_method="english_filler_removal",
        )

    @pytest.mark.asyncio
    async def test_parse_search_results_success(
        self, web_search_pass, sample_search_results, sample_query_info
    ):
        """Test successful parsing of search results."""
        result = await web_search_pass._parse_search_results(
            sample_search_results, sample_query_info, "Original Title"
        )

        assert result is not None
        assert result.original_artist == "Test Artist"
        assert result.song_title == "Test Song"
        assert result.method == "web_search_parsing"
        assert "search_ranking" in result.metadata
        assert "query_cleaning_boost" in result.metadata

    @pytest.mark.asyncio
    async def test_parse_search_results_empty(self, web_search_pass, sample_query_info):
        """Test parsing with empty search results."""
        result = await web_search_pass._parse_search_results(
            [], sample_query_info, "Original Title"
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_parse_search_results_low_confidence(
        self, web_search_pass, sample_search_results, sample_query_info
    ):
        """Test parsing with low confidence results."""
        # Mock parser to return low confidence
        web_search_pass.advanced_parser.parse_title.return_value = ParseResult(
            artist="Test Artist",
            song_title="Test Song",
            confidence=0.3,  # Low confidence
        )

        result = await web_search_pass._parse_search_results(
            sample_search_results, sample_query_info, "Original Title"
        )

        # Should not return results below threshold
        assert result is None

    @pytest.mark.asyncio
    async def test_parse_search_results_ranking_adjustment(
        self, web_search_pass, sample_search_results, sample_query_info
    ):
        """Test that ranking affects confidence."""
        result = await web_search_pass._parse_search_results(
            sample_search_results, sample_query_info, "Original Title"
        )

        assert result is not None
        # First result should have ranking bonus
        assert result.metadata["search_ranking"] == 1
        # Confidence should be adjusted by ranking and relevance factors
        assert result.confidence != 0.8  # Should be different from base confidence

    def test_select_consensus_result_single_group(self, web_search_pass):
        """Test consensus selection with single result group."""
        parse_results = [
            ParseResult(artist="Artist", song_title="Song", confidence=0.8),
            ParseResult(artist="Artist", song_title="Song", confidence=0.7),
            ParseResult(artist="Artist", song_title="Song", confidence=0.9),
        ]

        query_info = QueryCleaningResult(
            original_query="test", cleaned_query="test", removed_terms=[]
        )

        result = web_search_pass._select_consensus_result(parse_results, query_info)

        assert result is not None
        assert result.confidence == 0.9  # Highest confidence from the group
        assert result.metadata["consensus_group_size"] == 3

    def test_select_consensus_result_multiple_groups(self, web_search_pass):
        """Test consensus selection with multiple result groups."""
        parse_results = [
            ParseResult(artist="Artist1", song_title="Song1", confidence=0.6),
            ParseResult(artist="Artist1", song_title="Song1", confidence=0.5),
            ParseResult(artist="Artist2", song_title="Song2", confidence=0.8),
        ]

        query_info = QueryCleaningResult(
            original_query="test", cleaned_query="test", removed_terms=[]
        )

        result = web_search_pass._select_consensus_result(parse_results, query_info)

        assert result is not None
        # Should prefer the group with consensus even if individual confidence is lower
        # Group 1 has 2 results (avg 0.55 + 0.2 consensus bonus = 0.75)
        # Group 2 has 1 result (0.8 + 0.1 consensus bonus = 0.9)
        # Group 2 should win
        assert result.original_artist == "Artist2"

    def test_select_consensus_result_empty(self, web_search_pass):
        """Test consensus selection with empty results."""
        query_info = QueryCleaningResult(
            original_query="test", cleaned_query="test", removed_terms=[]
        )

        result = web_search_pass._select_consensus_result([], query_info)

        assert result is None

    def test_select_consensus_result_fallback(self, web_search_pass):
        """Test fallback to highest individual confidence."""
        parse_results = [
            ParseResult(artist="", song_title="", confidence=0.5),  # Empty result
            ParseResult(artist="Artist", song_title="Song", confidence=0.8),
        ]

        query_info = QueryCleaningResult(
            original_query="test", cleaned_query="test", removed_terms=[]
        )

        result = web_search_pass._select_consensus_result(parse_results, query_info)

        assert result is not None
        assert result.confidence == 0.8  # Should pick highest confidence as fallback


class TestMainParseMethod:
    """Test the main parse method."""

    @pytest.fixture
    def web_search_pass(self, mock_advanced_parser, mock_search_engine):
        return EnhancedWebSearchPass(mock_advanced_parser, mock_search_engine)

    @pytest.mark.asyncio
    async def test_parse_successful(self, web_search_pass):
        """Test successful parse execution."""
        title = "Test Artist - Test Song (Karaoke)"

        result = await web_search_pass.parse(title)

        assert result is not None
        assert result.original_artist == "Test Artist"
        assert result.song_title == "Test Song"
        assert web_search_pass.search_stats["total_searches"] == 1
        assert web_search_pass.search_stats["successful_parses"] == 1

    @pytest.mark.asyncio
    async def test_parse_no_queries_generated(self, web_search_pass):
        """Test parse when no queries are generated."""
        # Very short title that generates no queries
        title = "ab"

        result = await web_search_pass.parse(title)

        assert result is None

    @pytest.mark.asyncio
    async def test_parse_with_caching(self, web_search_pass):
        """Test parse with SERP caching."""
        title = "Test Artist - Test Song"

        # First call should not hit cache
        result1 = await web_search_pass.parse(title)
        assert web_search_pass.search_stats["cache_hits"] == 0

        # Second call should hit cache
        result2 = await web_search_pass.parse(title)
        assert web_search_pass.search_stats["cache_hits"] == 1

        # Results should be similar
        assert result1.original_artist == result2.original_artist

    @pytest.mark.asyncio
    async def test_parse_early_exit_high_confidence(self, web_search_pass):
        """Test early exit with high confidence result."""
        # Mock parser to return very high confidence
        web_search_pass.advanced_parser.parse_title.return_value = ParseResult(
            artist="Test Artist",
            song_title="Test Song",
            confidence=0.95,  # Very high confidence
        )

        title = "Test Artist - Test Song"
        result = await web_search_pass.parse(title)

        assert result is not None
        assert result.confidence > 0.85  # Should trigger early exit

    @pytest.mark.asyncio
    async def test_parse_exception_handling(self, web_search_pass):
        """Test exception handling in parse method."""
        # Mock query generation to raise exception
        with patch.object(web_search_pass, "_generate_search_queries") as mock_generate:
            mock_generate.side_effect = Exception("Test error")

            result = await web_search_pass.parse("Test Title")

            assert result is None

    @pytest.mark.asyncio
    async def test_parse_slow_search_logging(self, web_search_pass):
        """Test logging of slow searches."""
        with patch("time.time") as mock_time, patch(
            "collector.passes.web_search_pass.logger"
        ) as mock_logger:

            mock_time.side_effect = [0, 15]  # 15 second processing time

            await web_search_pass.parse("Test Title")

            # Should log warning for slow search
            mock_logger.warning.assert_called()


class TestStatisticsAndUtilities:
    """Test statistics and utility methods."""

    @pytest.fixture
    def web_search_pass(self, mock_advanced_parser, mock_search_engine):
        return EnhancedWebSearchPass(mock_advanced_parser, mock_search_engine)

    def test_get_statistics_initial(self, web_search_pass):
        """Test initial statistics."""
        stats = web_search_pass.get_statistics()

        assert "search_performance" in stats
        assert "cache_statistics" in stats
        assert stats["success_rate"] == 0.0
        assert stats["cache_hit_rate"] == 0.0
        assert "configuration" in stats

    def test_get_statistics_with_data(self, web_search_pass):
        """Test statistics with data."""
        # Simulate some activity
        web_search_pass.search_stats["total_searches"] = 10
        web_search_pass.search_stats["successful_parses"] = 7
        web_search_pass.search_stats["cache_hits"] = 3

        stats = web_search_pass.get_statistics()

        assert stats["success_rate"] == 0.7
        assert stats["cache_hit_rate"] == 0.3
        assert stats["search_performance"]["total_searches"] == 10

    def test_clear_cache(self, web_search_pass):
        """Test cache clearing."""
        # Add some cache entries
        web_search_pass.serp_cache.put("query1", [{"id": "1"}])
        web_search_pass.serp_cache.put("query2", [{"id": "2"}])

        assert len(web_search_pass.serp_cache.cache) == 2

        web_search_pass.clear_cache()

        assert len(web_search_pass.serp_cache.cache) == 0

    def test_get_cache_size(self, web_search_pass):
        """Test getting cache size."""
        assert web_search_pass.get_cache_size() == 0

        # Add some entries
        web_search_pass.serp_cache.put("query1", [{"id": "1"}])
        web_search_pass.serp_cache.put("query2", [{"id": "2"}])

        assert web_search_pass.get_cache_size() == 2


@pytest.mark.integration
class TestWebSearchPassIntegration:
    """Integration tests for the web search pass."""

    @pytest.fixture
    def full_web_search_pass(self):
        """Create a fully functional web search pass."""
        parser = AdvancedTitleParser()
        search_engine = MagicMock(spec=MultiStrategySearchEngine)
        search_engine.search_videos = AsyncMock(
            return_value=[
                SearchResult(
                    video_id="test_video",
                    url="http://test.com",
                    title="Ed Sheeran - Shape of You (Karaoke Version)",
                    channel="Karaoke Channel",
                    channel_id="karaoke_channel",
                )
            ]
        )
        return EnhancedWebSearchPass(parser, search_engine)

    @pytest.mark.asyncio
    async def test_full_search_workflow(self, full_web_search_pass):
        """Test full search workflow."""
        title = "Ed Sheeran - Shape of You Karaoke HD"

        result = await full_web_search_pass.parse(title)

        assert result is not None
        assert result.original_artist is not None
        assert result.song_title is not None
        assert result.method == "web_search_parsing"
        assert "search_ranking" in result.metadata

    def test_filler_word_processing_comprehensive(self):
        """Test comprehensive filler word processing."""
        processor = FillerWordProcessor()

        test_cases = [
            # (input, language, expected_removed_terms)
            ("Artist - Song Karaoke HD", "english", ["karaoke", "hd"]),
            ("Artista - Canción Karaoké", "french", ["karaoké"]),
            ("Song Title Official Music Video", "english", ["official", "music", "video"]),
            ("Artist - Song 123456 (HD Quality)", "english", ["hd", "quality"]),
            ("DE Artist - Song", "english", []),  # DE should be removed but not tracked
        ]

        for input_query, language, expected_terms in test_cases:
            result = processor.clean_query(input_query, language)

            # Check that expected terms were removed
            for term in expected_terms:
                assert term.lower() in [t.lower() for t in result.removed_terms]
                assert term.lower() not in result.cleaned_query.lower()

    def test_serp_cache_comprehensive(self):
        """Test comprehensive SERP cache functionality."""
        cache = SERPCache(max_entries=3, default_ttl_hours=1)

        # Test basic operations
        cache.put("query1", [{"id": "1"}])
        cache.put("query2", [{"id": "2"}])
        cache.put("query3", [{"id": "3"}])

        assert len(cache.cache) == 3

        # Test eviction
        cache.put("query4", [{"id": "4"}])
        assert len(cache.cache) == 3  # Should evict LRU

        # Test access pattern affects eviction
        cache.get("query1")  # Make query1 recently used
        cache.put("query5", [{"id": "5"}])

        # query1 should still be there due to recent access
        assert cache.get("query1") is not None

    @pytest.mark.asyncio
    async def test_query_generation_and_ranking(self, full_web_search_pass):
        """Test query generation and ranking system."""
        test_titles = [
            "Simple Title",
            "Artist - Song (Karaoke HD)",
            'Artist - "Song Title" Official Video',
            "Artista - Canción Karaoke",  # Spanish
            "Very Long Title With Many Filler Words HD 4K Official Music Video",
        ]

        for title in test_titles:
            queries = full_web_search_pass._generate_search_queries(title, "", "")

            # Should generate meaningful queries
            assert len(queries) > 0
            assert all(len(q.cleaned_query) > 3 for q in queries)

            # Should be sorted by confidence boost
            confidence_boosts = [q.confidence_boost for q in queries]
            assert confidence_boosts == sorted(confidence_boosts, reverse=True)

            # Should not exceed limit
            assert len(queries) <= 5
