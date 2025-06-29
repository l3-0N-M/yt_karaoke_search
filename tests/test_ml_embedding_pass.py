"""Comprehensive tests for the ML embedding pass module."""

import time
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from collector.advanced_parser import AdvancedTitleParser, ParseResult
from collector.passes.base import PassType
from collector.passes.ml_embedding_pass import (
    EmbeddingMatch,
    EnhancedMLEmbeddingPass,
    SemanticCandidate,
)
from collector.search.fuzzy_matcher import FuzzyMatcher


class TestEmbeddingMatch:
    """Test the EmbeddingMatch dataclass."""

    def test_embedding_match_creation(self):
        """Test creating an EmbeddingMatch with required values."""
        match = EmbeddingMatch(
            query="test query",
            matched_text="matched text",
            similarity_score=0.85,
            method="semantic_cosine",
        )
        assert match.query == "test query"
        assert match.matched_text == "matched text"
        assert match.similarity_score == 0.85
        assert match.method == "semantic_cosine"
        assert match.artist_candidate is None
        assert match.song_candidate is None
        assert match.confidence == 0.0
        assert match.metadata == {}

    def test_embedding_match_with_all_values(self):
        """Test creating an EmbeddingMatch with all values."""
        metadata = {"score": 0.9, "method": "test"}
        match = EmbeddingMatch(
            query="test query",
            matched_text="matched text",
            similarity_score=0.85,
            method="semantic_cosine",
            artist_candidate="Test Artist",
            song_candidate="Test Song",
            confidence=0.9,
            metadata=metadata,
        )
        assert match.artist_candidate == "Test Artist"
        assert match.song_candidate == "Test Song"
        assert match.confidence == 0.9
        assert match.metadata == metadata

    def test_embedding_match_serializable(self):
        """Test that EmbeddingMatch can be converted to dict."""
        match = EmbeddingMatch(
            query="test query",
            matched_text="matched text",
            similarity_score=0.85,
            method="semantic_cosine",
        )
        match_dict = asdict(match)
        assert isinstance(match_dict, dict)
        assert match_dict["query"] == "test query"
        assert match_dict["similarity_score"] == 0.85


class TestSemanticCandidate:
    """Test the SemanticCandidate dataclass."""

    def test_semantic_candidate_creation(self):
        """Test creating a SemanticCandidate with default values."""
        candidate = SemanticCandidate(
            text="test text",
            category="artist",
        )
        assert candidate.text == "test text"
        assert candidate.category == "artist"
        assert candidate.embedding is None
        assert candidate.frequency == 1
        assert candidate.last_seen == 0.0
        assert candidate.aliases == set()

    def test_semantic_candidate_with_values(self):
        """Test creating a SemanticCandidate with specific values."""
        embedding = np.array([0.1, 0.2, 0.3])
        aliases = {"alias1", "alias2"}
        now = time.time()

        candidate = SemanticCandidate(
            text="test text",
            category="song",
            embedding=embedding,
            frequency=5,
            last_seen=now,
            aliases=aliases,
        )
        assert candidate.category == "song"
        assert np.array_equal(candidate.embedding, embedding)
        assert candidate.frequency == 5
        assert candidate.last_seen == now
        assert candidate.aliases == aliases

    def test_semantic_candidate_serializable(self):
        """Test that SemanticCandidate can be converted to dict."""
        candidate = SemanticCandidate(
            text="test text",
            category="artist",
            frequency=3,
        )
        candidate_dict = asdict(candidate)
        assert isinstance(candidate_dict, dict)
        assert candidate_dict["text"] == "test text"
        assert candidate_dict["frequency"] == 3


class TestEnhancedMLEmbeddingPassInitialization:
    """Test EnhancedMLEmbeddingPass initialization."""

    @pytest.fixture
    def mock_advanced_parser(self):
        """Create a mock AdvancedTitleParser."""
        parser = MagicMock(spec=AdvancedTitleParser)
        parser.known_artists = {"Artist1", "Artist2"}
        parser.known_songs = {"Song1", "Song2"}
        return parser

    @pytest.fixture
    def mock_fuzzy_matcher(self):
        """Create a mock FuzzyMatcher."""
        return MagicMock(spec=FuzzyMatcher)

    @pytest.fixture
    def mock_db_manager(self):
        """Create a mock database manager."""
        return MagicMock()

    def test_pass_type(self, mock_advanced_parser, mock_fuzzy_matcher):
        """Test that the pass type is correctly set."""
        ml_pass = EnhancedMLEmbeddingPass(mock_advanced_parser, mock_fuzzy_matcher)
        assert ml_pass.pass_type == PassType.ML_EMBEDDING

    def test_initialization_without_dependencies(self, mock_advanced_parser, mock_fuzzy_matcher):
        """Test initialization without ML dependencies."""
        with patch("collector.passes.ml_embedding_pass.HAS_SENTENCE_TRANSFORMERS", False):
            ml_pass = EnhancedMLEmbeddingPass(mock_advanced_parser, mock_fuzzy_matcher)

            assert ml_pass.advanced_parser == mock_advanced_parser
            assert ml_pass.fuzzy_matcher == mock_fuzzy_matcher
            assert ml_pass.embedding_model is None
            assert isinstance(ml_pass.artist_candidates, dict)
            assert isinstance(ml_pass.song_candidates, dict)
            assert isinstance(ml_pass.embedding_cache, dict)

    def test_initialization_with_db(
        self, mock_advanced_parser, mock_fuzzy_matcher, mock_db_manager
    ):
        """Test initialization with database manager."""
        ml_pass = EnhancedMLEmbeddingPass(mock_advanced_parser, mock_fuzzy_matcher, mock_db_manager)
        assert ml_pass.db_manager == mock_db_manager

    def test_configuration_parameters(self, mock_advanced_parser, mock_fuzzy_matcher):
        """Test that configuration parameters are set correctly."""
        ml_pass = EnhancedMLEmbeddingPass(mock_advanced_parser, mock_fuzzy_matcher)

        assert ml_pass.min_semantic_similarity == 0.75
        assert ml_pass.min_fuzzy_similarity == 0.8
        assert ml_pass.embedding_weight == 0.6
        assert ml_pass.fuzzy_weight == 0.4
        assert ml_pass.max_embedding_cache_size == 10000

    def test_entity_patterns_loaded(self, mock_advanced_parser, mock_fuzzy_matcher):
        """Test that entity patterns are loaded."""
        ml_pass = EnhancedMLEmbeddingPass(mock_advanced_parser, mock_fuzzy_matcher)

        assert "artist_indicators" in ml_pass.entity_patterns
        assert "song_indicators" in ml_pass.entity_patterns
        assert "noise_patterns" in ml_pass.entity_patterns


class TestEntityExtraction:
    """Test entity extraction methods."""

    @pytest.fixture
    def ml_pass(self):
        """Create an EnhancedMLEmbeddingPass instance."""
        parser = MagicMock(spec=AdvancedTitleParser)
        parser.known_artists = set()
        parser.known_songs = set()
        fuzzy_matcher = MagicMock(spec=FuzzyMatcher)
        return EnhancedMLEmbeddingPass(parser, fuzzy_matcher)

    def test_extract_entities_basic(self, ml_pass):
        """Test basic entity extraction."""
        title = "Test Artist - Test Song (Karaoke)"
        entities = ml_pass._extract_entities(title, "", "")

        assert "potential_artists" in entities
        assert "potential_songs" in entities
        assert "quoted_text" in entities
        assert "capitalized_words" in entities

        # Should extract capitalized words
        assert len(entities["capitalized_words"]) > 0

    def test_extract_entities_quoted_text(self, ml_pass):
        """Test extraction of quoted text."""
        title = 'Artist - "Song Title" (Karaoke)'
        entities = ml_pass._extract_entities(title, "", "")

        assert "Song Title" in entities["quoted_text"]

    def test_extract_entities_capitalized_words(self, ml_pass):
        """Test extraction of capitalized words."""
        title = "Famous Artist - Great Song Title"
        entities = ml_pass._extract_entities(title, "", "")

        # Should extract proper nouns
        cap_words = entities["capitalized_words"]
        assert any("Famous" in word for word in cap_words)
        assert any("Artist" in word for word in cap_words)

    def test_extract_entities_with_description(self, ml_pass):
        """Test entity extraction with description."""
        title = "Song Title"
        description = "Performed by Famous Artist, great vocals"
        entities = ml_pass._extract_entities(title, description, "")

        # Should find artist indicators in description
        assert len(entities["potential_artists"]) > 0

    def test_extract_entities_noise_filtering(self, ml_pass):
        """Test that noise words are filtered out."""
        title = "Artist - Song Karaoke HD Official Video"
        entities = ml_pass._extract_entities(title, "", "")

        # Should not include noise words in potential entities
        all_candidates = []
        for entity_list in entities.values():
            all_candidates.extend(entity_list)

        # Noise words should not appear in clean candidates
        combined_text = " ".join(all_candidates).lower()
        assert "karaoke" not in combined_text
        assert "official" not in combined_text

    def test_extract_entities_word_combinations(self, ml_pass):
        """Test extraction of word combinations."""
        title = "Artist Name - Song Title Here"
        entities = ml_pass._extract_entities(title, "", "")

        # Should create different combinations of words
        assert len(entities["potential_artists"]) > 0
        assert len(entities["potential_songs"]) > 0

    @pytest.mark.parametrize(
        "title,expected_quoted",
        [
            ('Artist - "Song Title"', ["Song Title"]),
            ('"Artist Name" - Song', ["Artist Name"]),
            ("No quotes here", []),
            ('"First" and "Second"', ["First", "Second"]),
        ],
    )
    def test_extract_entities_quoted_patterns(self, ml_pass, title, expected_quoted):
        """Test various quoted text patterns."""
        entities = ml_pass._extract_entities(title, "", "")
        assert entities["quoted_text"] == expected_quoted


class TestFuzzyMatching:
    """Test enhanced fuzzy matching methods."""

    @pytest.fixture
    def ml_pass_with_fuzzy(self):
        """Create an ML pass with working fuzzy matcher."""
        parser = MagicMock(spec=AdvancedTitleParser)
        parser.known_artists = set()
        parser.known_songs = set()

        fuzzy_matcher = MagicMock(spec=FuzzyMatcher)
        fuzzy_matcher.find_best_matches.return_value = [
            MagicMock(text="Test Artist", score=0.85, category="artist"),
            MagicMock(text="Test Song", score=0.9, category="song"),
        ]

        return EnhancedMLEmbeddingPass(parser, fuzzy_matcher)

    @pytest.mark.asyncio
    async def test_enhanced_fuzzy_matching_success(self, ml_pass_with_fuzzy):
        """Test successful enhanced fuzzy matching."""
        title = "Test Title"
        entities = {
            "potential_artists": ["Test Artist"],
            "potential_songs": ["Test Song"],
            "quoted_text": [],
            "capitalized_words": [],
        }

        result = await ml_pass_with_fuzzy._enhanced_fuzzy_matching(title, entities)

        assert result is not None
        assert result.original_artist == "Test Artist"
        assert result.song_title == "Test Song"
        assert result.method == "enhanced_fuzzy"

    @pytest.mark.asyncio
    async def test_enhanced_fuzzy_matching_no_matcher(self):
        """Test fuzzy matching without fuzzy matcher."""
        parser = MagicMock(spec=AdvancedTitleParser)
        ml_pass = EnhancedMLEmbeddingPass(parser, None)

        entities = {"potential_artists": ["Test Artist"]}
        result = await ml_pass._enhanced_fuzzy_matching("title", entities)

        assert result is None

    @pytest.mark.asyncio
    async def test_enhanced_fuzzy_matching_low_confidence(self, ml_pass_with_fuzzy):
        """Test fuzzy matching with low confidence results."""
        # Mock low confidence results
        ml_pass_with_fuzzy.fuzzy_matcher.find_best_matches.return_value = [
            MagicMock(text="Test Artist", score=0.3, category="artist"),  # Low score
        ]

        entities = {"potential_artists": ["Test Artist"]}
        result = await ml_pass_with_fuzzy._enhanced_fuzzy_matching("title", entities)

        # Should return None for low confidence
        assert result is None


class TestSemanticSimilarity:
    """Test semantic similarity matching methods."""

    @pytest.fixture
    def ml_pass_with_embeddings(self):
        """Create an ML pass with mock embedding model."""
        parser = MagicMock(spec=AdvancedTitleParser)
        parser.known_artists = set()
        parser.known_songs = set()
        fuzzy_matcher = MagicMock(spec=FuzzyMatcher)

        ml_pass = EnhancedMLEmbeddingPass(parser, fuzzy_matcher)

        # Mock embedding model
        ml_pass.embedding_model = MagicMock()
        ml_pass.embedding_model.encode.return_value = [np.array([0.1, 0.2, 0.3])]

        # Add some known entities with embeddings
        ml_pass.artist_candidates["Test Artist"] = SemanticCandidate(
            text="Test Artist",
            category="artist",
            embedding=np.array([0.1, 0.2, 0.3]),
        )
        ml_pass.song_candidates["Test Song"] = SemanticCandidate(
            text="Test Song",
            category="song",
            embedding=np.array([0.9, 0.8, 0.7]),
        )

        return ml_pass

    def test_get_embeddings_with_cache(self, ml_pass_with_embeddings):
        """Test embedding generation with caching."""
        texts = ["text1", "text2"]

        # First call - should generate embeddings
        embeddings1 = ml_pass_with_embeddings._get_embeddings(texts)
        assert len(embeddings1) == 2

        # Check cache
        assert len(ml_pass_with_embeddings.embedding_cache) == 2

        # Second call - should use cache
        embeddings2 = ml_pass_with_embeddings._get_embeddings(texts)
        assert len(embeddings2) == 2

        # Should be same embeddings from cache
        assert np.array_equal(embeddings1["text1"], embeddings2["text1"])

    def test_get_embeddings_no_model(self):
        """Test embedding generation without model."""
        parser = MagicMock(spec=AdvancedTitleParser)
        ml_pass = EnhancedMLEmbeddingPass(parser, MagicMock())
        ml_pass.embedding_model = None

        embeddings = ml_pass._get_embeddings(["text1", "text2"])
        assert embeddings == {}

    def test_get_embeddings_cache_limit(self, ml_pass_with_embeddings):
        """Test embedding cache size limit."""
        ml_pass_with_embeddings.max_embedding_cache_size = 2

        texts = ["text1", "text2", "text3"]
        ml_pass_with_embeddings._get_embeddings(texts)

        # Should only cache up to limit
        assert len(ml_pass_with_embeddings.embedding_cache) <= 2

    def test_cosine_similarity_with_sklearn(self, ml_pass_with_embeddings):
        """Test cosine similarity calculation with sklearn."""
        embedding1 = np.array([1.0, 0.0, 0.0])
        embedding2 = np.array([0.0, 1.0, 0.0])

        with patch("collector.passes.ml_embedding_pass.HAS_SKLEARN", True), patch(
            "collector.passes.ml_embedding_pass.sklearn_metrics"
        ) as mock_sklearn:

            mock_sklearn.cosine_similarity.return_value = [[0.5]]

            similarity = ml_pass_with_embeddings._cosine_similarity(embedding1, embedding2)
            assert similarity == 0.5

    def test_cosine_similarity_fallback(self, ml_pass_with_embeddings):
        """Test fallback cosine similarity calculation."""
        embedding1 = np.array([1.0, 0.0, 0.0])
        embedding2 = np.array([1.0, 0.0, 0.0])  # Same vector

        with patch("collector.passes.ml_embedding_pass.HAS_SKLEARN", False):
            similarity = ml_pass_with_embeddings._cosine_similarity(embedding1, embedding2)
            assert similarity == 1.0  # Should be 1.0 for identical vectors

    def test_cosine_similarity_zero_norm(self, ml_pass_with_embeddings):
        """Test cosine similarity with zero norm vectors."""
        embedding1 = np.array([0.0, 0.0, 0.0])
        embedding2 = np.array([1.0, 0.0, 0.0])

        with patch("collector.passes.ml_embedding_pass.HAS_SKLEARN", False):
            similarity = ml_pass_with_embeddings._cosine_similarity(embedding1, embedding2)
            assert similarity == 0.0

    def test_find_best_semantic_match_artist(self, ml_pass_with_embeddings):
        """Test finding best semantic match for artist."""
        candidate_embeddings = {
            "Similar Artist": np.array([0.1, 0.2, 0.3]),  # Should match Test Artist
        }
        candidates = ["Similar Artist"]

        match = ml_pass_with_embeddings._find_best_semantic_match(
            candidate_embeddings, candidates, "artist"
        )

        assert match is not None
        assert match.matched_text == "Test Artist"
        assert match.method == "semantic_cosine"
        assert match.similarity_score > 0.0

    def test_find_best_semantic_match_no_entities(self, ml_pass_with_embeddings):
        """Test semantic matching with no known entities."""
        ml_pass_with_embeddings.artist_candidates = {}

        candidate_embeddings = {"Artist": np.array([0.1, 0.2, 0.3])}
        match = ml_pass_with_embeddings._find_best_semantic_match(
            candidate_embeddings, ["Artist"], "artist"
        )

        assert match is None

    def test_find_best_semantic_match_low_similarity(self, ml_pass_with_embeddings):
        """Test semantic matching with low similarity."""
        candidate_embeddings = {
            "Very Different": np.array([0.9, 0.8, 0.7]),  # Very different from Test Artist
        }

        # Set high threshold
        ml_pass_with_embeddings.min_semantic_similarity = 0.95

        match = ml_pass_with_embeddings._find_best_semantic_match(
            candidate_embeddings, ["Very Different"], "artist"
        )

        assert match is None

    @pytest.mark.asyncio
    async def test_semantic_similarity_matching_success(self, ml_pass_with_embeddings):
        """Test successful semantic similarity matching."""
        entities = {
            "potential_artists": ["Similar Artist"],
            "potential_songs": ["Similar Song"],
            "quoted_text": [],
            "capitalized_words": [],
        }

        # Mock to return high similarity
        with patch.object(ml_pass_with_embeddings, "_find_best_semantic_match") as mock_find:
            mock_find.side_effect = [
                EmbeddingMatch("query", "Test Artist", 0.9, "semantic", confidence=0.9),
                EmbeddingMatch("query", "Test Song", 0.85, "semantic", confidence=0.85),
            ]

            result = await ml_pass_with_embeddings._semantic_similarity_matching("title", entities)

            assert result is not None
            assert result.original_artist == "Test Artist"
            assert result.song_title == "Test Song"
            assert result.method == "semantic_similarity"

    @pytest.mark.asyncio
    async def test_semantic_similarity_matching_no_model(self):
        """Test semantic matching without embedding model."""
        parser = MagicMock(spec=AdvancedTitleParser)
        ml_pass = EnhancedMLEmbeddingPass(parser, MagicMock())
        ml_pass.embedding_model = None

        entities = {"potential_artists": ["Artist"]}
        result = await ml_pass._semantic_similarity_matching("title", entities)

        assert result is None

    @pytest.mark.asyncio
    async def test_semantic_similarity_matching_no_candidates(self, ml_pass_with_embeddings):
        """Test semantic matching with no candidates."""
        entities = {
            "potential_artists": [],
            "potential_songs": [],
            "quoted_text": [],
            "capitalized_words": [],
        }

        result = await ml_pass_with_embeddings._semantic_similarity_matching("title", entities)

        assert result is None


class TestHybridMatching:
    """Test hybrid matching methods."""

    @pytest.fixture
    def ml_pass(self):
        """Create an ML pass for hybrid testing."""
        parser = MagicMock(spec=AdvancedTitleParser)
        parser.known_artists = set()
        parser.known_songs = set()
        fuzzy_matcher = MagicMock(spec=FuzzyMatcher)
        return EnhancedMLEmbeddingPass(parser, fuzzy_matcher)

    @pytest.mark.asyncio
    async def test_hybrid_matching_both_results(self, ml_pass):
        """Test hybrid matching with both fuzzy and semantic results."""
        fuzzy_result = ParseResult(
            original_artist="Fuzzy Artist",
            song_title="Fuzzy Song",
            confidence=0.8,
            method="enhanced_fuzzy",
        )

        semantic_result = ParseResult(
            original_artist="Semantic Artist",
            song_title="Semantic Song",
            confidence=0.7,
            method="semantic_similarity",
        )

        with patch.object(ml_pass, "_enhanced_fuzzy_matching") as mock_fuzzy, patch.object(
            ml_pass, "_semantic_similarity_matching"
        ) as mock_semantic:

            mock_fuzzy.return_value = fuzzy_result
            mock_semantic.return_value = semantic_result

            result = await ml_pass._hybrid_matching("title", {})

            assert result is not None
            assert result.method == "hybrid_fuzzy_semantic"
            # Should use fuzzy result (higher confidence)
            assert result.original_artist == "Fuzzy Artist"
            assert result.song_title == "Fuzzy Song"

    @pytest.mark.asyncio
    async def test_hybrid_matching_fuzzy_only(self, ml_pass):
        """Test hybrid matching with only fuzzy result."""
        fuzzy_result = ParseResult(
            original_artist="Fuzzy Artist",
            song_title="Fuzzy Song",
            confidence=0.8,
            method="enhanced_fuzzy",
        )

        with patch.object(ml_pass, "_enhanced_fuzzy_matching") as mock_fuzzy, patch.object(
            ml_pass, "_semantic_similarity_matching"
        ) as mock_semantic:

            mock_fuzzy.return_value = fuzzy_result
            mock_semantic.return_value = None

            result = await ml_pass._hybrid_matching("title", {})

            assert result is not None
            assert result.original_artist == "Fuzzy Artist"

    @pytest.mark.asyncio
    async def test_hybrid_matching_semantic_only(self, ml_pass):
        """Test hybrid matching with only semantic result."""
        semantic_result = ParseResult(
            original_artist="Semantic Artist",
            song_title="Semantic Song",
            confidence=0.7,
            method="semantic_similarity",
        )

        with patch.object(ml_pass, "_enhanced_fuzzy_matching") as mock_fuzzy, patch.object(
            ml_pass, "_semantic_similarity_matching"
        ) as mock_semantic:

            mock_fuzzy.return_value = None
            mock_semantic.return_value = semantic_result

            result = await ml_pass._hybrid_matching("title", {})

            assert result is not None
            assert result.original_artist == "Semantic Artist"

    @pytest.mark.asyncio
    async def test_hybrid_matching_no_results(self, ml_pass):
        """Test hybrid matching with no results."""
        with patch.object(ml_pass, "_enhanced_fuzzy_matching") as mock_fuzzy, patch.object(
            ml_pass, "_semantic_similarity_matching"
        ) as mock_semantic:

            mock_fuzzy.return_value = None
            mock_semantic.return_value = None

            result = await ml_pass._hybrid_matching("title", {})

            assert result is None

    @pytest.mark.asyncio
    async def test_hybrid_matching_confidence_weighting(self, ml_pass):
        """Test confidence weighting in hybrid matching."""
        fuzzy_result = ParseResult(
            original_artist="Fuzzy Artist",
            song_title="Fuzzy Song",
            confidence=0.6,
            method="enhanced_fuzzy",
        )

        semantic_result = ParseResult(
            original_artist="Semantic Artist",
            song_title="Semantic Song",
            confidence=0.8,
            method="semantic_similarity",
        )

        with patch.object(ml_pass, "_enhanced_fuzzy_matching") as mock_fuzzy, patch.object(
            ml_pass, "_semantic_similarity_matching"
        ) as mock_semantic:

            mock_fuzzy.return_value = fuzzy_result
            mock_semantic.return_value = semantic_result

            result = await ml_pass._hybrid_matching("title", {})

            # Combined confidence should be weighted average
            expected_confidence = 0.6 * ml_pass.fuzzy_weight + 0.8 * ml_pass.embedding_weight
            assert abs(result.confidence - expected_confidence) < 0.01

            # Should use semantic result (higher confidence)
            assert result.original_artist == "Semantic Artist"


class TestMainParseMethod:
    """Test the main parse method."""

    @pytest.fixture
    def ml_pass(self):
        """Create an ML pass for testing."""
        parser = MagicMock(spec=AdvancedTitleParser)
        parser.known_artists = set()
        parser.known_songs = set()
        fuzzy_matcher = MagicMock(spec=FuzzyMatcher)
        return EnhancedMLEmbeddingPass(parser, fuzzy_matcher)

    @pytest.mark.asyncio
    async def test_parse_no_entities(self, ml_pass):
        """Test parse when no entities are extracted."""
        with patch.object(ml_pass, "_extract_entities") as mock_extract:
            mock_extract.return_value = {}

            result = await ml_pass.parse("empty title")
            assert result is None

    @pytest.mark.asyncio
    async def test_parse_fuzzy_high_confidence(self, ml_pass):
        """Test parse with high confidence fuzzy result."""
        entities = {"potential_artists": ["Test Artist"]}
        fuzzy_result = ParseResult(confidence=0.85, method="enhanced_fuzzy")

        with patch.object(ml_pass, "_extract_entities") as mock_extract, patch.object(
            ml_pass, "_enhanced_fuzzy_matching"
        ) as mock_fuzzy:

            mock_extract.return_value = entities
            mock_fuzzy.return_value = fuzzy_result

            result = await ml_pass.parse("test title")

            assert result == fuzzy_result
            # Should not call other methods due to early exit

    @pytest.mark.asyncio
    async def test_parse_semantic_high_confidence(self, ml_pass):
        """Test parse with high confidence semantic result."""
        entities = {"potential_artists": ["Test Artist"]}
        semantic_result = ParseResult(confidence=0.8, method="semantic_similarity")

        ml_pass.embedding_model = MagicMock()  # Enable semantic matching

        with patch.object(ml_pass, "_extract_entities") as mock_extract, patch.object(
            ml_pass, "_enhanced_fuzzy_matching"
        ) as mock_fuzzy, patch.object(ml_pass, "_semantic_similarity_matching") as mock_semantic:

            mock_extract.return_value = entities
            mock_fuzzy.return_value = None
            mock_semantic.return_value = semantic_result

            result = await ml_pass.parse("test title")

            assert result == semantic_result

    @pytest.mark.asyncio
    async def test_parse_hybrid_matching(self, ml_pass):
        """Test parse falling back to hybrid matching."""
        entities = {"potential_artists": ["Test Artist"]}
        hybrid_result = ParseResult(confidence=0.75, method="hybrid_fuzzy_semantic")

        with patch.object(ml_pass, "_extract_entities") as mock_extract, patch.object(
            ml_pass, "_enhanced_fuzzy_matching"
        ) as mock_fuzzy, patch.object(
            ml_pass, "_semantic_similarity_matching"
        ) as mock_semantic, patch.object(
            ml_pass, "_hybrid_matching"
        ) as mock_hybrid:

            mock_extract.return_value = entities
            mock_fuzzy.return_value = ParseResult(confidence=0.6)  # Low confidence
            mock_semantic.return_value = ParseResult(confidence=0.5)  # Low confidence
            mock_hybrid.return_value = hybrid_result

            result = await ml_pass.parse("test title")

            assert result == hybrid_result

    @pytest.mark.asyncio
    async def test_parse_pattern_matching_fallback(self, ml_pass):
        """Test parse falling back to pattern matching."""
        entities = {"potential_artists": ["Test Artist"]}
        pattern_result = ParseResult(confidence=0.7, method="entity_pattern")

        with patch.object(ml_pass, "_extract_entities") as mock_extract, patch.object(
            ml_pass, "_enhanced_fuzzy_matching"
        ) as mock_fuzzy, patch.object(
            ml_pass, "_semantic_similarity_matching"
        ) as mock_semantic, patch.object(
            ml_pass, "_hybrid_matching"
        ) as mock_hybrid, patch.object(
            ml_pass, "_entity_pattern_matching"
        ) as mock_pattern:

            mock_extract.return_value = entities
            mock_fuzzy.return_value = ParseResult(confidence=0.5)  # Low confidence
            mock_semantic.return_value = None
            mock_hybrid.return_value = ParseResult(confidence=0.6)  # Low confidence
            mock_pattern.return_value = pattern_result

            result = await ml_pass.parse("test title")

            assert result == pattern_result

    @pytest.mark.asyncio
    async def test_parse_no_results(self, ml_pass):
        """Test parse when no methods return results."""
        entities = {"potential_artists": ["Test Artist"]}

        with patch.object(ml_pass, "_extract_entities") as mock_extract, patch.object(
            ml_pass, "_enhanced_fuzzy_matching"
        ) as mock_fuzzy, patch.object(
            ml_pass, "_semantic_similarity_matching"
        ) as mock_semantic, patch.object(
            ml_pass, "_hybrid_matching"
        ) as mock_hybrid, patch.object(
            ml_pass, "_entity_pattern_matching"
        ) as mock_pattern:

            mock_extract.return_value = entities
            mock_fuzzy.return_value = None
            mock_semantic.return_value = None
            mock_hybrid.return_value = None
            mock_pattern.return_value = None

            result = await ml_pass.parse("test title")

            assert result is None

    @pytest.mark.asyncio
    async def test_parse_exception_handling(self, ml_pass):
        """Test exception handling in parse method."""
        with patch.object(ml_pass, "_extract_entities") as mock_extract:
            mock_extract.side_effect = Exception("Test error")

            result = await ml_pass.parse("test title")

            assert result is None

    @pytest.mark.asyncio
    async def test_parse_slow_processing_logging(self, ml_pass):
        """Test logging of slow processing."""
        entities = {"potential_artists": ["Test Artist"]}

        with patch.object(ml_pass, "_extract_entities") as mock_extract, patch(
            "time.time"
        ) as mock_time, patch("collector.passes.ml_embedding_pass.logger") as mock_logger:

            mock_extract.return_value = entities
            mock_time.side_effect = [0, 10]  # 10 second processing time

            await ml_pass.parse("test title")

            # Should log warning for slow processing
            mock_logger.warning.assert_called()


class TestKnowledgeBaseManagement:
    """Test knowledge base management methods."""

    @pytest.fixture
    def ml_pass(self):
        """Create an ML pass for testing."""
        parser = MagicMock(spec=AdvancedTitleParser)
        parser.known_artists = {"Known Artist"}
        parser.known_songs = {"Known Song"}
        fuzzy_matcher = MagicMock(spec=FuzzyMatcher)
        return EnhancedMLEmbeddingPass(parser, fuzzy_matcher)

    def test_load_knowledge_base(self, ml_pass):
        """Test loading knowledge base from parser."""
        # Should load from advanced parser
        assert "Known Artist" in ml_pass.artist_candidates
        assert "Known Song" in ml_pass.song_candidates

        # Check candidate properties
        artist_candidate = ml_pass.artist_candidates["Known Artist"]
        assert artist_candidate.text == "Known Artist"
        assert artist_candidate.category == "artist"
        assert artist_candidate.frequency == 1

    def test_add_entity_new_artist(self, ml_pass):
        """Test adding new artist entity."""
        ml_pass.add_entity("New Artist", "artist", 0.9)

        assert "New Artist" in ml_pass.artist_candidates
        candidate = ml_pass.artist_candidates["New Artist"]
        assert candidate.text == "New Artist"
        assert candidate.category == "artist"
        assert candidate.frequency == 1

    def test_add_entity_existing_artist(self, ml_pass):
        """Test adding existing artist entity."""
        initial_frequency = ml_pass.artist_candidates["Known Artist"].frequency

        ml_pass.add_entity("Known Artist", "artist", 0.9)

        # Should update frequency
        assert ml_pass.artist_candidates["Known Artist"].frequency == initial_frequency + 1

    def test_add_entity_new_song(self, ml_pass):
        """Test adding new song entity."""
        ml_pass.add_entity("New Song", "song", 0.8)

        assert "New Song" in ml_pass.song_candidates
        candidate = ml_pass.song_candidates["New Song"]
        assert candidate.text == "New Song"
        assert candidate.category == "song"

    def test_add_entity_invalid_category(self, ml_pass):
        """Test adding entity with invalid category."""
        initial_artist_count = len(ml_pass.artist_candidates)
        initial_song_count = len(ml_pass.song_candidates)

        ml_pass.add_entity("Invalid Entity", "invalid_category", 0.9)

        # Should not add anything
        assert len(ml_pass.artist_candidates) == initial_artist_count
        assert len(ml_pass.song_candidates) == initial_song_count

    def test_add_entity_with_embedding_model(self, ml_pass):
        """Test adding entity with embedding model."""
        ml_pass.embedding_model = MagicMock()
        ml_pass.embedding_model.encode.return_value = [np.array([0.1, 0.2, 0.3])]

        ml_pass.add_entity("New Artist", "artist", 0.9)

        candidate = ml_pass.artist_candidates["New Artist"]
        assert candidate.embedding is not None
        assert np.array_equal(candidate.embedding, np.array([0.1, 0.2, 0.3]))

    def test_add_entity_embedding_failure(self, ml_pass):
        """Test adding entity when embedding generation fails."""
        ml_pass.embedding_model = MagicMock()
        ml_pass.embedding_model.encode.side_effect = Exception("Embedding failed")

        ml_pass.add_entity("New Artist", "artist", 0.9)

        # Should still add entity without embedding
        assert "New Artist" in ml_pass.artist_candidates
        candidate = ml_pass.artist_candidates["New Artist"]
        assert candidate.embedding is None


class TestStatistics:
    """Test statistics functionality."""

    @pytest.fixture
    def ml_pass(self):
        """Create an ML pass for testing."""
        parser = MagicMock(spec=AdvancedTitleParser)
        parser.known_artists = {"Artist1", "Artist2"}
        parser.known_songs = {"Song1"}
        fuzzy_matcher = MagicMock(spec=FuzzyMatcher)
        return EnhancedMLEmbeddingPass(parser, fuzzy_matcher)

    def test_get_statistics_without_model(self, ml_pass):
        """Test statistics without embedding model."""
        ml_pass.embedding_model = None

        stats = ml_pass.get_statistics()

        assert stats["has_embedding_model"] is False
        assert stats["embedding_model_name"] is None
        assert stats["artist_candidates"] == 2
        assert stats["song_candidates"] == 1
        assert stats["embedding_cache_size"] == 0
        assert stats["entities_with_embeddings"] == 0
        assert "configuration" in stats

    def test_get_statistics_with_model(self, ml_pass):
        """Test statistics with embedding model."""
        ml_pass.embedding_model = MagicMock()

        # Add some embeddings to entities
        ml_pass.artist_candidates["Artist1"].embedding = np.array([0.1, 0.2])
        ml_pass.song_candidates["Song1"].embedding = np.array([0.3, 0.4])

        # Add some cache entries
        ml_pass.embedding_cache["cached_text"] = np.array([0.5, 0.6])

        stats = ml_pass.get_statistics()

        assert stats["has_embedding_model"] is True
        assert stats["embedding_model_name"] == ml_pass.embedding_model_name
        assert stats["artist_candidates"] == 2
        assert stats["song_candidates"] == 1
        assert stats["embedding_cache_size"] == 1
        assert stats["entities_with_embeddings"] == 2  # Artist1 + Song1

    def test_get_statistics_configuration(self, ml_pass):
        """Test statistics configuration section."""
        stats = ml_pass.get_statistics()

        config = stats["configuration"]
        assert config["min_semantic_similarity"] == ml_pass.min_semantic_similarity
        assert config["min_fuzzy_similarity"] == ml_pass.min_fuzzy_similarity
        assert config["embedding_weight"] == ml_pass.embedding_weight
        assert config["fuzzy_weight"] == ml_pass.fuzzy_weight


@pytest.mark.integration
class TestMLEmbeddingPassIntegration:
    """Integration tests for the ML embedding pass."""

    @pytest.fixture
    def full_ml_pass(self):
        """Create a fully functional ML embedding pass."""
        parser = AdvancedTitleParser()
        parser.known_artists = {"Ed Sheeran", "Taylor Swift", "Beatles"}
        parser.known_songs = {"Shape of You", "Shake It Off", "Hey Jude"}

        fuzzy_matcher = MagicMock(spec=FuzzyMatcher)

        return EnhancedMLEmbeddingPass(parser, fuzzy_matcher)

    @pytest.mark.asyncio
    async def test_full_parsing_workflow_mocked(self, full_ml_pass):
        """Test full parsing workflow with mocked components."""
        # Mock fuzzy matcher to return good results
        full_ml_pass.fuzzy_matcher.find_best_matches.return_value = [
            MagicMock(text="Ed Sheeran", score=0.9, category="artist"),
            MagicMock(text="Shape of You", score=0.85, category="song"),
        ]

        title = "Ed Sheeran - Shape of You (Karaoke Version)"
        result = await full_ml_pass.parse(title)

        assert result is not None
        assert result.original_artist == "Ed Sheeran"
        assert result.song_title == "Shape of You"
        assert result.confidence > 0.8

    def test_entity_extraction_comprehensive(self, full_ml_pass):
        """Test comprehensive entity extraction."""
        test_cases = [
            # (title, expected_artist_candidates, expected_song_candidates)
            ("Ed Sheeran - Shape of You", ["Ed Sheeran"], ["Shape of You"]),
            ('"Perfect" by Ed Sheeran', ["Ed Sheeran"], ["Perfect"]),
            ("Taylor Swift Shake It Off Official", ["Taylor Swift"], ["Shake It Off"]),
            ("Beatles - Hey Jude (Live Version)", ["Beatles"], ["Hey Jude"]),
        ]

        for title, expected_artists, expected_songs in test_cases:
            entities = full_ml_pass._extract_entities(title, "", "")

            # Check that expected entities are found
            all_artist_candidates = []
            all_song_candidates = []

            for candidate_list in entities.values():
                for candidate in candidate_list:
                    if any(artist in candidate for artist in expected_artists):
                        all_artist_candidates.append(candidate)
                    if any(song in candidate for song in expected_songs):
                        all_song_candidates.append(candidate)

            assert len(all_artist_candidates) > 0
            assert len(all_song_candidates) > 0

    def test_knowledge_base_growth(self, full_ml_pass):
        """Test that knowledge base grows over time."""
        initial_artist_count = len(full_ml_pass.artist_candidates)
        initial_song_count = len(full_ml_pass.song_candidates)

        # Add new entities
        full_ml_pass.add_entity("New Artist", "artist")
        full_ml_pass.add_entity("New Song", "song")

        assert len(full_ml_pass.artist_candidates) == initial_artist_count + 1
        assert len(full_ml_pass.song_candidates) == initial_song_count + 1

        # Add existing entity - should increase frequency
        full_ml_pass.add_entity("New Artist", "artist")
        assert full_ml_pass.artist_candidates["New Artist"].frequency == 2

    def test_embedding_caching_behavior(self, full_ml_pass):
        """Test embedding caching behavior."""
        if not full_ml_pass.embedding_model:
            pytest.skip("No embedding model available")

        # Generate embeddings for same texts multiple times
        texts = ["test text 1", "test text 2"]

        # First call
        embeddings1 = full_ml_pass._get_embeddings(texts)
        cache_size_1 = len(full_ml_pass.embedding_cache)

        # Second call - should use cache
        embeddings2 = full_ml_pass._get_embeddings(texts)
        cache_size_2 = len(full_ml_pass.embedding_cache)

        # Cache size should not increase
        assert cache_size_2 == cache_size_1

        # Embeddings should be the same
        for text in texts:
            if text in embeddings1 and text in embeddings2:
                assert np.array_equal(embeddings1[text], embeddings2[text])

    def test_configuration_flexibility(self, full_ml_pass):
        """Test that configuration parameters work correctly."""
        # Test different similarity thresholds
        original_threshold = full_ml_pass.min_semantic_similarity

        # Set very high threshold
        full_ml_pass.min_semantic_similarity = 0.99

        # Mock semantic matching to return medium similarity
        with patch.object(full_ml_pass, "_cosine_similarity") as mock_cosine:
            mock_cosine.return_value = 0.8  # Below new threshold

            # Should not return results due to high threshold
            candidate_embeddings = {"test": np.array([0.1, 0.2])}
            match = full_ml_pass._find_best_semantic_match(candidate_embeddings, ["test"], "artist")
            assert match is None

        # Restore original threshold
        full_ml_pass.min_semantic_similarity = original_threshold
