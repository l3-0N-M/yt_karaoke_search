"""Pass 2: Enhanced ML/embedding nearest-neighbour with semantic similarity."""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np

from ..advanced_parser import AdvancedTitleParser, ParseResult
from ..search.fuzzy_matcher import FuzzyMatcher
from .base import ParsingPass, PassType

logger = logging.getLogger(__name__)

# Optional imports for enhanced functionality
try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None  # type: ignore
    HAS_SENTENCE_TRANSFORMERS = False
    logger.info("sentence-transformers not available, using fallback methods")

try:
    import sklearn.metrics.pairwise as sklearn_metrics

    HAS_SKLEARN = True
except ImportError:
    sklearn_metrics = None  # type: ignore
    HAS_SKLEARN = False
    logger.info("scikit-learn not available, using basic similarity measures")


@dataclass
class EmbeddingMatch:
    """Result of embedding-based matching."""

    query: str
    matched_text: str
    similarity_score: float
    method: str
    artist_candidate: Optional[str] = None
    song_candidate: Optional[str] = None
    confidence: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class SemanticCandidate:
    """A candidate artist or song with semantic information."""

    text: str
    category: str  # 'artist' or 'song'
    embedding: Optional[np.ndarray] = None
    frequency: int = 1
    last_seen: float = 0.0
    aliases: Set[str] = field(default_factory=set)


class EnhancedMLEmbeddingPass(ParsingPass):
    """Pass 2: Enhanced ML/embedding similarity with semantic understanding."""

    def __init__(
        self, advanced_parser: AdvancedTitleParser, fuzzy_matcher: FuzzyMatcher, db_manager=None
    ):
        self.advanced_parser = advanced_parser
        self.fuzzy_matcher = fuzzy_matcher
        self.db_manager = db_manager

        # Embedding model initialization
        self.embedding_model = None
        self.embedding_model_name = "all-MiniLM-L6-v2"  # Lightweight, good quality
        self.max_embedding_cache_size = 10000

        # Knowledge base
        self.artist_candidates: Dict[str, SemanticCandidate] = {}
        self.song_candidates: Dict[str, SemanticCandidate] = {}
        self.embedding_cache: Dict[str, np.ndarray] = {}

        # Configuration
        self.min_semantic_similarity = 0.75
        self.min_fuzzy_similarity = 0.8
        self.embedding_weight = 0.6  # Weight for embedding vs fuzzy matching
        self.fuzzy_weight = 0.4

        # Entity extraction patterns
        self.entity_patterns = self._load_entity_patterns()

        # Initialize components
        self._initialize_embedding_model()
        self._load_knowledge_base()

    def _initialize_embedding_model(self):
        """Initialize the sentence transformer model if available."""

        if not HAS_SENTENCE_TRANSFORMERS:
            logger.info("Using fallback semantic similarity without embeddings")
            return

        try:
            if SentenceTransformer:
                self.embedding_model = SentenceTransformer(self.embedding_model_name, device="cpu")
                logger.info(f"Initialized embedding model: {self.embedding_model_name}")
                self.has_embedding_model = True
            else:
                logger.warning("SentenceTransformer not available")
                self.embedding_model = None
        except Exception as e:
            logger.warning(f"Failed to initialize embedding model: {e}")
            self.embedding_model = None

    def _load_entity_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for entity extraction."""

        return {
            "artist_indicators": [
                r"\b(?:by|performed by|artist|singer|band|group)\b",
                r"\b(?:feat|featuring|ft)\b\.?\s+([^,\(\)]+)",
                r"\b(?:with|w/)\s+([^,\(\)]+)",
                r"@(\w+)",  # Social media handles
            ],
            "song_indicators": [
                r"\b(?:song|track|single|hit|cover|version)\b",
                r"\b(?:original|acoustic|live|remix)\b",
                r"['\"\u201c\u201d]([^'\"]+)['\"\u201c\u201d]",  # Quoted text
            ],
            "noise_patterns": [
                r"\b(?:karaoke|instrumental|backing|track|minus|one|mr|inst)\b",
                r"\b(?:hd|4k|1080p|720p|high quality|hq)\b",
                r"\b(?:official|video|audio|music|clip)\b",
                r"\([^)]*(?:key|tempo|bpm)[^)]*\)",
            ],
        }

    @property
    def pass_type(self) -> PassType:
        return PassType.ML_EMBEDDING

    async def parse(
        self,
        title: str,
        description: str = "",
        tags: str = "",
        channel_name: str = "",
        channel_id: str = "",
        metadata: Optional[Dict] = None,
    ) -> Optional[ParseResult]:
        """Execute enhanced ML/embedding similarity matching."""

        start_time = time.time()

        try:
            # Step 1: Extract potential entities from title
            entities = self._extract_entities(title, description, tags)
            if not entities:
                return None

            # Step 2: Enhanced fuzzy matching with existing system
            fuzzy_result = await self._enhanced_fuzzy_matching(title, entities)
            if fuzzy_result and fuzzy_result.confidence > 0.8:
                return fuzzy_result

            # Step 3: Semantic similarity matching
            if self.embedding_model:
                semantic_result = await self._semantic_similarity_matching(title, entities)
                if semantic_result and semantic_result.confidence > 0.75:
                    return semantic_result

            # Step 4: Hybrid matching (combine fuzzy + semantic)
            hybrid_result = await self._hybrid_matching(title, entities)
            if hybrid_result and hybrid_result.confidence > 0.7:
                return hybrid_result

            # Step 5: Entity-based pattern matching
            pattern_result = self._entity_pattern_matching(title, entities)
            if pattern_result and pattern_result.confidence > 0.65:
                return pattern_result

            return None

        except Exception as e:
            logger.error(f"ML embedding pass failed: {e}")
            return None
        finally:
            processing_time = time.time() - start_time
            if processing_time > 5.0:  # Log slow operations
                logger.warning(f"ML embedding pass took {processing_time:.2f}s")

    def _extract_entities(self, title: str, description: str, tags: str) -> Dict[str, List[str]]:
        """Extract potential artist and song entities from text."""

        entities = {
            "potential_artists": [],
            "potential_songs": [],
            "quoted_text": [],
            "capitalized_words": [],
        }

        # Combine all text sources
        full_text = f"{title} {description} {tags}".lower()

        # Extract quoted text (often song titles)
        import re

        quoted_matches = re.findall(r'["\u201c\u201d]([^"\u201c\u201d]+)["\u201c\u201d]', title)
        entities["quoted_text"].extend(quoted_matches)

        # Extract capitalized words/phrases (often proper nouns)
        cap_words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", title)
        entities["capitalized_words"].extend(cap_words)

        # Look for artist indicators
        for pattern in self.entity_patterns["artist_indicators"]:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            if matches:
                entities["potential_artists"].extend(matches)

        # Extract words that aren't noise
        words = re.findall(r"\b\w+\b", title)
        clean_words = []
        for word in words:
            is_noise = any(
                re.search(noise_pattern, word, re.IGNORECASE)
                for noise_pattern in self.entity_patterns["noise_patterns"]
            )
            if not is_noise and len(word) > 2:
                clean_words.append(word)

        # Group clean words into potential entities
        if len(clean_words) >= 2:
            # Try different combinations
            entities["potential_artists"].append(" ".join(clean_words[:2]))
            entities["potential_songs"].append(" ".join(clean_words[-2:]))
            if len(clean_words) >= 3:
                entities["potential_artists"].append(" ".join(clean_words[:3]))
                entities["potential_songs"].append(" ".join(clean_words[-3:]))

        return entities

    async def _enhanced_fuzzy_matching(
        self, title: str, entities: Dict[str, List[str]]
    ) -> Optional[ParseResult]:
        """Enhanced fuzzy matching using the existing fuzzy matcher."""

        if not self.fuzzy_matcher:
            return None

        best_match = None
        best_confidence = 0.0

        # Get all known artists and songs from the advanced parser
        known_artists = list(self.advanced_parser.known_artists)
        known_songs = list(self.advanced_parser.known_songs)

        if not known_artists and not known_songs:
            return None

        # Try fuzzy matching against all potential entities
        all_candidates = (
            entities.get("potential_artists", [])
            + entities.get("potential_songs", [])
            + entities.get("quoted_text", [])
            + entities.get("capitalized_words", [])
        )

        for candidate in all_candidates:
            if len(candidate.strip()) < 2:
                continue

            # Try as artist
            if known_artists:
                artist_match = self.fuzzy_matcher.find_best_match(
                    candidate, known_artists, "artist", min_score=self.min_fuzzy_similarity
                )
                if artist_match and artist_match.score > best_confidence:
                    # Look for song in remaining text
                    remaining_text = title.replace(candidate, "").strip()
                    song_candidates = self._extract_song_candidates_from_text(remaining_text)

                    for song_candidate in song_candidates:
                        song_match = self.fuzzy_matcher.find_best_match(
                            song_candidate, known_songs, "song", min_score=0.7
                        )
                        if song_match:
                            combined_confidence = artist_match.score * 0.6 + song_match.score * 0.4
                            if combined_confidence > best_confidence:
                                best_confidence = combined_confidence
                                best_match = ParseResult(
                                    artist=artist_match.matched,
                                    song_title=song_match.matched,
                                    confidence=combined_confidence,
                                    method="enhanced_fuzzy_matching",
                                    pattern_used=f"fuzzy_artist:{artist_match.score:.2f}_song:{song_match.score:.2f}",
                                    metadata={
                                        "artist_match": artist_match.score,
                                        "song_match": song_match.score,
                                        "method": "enhanced_fuzzy",
                                    },
                                )

            # Try as song
            if known_songs:
                song_match = self.fuzzy_matcher.find_best_match(
                    candidate, known_songs, "song", min_score=self.min_fuzzy_similarity
                )
                if song_match and song_match.score > best_confidence:
                    best_confidence = song_match.score
                    best_match = ParseResult(
                        song_title=song_match.matched,
                        confidence=song_match.score * 0.9,  # Slight penalty for song-only
                        method="enhanced_fuzzy_matching",
                        pattern_used=f"fuzzy_song:{song_match.score:.2f}",
                        metadata={
                            "song_match": song_match.score,
                            "method": "enhanced_fuzzy_song_only",
                        },
                    )

        return best_match

    def _extract_song_candidates_from_text(self, text: str) -> List[str]:
        """Extract potential song titles from remaining text."""

        import re

        # Clean the text
        text = re.sub(r"\([^)]*(?:karaoke|instrumental)[^)]*\)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\[[^\]]*(?:karaoke|instrumental)[^\]]*\]", "", text, flags=re.IGNORECASE)
        text = text.strip()

        candidates = []

        # Look for quoted text first
        quoted = re.findall(r'["\u201c\u201d]([^"\u201c\u201d]+)["\u201c\u201d]', text)
        candidates.extend(quoted)

        # Split by common separators
        for separator in ["-", "–", "—", "by", "from"]:
            if separator in text.lower():
                parts = re.split(rf"\s*{re.escape(separator)}\s*", text, flags=re.IGNORECASE)
                candidates.extend([part.strip() for part in parts if len(part.strip()) > 2])

        # Add the whole text as a candidate
        if len(text.strip()) > 2:
            candidates.append(text.strip())

        return candidates

    async def _semantic_similarity_matching(
        self, title: str, entities: Dict[str, List[str]]
    ) -> Optional[ParseResult]:
        """Semantic similarity matching using embeddings."""

        if not self.embedding_model:
            return None

        try:
            # Get embeddings for all candidates
            all_candidates = []
            for entity_list in entities.values():
                all_candidates.extend(entity_list)

            if not all_candidates:
                return None

            candidate_embeddings = self._get_embeddings(all_candidates)

            # Compare against known entities
            best_artist_match = self._find_best_semantic_match(
                candidate_embeddings, all_candidates, "artist"
            )
            best_song_match = self._find_best_semantic_match(
                candidate_embeddings, all_candidates, "song"
            )

            if best_artist_match or best_song_match:
                # Calculate combined confidence
                artist_score = best_artist_match.similarity_score if best_artist_match else 0.0
                song_score = best_song_match.similarity_score if best_song_match else 0.0

                combined_confidence = max(artist_score, song_score)
                if best_artist_match and best_song_match:
                    combined_confidence = artist_score * 0.6 + song_score * 0.4
                elif best_artist_match or best_song_match:
                    combined_confidence *= 0.85  # Penalty for single match

                return ParseResult(
                    artist=best_artist_match.matched_text if best_artist_match else None,
                    song_title=best_song_match.matched_text if best_song_match else None,
                    confidence=combined_confidence,
                    method="semantic_similarity",
                    pattern_used=f"semantic_artist:{artist_score:.2f}_song:{song_score:.2f}",
                    metadata={
                        "artist_semantic_score": artist_score,
                        "song_semantic_score": song_score,
                        "method": "semantic_embeddings",
                    },
                )

        except Exception as e:
            logger.warning(f"Semantic similarity matching failed: {e}")

        return None

    def _get_embeddings(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Get embeddings for a list of texts with caching."""

        if self.embedding_model is None:
            return {}

        embeddings = {}

        for text in texts:
            # Check cache first
            if text in self.embedding_cache:
                embeddings[text] = self.embedding_cache[text]
                continue

            try:
                # Generate embedding
                embedding = self.embedding_model.encode([text])[0]
                embeddings[text] = embedding

                # Cache with size limit
                if len(self.embedding_cache) < self.max_embedding_cache_size:
                    self.embedding_cache[text] = embedding

            except Exception as e:
                logger.warning(f"Failed to generate embedding for '{text}': {e}")
                continue

        return embeddings

    def _find_best_semantic_match(
        self, candidate_embeddings: Dict[str, np.ndarray], candidates: List[str], match_type: str
    ) -> Optional[EmbeddingMatch]:
        """Find the best semantic match for a given type."""

        if match_type == "artist":
            known_entities = self.artist_candidates
        elif match_type == "song":
            known_entities = self.song_candidates
        else:
            return None

        if not known_entities:
            return None

        best_match = None
        best_similarity = 0.0

        for candidate_text, candidate_embedding in candidate_embeddings.items():
            for known_entity_text, known_entity in known_entities.items():
                if known_entity.embedding is None:
                    continue

                # Calculate cosine similarity
                similarity = self._cosine_similarity(candidate_embedding, known_entity.embedding)

                if similarity > best_similarity and similarity >= self.min_semantic_similarity:
                    best_similarity = similarity
                    best_match = EmbeddingMatch(
                        query=candidate_text,
                        matched_text=known_entity_text,
                        similarity_score=similarity,
                        method="semantic_cosine",
                        confidence=similarity,
                    )

        return best_match

    def _cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""

        if HAS_SKLEARN and sklearn_metrics:
            return sklearn_metrics.cosine_similarity(
                embedding1.reshape(1, -1), embedding2.reshape(1, -1)
            )[0][0]

        logger.warning("scikit-learn is not available, using fallback cosine similarity")

        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

    async def _hybrid_matching(
        self, title: str, entities: Dict[str, List[str]]
    ) -> Optional[ParseResult]:
        """Hybrid matching combining fuzzy and semantic approaches."""

        # Get results from both approaches
        fuzzy_result = await self._enhanced_fuzzy_matching(title, entities)
        semantic_result = await self._semantic_similarity_matching(title, entities)

        if not fuzzy_result and not semantic_result:
            return None

        # Determine the better result and calculate combined confidence
        if fuzzy_result and semantic_result:
            fuzzy_confidence = getattr(fuzzy_result, "confidence", 0.0)
            semantic_confidence = getattr(semantic_result, "confidence", 0.0)
            combined_confidence = (
                fuzzy_confidence * self.fuzzy_weight + semantic_confidence * self.embedding_weight
            )
            better_result = (
                fuzzy_result if fuzzy_confidence > semantic_confidence else semantic_result
            )
        elif fuzzy_result:
            better_result = fuzzy_result
            combined_confidence = getattr(fuzzy_result, "confidence", 0.0)
        elif semantic_result:
            better_result = semantic_result
            combined_confidence = getattr(semantic_result, "confidence", 0.0)
        else:
            return None

        # Apply confidence adjustments
        if better_result.method != "hybrid_combination":
            combined_confidence *= 0.9  # Slight penalty for non-hybrid results

        if (
            better_result is None
            or not hasattr(better_result, "artist")
            or not hasattr(better_result, "song_title")
        ):
            return None

        hybrid_result = ParseResult(
            artist=better_result.original_artist,
            song_title=better_result.song_title,
            confidence=combined_confidence,
            method="hybrid_fuzzy_semantic",
            pattern_used=(
                f"hybrid_fuzzy:{getattr(fuzzy_result, 'confidence', 0.0):.2f}_semantic:{getattr(semantic_result, 'confidence', 0.0):.2f}"
            ),
            metadata={
                "fuzzy_confidence": getattr(fuzzy_result, "confidence", 0.0),
                "semantic_confidence": getattr(semantic_result, "confidence", 0.0),
                "combined_confidence": combined_confidence,
                "method": "hybrid_combination",
            },
        )

        return hybrid_result

    def _entity_pattern_matching(
        self, title: str, entities: Dict[str, List[str]]
    ) -> Optional[ParseResult]:
        """Pattern-based matching using extracted entities."""

        # This is a simpler fallback approach using entity extraction
        quoted_texts = entities.get("quoted_text", [])
        capitalized = entities.get("capitalized_words", [])

        if len(quoted_texts) >= 2:
            # Assume first quoted text is artist, second is song
            return ParseResult(
                artist=quoted_texts[0],
                song_title=quoted_texts[1],
                confidence=0.65,
                method="entity_pattern_quoted",
                pattern_used="quoted_text_extraction",
            )

        if len(quoted_texts) == 1 and len(capitalized) >= 1:
            # One quoted (probably song), one capitalized (probably artist)
            return ParseResult(
                artist=capitalized[0],
                song_title=quoted_texts[0],
                confidence=0.6,
                method="entity_pattern_mixed",
                pattern_used="mixed_entity_extraction",
            )

        if len(capitalized) >= 2:
            # Assume first capitalized is artist, last is song
            return ParseResult(
                artist=capitalized[0],
                song_title=capitalized[-1],
                confidence=0.55,
                method="entity_pattern_capitalized",
                pattern_used="capitalized_entity_extraction",
            )

        return None

    def _load_knowledge_base(self):
        """Load knowledge base of artists and songs."""

        # Initialize from advanced parser's known entities
        if self.advanced_parser.known_artists:
            for artist in self.advanced_parser.known_artists:
                self.artist_candidates[artist] = SemanticCandidate(
                    text=artist, category="artist", frequency=1, last_seen=time.time()
                )

        if self.advanced_parser.known_songs:
            for song in self.advanced_parser.known_songs:
                self.song_candidates[song] = SemanticCandidate(
                    text=song, category="song", frequency=1, last_seen=time.time()
                )

        # Generate embeddings for known entities if model is available
        if self.embedding_model and (self.artist_candidates or self.song_candidates):
            self._generate_knowledge_base_embeddings()

    def _generate_knowledge_base_embeddings(self):
        """Generate embeddings for all entities in the knowledge base."""

        if self.embedding_model is None:
            return

        try:
            # Generate artist embeddings
            artist_texts = list(self.artist_candidates.keys())
            if artist_texts:
                artist_embeddings = self.embedding_model.encode(artist_texts)
                for i, text in enumerate(artist_texts):
                    self.artist_candidates[text].embedding = artist_embeddings[i]

            # Generate song embeddings
            song_texts = list(self.song_candidates.keys())
            if song_texts:
                song_embeddings = self.embedding_model.encode(song_texts)
                for i, text in enumerate(song_texts):
                    self.song_candidates[text].embedding = song_embeddings[i]

            logger.info(
                f"Generated embeddings for {len(artist_texts)} artists and {len(song_texts)} songs"
            )

        except Exception as e:
            logger.warning(f"Failed to generate knowledge base embeddings: {e}")

    def add_entity(self, text: str, category: str, confidence: float = 1.0):
        """Add a new entity to the knowledge base."""

        if category == "artist":
            candidates = self.artist_candidates
        elif category == "song":
            candidates = self.song_candidates
        else:
            return

        if text in candidates:
            # Update existing
            candidates[text].frequency += 1
            candidates[text].last_seen = time.time()
        else:
            # Add new
            candidate = SemanticCandidate(
                text=text, category=category, frequency=1, last_seen=time.time()
            )

            # Generate embedding if model available
            if self.embedding_model:
                try:
                    candidate.embedding = self.embedding_model.encode([text])[0]
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for new entity '{text}': {e}")

            candidates[text] = candidate

    def get_statistics(self) -> Dict:
        """Get statistics for the ML embedding pass."""

        return {
            "has_embedding_model": self.embedding_model is not None,
            "embedding_model_name": self.embedding_model_name if self.embedding_model else None,
            "artist_candidates": len(self.artist_candidates),
            "song_candidates": len(self.song_candidates),
            "embedding_cache_size": len(self.embedding_cache),
            "entities_with_embeddings": sum(
                1
                for candidate in list(self.artist_candidates.values())
                + list(self.song_candidates.values())
                if candidate.embedding is not None
            ),
            "configuration": {
                "min_semantic_similarity": self.min_semantic_similarity,
                "min_fuzzy_similarity": self.min_fuzzy_similarity,
                "embedding_weight": self.embedding_weight,
                "fuzzy_weight": self.fuzzy_weight,
            },
        }
