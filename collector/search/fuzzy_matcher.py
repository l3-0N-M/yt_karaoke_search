"""Fuzzy matching system for handling artist/song name variations and typos."""

import logging
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

try:
    import jellyfish  # type: ignore

    HAS_JELLYFISH = True
except ImportError:
    HAS_JELLYFISH = False
    logger.info("jellyfish not available, using basic fuzzy matching")


@dataclass
class FuzzyMatch:
    """Result of fuzzy matching operation."""

    original: str
    matched: str
    score: float
    method: str
    normalized_original: str = ""
    normalized_matched: str = ""
    distance: int = 0


class FuzzyMatcher:
    """Advanced fuzzy matching with multiple algorithms and normalization strategies."""

    def __init__(self, config: Optional[object] = None) -> None:
        """Initialize the fuzzy matcher with optional configuration."""
        self.config = config or {}

        # Handle different config types
        if hasattr(config, "fuzzy_matching"):
            # Handle structured config object
            fuzzy_config = getattr(config, "fuzzy_matching", {})
            if hasattr(fuzzy_config, "__dict__"):
                fuzzy_config = fuzzy_config.__dict__
        elif hasattr(config, "__dict__"):
            # Handle other config objects
            fuzzy_config = config.__dict__.get("fuzzy_matching", {})
        elif isinstance(config, dict):
            # Handle dict config
            fuzzy_config = config.get("fuzzy_matching", {})
        else:
            fuzzy_config = {}

        # Matching thresholds
        self.min_similarity_threshold = fuzzy_config.get("min_similarity", 0.7)
        self.min_phonetic_threshold = fuzzy_config.get("min_phonetic", 0.8)
        self.max_edit_distance = fuzzy_config.get("max_edit_distance", 3)

        # Known normalization patterns
        self._load_normalization_patterns()

        # Cache for expensive operations
        self._normalization_cache = {}
        self._soundex_cache = {}
        self._metaphone_cache = {}

    def _load_normalization_patterns(self):
        """Load patterns for text normalization."""

        # Common artist name variations
        self.artist_normalizations = {
            # Articles
            r"^the\s+": "",
            r"^a\s+": "",
            r"^an\s+": "",
            # Punctuation and symbols
            r"[&]": "and",
            r"[\.,\-_]": " ",
            r'[\'\""]': "",
            r"\s+": " ",
            # Common abbreviations
            r"\bft\.?\b": "featuring",
            r"\bfeat\.?\b": "featuring",
            r"\bvs\.?\b": "versus",
            r"\bw/\b": "with",
        }

        # Common song title variations
        self.song_normalizations = {
            # Remove parentheticals that don't affect matching
            r"\s*\([^)]*(?:remix|edit|version|mix|remaster)[^)]*\)": "",
            r"\s*\([^)]*(?:live|acoustic|demo|instrumental)[^)]*\)": "",
            r"\s*\[[^\]]*(?:remix|edit|version|mix|remaster)[^\]]*\]": "",
            # Normalize punctuation
            r'[\'\""]': "",
            r"[\-_]": " ",
            r"\s+": " ",
        }

        # Phonetically similar replacements
        self.phonetic_replacements = {
            "ph": "f",
            "ck": "k",
            "qu": "kw",
            "tion": "shun",
            "sion": "shun",
            "c": "k",  # context-dependent
        }

    def normalize_text(self, text: str, text_type: str = "general") -> str:
        """Normalize text for better matching."""
        if not isinstance(text, str):
            text = str(text)

        cache_key = (text_type, text)
        if cache_key in self._normalization_cache:
            return self._normalization_cache[cache_key]

        # Unicode normalization
        normalized = unicodedata.normalize("NFKD", text.lower().strip())

        # Remove accents
        normalized = "".join(c for c in normalized if not unicodedata.combining(c))

        # Apply type-specific normalizations
        patterns = {}
        if text_type == "artist":
            patterns = self.artist_normalizations
        elif text_type == "song":
            patterns = self.song_normalizations
        else:
            patterns = {**self.artist_normalizations, **self.song_normalizations}

        for pattern, replacement in patterns.items():
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

        normalized = normalized.strip()
        self._normalization_cache[cache_key] = normalized
        return normalized

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate string similarity using multiple methods."""
        if not text1 or not text2:
            return 0.0

        # Quick exact match check
        if text1.lower() == text2.lower():
            return 1.0

        # Normalize both texts
        norm1 = self.normalize_text(text1)
        norm2 = self.normalize_text(text2)

        if norm1 == norm2:
            return 0.95

        # Use multiple similarity measures
        similarities = []

        # 1. Sequence matcher (Ratcliff-Obershelp)
        seq_sim = SequenceMatcher(None, norm1, norm2).ratio()
        similarities.append(("sequence", seq_sim))

        # 2. Jaro-Winkler similarity (if available)
        if HAS_JELLYFISH:
            jaro_sim = jellyfish.jaro_winkler_similarity(norm1, norm2)
            similarities.append(("jaro_winkler", jaro_sim))

        # 3. Levenshtein distance normalized
        if HAS_JELLYFISH:
            lev_distance = jellyfish.levenshtein_distance(norm1, norm2)
            max_len = max(len(norm1), len(norm2))
            lev_sim = 1.0 - (lev_distance / max_len) if max_len > 0 else 0.0
            similarities.append(("levenshtein", lev_sim))

        # 4. Token-based similarity for multi-word strings
        if " " in norm1 or " " in norm2:
            token_sim = self._calculate_token_similarity(norm1, norm2)
            similarities.append(("token", token_sim))

        # Weighted average of similarities
        if not similarities:
            return 0.0

        weights = {"sequence": 0.3, "jaro_winkler": 0.4, "levenshtein": 0.2, "token": 0.1}
        weighted_sum = sum(weights.get(method, 0.25) * score for method, score in similarities)
        total_weight = sum(weights.get(method, 0.25) for method, _ in similarities)

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _calculate_token_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity based on token overlap."""
        tokens1 = set(text1.split())
        tokens2 = set(text2.split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)

        # Jaccard similarity
        jaccard = len(intersection) / len(union) if union else 0.0

        # Bonus for ordered similarity
        ordered_bonus = 0.0
        list1, list2 = text1.split(), text2.split()
        if len(list1) == len(list2) and len(list1) > 0:
            matches = sum(1 for a, b in zip(list1, list2) if a == b)
            ordered_bonus = matches / len(list1) * 0.2
            ordered_bonus = min(ordered_bonus, 1.0 - jaccard)

        return jaccard + ordered_bonus

    def phonetic_similarity(self, text1: str, text2: str) -> float:
        """Calculate phonetic similarity using Soundex and Metaphone.

        Falls back to simplified algorithms when :mod:`jellyfish` is unavailable.
        """

        norm1 = self.normalize_text(text1)
        norm2 = self.normalize_text(text2)

        similarities = []

        # Soundex comparison
        try:
            soundex1 = self._get_soundex(norm1)
            soundex2 = self._get_soundex(norm2)
            if soundex1 == soundex2:
                similarities.append(1.0)
            else:
                # Partial soundex similarity
                seq_sim = SequenceMatcher(None, soundex1, soundex2).ratio()
                similarities.append(seq_sim)
        except Exception as exc:
            logger.debug("Soundex comparison failed: %s", exc)

        # Metaphone comparison
        try:
            metaphone1 = self._get_metaphone(norm1)
            metaphone2 = self._get_metaphone(norm2)
            if metaphone1 == metaphone2:
                similarities.append(1.0)
            else:
                seq_sim = SequenceMatcher(None, metaphone1, metaphone2).ratio()
                similarities.append(seq_sim)
        except Exception as exc:
            logger.debug("Metaphone comparison failed: %s", exc)

        return max(similarities) if similarities else 0.0

    def _get_soundex(self, text: str) -> str:
        """Get Soundex encoding with caching."""
        if text in self._soundex_cache:
            return self._soundex_cache[text]

        try:
            soundex = jellyfish.soundex(text) if HAS_JELLYFISH else self._basic_soundex(text)
            self._soundex_cache[text] = soundex
            return soundex
        except Exception as exc:
            logger.debug("Soundex generation failed for '%s': %s", text, exc)
            return ""

    def _get_metaphone(self, text: str) -> str:
        """Get Metaphone encoding with caching."""
        if text in self._metaphone_cache:
            return self._metaphone_cache[text]

        try:
            metaphone = jellyfish.metaphone(text) if HAS_JELLYFISH else self._basic_metaphone(text)
            self._metaphone_cache[text] = metaphone
            return metaphone
        except Exception as exc:
            logger.debug("Metaphone generation failed for '%s': %s", text, exc)
            return ""

    def _basic_soundex(self, text: str) -> str:
        """Simple Soundex implementation used as a fallback."""
        text = re.sub(r"[^A-Za-z]", "", text).upper()
        if not text:
            return ""
        first = text[0]
        mapping = {
            "B": "1",
            "F": "1",
            "P": "1",
            "V": "1",
            "C": "2",
            "G": "2",
            "J": "2",
            "K": "2",
            "Q": "2",
            "S": "2",
            "X": "2",
            "Z": "2",
            "D": "3",
            "T": "3",
            "L": "4",
            "M": "5",
            "N": "5",
            "R": "6",
        }
        encoded = [mapping.get(ch, "") for ch in text[1:]]
        result = [first]
        prev = mapping.get(first, "")
        for code in encoded:
            if code != prev and code != "":
                result.append(code)
            prev = code
        result = "".join(result).ljust(4, "0")[:4]
        return result

    def _basic_metaphone(self, text: str) -> str:
        """Very small Metaphone-like fallback implementation."""
        text = re.sub(r"[^A-Za-z]", "", text).lower()
        if not text:
            return ""
        # drop duplicate adjacent letters
        deduped = [text[0]]
        for ch in text[1:]:
            if ch != deduped[-1]:
                deduped.append(ch)
        text = "".join(deduped)
        # remove vowels except first letter
        first = text[0]
        remainder = re.sub(r"[aeiou]", "", text[1:])
        return (first + remainder)[:4].upper()

    def find_best_match(
        self, query: str, candidates: List[str], text_type: str = "general", min_score: float = None
    ) -> Optional[FuzzyMatch]:
        """Find the best fuzzy match from a list of candidates."""
        if not query or not candidates:
            return None

        min_score = min_score or self.min_similarity_threshold
        best_match = None
        best_score = 0.0

        for candidate in candidates:
            # Calculate combined similarity
            string_sim = self.calculate_similarity(query, candidate)
            phonetic_sim = self.phonetic_similarity(query, candidate)

            # Weighted combination
            combined_score = string_sim * 0.8 + phonetic_sim * 0.2

            if combined_score > best_score and combined_score >= min_score:
                best_score = combined_score
                method = "combined" if phonetic_sim > 0 else "string"

                best_match = FuzzyMatch(
                    original=query,
                    matched=candidate,
                    score=combined_score,
                    method=method,
                    normalized_original=self.normalize_text(query, text_type),
                    normalized_matched=self.normalize_text(candidate, text_type),
                )

        return best_match

    def find_all_matches(
        self,
        query: str,
        candidates: List[str],
        text_type: str = "general",
        min_score: float = None,
        max_results: int = 10,
    ) -> List[FuzzyMatch]:
        """Find all matches above threshold, sorted by score."""
        if not query or not candidates:
            return []

        min_score = min_score or self.min_similarity_threshold
        matches = []

        for candidate in candidates:
            string_sim = self.calculate_similarity(query, candidate)
            phonetic_sim = self.phonetic_similarity(query, candidate)
            combined_score = string_sim * 0.8 + phonetic_sim * 0.2

            if combined_score >= min_score:
                method = "combined" if phonetic_sim > 0 else "string"

                match = FuzzyMatch(
                    original=query,
                    matched=candidate,
                    score=combined_score,
                    method=method,
                    normalized_original=self.normalize_text(query, text_type),
                    normalized_matched=self.normalize_text(candidate, text_type),
                )
                matches.append(match)

        # Sort by score descending
        matches.sort(key=lambda x: x.score, reverse=True)
        return matches[:max_results]

    def match_artist_song_pair(
        self,
        query_artist: str,
        query_song: str,
        candidate_pairs: List[Tuple[str, str]],
        min_score: float = None,
    ) -> Optional[Tuple[FuzzyMatch, FuzzyMatch]]:
        """Match artist-song pairs with combined scoring."""
        if not query_artist or not query_song or not candidate_pairs:
            return None

        min_score = min_score or self.min_similarity_threshold
        best_match = None
        best_combined_score = 0.0

        for candidate_artist, candidate_song in candidate_pairs:
            # Match both artist and song
            artist_match = self.find_best_match(
                query_artist, [candidate_artist], "artist", min_score=0.5
            )
            song_match = self.find_best_match(query_song, [candidate_song], "song", min_score=0.5)

            if artist_match and song_match:
                # Combined score with artist weighted higher
                combined_score = artist_match.score * 0.6 + song_match.score * 0.4

                if combined_score > best_combined_score and combined_score >= min_score:
                    best_combined_score = combined_score
                    best_match = (artist_match, song_match)

        return best_match

    def get_statistics(self) -> Dict:
        """Get fuzzy matching statistics."""
        return {
            "normalization_cache_size": len(self._normalization_cache),
            "soundex_cache_size": len(self._soundex_cache),
            "metaphone_cache_size": len(self._metaphone_cache),
            "has_jellyfish": HAS_JELLYFISH,
            "thresholds": {
                "similarity": self.min_similarity_threshold,
                "phonetic": self.min_phonetic_threshold,
                "edit_distance": self.max_edit_distance,
            },
        }

    def clear_caches(self):
        """Clear all internal caches."""
        self._normalization_cache.clear()
        self._soundex_cache.clear()
        self._metaphone_cache.clear()
