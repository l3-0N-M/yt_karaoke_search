"""Advanced title parsing with machine learning-inspired techniques and robust handling."""

import logging
import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from .validation_corrector import ValidationResult

try:
    from .search.fuzzy_matcher import FuzzyMatcher

    HAS_FUZZY_MATCHER = True
except ImportError:
    HAS_FUZZY_MATCHER = False

logger = logging.getLogger(__name__)


@dataclass
class ParseResult:
    """Enhanced parsing result with detailed metadata."""

    original_artist: Optional[str] = None
    song_title: Optional[str] = None
    featured_artists: Optional[str] = None
    confidence: float = 0.0
    method: str = ""
    pattern_used: str = ""
    validation_score: float = 0.0
    alternative_results: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class PatternStats:
    """Statistics for pattern effectiveness."""

    pattern: str
    success_count: int = 0
    total_attempts: int = 0
    avg_confidence: float = 0.0
    common_failures: List[str] = field(default_factory=list)


class AdvancedTitleParser:
    """Advanced parser with multiple strategies and adaptive learning."""

    def __init__(self, config=None):
        self.config = config
        self.pattern_stats = defaultdict(lambda: PatternStats(""))
        self.known_artists = set()
        self.known_songs = set()
        self.channel_patterns = {}
        self.language_patterns = {}

        # Initialize fuzzy matcher if available
        if HAS_FUZZY_MATCHER:
            self.fuzzy_matcher = FuzzyMatcher(config)
        else:
            self.fuzzy_matcher = None
            logger.info("FuzzyMatcher not available, using basic fuzzy matching")

        # Load comprehensive patterns
        self._load_patterns()

        # Initialize validation datasets
        self._init_validation_data()

    def _load_patterns(self):
        """Load all pattern categories with enhanced coverage."""

        # Core karaoke patterns (existing + new)
        self.core_patterns = [
            # Specific karaoke format - Highest priority
            (
                r"^[Kk]araoke\s*[-–—]\s*([^-]+?)\s*[-–—]\s*(.+)$",
                2,
                1,
                0.98,
                "karaoke_song_artist",
            ),
            # Quoted patterns - Highest priority
            (
                r'^"([^"]+)"\s*[-–—]\s*"([^"]+)"\s*\([^)]*[Kk]araoke[^)]*\)',
                1,
                2,
                0.95,
                "quoted_artist_title",
            ),
            (r'^[Kk]araoke\s*:?\s*"([^"]+)"\s*[-–—]\s*"([^"]+)"', 2, 1, 0.95, "karaoke_quoted"),
            (r'^"([^"]+)"\s*\([^)]*[Ss]tyle\s+of\s+"([^"]+)"[^)]*\)', 2, 1, 0.9, "style_of_quoted"),
            # Channel-specific patterns
            (
                r'^([^-]+)\s*[-–—]\s*"([^"]+)"\s*\([^)]*[Ss]tyle\s+of\s+"([^"]+)"[^)]*\)',
                3,
                2,
                0.9,
                "channel_style_of",
            ),
            (
                r'^([^-]+)\s*[-–—]\s*"([^"]+)"\s*[-–—]\s*"([^"]+)"\s*\([^)]*[Kk]araoke[^)]*\)',
                2,
                3,
                0.85,
                "channel_artist_title",
            ),
            # Complex multi-part patterns
            (
                r'^"([^"]+)"-"([^"]+)"\s*(?:"[^"]*")*\s*\([^)]*[Kk]araoke[^)]*\)',
                1,
                2,
                0.85,
                "complex_quoted",
            ),
            (
                r'^"([^"]+)"\s*[-–—]\s*"([^"]+)"\s*[-–—]\s*"[^"]*[Kk]araoke[^"]*"',
                1,
                2,
                0.9,
                "triple_quoted",
            ),
            # Bracket patterns
            (
                r"^\[[^\]]*[Kk]araoke[^\]]*\]\s*([^-–—]+)[-–—]([^[\]]+)\s*\[[^\]]*\]",
                1,
                2,
                0.9,
                "bracket_karaoke",
            ),
            (
                r"^\[[^\]]*\]\s*([^-–—]+)[-–—]([^[\]]+)\s*\[[^\]]*[Kk]araoke[^\]]*\]",
                1,
                2,
                0.85,
                "bracket_suffix",
            ),
            # Standard patterns
            (
                r"^([^-–—]+?)\s*[-–—]\s*([^([\]]+?)\s*\([^)]*[Kk]araoke[^)]*\)",
                1,
                2,
                0.8,
                "standard_artist_title",
            ),
            (r"^([^(]+?)\s*\([^)]*[Ss]tyle\s+of\s+([^)]+?)\)", 2, 1, 0.75, "style_of_standard"),
            (r"^([^(]+?)\s+by\s+([^(]+?)\s*\([^)]*[Kk]araoke[^)]*\)", 2, 1, 0.75, "by_pattern"),
            # Fallback patterns
            (
                r"^([^-–—]+?)\s*[-–—]\s*([^([\]]+)(?:\s*[\(\[][^)\]]*[Kk]araoke[^)\]]*[\)\]])?",
                1,
                2,
                0.6,
                "basic_dash",
            ),
            (r"^([^(]+?)\s*\([^)]*[Kk]araoke[^)]*\)", None, 1, 0.5, "title_only"),
        ]

        # Language-specific patterns
        self.language_patterns = {
            "korean": [
                (
                    r"^\[[^\]]*가라오케[^\]]*\]\s*([^-–—]+)[-–—]([^[\]]+)",
                    1,
                    2,
                    0.9,
                    "korean_bracket",
                ),
                (r"^([^-–—]+)[-–—]([^(]+)\s*\([^)]*가라오케[^)]*\)", 1, 2, 0.85, "korean_standard"),
            ],
            "japanese": [
                (
                    r"^\[[^\]]*カラオケ[^\]]*\]\s*([^-–—]+)[-–—]([^[\]]+)",
                    1,
                    2,
                    0.9,
                    "japanese_bracket",
                ),
                (
                    r"^([^-–—]+)[-–—]([^(]+)\s*\([^)]*カラオケ[^)]*\)",
                    1,
                    2,
                    0.85,
                    "japanese_standard",
                ),
            ],
            "chinese": [
                (
                    r"^\[[^\]]*卡拉OK[^\]]*\]\s*([^-–—]+)[-–—]([^[\]]+)",
                    1,
                    2,
                    0.9,
                    "chinese_bracket",
                ),
                (r"^([^-–—]+)[-–—]([^(]+)\s*\([^)]*卡拉OK[^)]*\)", 1, 2, 0.85, "chinese_standard"),
            ],
        }

        # Channel-specific patterns (can be learned)
        self.channel_patterns = {
            "sing_king": [
                (
                    r'^Sing King Karaoke\s*[-–—]\s*"([^"]+)"\s*\([^)]*[Ss]tyle\s+of\s+"([^"]+)"[^)]*\)',
                    2,
                    1,
                    0.95,
                    "sing_king_style",
                ),
                (
                    r"^Sing King.*?[-–—]\s*([^(]+?)\s*\([^)]*[Kk]araoke[^)]*\)",
                    None,
                    1,
                    0.8,
                    "sing_king_basic",
                ),
            ],
            "zoom_karaoke": [
                (
                    r'^"([^"]+)"\s*[-–—]\s*"([^"]+)"\s*[-–—]\s*"[^"]*Zoom\s*Karaoke[^"]*"',
                    1,
                    2,
                    0.9,
                    "zoom_triple",
                ),
                (
                    r"^([^-–—]+)\s*[-–—]\s*([^-–—]+)\s*[-–—]\s*.*?Zoom\s*Karaoke",
                    1,
                    2,
                    0.8,
                    "zoom_basic",
                ),
            ],
        }

    def _init_validation_data(self):
        """Initialize validation datasets for artist/song validation."""

        # Common artist indicators
        self.artist_indicators = {
            "prefixes": {"the", "a", "an", "los", "las", "le", "la", "les", "der", "die", "das"},
            "suffixes": {"band", "group", "orchestra", "choir", "ensemble", "quartet", "trio"},
            "conjunctions": {"and", "&", "feat", "ft", "featuring", "with", "vs", "versus"},
        }

        # Common song title indicators
        self.song_indicators = {
            "parentheticals": {
                "remix",
                "extended",
                "radio edit",
                "album version",
                "single",
                "live",
            },
            "numbering": re.compile(r"\b(part|pt\.?)\s*\d+\b|\b\d+\.\s*|\(\d+\)"),
        }

        # Invalid terms that definitely aren't artists/songs
        self.definitely_invalid = {
            "technical": {"mp3", "mp4", "wav", "flac", "audio", "video", "hd", "4k", "1080p"},
            "karaoke_terms": {"karaoke", "instrumental", "backing track", "minus one", "playback"},
            "generic": {"music", "song", "track", "number", "piece", "composition"},
        }

    def parse_title(
        self, title: str, description: str = "", tags: str = "", channel_name: str = ""
    ) -> ParseResult:
        """Main parsing method with multi-strategy approach."""

        # Clean and normalize input
        clean_title = self._advanced_clean_title(title)

        # Multi-pass parsing
        results = []

        # Pass 1: Channel-specific patterns
        if channel_name:
            channel_result = self._parse_with_channel_patterns(clean_title, channel_name)
            if channel_result.confidence > 0.7:
                return channel_result
            results.append(channel_result)

        # Pass 2: Language-specific patterns
        detected_language = self._detect_language(clean_title)
        if detected_language != "english":
            lang_result = self._parse_with_language_patterns(clean_title, detected_language)
            if lang_result.confidence > 0.8:
                return lang_result
            results.append(lang_result)

        # Pass 3: Core pattern matching
        core_result = self._parse_with_core_patterns(clean_title)
        results.append(core_result)

        # Pass 4: ML-inspired heuristic parsing
        heuristic_result = self._parse_with_heuristics(clean_title, description, tags)
        results.append(heuristic_result)

        # Pass 5: Advanced fuzzy matching against known data
        if (self.known_artists or self.known_songs) and self.fuzzy_matcher:
            fuzzy_result = self._parse_with_advanced_fuzzy_matching(clean_title)
            results.append(fuzzy_result)
        elif self.known_artists or self.known_songs:
            # Fallback to basic fuzzy matching
            fuzzy_result = self._parse_with_fuzzy_matching(clean_title)
            results.append(fuzzy_result)

        # Select best result using confidence weighting
        best_result = self._select_best_result(results, clean_title)

        # Enhance with external validation
        self._validate_and_enhance_result(best_result, description, tags)

        # Learn from this parse for future improvement
        self._learn_from_parse(title, best_result)

        return best_result

    def _advanced_clean_title(self, title: str) -> str:
        """Advanced title cleaning with Unicode normalization."""

        # Unicode normalization
        cleaned = unicodedata.normalize("NFKC", title)

        # Remove common prefixes/suffixes
        prefixes = [
            r"^\[[^\]]*\]\s*",  # [Any brackets]
            r"^【[^】]*】\s*",  # 【CJK brackets】
            r"^.*?[Kk]araoke[^:]*:\s*",  # "Channel Karaoke:"
            r"^.*?presents\s*:?\s*",  # "Channel presents:"
            r"^Official\s+",  # "Official "
            r"^HD\s+",  # "HD "
        ]

        for prefix in prefixes:
            cleaned = re.sub(prefix, "", cleaned, flags=re.IGNORECASE)

        # Normalize separators
        cleaned = re.sub(r"[-–—]", "-", cleaned)  # Normalize dashes
        cleaned = re.sub(r"\s+", " ", cleaned).strip()  # Normalize whitespace

        return cleaned

    def _detect_language(self, title: str) -> str:
        """Detect the primary language of the title."""

        # Simple heuristic based on character sets
        if re.search(r"[\u4e00-\u9fff]", title):  # Chinese characters
            return "chinese"
        elif re.search(r"[\u3040-\u309f\u30a0-\u30ff]", title):  # Japanese hiragana/katakana
            return "japanese"
        elif re.search(r"[\uac00-\ud7af]", title):  # Korean hangul
            return "korean"
        elif re.search(r"[\u0400-\u04ff]", title):  # Cyrillic
            return "russian"
        else:
            return "english"

    def _parse_with_channel_patterns(self, title: str, channel_name: str) -> ParseResult:
        """Parse using channel-specific patterns."""

        # Identify channel type
        channel_lower = channel_name.lower()
        channel_type = None

        if "sing king" in channel_lower:
            channel_type = "sing_king"
        elif "zoom" in channel_lower and "karaoke" in channel_lower:
            channel_type = "zoom_karaoke"

        if channel_type and channel_type in self.channel_patterns:
            for (
                pattern,
                artist_group,
                title_group,
                confidence,
                pattern_name,
            ) in self.channel_patterns[channel_type]:
                match = re.search(pattern, title, re.IGNORECASE | re.UNICODE)
                if match:
                    return self._create_result_from_match(
                        match,
                        artist_group,
                        title_group,
                        confidence,
                        f"channel_{channel_type}",
                        pattern,
                    )

        return ParseResult(method="channel_specific")

    def _parse_with_language_patterns(self, title: str, language: str) -> ParseResult:
        """Parse using language-specific patterns."""

        if language in self.language_patterns:
            for (
                pattern,
                artist_group,
                title_group,
                confidence,
                pattern_name,
            ) in self.language_patterns[language]:
                match = re.search(pattern, title, re.IGNORECASE | re.UNICODE)
                if match:
                    return self._create_result_from_match(
                        match,
                        artist_group,
                        title_group,
                        confidence,
                        f"language_{language}",
                        pattern,
                    )

        return ParseResult(method=f"language_{language}")

    def _parse_with_core_patterns(self, title: str) -> ParseResult:
        """Parse using core regex patterns."""

        for pattern, artist_group, title_group, confidence, pattern_name in self.core_patterns:
            match = re.search(pattern, title, re.IGNORECASE | re.UNICODE)
            if match:
                result = self._create_result_from_match(
                    match, artist_group, title_group, confidence, "core_patterns", pattern
                )
                if result.confidence > 0:
                    return result

        return ParseResult(method="core_patterns")

    def _parse_with_heuristics(self, title: str, description: str, tags: str) -> ParseResult:
        """ML-inspired heuristic parsing using statistical analysis."""

        # Analyze word frequencies and positions
        words = re.findall(r"\b\w+\b", title.lower())

        # Look for artist/title separation indicators
        separators = ["-", "–", "—", "by", "from", "feat", "ft", "featuring"]
        separator_positions = []

        for i, word in enumerate(words):
            if word in separators or re.match(
                r"^[-–—]$", title[title.find(word) : title.find(word) + 1]
            ):
                separator_positions.append(i)

        if separator_positions:
            # Use most likely separator
            sep_pos = separator_positions[0]

            # Heuristic: shorter part is usually artist, longer is title
            before_sep = " ".join(words[:sep_pos])
            after_sep = " ".join(words[sep_pos + 1 :])

            # Remove karaoke indicators
            after_sep_clean = re.sub(
                r"\b(karaoke|instrumental|backing|track)\b.*", "", after_sep
            ).strip()

            if len(before_sep) < len(after_sep_clean) and len(before_sep) > 0:
                return ParseResult(
                    original_artist=before_sep.title(),
                    song_title=after_sep_clean.title(),
                    confidence=0.6,
                    method="heuristic_length",
                    pattern_used="length_based_separation",
                )
            elif len(after_sep_clean) > 0:
                return ParseResult(
                    original_artist=after_sep_clean.title(),
                    song_title=before_sep.title(),
                    confidence=0.5,
                    method="heuristic_length",
                    pattern_used="reverse_length_based",
                )

        # Look in description for artist info
        desc_patterns = [
            r"(?:by|artist|performed by|original artist):\s*([^\n\r]+)",
            r"Artist:\s*([^\n\r]+)",
            r"Song:\s*([^\n\r]+)",
        ]

        for pattern in desc_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                artist = self._clean_extracted_text(match.group(1))
                if self._is_valid_artist_name(artist):
                    return ParseResult(
                        original_artist=artist,
                        confidence=0.7,
                        method="description_extraction",
                        pattern_used=pattern,
                    )

        return ParseResult(method="heuristics")

    def _parse_with_fuzzy_matching(self, title: str) -> ParseResult:
        """Parse using fuzzy matching against known artists/songs."""

        # Extract potential artist/song candidates
        candidates = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", title)

        best_artist_match = None
        best_song_match = None
        best_artist_score = 0
        best_song_score = 0

        for candidate in candidates:
            # Check against known artists
            for known_artist in self.known_artists:
                score = SequenceMatcher(None, candidate.lower(), known_artist.lower()).ratio()
                if score > best_artist_score and score > 0.8:
                    best_artist_score = score
                    best_artist_match = known_artist

            # Check against known songs
            for known_song in self.known_songs:
                score = SequenceMatcher(None, candidate.lower(), known_song.lower()).ratio()
                if score > best_song_score and score > 0.8:
                    best_song_score = score
                    best_song_match = known_song

        if best_artist_match or best_song_match:
            return ParseResult(
                original_artist=best_artist_match,
                song_title=best_song_match,
                confidence=max(best_artist_score, best_song_score)
                * 0.9,  # Slight penalty for fuzzy
                method="fuzzy_matching",
                pattern_used=f"fuzzy_artist:{best_artist_score:.2f}_song:{best_song_score:.2f}",
            )

        return ParseResult(method="fuzzy_matching")

    def _parse_with_advanced_fuzzy_matching(self, title: str) -> ParseResult:
        """Parse using advanced fuzzy matching with phonetic similarity."""
        if not self.fuzzy_matcher:
            return ParseResult(method="advanced_fuzzy_matching")

        # Extract potential artist/song candidates
        candidates = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", title)

        best_artist_match = None
        best_song_match = None
        best_combined_score = 0.0

        # Try fuzzy matching against known artists
        for candidate in candidates:
            artist_match = self.fuzzy_matcher.find_best_match(
                candidate, list(self.known_artists), "artist", min_score=0.7
            )

            if artist_match:
                # Try to find corresponding song from remaining title
                remaining_title = title.replace(candidate, "").strip()
                remaining_candidates = re.findall(r"\b\w+(?:\s+\w+)*\b", remaining_title)

                for song_candidate in remaining_candidates:
                    song_match = self.fuzzy_matcher.find_best_match(
                        song_candidate, list(self.known_songs), "song", min_score=0.7
                    )

                    if song_match:
                        combined_score = artist_match.score * 0.6 + song_match.score * 0.4
                        if combined_score > best_combined_score:
                            best_combined_score = combined_score
                            best_artist_match = artist_match
                            best_song_match = song_match

        # Also try direct song matching
        song_match = self.fuzzy_matcher.find_best_match(
            title, list(self.known_songs), "song", min_score=0.8
        )

        if song_match and song_match.score > best_combined_score:
            best_song_match = song_match
            best_artist_match = None
            best_combined_score = song_match.score

        if best_artist_match or best_song_match:
            confidence = best_combined_score * 0.95  # Slight penalty for fuzzy matching

            return ParseResult(
                original_artist=best_artist_match.matched if best_artist_match else None,
                song_title=best_song_match.matched if best_song_match else None,
                confidence=confidence,
                method="advanced_fuzzy_matching",
                pattern_used=f"fuzzy_artist:{best_artist_match.score if best_artist_match else 0:.2f}_song:{best_song_match.score if best_song_match else 0:.2f}",
                metadata={
                    "artist_match": asdict(best_artist_match) if best_artist_match else None,
                    "song_match": asdict(best_song_match) if best_song_match else None,
                    "fuzzy_method": "advanced_with_phonetic",
                },
            )

        return ParseResult(method="advanced_fuzzy_matching")

    def _create_result_from_match(
        self, match, artist_group, title_group, confidence, method, pattern
    ) -> ParseResult:
        """Create ParseResult from regex match."""

        result = ParseResult(method=method, pattern_used=pattern)

        if artist_group and artist_group <= len(match.groups()):
            artist = self._clean_extracted_text(match.group(artist_group))
            if self._is_valid_artist_name(artist):
                result.original_artist = artist

        if title_group and title_group <= len(match.groups()):
            song_title = self._clean_extracted_text(match.group(title_group))
            if self._is_valid_song_title(song_title):
                result.song_title = song_title

        # Calculate confidence based on extraction success
        if result.original_artist and result.song_title:
            result.confidence = confidence
        elif result.original_artist or result.song_title:
            result.confidence = confidence * 0.7
        else:
            result.confidence = 0

        return result

    def _select_best_result(self, results: List[ParseResult], original_title: str) -> ParseResult:
        """Select the best result from multiple parsing attempts."""

        # Filter out empty results
        valid_results = [r for r in results if r.confidence > 0]

        if not valid_results:
            return ParseResult(method="no_match")

        # Weight results by confidence and completeness
        scored_results = []
        for result in valid_results:
            score = result.confidence

            # Bonus for having both artist and title
            if result.original_artist and result.song_title:
                score *= 1.2

            # Bonus for high-confidence methods
            if result.method in ["channel_specific", "language_specific"]:
                score *= 1.1

            # Penalty for very short extractions
            if result.original_artist and len(result.original_artist) < 3:
                score *= 0.8
            if result.song_title and len(result.song_title) < 3:
                score *= 0.8

            scored_results.append((score, result))

        # Return highest scored result
        best_score, best_result = max(scored_results, key=lambda x: x[0])
        best_result.validation_score = best_score

        # Store alternatives
        best_result.alternative_results = [
            {
                "method": r.method,
                "confidence": r.confidence,
                "artist": r.original_artist,
                "title": r.song_title,
            }
            for _, r in sorted(scored_results, key=lambda x: x[0], reverse=True)[1:3]
        ]

        return best_result

    def _validate_and_enhance_result(self, result: ParseResult, description: str, tags: str):
        """Validate and enhance the result using external information."""

        # Cross-validate with description/tags
        if result.original_artist:
            # Check if artist appears in description
            if result.original_artist.lower() in description.lower():
                result.confidence *= 1.1
                result.metadata["description_validation"] = True

        if result.song_title:
            # Check if song title appears in description
            if result.song_title.lower() in description.lower():
                result.confidence *= 1.1
                result.metadata["description_validation"] = True

        # Extract featured artists
        if result.original_artist or result.song_title:
            featured = self._extract_featured_artists_advanced(
                f"{result.original_artist or ''} {result.song_title or ''}", description
            )
            if featured:
                result.featured_artists = featured

    def _extract_featured_artists_advanced(self, title: str, description: str) -> Optional[str]:
        """Advanced featured artist extraction."""

        combined_text = f"{title} {description}".lower()

        patterns = [
            r"feat\.?\s+([^(\[,]+)",
            r"featuring\s+([^(\[,]+)",
            r"ft\.?\s+([^(\[,]+)",
            r"with\s+([^(\[,]+)",
            r"&\s+([^(\[,]+)",
            r"\+\s+([^(\[,]+)",
            r"vs\.?\s+([^(\[,]+)",
            r"x\s+([^(\[,]+)",
        ]

        featured = set()
        for pattern in patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            for match in matches:
                cleaned = re.sub(r"[^\w\s,&]", "", match.strip())
                parts = re.split(r"\s*[,&]\s*|\s+and\s+", cleaned)
                for part in parts:
                    artist = part.strip()
                    if self._is_valid_artist_name(artist) and len(artist) > 2:
                        featured.add(artist.title())

        return ", ".join(sorted(featured)) if featured else None

    def _learn_from_parse(self, original_title: str, result: ParseResult):
        """Learn from parsing results to improve future performance."""

        # Update pattern statistics
        if result.pattern_used:
            if result.pattern_used not in self.pattern_stats:
                self.pattern_stats[result.pattern_used] = PatternStats(pattern=result.pattern_used)

            stats = self.pattern_stats[result.pattern_used]
            stats.total_attempts += 1
            if result.confidence > 0.7:
                stats.success_count += 1
            stats.avg_confidence = (
                stats.avg_confidence * (stats.total_attempts - 1) + result.confidence
            ) / stats.total_attempts

        # Add to known artists/songs if high confidence
        if result.confidence > 0.8:
            if result.original_artist:
                self.known_artists.add(result.original_artist)
            if result.song_title:
                self.known_songs.add(result.song_title)

    def apply_validation_feedback(
        self, artist: str, song: str, validation_result: "ValidationResult"
    ) -> None:
        """Update internal datasets based on validation results."""
        if validation_result.artist_valid and artist:
            self.known_artists.add(artist)
        if validation_result.song_valid and song:
            self.known_songs.add(song)

    def _clean_extracted_text(self, text: str) -> str:
        """Clean extracted text with advanced techniques."""

        if not text:
            return ""

        # Remove quotes and brackets
        cleaned = re.sub(r'^["\'`]+|["\'`]+$', "", text.strip())
        cleaned = re.sub(r"^\([^)]*\)|^\[[^\]]*\]", "", cleaned).strip()

        # Remove trailing noise
        noise_patterns = [
            r"\s*\([^)]*(?:[Kk]araoke|[Ii]nstrumental|[Mm]inus|[Mm][Rr])[^)]*\)$",
            r"\s*\[[^\]]*(?:[Kk]araoke|[Ii]nstrumental|[Mm]inus|[Mm][Rr])[^\]]*\]$",
            r"\s*-\s*[Kk]araoke.*$",
            r"\s*[Mm][Rr]$",
            r"\s*[Ii]nst\.?$",
            r"\s*\([^)]*[Kk]ey\)$",
            r"\s*\([^)]*\d+[Kk]ey\)$",
            r"\s*\(.*?[Bb]acking.*?\)$",
        ]

        for pattern in noise_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

        # Normalize capitalization
        if cleaned.isupper() or cleaned.islower():
            cleaned = cleaned.title()

        # Clean up whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned

    def _is_valid_artist_name(self, artist: str) -> bool:
        """Enhanced artist name validation."""

        if not artist or len(artist.strip()) < 1:
            return False

        artist_lower = artist.lower().strip()

        # Check against definitely invalid terms
        for category in self.definitely_invalid.values():
            if artist_lower in category:
                return False

        # Check for reasonable character composition
        word_chars = len(re.findall(r"\w", artist))
        if word_chars < len(artist) * 0.3:
            return False

        # Check for realistic length
        if len(artist) > 100 or len(artist) < 1:
            return False

        # Allow known artist patterns
        if any(indicator in artist_lower for indicator in self.artist_indicators["conjunctions"]):
            return True

        return True

    def _is_valid_song_title(self, title: str) -> bool:
        """Enhanced song title validation."""

        if not title or len(title.strip()) < 1:
            return False

        title_lower = title.lower().strip()

        # Check against karaoke-specific invalid terms
        if title_lower in {"karaoke", "instrumental", "backing track", "minus one", "mr", "inst"}:
            return False

        # Allow reasonable length
        return len(title) <= 200

    def get_statistics(self) -> Dict:
        """Get parser performance statistics."""

        total_attempts = sum(stats.total_attempts for stats in self.pattern_stats.values())
        total_successes = sum(stats.success_count for stats in self.pattern_stats.values())

        return {
            "total_parses": total_attempts,
            "success_rate": total_successes / total_attempts if total_attempts > 0 else 0,
            "known_artists": len(self.known_artists),
            "known_songs": len(self.known_songs),
            "pattern_performance": {
                pattern: {
                    "success_rate": (
                        stats.success_count / stats.total_attempts
                        if stats.total_attempts > 0
                        else 0
                    ),
                    "avg_confidence": stats.avg_confidence,
                    "total_uses": stats.total_attempts,
                }
                for pattern, stats in self.pattern_stats.items()
            },
        }
