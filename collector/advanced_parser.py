"""Advanced title parsing with machine learning-inspired techniques and robust handling."""

import logging
import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from .channel_formats import (
    CHANNEL_FORMAT_CONFIDENCE_BOOST,
    TitleFormat,
    get_channel_format,
    should_trust_channel_format,
)
from .validation_corrector import ValidationResult

try:
    from .search.fuzzy_matcher import FuzzyMatcher

    HAS_FUZZY_MATCHER = True
except ImportError:
    HAS_FUZZY_MATCHER = False

# FillerWordProcessor was removed with web search pass
FillerWordProcessor = None  # type: ignore
HAS_FILLER_PROCESSOR = False

logger = logging.getLogger(__name__)


@dataclass
class ParseResult:
    """Enhanced parsing result with detailed metadata."""

    artist: Optional[str] = None  # renamed from original_artist
    song_title: Optional[str] = None
    featured_artists: Optional[str] = None
    confidence: float = 0.0
    method: str = ""
    pattern_used: str = ""
    validation_score: float = 0.0
    alternative_results: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    # Backward compatibility property
    @property
    def original_artist(self) -> Optional[str]:
        """Backward compatibility for original_artist field name."""
        return self.artist

    @original_artist.setter
    def original_artist(self, value: Optional[str]):
        """Backward compatibility setter for original_artist field name."""
        self.artist = value


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
            self.fuzzy_matcher = FuzzyMatcher(config)  # type: ignore
        else:
            self.fuzzy_matcher = None
            logger.info("FuzzyMatcher not available, using basic fuzzy matching")

        # Initialize filler word processor for enhanced cleaning
        if HAS_FILLER_PROCESSOR and FillerWordProcessor is not None:
            self.filler_processor = FillerWordProcessor()
        else:
            self.filler_processor = None

        # Load comprehensive patterns
        self._load_patterns()

        # Initialize validation datasets
        self._init_validation_data()

    def _load_patterns(self):
        """Load all pattern categories with enhanced coverage."""

        # Core karaoke patterns (existing + new)
        self.core_patterns = [
            # KaraFun Deutschland specific format: "DE artist song number"
            # Examples: "DE sarah lombardi wohin gehst du 57948"
            #           "DE helene fischer driving home for christmas 56433"
            (
                r"^DE\s+(.+?)\s+\d{4,}$",
                "custom_karafun_de",  # Special handling to split artist/song
                None,  # No title group since we need custom splitting
                0.85,
                "karafun_de_format",
            ),
            # Channel-specific patterns - Highest priority
            # Let's Sing Karaoke format: "LastName, FirstName - Song Title (Karaoke & Lyrics)"
            (
                r"^([^,]+),\s*([^-]+)\s*[-–—]\s*([^(]+?)\s*\([^)]*[Kk]araoke[^)]*\)",
                "custom_lets_sing",  # Special handling needed for name reordering
                3,
                0.95,
                "lets_sing_karaoke_format",
            ),
            # ZZang format: "Song (Karaoke) - Artist"
            (
                r"^([^(]+?)\s*\([Kk]araoke\)\s*[-–—]\s*(.+)$",
                2,  # Artist is after dash
                1,  # Song is before parentheses
                0.95,
                "zzang_karaoke_format",
            ),
            # BandaisuanKaraoke format: "Song / Artist (Karaoke)"
            (
                r"^([^/]+?)\s*/\s*([^(]+?)\s*\([^)]*[Kk]araoke[^)]*\)",
                2,  # Artist is after slash
                1,  # Song is before slash
                0.95,
                "bandaisuan_slash_format",
            ),
            # KaraokeyTV format: "Song Artist (Karaoke)" - no separator!
            # Look for pattern like "My Heart Will Go On Celine Dion (Karaoke)"
            # Strategy: Last 1-3 capitalized words before (Karaoke) are likely the artist
            (
                r"^(.+?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s*\([Kk]araoke\)(?:\s*#\d+)?$",
                2,  # Last 1-3 capitalized words before (Karaoke) are artist
                1,  # Everything before that is song
                0.75,  # Lower confidence due to ambiguity
                "karaokeyTV_no_separator",
            ),
            # Pipe-separated format with Karaoke indicator: "Song - Artist | Karaoke..."
            # More specific pattern to avoid false matches
            (
                r"^([^-]+?)\s*[-–—]\s*([^|]+?)\s*\|.*[Kk]araoke",
                2,  # Artist is after dash
                1,  # Song is before dash
                0.92,
                "pipe_separated_song_artist_karaoke",
            ),
            # Pipe format with parentheses in first part: "Song (version) - Artist | ..."
            (
                r"^([^-]+?\([^)]+?\)[^-]*?)\s*[-–—]\s*([^|]+?)\s*\|",
                2,  # Artist is after dash
                1,  # Song is before dash (includes parentheses)
                0.91,
                "pipe_separated_song_with_parens_artist",
            ),
            # Generic pipe format - lower priority
            (
                r"^([^-|]+?)\s*[-–—]\s*([^|]+?)\s*\|",
                1,  # Artist is before dash
                2,  # Song is after dash
                0.85,
                "pipe_separated_artist_song",
            ),
            # Lugn format: "ARTIST • Song Title • Karaoke"
            (
                r"^([^•]+?)\s*•\s*([^•]+?)\s*•\s*[Kk]araoke",
                1,
                2,
                0.95,
                "lugn_artist_song_karaoke",
            ),
            # KaraFun Deutschland format: "Karaoke Song Title - Artist Name *"
            (
                r"^[Kk]araoke\s+([^-]+?)\s*[-–—]\s*([^*]+?)\s*\*",
                2,
                1,
                0.95,
                "karafun_deutschland_format",
            ),
            # Specific karaoke format - Highest priority
            (
                r"^[Kk]araoke\s+(.+?)\s*[-–—]\s*(.+)$",
                2,
                1,
                0.85,
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
            # Song - Artist (Karaoke) pattern - must come before standard pattern
            # Detects when text after dash is likely an artist name
            (
                r"^([^-–—]+?)\s*[-–—]\s*([^([\]]+?)\s*\([^)]*[Kk]araoke[^)]*\)$",
                2,  # Artist is after dash
                1,  # Song is before dash
                0.82,  # Slightly higher than standard to take precedence
                "song_artist_karaoke_pattern",
            ),
            # Pattern for "Song - Artist Karaoke" (no parentheses)
            (
                r"^([^-–—]+?)\s*[-–—]\s*([^([\]|]+?)\s*[Kk]araoke\s*$",
                2,  # Artist is before "Karaoke"
                1,  # Song is before dash
                0.82,
                "song_artist_karaoke_no_parens",
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
            # Known artist patterns (highest confidence for cleaned German titles)
            (
                r"^(helene fischer|unheilig|beatrice egli|peter wackel|andrea berg|gregor meyle)\s+(.+)$",
                1,
                2,
                0.75,
                "known_artist_pattern",
            ),
            # Special case for hyphenated titles that are likely single songs
            # e.g., "no na-shoot (Karaoke Version)"
            (
                r"^([a-zA-Z]+(?:\s+[a-zA-Z]+)*-[a-zA-Z]+)\s*\(",
                None,  # No artist
                1,  # Whole match is the song title
                0.75,
                "hyphenated_song_title",
            ),
            # Fallback patterns
            (
                r"^([^-–—]+?)\s*[-–—]\s*([^([\]]+)(?:\s*[\(\[][^)\]]*[Kk]araoke[^)\]]*[\)\]])?",
                1,
                2,
                0.6,
                "basic_dash",
            ),
            # Space-separated artist-song patterns (for cleaned titles)
            (
                r"^([a-zA-Z\s]+?)\s+([a-zA-Z\s]{3,}?)$",
                1,
                2,
                0.5,
                "space_separated_artist_song",
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
            "lets_sing_karaoke": [
                (
                    r"^([^,]+),\s*([^-]+)\s*[-–—]\s*([^(]+?)\s*\([^)]*[Kk]araoke[^)]*\)",
                    "custom_lets_sing",
                    3,
                    0.95,
                    "lets_sing_lastname_firstname",
                ),
            ],
            "lugn": [
                (
                    r"^([^•]+?)\s*•\s*([^•]+?)\s*•\s*[Kk]araoke",
                    1,
                    2,
                    0.95,
                    "lugn_bullet_format",
                ),
            ],
            "karafun_deutschland": [
                (
                    r"^[Kk]araoke\s+([^-]+?)\s*[-–—]\s*([^*]+?)\s*\*",
                    2,
                    1,
                    0.95,
                    "karafun_song_artist_star",
                ),
            ],
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

        # IMPORTANT: Preserve original title for special pattern matching
        # Some patterns need to see pipe characters before they're removed
        original_title = title

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
        # Use original title to preserve pipe characters for specific patterns
        core_result = self._parse_with_core_patterns(original_title, channel_name)
        # If that fails, try with clean title
        if core_result.confidence == 0:
            core_result = self._parse_with_core_patterns(clean_title, channel_name)
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
        """Advanced title cleaning with Unicode normalization and enhanced filler removal."""

        # Unicode normalization
        cleaned = unicodedata.normalize("NFKC", title)

        # Apply enhanced filler word removal if available
        if self.filler_processor:
            try:
                filler_result = self.filler_processor.clean_query(cleaned, "english")
                cleaned = filler_result.cleaned_query
                # If filler removal produced empty result, fall back to original
                if not cleaned.strip():
                    cleaned = title
            except Exception as e:
                logger.warning(f"Filler word processing failed: {e}")
                # Continue with original cleaning logic

        # Remove common prefixes/suffixes (fallback if filler processor unavailable)
        prefixes = [
            r"^\[[^\]]*\]\s*",  # [Any brackets]
            r"^【[^】]*】\s*",  # 【CJK brackets】
            r"^.*?[Kk]araoke[^:]*:\s*",  # "Channel Karaoke:"
            r"^.*?presents\s*:?\s*",  # "Channel presents:"
            r"^Official\s+",  # "Official "
            r"^HD\s+",  # "HD "
            r"^(?:DE|EN|FR|ES|IT|PT|NL|PL|RU|JP|KR|CN)\s+",  # Language prefixes
        ]

        for prefix in prefixes:
            cleaned = re.sub(prefix, "", cleaned, flags=re.IGNORECASE)

        # Remove common pipe-separated suffixes
        # These often appear at the end of titles and confuse parsing
        pipe_suffixes = [
            r"\s*\|\s*[Kk]araoke\s+[Vv]ersion\s*\|\s*[Kk]ara[Ff]un$",  # | Karaoke Version | KaraFun
            r"\s*\|\s*[Kk]araoke\s+[Vv]ersion$",  # | Karaoke Version
            r"\s*\|\s*[Kk]ara[Ff]un$",  # | KaraFun
            r"\s*\|\s*[Oo]fficial\s+[Kk]araoke.*$",  # | Official Karaoke...
        ]

        for suffix in pipe_suffixes:
            cleaned = re.sub(suffix, "", cleaned, flags=re.IGNORECASE)

        # Additional fallback cleaning for problematic patterns
        if not self.filler_processor:
            # Remove large ID numbers (fallback)
            cleaned = re.sub(r"\b\d{4,}\b", "", cleaned)
            # Remove quality indicators (fallback)
            cleaned = re.sub(r"\b(?:HD|HQ|4K|1080p|720p|480p)\b", "", cleaned, flags=re.IGNORECASE)

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

        if "let's sing karaoke" in channel_lower or "lets sing karaoke" in channel_lower:
            channel_type = "lets_sing_karaoke"
        elif "lugn" in channel_lower:
            channel_type = "lugn"
        elif "karafun deutschland" in channel_lower or "karafun" in channel_lower:
            channel_type = "karafun_deutschland"
        elif "sing king" in channel_lower:
            channel_type = "sing_king"
        elif "zoom" in channel_lower and "karaoke" in channel_lower:
            channel_type = "zoom_karaoke"

        if channel_type and channel_type in self.channel_patterns:
            for (
                pattern,
                artist_group,
                title_group,
                confidence,
                _,  # pattern_name
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
                _,  # pattern_name
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

    def _parse_with_core_patterns(self, title: str, channel_name: str = "") -> ParseResult:
        """Parse using core regex patterns with channel-aware selection."""

        # Check if this channel has a known format preference
        channel_format = get_channel_format(channel_name) if channel_name else None

        # If channel has known format, prioritize matching patterns
        if channel_format in [TitleFormat.ARTIST_SONG, TitleFormat.SONG_ARTIST]:
            # Try channel-preferred patterns first
            for pattern, artist_group, title_group, confidence, pattern_name in self.core_patterns:
                # Skip patterns that don't match the channel's known format
                if channel_format == TitleFormat.ARTIST_SONG:
                    # For Artist-Song channels, skip Song-Artist patterns
                    if pattern_name in [
                        "song_artist_karaoke_pattern",
                        "pipe_separated_song_artist_karaoke",
                        "pipe_separated_song_with_parens_artist",
                        "by_pattern",
                        "style_of_standard",
                    ]:
                        continue
                elif channel_format == TitleFormat.SONG_ARTIST:
                    # For Song-Artist channels, prioritize those patterns
                    if pattern_name not in [
                        "song_artist_karaoke_pattern",
                        "pipe_separated_song_artist_karaoke",
                        "pipe_separated_song_with_parens_artist",
                        "by_pattern",
                        "style_of_standard",
                    ]:
                        continue

                match = re.search(pattern, title, re.IGNORECASE | re.UNICODE)
                if match:
                    result = self._create_result_from_match(
                        match,
                        artist_group,
                        title_group,
                        confidence + CHANNEL_FORMAT_CONFIDENCE_BOOST,
                        "core_patterns",
                        pattern,
                        pattern_name,
                        title,
                        channel_name,
                    )
                    if result.confidence > 0:
                        return result

        # Standard pattern matching for unknown channels or if preferred patterns didn't match
        for pattern, artist_group, title_group, confidence, pattern_name in self.core_patterns:
            match = re.search(pattern, title, re.IGNORECASE | re.UNICODE)
            if match:
                result = self._create_result_from_match(
                    match,
                    artist_group,
                    title_group,
                    confidence,
                    "core_patterns",
                    pattern,
                    pattern_name,
                    title,
                    channel_name,
                )
                if result.confidence > 0:
                    return result

        return ParseResult(method="core_patterns")

    def _parse_with_heuristics(self, title: str, description: str, _: str) -> ParseResult:
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
                    artist=before_sep.title(),
                    song_title=after_sep_clean.title(),
                    confidence=0.6,
                    method="heuristic_length",
                    pattern_used="length_based_separation",
                )
            elif len(after_sep_clean) > 0:
                return ParseResult(
                    artist=after_sep_clean.title(),
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
                        artist=artist,
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
                artist=best_artist_match,
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
                artist=best_artist_match.matched if best_artist_match else None,
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
        self,
        match,
        artist_group,
        title_group,
        confidence,
        method,
        pattern,
        pattern_name=None,
        original_title="",
        channel_name="",
    ) -> ParseResult:
        """Create ParseResult from regex match."""

        result = ParseResult(method=method, pattern_used=pattern)

        # Check if channel has known format - if so, trust it over validation
        if channel_name and should_trust_channel_format(channel_name):
            channel_format = get_channel_format(channel_name)

            # For known Song-Artist channels, ensure we use Song-Artist extraction
            if channel_format == TitleFormat.SONG_ARTIST:
                # If pattern assumes Artist-Song but channel uses Song-Artist, switch
                if pattern_name in ["standard_artist_title", "pipe_separated_artist_song"]:
                    artist_group, title_group = title_group, artist_group

            # For known Artist-Song channels, ensure we use Artist-Song extraction
            elif channel_format == TitleFormat.ARTIST_SONG:
                # If pattern assumes Song-Artist but channel uses Artist-Song, switch
                if pattern_name in [
                    "song_artist_karaoke_pattern",
                    "song_artist_karaoke_no_parens",
                    "pipe_separated_song_artist_karaoke",
                    "pipe_separated_song_with_parens_artist",
                ]:
                    artist_group, title_group = title_group, artist_group
                # Also handle patterns that might extract in wrong order for non-karaoke titles
                elif (
                    pattern_name in ["basic_dash", "pipe_separated_artist_song"]
                    and title_group < artist_group
                ):
                    # These patterns might have wrong group order, ensure artist comes first
                    artist_group, title_group = title_group, artist_group

        # For unknown channels, use validation logic
        elif pattern_name not in [
            "song_artist_karaoke_pattern",
            "song_artist_karaoke_no_parens",
            "pipe_separated_song_artist_karaoke",
            "pipe_separated_song_with_parens_artist",
            "song_by_artist_pattern",
            "style_of_standard",
            "by_pattern",
            "zzang_karaoke_format",
            "bandaisuan_slash_format",
            "karaokeyTV_no_separator",
        ]:
            # For other patterns, check if we need to switch based on content
            if pattern_name in ["standard_artist_title", "pipe_separated_artist_song"]:
                # Use validation to determine if this is actually Song - Artist format
                should_switch = self._should_switch_artist_song(
                    match, artist_group, title_group, original_title
                )
                if should_switch:
                    # Switch the groups
                    artist_group, title_group = title_group, artist_group

        # Special handling for Let's Sing Karaoke format
        if artist_group == "custom_lets_sing":
            # Format: "LastName, FirstName - Song Title (Karaoke & Lyrics)"
            # Group 1: LastName, Group 2: FirstName, Group 3: Song Title
            if len(match.groups()) >= 3:
                last_name = self._clean_extracted_text(match.group(1))
                first_name = self._clean_extracted_text(match.group(2))
                # Reorder to "FirstName LastName"
                artist = f"{first_name.strip()} {last_name.strip()}".strip()
                if self._is_valid_artist_name(artist):
                    result.artist = artist
        # Special handling for KaraFun Deutschland format
        elif artist_group == "custom_karafun_de":
            # Format: "DE artist_and_song number"
            # Need to intelligently split artist and song
            if len(match.groups()) >= 1:
                artist_and_song = self._clean_extracted_text(match.group(1))
                artist, song = self._split_karafun_de_title(artist_and_song)
                if artist and self._is_valid_artist_name(artist):
                    result.artist = artist
                if song and self._is_valid_song_title(song):
                    result.song_title = song
        elif artist_group and artist_group <= len(match.groups()):
            artist = self._clean_extracted_text(match.group(artist_group))
            if self._is_valid_artist_name(artist):
                result.artist = artist

        if title_group and title_group <= len(match.groups()):
            song_title = self._clean_extracted_text(match.group(title_group))
            if self._is_valid_song_title(song_title):
                result.song_title = song_title

        # Calculate confidence based on extraction success
        if result.artist and result.song_title:
            result.confidence = confidence
        elif result.artist or result.song_title:
            result.confidence = confidence * 0.7
        else:
            result.confidence = 0

        return result

    def _select_best_result(self, results: List[ParseResult], _: str) -> ParseResult:
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
            if result.artist and result.song_title:
                score *= 1.2

            # Bonus for high-confidence methods
            if result.method in ["channel_specific", "language_specific"]:
                score *= 1.1

            # Penalty for very short extractions
            if result.artist and len(result.artist) < 3:
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
                "artist": r.artist,
                "title": r.song_title,
            }
            for _, r in sorted(scored_results, key=lambda x: x[0], reverse=True)[1:3]
        ]

        return best_result

    def _validate_and_enhance_result(self, result: ParseResult, description: str, _: str):
        """Validate and enhance the result using external information."""

        # Cross-validate with description/tags
        if result.artist:
            # Check if artist appears in description
            if result.artist.lower() in description.lower():
                result.confidence *= 1.1
                result.metadata["description_validation"] = True

        if result.song_title:
            # Check if song title appears in description
            if result.song_title.lower() in description.lower():
                result.confidence *= 1.1
                result.metadata["description_validation"] = True

        # Extract featured artists
        if result.artist or result.song_title:
            featured = self._extract_featured_artists_advanced(
                f"{result.artist or ''} {result.song_title or ''}", description
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

    def _learn_from_parse(self, _: str, result: ParseResult):
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
            if result.artist:
                self.known_artists.add(result.artist)
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

        # Remove leading "Karaoke" prefix
        cleaned = re.sub(r"^[Kk]araoke\s+", "", cleaned).strip()

        # Remove trailing noise
        # IMPORTANT: Be careful not to remove legitimate parentheses content
        # like "(From F1® The Movie)" which is part of the song title
        noise_patterns = [
            r"\s*\|.*$",  # Remove everything after pipe character
            # Only remove parentheses if they contain ONLY karaoke-related terms
            r"\s*\(\s*(?:[Kk]araoke|[Ii]nstrumental|[Mm]inus|[Mm][Rr]|[Bb]acking [Tt]rack|[Vv]ersion)\s*\)$",
            r"\s*\[\s*(?:[Kk]araoke|[Ii]nstrumental|[Mm]inus|[Mm][Rr])\s*\]$",
            # Combined karaoke indicators in parentheses
            r"\s*\([^)]*[Kk]araoke\s+[Vv]ersion\s*\)$",
            r"\s*\([Oo]fficial\s+[Kk]araoke\s+[Ii]nstrumental\s*\)$",  # Remove (Official Karaoke Instrumental)
            # Version indicators - can be configured to keep or remove
            r"\s*\(\s*(?:[Rr]ock|[Aa]coustic|[Ll]ive|[Dd]emo|[Rr]adio|[Pp]iano|[Jj]azz|[Cc]lub|[Dd]ance|[Ee]dit|[Mm]ix)\s+[Vv]ersion\s*\)$",
            r"\s*\([^)]*[Mm]elody\s*\)$",
            r"\s*\([^)]*MR/Instrumental\s*\)$",
            r"\s*-\s*[Kk]araoke.*$",
            r"\s*[Mm][Rr]$",
            r"\s*[Ii]nst\.?$",
            r"\s*\([^)]*[Kk]ey\)$",
            r"\s*\([^)]*\d+[Kk]ey\)$",
            r"\s*\(.*?[Bb]acking.*?\)$",
            r"\s*\*+\s*$",
        ]

        for pattern in noise_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

        # Normalize capitalization (conservative approach)
        # Only normalize if it's clearly all lowercase and appears to be a normal sentence
        # Preserve intentional all-caps artist names like "AYLIVA"
        if cleaned.islower() and len(cleaned) > 3:
            cleaned = cleaned.title()
        # Don't normalize all-uppercase names as they might be intentional stylization

        # Clean up whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Remove title repetitions (e.g., "My Girl - MY GIRL" -> "My Girl")
        cleaned = self._remove_title_repetitions(cleaned)

        # Remove trailing non-alphanumeric characters (but preserve intentional punctuation)
        # Skip if it ends with common punctuation that might be intentional
        if not re.search(r"[!?\.]$", cleaned):
            cleaned = re.sub(r"[^\w\s]+$", "", cleaned).strip()

        return cleaned

    def _remove_title_repetitions(self, text: str) -> str:
        """Remove title repetitions like 'My Girl - MY GIRL' -> 'My Girl'."""
        if not text or " - " not in text:
            return text

        # Split by common separators
        separators = [" - ", " – ", " — ", " | ", " / "]
        for sep in separators:
            if sep in text:
                parts = text.split(sep)
                if len(parts) == 2:
                    part1 = parts[0].strip()
                    part2 = parts[1].strip()

                    # Check if they're the same ignoring case
                    if part1.lower() == part2.lower():
                        # Return the better formatted one (prefer title case or original case)
                        if part1.istitle() and not part2.istitle():
                            return part1
                        elif part2.istitle() and not part1.istitle():
                            return part2
                        else:
                            # Default to first part
                            return part1

        # Also check for repetitions without separator (e.g., "Love Love" -> "Love")
        words = text.split()
        if len(words) >= 2:
            # Check for exact word repetition
            cleaned_words = []
            prev_word_lower = ""
            for word in words:
                word_lower = word.lower()
                if word_lower != prev_word_lower:
                    cleaned_words.append(word)
                    prev_word_lower = word_lower

            if len(cleaned_words) < len(words):
                return " ".join(cleaned_words)

        return text

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

    def _should_switch_artist_song(
        self, match, artist_group, title_group, original_title: str
    ) -> bool:
        """Determine if artist and song groups should be switched based on content analysis."""
        if (
            not artist_group
            or not title_group
            or artist_group > len(match.groups())
            or title_group > len(match.groups())
        ):
            return False

        potential_artist = self._clean_extracted_text(match.group(artist_group))
        potential_song = self._clean_extracted_text(match.group(title_group))

        # Common artist name indicators
        artist_indicators = [
            r"\b(band|boys|girls|brothers|sisters|orchestra|ensemble|duo|trio|quartet)\b",
            r"\b(ft\.|feat\.|featuring|with|and|&)\b",
            r"^(the|los|las|le|la|der|die|das)\s+\w+",  # Articles often start band names
            r"^\w+\s+(and|&)\s+\w+$",  # "X and Y" format common for duos
        ]

        # Common song title indicators
        song_indicators = [
            r"\b(love|heart|soul|dream|night|day|time|life|world|baby|girl|boy)\b",
            r"\b(dance|sing|song|music|melody|rhythm)\b",
            r"(n't|'s|'ve|'re|'ll|'d)\b",  # Contractions more common in song titles
            r"^(i|you|we|they|he|she|it)\s+",  # Personal pronouns often start songs
            r"\?$",  # Questions are often song titles
        ]

        artist_score = 0
        song_score = 0

        # Check artist indicators
        for pattern in artist_indicators:
            if re.search(pattern, potential_artist, re.IGNORECASE):
                artist_score += 1
            if re.search(pattern, potential_song, re.IGNORECASE):
                song_score += 1

        # Check song indicators
        for pattern in song_indicators:
            if re.search(pattern, potential_song, re.IGNORECASE):
                artist_score += 1
            if re.search(pattern, potential_artist, re.IGNORECASE):
                song_score += 1

        # Check if text after dash in original is in parentheses (often indicates it's the artist)
        if " - " in original_title:
            parts = original_title.split(" - ", 1)
            if len(parts) == 2:
                after_dash = parts[1].strip()
                # Remove everything after first parenthesis or pipe
                after_dash_clean = re.sub(r"[\(\|].*", "", after_dash).strip()

                # If what we have as potential_song matches what's after the dash, it's likely the artist
                if after_dash_clean.lower() == potential_song.lower():
                    return True

        # If song indicators strongly favor switching
        return song_score > artist_score + 1

    def _split_karafun_de_title(self, text: str) -> tuple[str, str]:
        """Split KaraFun Deutschland format intelligently.

        Examples:
        - "sarah lombardi wohin gehst du" -> ("Sarah Lombardi", "Wohin Gehst Du")
        - "helene fischer driving home for christmas" -> ("Helene Fischer", "Driving Home For Christmas")
        - "the little mermaid 1989 film unter dem meer" -> ("The Little Mermaid", "Unter Dem Meer")
        """
        words = text.strip().split()

        # Known German artists and patterns
        known_artists = {
            "helene fischer": 2,
            "sarah lombardi": 2,
            "andrea berg": 2,
            "beatrice egli": 2,
            "unheilig": 1,
            "peter wackel": 2,
            "gregor meyle": 2,
            "de toppers": 2,
            "german nursery rhyme": 3,
            "the little mermaid": 3,
        }

        # Check for known artists
        for artist_pattern, word_count in known_artists.items():
            if text.lower().startswith(artist_pattern):
                artist = " ".join(words[:word_count]).title()
                song = " ".join(words[word_count:]).title()
                return artist, song

        # Special case for movie soundtracks
        if "film" in text.lower() or "movie" in text.lower():
            # Find the word "film" or "movie" and split there
            film_idx = next((i for i, w in enumerate(words) if w.lower() in ["film", "movie"]), -1)
            if film_idx > 0:
                # Include "film" in the artist/movie name, song comes after
                artist = " ".join(words[: film_idx + 1]).title()
                song = " ".join(words[film_idx + 1 :]).title()
                return artist, song

        # Heuristic: First 1-2 words are usually the artist
        # Unless the first word is very common (the, a, etc.)
        common_words = {"the", "a", "an", "der", "die", "das", "ein", "eine"}

        if len(words) >= 3:
            if words[0].lower() in common_words:
                # If starts with common word, take first 3 words as potential artist
                # But if word 3 is also common, just take 2
                if len(words) > 3 and words[2].lower() not in common_words:
                    artist = " ".join(words[:3]).title()
                    song = " ".join(words[3:]).title()
                else:
                    artist = " ".join(words[:2]).title()
                    song = " ".join(words[2:]).title()
            else:
                # Standard case: first 2 words are artist
                artist = " ".join(words[:2]).title()
                song = " ".join(words[2:]).title()
        elif len(words) == 2:
            # Only 2 words - assume first is artist, second is song
            artist = words[0].title()
            song = words[1].title()
        else:
            # Single word or empty - can't split meaningfully
            artist = text.title()
            song = ""

        return artist, song

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
