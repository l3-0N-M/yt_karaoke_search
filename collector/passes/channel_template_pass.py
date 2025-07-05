"""Pass 0: Enhanced channel-template matching with dynamic pattern learning."""

import logging
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from ..advanced_parser import AdvancedTitleParser, ParseResult
from .base import ParsingPass, PassType

logger = logging.getLogger(__name__)


@dataclass
class ChannelPattern:
    """A learned pattern for a specific channel."""

    pattern: str
    artist_group: Optional[int]
    title_group: Optional[int]
    confidence: float
    success_count: int = 0
    total_attempts: int = 0
    last_used: datetime = field(default_factory=datetime.now)
    created: datetime = field(default_factory=datetime.now)
    examples: List[str] = field(default_factory=list)


@dataclass
class ChannelStats:
    """Statistics for a channel's parsing performance."""

    channel_id: str
    channel_name: str
    total_videos: int = 0
    successful_parses: int = 0
    patterns: Dict[str, ChannelPattern] = field(default_factory=dict)
    common_formats: Counter = field(default_factory=Counter)
    last_updated: datetime = field(default_factory=datetime.now)
    drift_detected: bool = False
    drift_threshold: float = 0.5


class EnhancedChannelTemplatePass(ParsingPass):
    """Enhanced Pass 0: Channel-template matching with learning and drift detection."""

    def __init__(self, advanced_parser: AdvancedTitleParser, db_manager=None):
        self.advanced_parser = advanced_parser
        self.db_manager = db_manager

        # Channel-specific learned patterns and statistics
        self.channel_stats: Dict[str, ChannelStats] = {}
        self.channel_patterns: Dict[str, List[ChannelPattern]] = defaultdict(list)

        # Global pattern library with weights
        self.global_pattern_weights: Dict[str, float] = {}

        # Learning configuration
        self.min_examples_for_pattern = 3
        self.max_patterns_per_channel = 10
        self.pattern_decay_days = 30
        self.drift_check_interval = timedelta(days=7)

        # Channels that intentionally use reversed formats (Song - Artist)
        # These channels should be exempt from swap detection
        self.swap_exempt_channels = {
            "@MusisiKaraoke",
            "UCJw1qyMF4m3ZIBWdhogkcsw",  # Musisi Karaoke
            "@AtomicKaraoke",
            "UCutZyApGOjqhOS-pp7yAj4Q",  # Atomic Karaoke
            # Note: Sing Karaoke removed - uses standard "Artist - Song" format
            "@karaokeytv0618",
            "UCNbFgUCJj2Ls6LVzBbL8fqA",  # KaraokeyTV
            "@karafun",
            "UCbqcG1rdt9LMwOJN4PyGTKg",  # KaraFun Karaoke
            "@BandaisuanKaraoke001",
            "UCuyBQQ2CISV0ptQRHBHzGuA",  # BandaisuanKaraoke001
            "@quantumkaraoke",
            "UCY_0l0AngUurGCwAqF4NkzA",  # Quantum Karaoke
            "@edkara",
            "UCRrNOLvQ1LztDKbXtxvDAEQ",  # EdKara
        }

        # Load existing patterns if available
        self._load_channel_patterns()

    @property
    def pass_type(self) -> PassType:
        return PassType.CHANNEL_TEMPLATE

    async def parse(
        self,
        title: str,
        description: str = "",
        tags: str = "",
        channel_name: str = "",
        channel_id: str = "",
        metadata: Optional[Dict] = None,
    ) -> Optional[ParseResult]:
        """Execute enhanced channel-template matching."""

        if not channel_id and not channel_name:
            return None

        start_time = time.time()

        try:
            logger.debug(
                f"Channel template pass starting for title: '{title}', channel_id: '{channel_id}', channel_name: '{channel_name}'"
            )
            # Step 1: Try channel-specific learned patterns first
            # Check both channel_id and channel_name (e.g., "@singkingkaraoke")
            channel_keys = [
                k for k in [channel_id, channel_name] if k and k in self.channel_patterns
            ]

            # Also check if channel_name maps to additional keys
            if channel_name in self.channel_name_mapping:
                mapped_keys = self.channel_name_mapping[channel_name]
                channel_keys.extend(
                    [k for k in mapped_keys if k in self.channel_patterns and k not in channel_keys]
                )

            logger.debug(
                f"Found channel keys: {channel_keys}, available pattern keys: {list(self.channel_patterns.keys())[:5]}"
            )
            for channel_key in channel_keys:
                result = self._try_channel_patterns(
                    channel_key, title, description, tags, channel_name
                )
                if result and result.confidence > 0.75:
                    self._update_pattern_success(channel_key, result.pattern_used, title)
                    return result

            # Step 2: Try enhanced channel name detection
            enhanced_result = self._enhanced_channel_detection(
                title, description, tags, channel_name, channel_id
            )
            if enhanced_result and enhanced_result.confidence > 0.75:
                self._learn_from_success(channel_id, channel_name, title, enhanced_result)
                return enhanced_result

            # Step 3: Fall back to existing advanced parser channel logic
            fallback_result = self.advanced_parser._parse_with_channel_patterns(title, channel_name)
            if fallback_result and fallback_result.confidence > 0.7:
                self._learn_from_success(channel_id, channel_name, title, fallback_result)
                return fallback_result

            # Step 4: Try to extract patterns from channel name context
            context_result = self._extract_with_channel_context(
                title, description, channel_name, channel_id
            )
            if context_result and context_result.confidence > 0.6:
                self._learn_from_success(channel_id, channel_name, title, context_result)
                return context_result

            return None

        except Exception as e:
            logger.error(f"Channel template pass failed: {e}")
            return None
        finally:
            processing_time = time.time() - start_time
            self._update_channel_stats(channel_id, channel_name, processing_time)

    def _try_channel_patterns(
        self, channel_id: str, title: str, description: str, tags: str, channel_name: str
    ) -> Optional[ParseResult]:
        """Try channel-specific learned patterns."""

        patterns = self.channel_patterns.get(channel_id, [])

        logger.debug(
            f"Trying channel patterns for channel_id: {channel_id}, found {len(patterns)} patterns"
        )
        if patterns:
            logger.debug(f"Available patterns: {[p.pattern for p in patterns[:3]]}")

        # Sort patterns by confidence first, then success rate and recency
        patterns.sort(
            key=lambda p: (p.confidence, p.success_count / max(p.total_attempts, 1), p.last_used),
            reverse=True,
        )

        # Try both original title and cleaned title
        titles_to_try = [title, self._clean_title_for_matching(title)]

        for test_title in titles_to_try:
            for pattern in patterns:
                try:
                    match = re.search(pattern.pattern, test_title, re.IGNORECASE | re.UNICODE)
                    if match:
                        logger.debug(
                            f"Pattern matched! Pattern: {pattern.pattern}, Title: {test_title}"
                        )
                        result = self._create_result_from_pattern_match(
                            match, pattern, f"learned_channel_{channel_id}", test_title, channel_id
                        )
                        if result and result.confidence > 0:
                            logger.debug(
                                f"Result created with confidence: {result.confidence}, artist: {result.artist}, song: {result.song_title}"
                            )
                            pattern.total_attempts += 1
                            pattern.last_used = datetime.now()
                            return result
                        else:
                            logger.debug("Result validation failed or zero confidence")
                    else:
                        logger.debug(
                            f"Pattern did NOT match. Pattern: {pattern.pattern}, Title: {test_title}"
                        )

                except re.error as e:
                    logger.warning(f"Invalid regex pattern for channel {channel_id}: {e}")
                    continue

        return None

    def _clean_title_for_matching(self, title: str) -> str:
        """Clean title by removing metadata brackets/parentheses while preserving artist names."""
        cleaned = title

        # Remove karaoke/technical metadata in brackets/parentheses (but preserve Korean artist names)
        karaoke_patterns = [
            r"\[ZZang KARAOKE\]",
            r"\[짱가라오케/노래방\]",
            r"\(Karaoke Version\)",
            r"\(MR/Instrumental\)",
            r"\(MR/\)",  # Handle incomplete MR pattern
            r"\(\s*Version\)",  # Handle incomplete Version pattern
            r"\(Melody\)",
            r"\(Instrumental\)",
            r"\(Karaoke\)",
            r"\[Karaoke\]",
            r"\[MR\]",
            r"\[Instrumental\]",
            r"\[Melody\]",
            r"\bMelody\b",  # Remove standalone "Melody" word
            r"\b노래방\b",  # Korean for karaoke
            r"\b반주\b",  # Korean for accompaniment
        ]

        # Remove specific karaoke metadata patterns
        for pattern in karaoke_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        # Remove only parentheses that contain obvious metadata (not Korean names)
        # Keep parentheses with Korean characters (Hangul) as they're likely artist names
        def should_keep_parentheses(match):
            content = match.group(1)
            # Keep if contains Korean characters (standardized range)
            if re.search(r"[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]", content):
                return match.group(0)
            # Keep if it looks like an artist name (letters, Korean characters, spaces, hyphens)
            # Added Unicode ranges for Korean characters: Hangul Syllables (AC00-D7AF), Hangul Jamo (1100-11FF), Hangul Compatibility Jamo (3130-318F)
            if (
                re.match(r"^[a-zA-Z\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF\s\-\(\)]+$", content)
                and len(content) < 50
            ):
                return match.group(0)
            # Remove if it looks like metadata
            metadata_indicators = [
                "version",
                "remix",
                "edit",
                "remaster",
                "feat",
                "ft",
                "featuring",
            ]
            if any(indicator in content.lower() for indicator in metadata_indicators):
                return ""
            return match.group(0)

        # Apply selective parentheses removal
        cleaned = re.sub(r"\(([^)]+)\)", should_keep_parentheses, cleaned)

        # Add spaces around dashes that don't have them (but preserve double dashes)
        # This helps with titles like "Artist-Song" → "Artist - Song"
        cleaned = re.sub(r"(?<=[a-zA-Z0-9])(?<!-)[-](?!-)(?=[a-zA-Z0-9])", " - ", cleaned)

        # Clean up extra whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _enhanced_channel_detection(
        self, title: str, description: str, tags: str, channel_name: str, channel_id: str
    ) -> Optional[ParseResult]:
        """Enhanced channel-specific pattern detection."""

        # Try both original title and cleaned title
        titles_to_try = [title, self._clean_title_for_matching(title)]

        # Enhanced patterns based on common karaoke channel formats
        enhanced_patterns = [
            # Channel branding patterns
            (
                rf"^{re.escape(channel_name)}[:\s]*[\"']([^\"']+)[\"']\s*[-–—]\s*[\"']([^\"']+)[\"']",
                1,
                2,
                0.9,
                "channel_branded_quoted",
            ),
            (
                rf"^\[{re.escape(channel_name)}\]\s*([^-–—]+)[-–—](.+)",
                1,
                2,
                0.85,
                "channel_bracketed",
            ),
            (
                rf"^{re.escape(channel_name)}\s*[-–—]\s*(.+?)\s*[-–—]\s*(.+?)\s*\([^)]*[Kk]araoke[^)]*\)",
                1,
                2,
                0.8,
                "channel_double_dash",
            ),
            # Complex title patterns with source references (lower priority)
            # These patterns are handled specially to extract source information
            # Pattern: Artist - Song Title From "Movie/Show"
            (
                r'^(.+?)\s*[-–—]\s*(.+?)\s+[Ff]rom\s+"([^"]+)"(?:\s*\([Kk]araoke[^)]*\))?$',
                1,  # artist
                2,  # song title (without the "from" part)
                0.7,
                "artist_song_from_quoted_source",
            ),
            # Pattern: Artist - Song Title (From "Movie/Show")
            (
                r'^(.+?)\s*[-–—]\s*(.+?)\s*\([Ff]rom\s+"([^"]+)"\)(?:\s*\([Kk]araoke[^)]*\))?$',
                1,  # artist
                2,  # song title (without the "from" part)
                0.7,
                "artist_song_from_quoted_source_parens",
            ),
            # Popular karaoke channel specific patterns
            (
                r'^"([^"]+)"\s*[-–—]\s*"([^"]+)"\s*[-–—]\s*[Kk]araoke',
                1,
                2,
                0.9,
                "double_quoted_karaoke",
            ),
            (
                r'^([^-–—]+)\s*[-–—]\s*"([^"]+)"\s*\([^)]*[Kk]araoke[^)]*\)',
                1,
                2,
                0.85,
                "artist_quoted_title",
            ),
            (
                r"^\[([^\]]+)\]\s*([^-–—]+)[-–—](.+?)\s*\[[^\]]*[Kk]araoke[^\]]*\]",
                2,
                3,
                0.8,
                "bracketed_artist_title",
            ),
        ]

        # Try enhanced patterns on both original and cleaned titles
        for test_title in titles_to_try:
            for pattern, artist_group, title_group, confidence, pattern_name in enhanced_patterns:
                try:
                    match = re.search(pattern, test_title, re.IGNORECASE | re.UNICODE)
                    if match:
                        result = self.advanced_parser._create_result_from_match(
                            match,
                            artist_group,
                            title_group,
                            confidence,
                            f"enhanced_channel_{pattern_name}",
                            pattern,
                        )
                        if result and result.confidence > 0:
                            return result

                except re.error:
                    continue

            # Simple dash pattern for cleaned titles
            if test_title != title:  # Only try this on cleaned title
                try:
                    # Simple Artist - Title pattern (after cleaning brackets)
                    match = re.search(
                        r"^([^-–—]+)\s*[-–—]\s*(.+)$", test_title, re.IGNORECASE | re.UNICODE
                    )
                    if match:
                        result = self.advanced_parser._create_result_from_match(
                            match,
                            1,
                            2,
                            0.75,
                            "enhanced_channel_cleaned_dash",
                            r"^([^-–—]+)\s*[-–—]\s*(.+)$",
                        )
                        if result and result.confidence > 0:
                            return result
                except re.error:
                    continue

        return None

    def _extract_with_channel_context(
        self, title: str, description: str, channel_name: str, channel_id: str
    ) -> Optional[ParseResult]:
        """Extract using channel name as context."""

        # Check if channel name suggests karaoke content
        channel_lower = channel_name.lower()
        karaoke_indicators = [
            "karaoke",
            "karafun",
            "karaoké",
            "karaokê",
            "караоке",
            "sing along",
            "backing track",
            "instrumental",
            "covers",
            "tribute",
            "piano version",
        ]

        is_karaoke_channel = any(indicator in channel_lower for indicator in karaoke_indicators)

        if not is_karaoke_channel:
            return None

        # Try to extract without requiring "karaoke" in title since channel is karaoke-focused
        simple_patterns = [
            (r"^([^-–—]+)\s*[-–—]\s*([^(\[]+)", 1, 2, 0.65, "simple_dash"),
            (r'^"([^"]+)"\s*[-–—]\s*"([^"]+)"', 1, 2, 0.7, "simple_quoted"),
            (r"^([^(]+?)\s*\([^)]*by\s+([^)]+)\)", 2, 1, 0.6, "by_in_parentheses"),
        ]

        for pattern, artist_group, title_group, base_confidence, pattern_name in simple_patterns:
            try:
                match = re.search(pattern, title, re.IGNORECASE | re.UNICODE)
                if match:
                    result = self.advanced_parser._create_result_from_match(
                        match,
                        artist_group,
                        title_group,
                        base_confidence,
                        f"channel_context_{pattern_name}",
                        pattern,
                    )
                    if result and result.confidence > 0:
                        # Boost confidence since it's from a karaoke channel
                        result.confidence = min(result.confidence * 1.1, 0.95)
                        result.metadata = result.metadata or {}
                        result.metadata["karaoke_channel_boost"] = True
                        return result

            except re.error:
                continue

        return None

    def _create_result_from_pattern_match(
        self,
        match,
        pattern: ChannelPattern,
        method: str,
        original_title: str,
        channel_id: Optional[str] = None,
    ) -> Optional[ParseResult]:
        """Create a ParseResult from a pattern match."""

        logger.debug(f"Creating result from pattern match: {pattern.pattern}")
        try:
            result = ParseResult(method=method, pattern_used=pattern.pattern)

            if pattern.artist_group and pattern.artist_group <= len(match.groups()):
                artist = self.advanced_parser._clean_extracted_text(
                    match.group(pattern.artist_group)
                )
                # Handle middot character as artist separator
                if "·" in artist:
                    # Replace middot with comma for multiple artists
                    artist = artist.replace(" · ", ", ").replace("·", ", ")

                if self.advanced_parser._is_valid_artist_name(artist):
                    result.artist = artist
                else:
                    logger.debug(f"Artist validation failed for: '{artist}'")

            if pattern.title_group and pattern.title_group <= len(match.groups()):
                song_title = self.advanced_parser._clean_extracted_text(
                    match.group(pattern.title_group)
                )
                if self.advanced_parser._is_valid_song_title(song_title):
                    result.song_title = song_title
                else:
                    logger.debug(f"Song title validation failed for: '{song_title}'")

            # Calculate confidence based on pattern history and extraction success
            base_confidence = pattern.confidence
            if result.artist and result.song_title:
                result.confidence = base_confidence
            elif result.artist or result.song_title:
                result.confidence = base_confidence * 0.7
            else:
                result.confidence = 0

            # Boost confidence for Korean content to match English content quality
            if result.artist and self._contains_korean_characters(result.artist):
                # Korean artist names in parentheses format are very reliable on ZZang channel
                if "(" in result.artist and ")" in result.artist:
                    result.confidence = min(result.confidence * 1.1, 0.95)
                # General Korean content boost
                result.confidence = min(result.confidence * 1.05, 0.95)
                logger.debug(f"Korean content confidence boost applied: {result.artist}")

            # Boost confidence based on pattern success history
            # For hardcoded patterns (high initial confidence), don't penalize for lack of history
            if pattern.total_attempts > 0 and pattern.confidence < 0.9:
                success_rate = pattern.success_count / pattern.total_attempts
                result.confidence *= 0.8 + 0.2 * success_rate  # 80-100% multiplier
            elif pattern.total_attempts == 0 and pattern.confidence >= 0.9:
                # Hardcoded patterns start with high confidence, keep it high
                result.confidence *= 0.95  # Small penalty for being untested

            result.metadata = {
                "pattern_success_rate": pattern.success_count / max(pattern.total_attempts, 1),
                "pattern_age_days": (datetime.now() - pattern.created).days,
                "pattern_last_used": pattern.last_used.isoformat(),
            }

            # Handle special patterns that extract source information
            # Check if the pattern has groups for source extraction
            if "from" in pattern.pattern.lower() and match.lastindex and match.lastindex >= 3:
                source = match.group(3)
                if source:
                    result.metadata["source"] = source.strip()
                    result.metadata["source_type"] = "movie/show"
                    logger.debug(f"Extracted source information: {source}")

                    # Clean the song title to remove "from" part if needed
                    if result.song_title and " from " in result.song_title.lower():
                        # Remove the "from X" part from the title
                        result.song_title = re.sub(
                            r"\s+[Ff]rom\s+.*$", "", result.song_title
                        ).strip()

            # Check for potential artist/song swap
            # Skip swap detection for:
            # 1. High confidence patterns (>= 0.95) - trust the pattern
            # 2. Channels in the exemption list - they use reversed formats intentionally
            should_check_swap = (
                result.artist
                and result.song_title
                and pattern.confidence < 0.95
                and (not channel_id or channel_id not in self.swap_exempt_channels)
            )

            if should_check_swap and result.artist is not None and result.song_title is not None:
                swap_detected, swap_confidence = self._detect_artist_song_swap(
                    result.artist, result.song_title
                )
                if swap_detected and swap_confidence > 0.8:  # Increased threshold from 0.7 to 0.8
                    # Swap the fields
                    logger.debug(
                        f"Detected artist/song swap with confidence {swap_confidence}: '{result.artist}' <-> '{result.song_title}'"
                    )
                    result.artist, result.song_title = result.song_title, result.artist
                    result.metadata = result.metadata or {}
                    result.metadata["swap_corrected"] = True
                    result.metadata["swap_confidence"] = swap_confidence

            return result

        except Exception as e:
            logger.warning(f"Failed to create result from pattern match: {e}")
            return None

    def _learn_from_success(
        self, channel_id: str, channel_name: str, title: str, result: ParseResult
    ):
        """Learn new patterns from successful parses."""

        if not channel_id or not result.pattern_used:
            return

        # Update or create channel stats
        if channel_id not in self.channel_stats:
            self.channel_stats[channel_id] = ChannelStats(
                channel_id=channel_id, channel_name=channel_name
            )

        stats = self.channel_stats[channel_id]
        stats.successful_parses += 1
        stats.last_updated = datetime.now()

        # Track format patterns
        if result.artist and result.song_title:
            # Try to generalize the successful pattern
            generalized_pattern = self._generalize_pattern(title, result)
            if generalized_pattern:
                self._add_or_update_pattern(channel_id, generalized_pattern, title)

    def _generalize_pattern(self, title: str, result: ParseResult) -> Optional[str]:
        """Generalize a successful parse into a reusable pattern."""

        if not result.artist or not result.song_title:
            return None

        # Escape the extracted parts for regex
        # Create a generalized pattern by replacing the specific parts with groups
        pattern = title

        # Replace artist with group 1
        pattern = pattern.replace(result.artist, "([^-–—\"']+?)")

        # Replace title with group 2
        pattern = pattern.replace(result.song_title, r"([^(\[]+?)")

        # Clean up the pattern
        pattern = pattern.strip()

        # Add anchors and clean up whitespace
        pattern = f"^{pattern}$"
        pattern = re.sub(r"\s+", r"\\s+", pattern)

        # Validate the pattern
        try:
            re.compile(pattern)
            return pattern
        except re.error:
            return None

    def _contains_korean_characters(self, text: str) -> bool:
        """Check if text contains Korean characters."""
        if not text:
            return False
        # Check for Hangul Syllables, Jamo, and Compatibility Jamo
        return bool(re.search(r"[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]", text))

    def _detect_artist_song_swap(self, artist: str, song_title: str) -> Tuple[bool, float]:
        """Detect if artist and song title are potentially swapped with improved heuristics."""
        swap_indicators = {
            # More specific song-related keywords
            "song_keywords_in_artist": [
                "live",
                "acoustic",
                "remix",
                "unplugged",
                "medley",
                "reprise",
                "interlude",
                "outro",
                "intro",
                "bonus track",
            ],
            # Strong artist indicators (less likely to be in a song title)
            "strong_artist_indicators": [
                "feat.",
                "ft.",
                "featuring",
                " x ",
                "prod.",
                "produced by",
                " & ",
                " and ",
                " with ",
            ],
            # Common band/artist prefixes/suffixes
            "band_indicators": [
                "The ",
                "DJ ",
                "MC ",
                "Sir ",
                "Lady ",
                "Dr. ",
                "Mr. ",
                "Ms. ",
                " Band",
                " Orchestra",
                " Quartet",
                " Trio",
                " Duo",
                " Project",
                " Experience",
            ],
            # Common English words that are often song titles (not artists)
            "common_song_words": [
                "love",
                "heart",
                "life",
                "time",
                "world",
                "dream",
                "night",
                "day",
                "home",
                "away",
                "gone",
                "back",
                "down",
                "up",
                "over",
                "under",
                "fire",
                "water",
                "rain",
                "sun",
                "moon",
                "star",
                "sky",
                "heaven",
                "hell",
                "angel",
                "devil",
                "god",
                "soul",
                "mind",
                "body",
                "eyes",
                "kiss",
                "touch",
                "feel",
                "hurt",
                "pain",
                "joy",
                "happy",
                "sad",
                "lonely",
                "alone",
                "together",
                "forever",
                "never",
                "always",
                "maybe",
                "halo",
                "hero",
                "zero",
                "one",
                "two",
                "three",
                "four",
                "five",
                "baby",
                "girl",
                "boy",
                "man",
                "woman",
                "lady",
                "crazy",
                "wild",
                "free",
                "young",
                "old",
                "new",
                "last",
                "first",
                "only",
                "real",
                "true",
                "false",
                "right",
                "wrong",
                "good",
                "bad",
                "better",
                "best",
                "beautiful",
                "perfect",
                "broken",
                "lost",
                "found",
                "gone",
                "stay",
                "leave",
                "run",
                "walk",
                "dance",
                "sing",
                "cry",
                "smile",
                "laugh",
            ],
            # Common song title patterns that should NOT trigger swaps
            "song_title_patterns": [
                # Religious/spiritual songs
                "just as i am",
                "as i am",
                "take me as i am",
                "come as you are",
                "here i am",
                "where i am",
                "who i am",
                "what i am",
                # Common song phrases with pronouns
                "love me",
                "hold me",
                "kiss me",
                "take me",
                "make me",
                "help me",
                "save me",
                "tell me",
                "show me",
                "teach me",
                "leave me",
                "let me",
                "call me",
                "find me",
                "miss me",
                "want me",
                "need me",
                "free me",
                # Titles ending with pronouns
                "with me",
                "for me",
                "to me",
                "by me",
                "in me",
                "on me",
                "without me",
                "about me",
                "around me",
                "before me",
                "after me",
                # "I" phrases
                "i am",
                "i was",
                "i will",
                "i can",
                "i do",
                "i don't",
                "i want",
                "i need",
                "i love",
                "i hate",
                "i know",
                "i think",
                "i feel",
                "i believe",
                "i remember",
                "i forget",
                "i miss",
                "i wish",
                # Other common patterns
                "my way",
                "my love",
                "my life",
                "my heart",
                "my soul",
                "my mind",
                "my baby",
                "my girl",
                "my boy",
                "my man",
                "my woman",
                "my world",
            ],
        }

        confidence = 0.0
        reasons = []
        artist_lower = artist.lower()
        song_lower = song_title.lower()

        # Check if we have a known artist in our cache
        if hasattr(self, "advanced_parser") and hasattr(self.advanced_parser, "known_artists"):
            if artist_lower in [ka.lower() for ka in self.advanced_parser.known_artists]:
                confidence -= 0.4
                reasons.append("known_artist_in_cache")

        # Heuristic 1: Song keywords in artist field (high confidence indicator)
        for keyword in swap_indicators["song_keywords_in_artist"]:
            if f"({keyword})" in artist_lower or f"[{keyword}]" in artist_lower:
                confidence += 0.5
                reasons.append(f"strong_song_keyword_in_artist:{keyword}")
                break

        # Heuristic 2: Strong artist indicators in song field
        for indicator in swap_indicators["strong_artist_indicators"]:
            if indicator in song_lower:
                confidence += 0.4
                reasons.append(f"strong_artist_indicator_in_song:{indicator}")
                break

        # Heuristic 3: Band/artist indicators in song field
        for indicator in swap_indicators["band_indicators"]:
            if song_lower.startswith(indicator.lower()) or song_lower.endswith(indicator.lower()):
                confidence += 0.3
                reasons.append(f"band_indicator_in_song:{indicator}")
                break

        # Heuristic 4: Modified all caps check - exclude short words and common song titles
        if artist.isupper() and len(artist) > 4:
            # Check if it's a common song word
            if artist_lower in swap_indicators["common_song_words"]:
                confidence += 0.15  # Much lower weight for common words
                reasons.append("common_song_word_in_caps")
            elif song_title.istitle():
                confidence += 0.1  # Reduced from 0.25
                reasons.append("artist_all_caps_vs_title_case_song")

        # Heuristic 5: Song title looks like a typical artist name
        if re.match(r"^[A-Z][a-z']+(\s[A-Z][a-z']+)+$", song_title):  # e.g., "Firstname Lastname"
            confidence += 0.2
            reasons.append("song_looks_like_artist_name")

        # Heuristic 6: Length mismatch (artist much longer than song)
        if len(artist) > len(song_title) * 2.5 and len(song_title) < 15:
            confidence += 0.2
            reasons.append("artist_much_longer_than_song")

        # Counter-Heuristic 1: Penalize if artist looks like a valid artist name
        if re.match(r"^[A-Z][a-z']+(\s[A-Z][a-z']+)+$", artist):
            confidence -= 0.25
            reasons.append("valid_artist_name_pattern_penalty")

        # Counter-Heuristic 2: Penalize if song contains year in parentheses
        if re.search(r"\(\d{4}\)", song_lower):
            confidence -= 0.3
            reasons.append("year_in_song_title_penalty")

        # Counter-Heuristic 3: Penalize if artist is a common song word
        if artist_lower in swap_indicators["common_song_words"]:
            confidence -= 0.2
            reasons.append("artist_is_common_song_word")

        # Counter-Heuristic 4: Strong penalty if artist matches common song title patterns
        for pattern in swap_indicators["song_title_patterns"]:
            if pattern in artist_lower or artist_lower == pattern:
                confidence -= 0.5  # Strong penalty
                reasons.append(f"artist_matches_song_pattern:{pattern}")
                break

        # Counter-Heuristic 5: Check for specific problematic patterns
        # Songs like "Just As I Am", "Take Me As I Am" etc.
        if (
            artist_lower.endswith(" as i am")
            or artist_lower.endswith(" me")
            or artist_lower.startswith("take me")
            or artist_lower.startswith("love me")
            or artist_lower.startswith("hold me")
        ):
            confidence -= 0.4
            reasons.append("artist_has_pronoun_ending")

        # Final decision with higher threshold
        swap_detected = confidence >= 0.8  # Increased from 0.7

        if swap_detected or confidence >= 0.4:
            logger.debug(
                f"Swap detection for '{artist}' <-> '{song_title}': confidence={confidence:.2f}, reasons={reasons}"
            )

        return swap_detected, confidence

    def _add_or_update_pattern(self, channel_id: str, pattern: str, example: str):
        """Add or update a channel pattern."""

        if channel_id not in self.channel_patterns:
            self.channel_patterns[channel_id] = []

        patterns = self.channel_patterns[channel_id]

        # Check if pattern already exists
        for existing_pattern in patterns:
            if existing_pattern.pattern == pattern:
                existing_pattern.success_count += 1
                existing_pattern.total_attempts += 1
                existing_pattern.last_used = datetime.now()
                if example not in existing_pattern.examples:
                    existing_pattern.examples.append(example)
                return

        # Add new pattern if we haven't reached the limit
        if len(patterns) < self.max_patterns_per_channel:
            new_pattern = ChannelPattern(
                pattern=pattern,
                artist_group=1,  # Assuming first group is artist
                title_group=2,  # Assuming second group is title
                confidence=0.7,  # Start with moderate confidence
                success_count=1,
                total_attempts=1,
                examples=[example],
            )
            patterns.append(new_pattern)

        # Remove old or ineffective patterns
        self._cleanup_patterns(channel_id)

    def _cleanup_patterns(self, channel_id: str):
        """Remove old or ineffective patterns."""

        if channel_id not in self.channel_patterns:
            return

        patterns = self.channel_patterns[channel_id]
        now = datetime.now()

        # Filter out patterns that are too old or have poor success rates
        cleaned_patterns = []
        for pattern in patterns:
            age_days = (now - pattern.created).days
            success_rate = pattern.success_count / max(pattern.total_attempts, 1)

            # Keep pattern if it's recent, successful, or used recently
            if (
                age_days < self.pattern_decay_days
                or success_rate > 0.6
                or (now - pattern.last_used).days < 7
            ):
                cleaned_patterns.append(pattern)

        self.channel_patterns[channel_id] = cleaned_patterns

    def _update_pattern_success(self, channel_id: str, pattern_used: str, title: str):
        """Update pattern success statistics."""

        if channel_id not in self.channel_patterns:
            return

        for pattern in self.channel_patterns[channel_id]:
            if pattern.pattern == pattern_used:
                pattern.success_count += 1
                pattern.total_attempts += 1
                pattern.last_used = datetime.now()
                if title not in pattern.examples:
                    pattern.examples.append(title)
                break

    def _update_channel_stats(self, channel_id: str, channel_name: str, processing_time: float):
        """Update channel processing statistics."""

        if not channel_id:
            return

        if channel_id not in self.channel_stats:
            self.channel_stats[channel_id] = ChannelStats(
                channel_id=channel_id, channel_name=channel_name
            )

        stats = self.channel_stats[channel_id]
        stats.total_videos += 1
        stats.last_updated = datetime.now()

        # Check for drift if enough time has passed
        if (datetime.now() - stats.last_updated) > self.drift_check_interval:
            self._check_for_drift(channel_id)

    def _check_for_drift(self, channel_id: str):
        """Check if channel patterns are becoming less effective (drift detection)."""

        if channel_id not in self.channel_stats:
            return

        stats = self.channel_stats[channel_id]

        if stats.total_videos < 10:  # Need enough data
            return

        current_success_rate = stats.successful_parses / stats.total_videos

        # Compare with expected success rate based on pattern history
        if current_success_rate < stats.drift_threshold:
            stats.drift_detected = True
            logger.warning(
                f"Drift detected for channel {channel_id} ({stats.channel_name}): "
                f"success rate {current_success_rate:.2f} below threshold {stats.drift_threshold:.2f}"
            )

            # Trigger pattern refresh
            self._refresh_channel_patterns(channel_id)

    def _refresh_channel_patterns(self, channel_id: str):
        """Refresh patterns for a channel showing drift."""

        if channel_id in self.channel_patterns:
            # Reduce confidence of all patterns
            for pattern in self.channel_patterns[channel_id]:
                pattern.confidence *= 0.8

            # Remove patterns with very low success rates
            patterns = self.channel_patterns[channel_id]
            self.channel_patterns[channel_id] = [
                p for p in patterns if (p.success_count / max(p.total_attempts, 1)) > 0.3
            ]

    def _load_channel_patterns(self):
        """Load existing channel patterns from database."""
        # Always load hardcoded patterns regardless of db_manager
        try:
            # Both possible channel IDs for zzangkaraoke
            zzang_channel_ids = ["UCzv4mCu3YS_9WjAWSt9Xg9Q", "@zzangkaraoke"]

            # More robust patterns for zzangkaraoke based on actual titles
            zzang_patterns = [
                # Korean format: Artist(한국이름) - Title
                ChannelPattern(
                    pattern=r"\[짱가라오케/노래방\]\s*([^-]+(?:\([^)]*\))?)\s*-\s*(.+?)\s*\((?:Melody|MR/Instrumental)\)",
                    artist_group=1,
                    title_group=2,
                    confidence=0.95,
                ),
                # English format with Korean artist: Artist - Title (Karaoke Version)
                ChannelPattern(
                    pattern=r"^([^-]+(?:\([^)]*\))?)\s*-\s*(.+?)\s*\((?:Melody|MR/Instrumental)\)\s*\(Karaoke Version\)",
                    artist_group=1,
                    title_group=2,
                    confidence=0.90,
                ),
                # Broader Korean format to catch more variations
                ChannelPattern(
                    pattern=r"\[짱가라오케/노래방\]\s*(.+?)\s*-\s*(.+?)\s*\((?:Melody|MR/Instrumental)\)",
                    artist_group=1,
                    title_group=2,
                    confidence=0.85,
                ),
                # Generic pattern: Artist - Title (any karaoke indicator)
                ChannelPattern(
                    pattern=r"^([^-–—]+(?:\([^)]*\))?)\s*[-–—]\s*(.+?)\s*\((?:Melody|MR|Instrumental|Karaoke)",
                    artist_group=1,
                    title_group=2,
                    confidence=0.80,
                ),
                # After cleaning metadata: Simple Artist - Title
                ChannelPattern(
                    pattern=r"^([^-–—]+(?:\([^)]*\))?)\s*[-–—]\s*(.+?)$",
                    artist_group=1,
                    title_group=2,
                    confidence=0.75,
                ),
                # Fallback: Any dash-separated format (preserve parentheses in artist names)
                ChannelPattern(
                    pattern=r"^([^-–—]+(?:\([^)]*\))?)\s*[-–—]\s*(.+?)(?:\s*\[.*\])?$",
                    artist_group=1,
                    title_group=2,
                    confidence=0.70,
                ),
            ]

            # Apply patterns to both channel IDs
            for channel_id in zzang_channel_ids:
                self.channel_patterns[channel_id] = zzang_patterns
            logger.info(f"Loaded {len(zzang_patterns)} hardcoded patterns for zzangkaraoke")

            # @singkingkaraoke patterns
            singking_patterns = [
                # Primary: Artist - Song Title (Karaoke Version)
                ChannelPattern(
                    pattern=r"^(.+?)\s*-\s*(.+?)\s*\(Karaoke Version\)",
                    artist_group=1,
                    title_group=2,
                    confidence=0.95,
                ),
                # With Lyrics variation
                ChannelPattern(
                    pattern=r"^(.+?)\s*-\s*(.+?)\s*\((?:Karaoke Version )?With Lyrics\)",
                    artist_group=1,
                    title_group=2,
                    confidence=0.90,
                ),
                # No Vocals/Instrumental variations
                ChannelPattern(
                    pattern=r"^(.+?)\s*-\s*(.+?)\s*\((?:No Vocals|Instrumental|Acoustic)\)",
                    artist_group=1,
                    title_group=2,
                    confidence=0.90,
                ),
                # Generic pattern after metadata removal
                ChannelPattern(
                    pattern=r"^(.+?)\s*-\s*(.+?)$",
                    artist_group=1,
                    title_group=2,
                    confidence=0.75,
                ),
            ]
            self.channel_patterns["@singkingkaraoke"] = singking_patterns
            self.channel_patterns["UCwTRjvjVge51X-ILJ4i22ew"] = (
                singking_patterns  # Correct Sing King channel ID
            )

            # @MusisiKaraoke patterns
            musisi_patterns = [
                # Primary: Song Title - Artist (Karaoke Songs With Lyrics - Original Key)
                ChannelPattern(
                    pattern=r"^(.+?)\s*-\s*([^()]+?)\s*\(Karaoke Songs With Lyrics - Original Key\)",
                    artist_group=2,
                    title_group=1,
                    confidence=0.98,  # Increased confidence for this very specific pattern
                ),
                # Variation: Karaoke Instrumental
                ChannelPattern(
                    pattern=r"^(.+?)\s*-\s*([^()]+?)\s*\(Karaoke Instrumental\)",
                    artist_group=2,
                    title_group=1,
                    confidence=0.95,  # Increased confidence
                ),
                # Generic pattern (reversed)
                ChannelPattern(
                    pattern=r"^(.+?)\s*-\s*([^()]+?)(?:\s*\(.*\))?$",
                    artist_group=2,
                    title_group=1,
                    confidence=0.85,  # Increased confidence for this channel's reversed format
                ),
            ]
            self.channel_patterns["@MusisiKaraoke"] = musisi_patterns
            self.channel_patterns["UCJw1qyMF4m3ZIBWdhogkcsw"] = (
                musisi_patterns  # Musisi Karaoke channel ID
            )

            # @AtomicKaraoke patterns
            atomic_patterns = [
                # Primary: Song Title - Artist (HD Karaoke)
                ChannelPattern(
                    pattern=r"^(.+?)\s*-\s*([^()]+?)\s*\(HD Karaoke\)",
                    artist_group=2,
                    title_group=1,
                    confidence=0.98,  # Increased confidence for specific pattern
                ),
                # Alternative: Song Title - Artist | Karaoke
                ChannelPattern(
                    pattern=r"^(.+?)\s*-\s*([^|]+?)\s*\|\s*Karaoke",
                    artist_group=2,
                    title_group=1,
                    confidence=0.95,  # Increased confidence
                ),
                # Generic pattern (reversed for this channel)
                ChannelPattern(
                    pattern=r"^(.+?)\s*-\s*([^()|]+?)(?:\s*[|(].*)?$",
                    artist_group=2,
                    title_group=1,
                    confidence=0.85,  # Increased confidence
                ),
            ]
            self.channel_patterns["@AtomicKaraoke"] = atomic_patterns
            self.channel_patterns["UCutZyApGOjqhOS-pp7yAj4Q"] = (
                atomic_patterns  # Atomic Karaoke channel ID
            )

            # @BandaisuanKaraoke001 patterns
            bandaisuan_patterns = [
                # Primary: Karaoke - Song Title - Artist
                ChannelPattern(
                    pattern=r"^Karaoke\s*-\s*(.+?)\s*-\s*([^-]+?)(?:\s*\[.*\])?$",
                    artist_group=2,
                    title_group=1,
                    confidence=0.95,
                ),
                # With year or context
                ChannelPattern(
                    pattern=r"^Karaoke\s*-\s*(.+?)\s*-\s*([^-]+?)\s*(?:\[|\().*$",
                    artist_group=2,
                    title_group=1,
                    confidence=0.90,
                ),
                # Fallback without Karaoke prefix
                ChannelPattern(
                    pattern=r"^(.+?)\s*-\s*([^-]+?)$",
                    artist_group=2,
                    title_group=1,
                    confidence=0.70,
                ),
            ]
            self.channel_patterns["@BandaisuanKaraoke001"] = bandaisuan_patterns
            self.channel_patterns["UCuyBQQ2CISV0ptQRHBHzGuA"] = (
                bandaisuan_patterns  # BandaisuanKaraoke001 channel ID
            )

            # @karafun patterns
            karafun_patterns = [
                # Consistent format: Song Title - Artist | Karaoke Version | KaraFun
                ChannelPattern(
                    pattern=r"^(.+?)\s*-\s*([^|]+?)\s*\|\s*Karaoke Version\s*\|\s*KaraFun",
                    artist_group=2,
                    title_group=1,
                    confidence=0.95,
                ),
                # Without full branding
                ChannelPattern(
                    pattern=r"^(.+?)\s*-\s*([^|]+?)\s*\|\s*Karaoke",
                    artist_group=2,
                    title_group=1,
                    confidence=0.85,
                ),
                # Generic pattern
                ChannelPattern(
                    pattern=r"^(.+?)\s*-\s*([^|]+?)(?:\s*\|.*)?$",
                    artist_group=2,
                    title_group=1,
                    confidence=0.75,
                ),
            ]
            self.channel_patterns["@karafun"] = karafun_patterns
            self.channel_patterns["UCbqcG1rdt9LMwOJN4PyGTKg"] = (
                karafun_patterns  # KaraFun Karaoke channel ID
            )

            # @singkaraoke9783 patterns
            singkaraoke_patterns = [
                # Primary: Song Title - Artist (Karaoke)
                ChannelPattern(
                    pattern=r"^(.+?)\s*-\s*([^()]+?)\s*\(Karaoke\)",
                    artist_group=2,
                    title_group=1,
                    confidence=0.95,
                ),
                # Without parentheses
                ChannelPattern(
                    pattern=r"^(.+?)\s*-\s*([^()]+?)\s+Karaoke$",
                    artist_group=2,
                    title_group=1,
                    confidence=0.90,
                ),
                # Generic pattern
                ChannelPattern(
                    pattern=r"^(.+?)\s*-\s*([^()]+?)$",
                    artist_group=2,
                    title_group=1,
                    confidence=0.75,
                ),
            ]
            self.channel_patterns["@singkaraoke9783"] = singkaraoke_patterns
            self.channel_patterns["UC1AgLpY5t66HaI3ejzLoyOg"] = (
                singkaraoke_patterns  # Sing Karaoke channel ID
            )

            # @karaokeytv0618 patterns
            karaokeytv_patterns = [
                # Primary: Song Title - Artist (Karaoke Version)
                ChannelPattern(
                    pattern=r"^(.+?)\s*-\s*([^()]+?)\s*\(Karaoke Version\)",
                    artist_group=2,
                    title_group=1,
                    confidence=0.95,
                ),
                # Cover version pattern
                ChannelPattern(
                    pattern=r"^.+?\s*-\s*(.+?)\s*-\s*([^()]+?)\s*\(Karaoke Version\)",
                    artist_group=2,
                    title_group=1,
                    confidence=0.85,
                ),
                # With numerical identifier
                ChannelPattern(
                    pattern=r"^(.+?)\s*-\s*([^()#]+?)(?:\s*#\d+)?\s*\(Karaoke",
                    artist_group=2,
                    title_group=1,
                    confidence=0.85,
                ),
                # Generic pattern
                ChannelPattern(
                    pattern=r"^(.+?)\s*-\s*([^()#]+?)(?:\s*[#(].*)?$",
                    artist_group=2,
                    title_group=1,
                    confidence=0.75,
                ),
            ]
            self.channel_patterns["@karaokeytv0618"] = karaokeytv_patterns
            self.channel_patterns["UCNbFgUCJj2Ls6LVzBbL8fqA"] = (
                karaokeytv_patterns  # KaraokeyTV channel ID
            )

            # @mibalmzkaraoke patterns
            mibalmz_patterns = [
                # Song Title - Artist (Karaoke)
                ChannelPattern(
                    pattern=r"^(.+?)\s*-\s*([^()]+?)\s*\(Karaoke\)",
                    artist_group=2,
                    title_group=1,
                    confidence=0.90,
                ),
                # Artist - Song Title (Karaoke)
                ChannelPattern(
                    pattern=r"^([^-]+?)\s*-\s*(.+?)\s*\(Karaoke\)",
                    artist_group=1,
                    title_group=2,
                    confidence=0.90,
                ),
                # With Instrumental variations
                ChannelPattern(
                    pattern=r"^(.+?)\s*-\s*([^()]+?)\s*\(Karaoke Instrumental(?:\s+(?:No Vocal|With Lyrics))?\)",
                    artist_group=2,
                    title_group=1,
                    confidence=0.95,
                ),
                # Generic pattern
                ChannelPattern(
                    pattern=r"^(.+?)\s*-\s*([^()]+?)$",
                    artist_group=2,
                    title_group=1,
                    confidence=0.70,
                ),
            ]
            self.channel_patterns["@mibalmzkaraoke"] = mibalmz_patterns
            self.channel_patterns["UCRoAoGqqLuOIWztkcxUiYoA"] = (
                mibalmz_patterns  # Mi Balmz Karaoke Tracks channel ID
            )

            # @thepropervolume patterns
            propervolume_patterns = [
                # Primary: Artist - Song Title | Instrumental Karaoke Version
                ChannelPattern(
                    pattern=r"^([^-]+?)\s*-\s*(.+?)\s*\|\s*Instrumental Karaoke Version",
                    artist_group=1,
                    title_group=2,
                    confidence=0.95,
                ),
                # Without Instrumental
                ChannelPattern(
                    pattern=r"^([^-]+?)\s*-\s*(.+?)\s*\|\s*Karaoke Version",
                    artist_group=1,
                    title_group=2,
                    confidence=0.90,
                ),
                # Backing Track/Play Along variations
                ChannelPattern(
                    pattern=r"^([^-]+?)\s*-\s*(.+?)\s*\((?:Instrumental )?(?:Backing Track|Play Along)\)",
                    artist_group=1,
                    title_group=2,
                    confidence=0.90,
                ),
                # Generic pattern
                ChannelPattern(
                    pattern=r"^([^-|]+?)\s*-\s*([^|(]+?)(?:\s*[|(].*)?$",
                    artist_group=1,
                    title_group=2,
                    confidence=0.75,
                ),
            ]
            self.channel_patterns["@thepropervolume"] = propervolume_patterns
            self.channel_patterns["UCw0WVzCSi9-X0RMxaj3gTpg"] = (
                propervolume_patterns  # The Proper Volume Karaoke Studio channel ID
            )

            # @sing2piano patterns
            sing2piano_patterns = [
                # Primary: Artist - Song Title (Piano Karaoke)
                ChannelPattern(
                    pattern=r"^([^-]+?)\s*-\s*(.+?)\s*\(Piano Karaoke\)",
                    artist_group=1,
                    title_group=2,
                    confidence=0.95,
                ),
                # Complex format with key/version info
                ChannelPattern(
                    pattern=r"^(.+?)\s*(?:\(.*?\))?\s*\[Originally Performed by ([^]]+?)\]\s*\[Piano Karaoke Version\]",
                    artist_group=2,
                    title_group=1,
                    confidence=0.90,
                ),
                # Generic pattern
                ChannelPattern(
                    pattern=r"^([^-]+?)\s*-\s*(.+?)(?:\s*\(.*)?$",
                    artist_group=1,
                    title_group=2,
                    confidence=0.75,
                ),
            ]
            self.channel_patterns["@sing2piano"] = sing2piano_patterns
            self.channel_patterns["UCIk6z4gxI5ADYK7HmNiJvNg"] = (
                sing2piano_patterns  # Sing2Piano channel ID
            )

            # @LugnKaraoke patterns
            lugn_patterns = [
                # Primary: Artist • Song Title • Karaoke
                ChannelPattern(
                    pattern=r"^([^•]+?)\s*•\s*(.+?)\s*•\s*Karaoke",
                    artist_group=1,
                    title_group=2,
                    confidence=0.95,
                ),
                # With tempo/key info
                ChannelPattern(
                    pattern=r"^([^•]+?)\s*•\s*(.+?)\s*•\s*Karaoke\s*\(.*\)",
                    artist_group=1,
                    title_group=2,
                    confidence=0.90,
                ),
                # Fallback with bullet points
                ChannelPattern(
                    pattern=r"^([^•]+?)\s*•\s*(.+?)(?:\s*•.*)?$",
                    artist_group=1,
                    title_group=2,
                    confidence=0.80,
                ),
            ]
            self.channel_patterns["@LugnKaraoke"] = lugn_patterns
            self.channel_patterns["UCS4Q7GGXKdZW9uZ6YySe34Q"] = lugn_patterns  # Lugn channel ID

            # @karafunde patterns
            karafunde_patterns = [
                # Song Title - Artist (Karaoke)
                ChannelPattern(
                    pattern=r"^(.+?)\s*-\s*([^()]+?)\s*\(Karaoke\)",
                    artist_group=2,
                    title_group=1,
                    confidence=0.95,
                ),
                # Song Title (Karaoke)
                ChannelPattern(
                    pattern=r"^(.+?)\s*\(Karaoke\)$",
                    artist_group=None,
                    title_group=1,
                    confidence=0.80,
                ),
                # Song Title - Artist Karaoke
                ChannelPattern(
                    pattern=r"^(.+?)\s*-\s*([^()]+?)\s+Karaoke$",
                    artist_group=2,
                    title_group=1,
                    confidence=0.85,
                ),
                # Generic pattern
                ChannelPattern(
                    pattern=r"^(.+?)\s*-\s*([^()]+?)$",
                    artist_group=2,
                    title_group=1,
                    confidence=0.75,
                ),
            ]
            self.channel_patterns["@karafunde"] = karafunde_patterns
            self.channel_patterns["UCzEav_eOAmp23-s_cqBDpbA"] = (
                karafunde_patterns  # KaraFun Deutschland channel ID
            )

            # @FrauKnoblauch patterns
            frauknoblauch_patterns = [
                # Artist - Song Title (Karaoke)
                ChannelPattern(
                    pattern=r"^([^-]+?)\s*-\s*(.+?)\s*\(Karaoke\)",
                    artist_group=1,
                    title_group=2,
                    confidence=0.95,
                ),
                # Artist - Song Title (Instrumental)
                ChannelPattern(
                    pattern=r"^([^-]+?)\s*-\s*(.+?)\s*\(Instrumental\)",
                    artist_group=1,
                    title_group=2,
                    confidence=0.90,
                ),
                # Generic pattern
                ChannelPattern(
                    pattern=r"^([^-]+?)\s*-\s*(.+?)$",
                    artist_group=1,
                    title_group=2,
                    confidence=0.75,
                ),
            ]
            self.channel_patterns["@FrauKnoblauch"] = frauknoblauch_patterns
            self.channel_patterns["UC-AdlzvbJi7LvBkja531tVQ"] = (
                frauknoblauch_patterns  # FrauKnoblauch channel ID
            )

            # @avd-karaoke patterns
            avd_patterns = [
                # Consistent: Artist - Song Title (Karaoke) [AVD-xxx]
                ChannelPattern(
                    pattern=r"^([^-]+?)\s*-\s*(.+?)\s*\(Karaoke\)\s*\[AVD-\d+\]",
                    artist_group=1,
                    title_group=2,
                    confidence=0.95,
                ),
                # With Lyrics variation
                ChannelPattern(
                    pattern=r"^([^-]+?)\s*-\s*(.+?)\s*\(Karaoke Version With Lyrics\)\s*\[AVD-\d+\]",
                    artist_group=1,
                    title_group=2,
                    confidence=0.95,
                ),
                # Without AVD code
                ChannelPattern(
                    pattern=r"^([^-]+?)\s*-\s*(.+?)\s*\(Karaoke",
                    artist_group=1,
                    title_group=2,
                    confidence=0.85,
                ),
            ]
            self.channel_patterns["@avd-karaoke"] = avd_patterns
            self.channel_patterns["UCvh3Hf-Ub4ZecC09RD2igjQ"] = (
                avd_patterns  # AVD Karaoke channel ID
            )

            # @quantumkaraoke patterns
            quantum_patterns = [
                # Primary: Song Title | Quantum Karaoke | Show/Special
                ChannelPattern(
                    pattern=r"^(.+?)\s*\|\s*Quantum Karaoke\s*\|",
                    artist_group=None,
                    title_group=1,
                    confidence=0.90,
                ),
                # Show format: Quantum Karaoke | a Sing-Along Special | Show S## E##
                ChannelPattern(
                    pattern=r"^Quantum Karaoke\s*\|\s*a Sing-Along Special\s*\|",
                    artist_group=None,
                    title_group=None,
                    confidence=0.70,
                ),
                # Simple format: Song Title by Artist (Karaoke)
                ChannelPattern(
                    pattern=r"^(.+?)\s+by\s+([^()]+?)\s*\(Karaoke\)",
                    artist_group=2,
                    title_group=1,
                    confidence=0.85,
                ),
            ]
            self.channel_patterns["@quantumkaraoke"] = quantum_patterns
            self.channel_patterns["UCY_0l0AngUurGCwAqF4NkzA"] = (
                quantum_patterns  # Quantum Karaoke channel ID
            )

            # @partytymekaraokechannel6967 patterns
            partytyme_patterns = [
                # Primary: Song Title (Made Popular By Artist) (Karaoke Version)
                ChannelPattern(
                    pattern=r"^(.+?)\s*\(Made Popular By ([^)]+?)\)\s*\(Karaoke Version\)",
                    artist_group=2,
                    title_group=1,
                    confidence=0.95,
                ),
                # Square bracket variation
                ChannelPattern(
                    pattern=r"^(.+?)\s*\(Made Popular By ([^)]+?)\)\s*\[Karaoke Version\]",
                    artist_group=2,
                    title_group=1,
                    confidence=0.95,
                ),
                # Standard format: Artist - Song Title (Karaoke Version)
                ChannelPattern(
                    pattern=r"^([^-]+?)\s*-\s*(.+?)\s*\(Karaoke Version\)",
                    artist_group=1,
                    title_group=2,
                    confidence=0.95,
                ),
                # With Party Tyme branding
                ChannelPattern(
                    pattern=r"^(.+?)\s*\(Made Popular By ([^)]+?)\)\s*\(Party Tyme Karaoke",
                    artist_group=2,
                    title_group=1,
                    confidence=0.90,
                ),
            ]
            self.channel_patterns["@partytymekaraokechannel6967"] = partytyme_patterns
            self.channel_patterns["UCWLqO9ztz16a_Ko4YB9PnFQ"] = (
                partytyme_patterns  # PARTY TYME KARAOKE CHANNEL ID
            )

            # @StingrayKaraoke patterns
            stingray_patterns = [
                # Primary: Artist - Song Title (Karaoke with Lyrics)
                ChannelPattern(
                    pattern=r"^([^-]+?)\s*-\s*(.+?)\s*\(Karaoke with Lyrics\)",
                    artist_group=1,
                    title_group=2,
                    confidence=0.95,
                ),
                # Alternative: Artist - Song Title (Stingray Karaoke)
                ChannelPattern(
                    pattern=r"^([^-]+?)\s*-\s*(.+?)\s*\(Stingray Karaoke",
                    artist_group=1,
                    title_group=2,
                    confidence=0.90,
                ),
                # Generic pattern
                ChannelPattern(
                    pattern=r"^([^-]+?)\s*-\s*(.+?)$",
                    artist_group=1,
                    title_group=2,
                    confidence=0.75,
                ),
            ]
            self.channel_patterns["@StingrayKaraoke"] = stingray_patterns
            self.channel_patterns["UCYi9TC1HC_U2kaRAK6I4FSQ"] = (
                stingray_patterns  # Stingray Karaoke channel ID
            )

            # @TheoMusicChannel patterns
            theo_patterns = [
                # Pipe separator format: Song Title | Artist · Featured Artist
                ChannelPattern(
                    pattern=r"^(.+?)\s*\|\s*(.+?)$",  # Capture all artists including middot
                    artist_group=2,
                    title_group=1,
                    confidence=0.95,
                ),
                # Primary: Artist - Song Title - Lyrics
                ChannelPattern(
                    pattern=r"^([^-]+?)\s*-\s*(.+?)\s*-\s*Lyrics$",
                    artist_group=1,
                    title_group=2,
                    confidence=0.95,
                ),
                # With Chords
                ChannelPattern(
                    pattern=r"^([^-]+?)\s*-\s*(.+?)\s*-\s*Lyrics and Chords$",
                    artist_group=1,
                    title_group=2,
                    confidence=0.95,
                ),
                # Instrumental/Karaoke/Cover
                ChannelPattern(
                    pattern=r"^([^-]+?)\s*-\s*(.+?)\s*\((?:Instrumental/Karaoke/Cover|Acoustic Instrumental)\)",
                    artist_group=1,
                    title_group=2,
                    confidence=0.90,
                ),
                # Generic two-dash pattern
                ChannelPattern(
                    pattern=r"^([^-]+?)\s*-\s*(.+?)(?:\s*-.*)?$",
                    artist_group=1,
                    title_group=2,
                    confidence=0.75,
                ),
            ]
            self.channel_patterns["@TheoMusicChannel"] = theo_patterns
            self.channel_patterns["UCWyWC9jEp_0ecel6Usj7j8Q"] = (
                theo_patterns  # Theo's Music channel ID
            )

            # @FakeyOke patterns
            fakeyoke_patterns = [
                # Primary: Artist - Song Title [Karaoke]
                ChannelPattern(
                    pattern=r"^([^-]+?)\s*-\s*(.+?)\s*\[Karaoke\]",
                    artist_group=1,
                    title_group=2,
                    confidence=0.95,
                ),
                # Alternative: Artist - Song Title (FakeyOke Karaoke)
                ChannelPattern(
                    pattern=r"^([^-]+?)\s*-\s*(.+?)\s*\(FakeyOke Karaoke\)",
                    artist_group=1,
                    title_group=2,
                    confidence=0.90,
                ),
                # Generic pattern
                ChannelPattern(
                    pattern=r"^([^-]+?)\s*-\s*(.+?)$",
                    artist_group=1,
                    title_group=2,
                    confidence=0.75,
                ),
            ]
            self.channel_patterns["@FakeyOke"] = fakeyoke_patterns
            self.channel_patterns["UCvtLVf1qXFe_hmxlaB4Gh8Q"] = (
                fakeyoke_patterns  # FakeyOke channel ID
            )

            # @songjam patterns
            songjam_patterns = [
                # Primary: Artist - Song Title (Official Karaoke Instrumental) | SongJam
                ChannelPattern(
                    pattern=r"^([^-]+?)\s*-\s*(.+?)\s*\(Official Karaoke Instrumental\)\s*\|\s*SongJam",
                    artist_group=1,
                    title_group=2,
                    confidence=0.95,
                ),
                # Backing Track variation
                ChannelPattern(
                    pattern=r"^([^-]+?)\s*-\s*(.+?)\s*\(Backing Track\)",
                    artist_group=1,
                    title_group=2,
                    confidence=0.90,
                ),
                # Generic pattern
                ChannelPattern(
                    pattern=r"^([^-]+?)\s*-\s*(.+?)(?:\s*[|(].*)?$",
                    artist_group=1,
                    title_group=2,
                    confidence=0.75,
                ),
            ]
            self.channel_patterns["@songjam"] = songjam_patterns
            self.channel_patterns["UCLYRNIeTQNmx1E9jVAFxSlA"] = (
                songjam_patterns  # Songjam channel ID
            )

            # @edkara patterns
            edkara_patterns = [
                # Primary: Karaoke♬ Song Title - Artist 【With Guide Melody】 Instrumental
                ChannelPattern(
                    pattern=r"^Karaoke♬\s*(.+?)\s*-\s*([^【]+?)\s*【With Guide Melody】\s*Instrumental",
                    artist_group=2,
                    title_group=1,
                    confidence=0.95,
                ),
                # No Guide Melody variation
                ChannelPattern(
                    pattern=r"^Karaoke♬\s*(.+?)\s*-\s*([^【]+?)\s*【No Guide Melody】\s*Instrumental",
                    artist_group=2,
                    title_group=1,
                    confidence=0.95,
                ),
                # Without the music note
                ChannelPattern(
                    pattern=r"^Karaoke\s*(.+?)\s*-\s*([^【]+?)\s*【.*Guide Melody】",
                    artist_group=2,
                    title_group=1,
                    confidence=0.90,
                ),
                # Generic pattern after removing Karaoke prefix
                ChannelPattern(
                    pattern=r"^(?:Karaoke♬?\s*)?(.+?)\s*-\s*([^【]+?)(?:\s*【.*)?$",
                    artist_group=2,
                    title_group=1,
                    confidence=0.75,
                ),
            ]
            self.channel_patterns["@edkara"] = edkara_patterns
            self.channel_patterns["UCRrNOLvQ1LztDKbXtxvDAEQ"] = edkara_patterns  # EdKara channel ID

            # @singkaraoke patterns (NOT @singkingkaraoke - different channel!)
            # Sing Karaoke uses standard "Artist - Song (Karaoke)" format
            singkaraoke_patterns = [
                # Primary: Artist - Song Title (Karaoke)
                ChannelPattern(
                    pattern=r"^(.+?)\s*-\s*(.+?)\s*\(Karaoke\)",
                    artist_group=1,  # Artist is FIRST (standard format)
                    title_group=2,  # Song is SECOND
                    confidence=0.95,
                ),
                # Without parentheses
                ChannelPattern(
                    pattern=r"^(.+?)\s*-\s*(.+?)\s+Karaoke",
                    artist_group=1,
                    title_group=2,
                    confidence=0.90,
                ),
                # Generic pattern
                ChannelPattern(
                    pattern=r"^(.+?)\s*-\s*(.+?)$",
                    artist_group=1,
                    title_group=2,
                    confidence=0.80,
                ),
            ]
            self.channel_patterns["@singkaraoke"] = singkaraoke_patterns
            self.channel_patterns["Sing Karaoke"] = singkaraoke_patterns  # Channel name fallback
            self.channel_patterns["@singkaraoke9783"] = (
                singkaraoke_patterns  # Actual channel handle
            )
            self.channel_patterns["UC1AgLpY5t66HaI3ejzLoyOg"] = (
                singkaraoke_patterns  # Sing Karaoke channel ID
            )

            logger.info("Loaded hardcoded patterns for 22 karaoke channels")

            # Create channel name to ID/handle mapping for easier lookup
            self.channel_name_mapping = {
                "Sing King": ["@singkingkaraoke", "UCwTRjvjVge51X-ILJ4i22ew"],
                "Musisi Karaoke": ["@MusisiKaraoke", "UCJw1qyMF4m3ZIBWdhogkcsw"],
                "ZZang KARAOKE": ["@zzangkaraoke", "UCzv4mCu3YS_9WjAWSt9Xg9Q"],
                "Atomic Karaoke...": ["@AtomicKaraoke", "UCutZyApGOjqhOS-pp7yAj4Q"],
                "BandaisuanKaraoke001": ["@BandaisuanKaraoke001", "UCuyBQQ2CISV0ptQRHBHzGuA"],
                "KaraFun Karaoke": ["@karafun", "UCbqcG1rdt9LMwOJN4PyGTKg"],
                "Sing Karaoke": ["@singkaraoke9783", "UC1AgLpY5t66HaI3ejzLoyOg"],
                "KaraokeyTV": ["@karaokeytv0618", "UCNbFgUCJj2Ls6LVzBbL8fqA"],
                "Mi Balmz Karaoke Tracks": ["@mibalmzkaraoke", "UCRoAoGqqLuOIWztkcxUiYoA"],
                "The Proper Volume Karaoke Studio": [
                    "@thepropervolume",
                    "UCw0WVzCSi9-X0RMxaj3gTpg",
                ],
                "Sing2Piano | Piano Karaoke Instrumentals": [
                    "@sing2piano",
                    "UCIk6z4gxI5ADYK7HmNiJvNg",
                ],
                "Lugn": ["@LugnKaraoke", "UCS4Q7GGXKdZW9uZ6YySe34Q"],
                "KaraFun Deutschland - Karaoke": ["@karafunde", "UCzEav_eOAmp23-s_cqBDpbA"],
                "FrauKnoblauch": ["@FrauKnoblauch", "UC-AdlzvbJi7LvBkja531tVQ"],
                "AVD Karaoke": ["@avd-karaoke", "UCvh3Hf-Ub4ZecC09RD2igjQ"],
                "Quantum Karaoke": ["@quantumkaraoke", "UCY_0l0AngUurGCwAqF4NkzA"],
                "PARTY TYME KARAOKE CHANNEL": [
                    "@partytymekaraokechannel6967",
                    "UCWLqO9ztz16a_Ko4YB9PnFQ",
                ],
                "Stingray Karaoke": ["@StingrayKaraoke", "UCYi9TC1HC_U2kaRAK6I4FSQ"],
                "Theo's Music": ["@TheoMusicChannel", "UCWyWC9jEp_0ecel6Usj7j8Q"],
                "FakeyOke": ["@FakeyOke", "UCvtLVf1qXFe_hmxlaB4Gh8Q"],
                "Songjam: Official Karaoke": ["@songjam", "UCLYRNIeTQNmx1E9jVAFxSlA"],
                "EdKara": ["@edkara", "UCRrNOLvQ1LztDKbXtxvDAEQ"],
            }

            # Load additional patterns from database if available
            if self.db_manager:
                # Here we would load learned patterns from database
                # For now, just log that we would do this
                logger.debug("Database manager available, would load learned patterns")

        except Exception as e:
            logger.warning(f"Failed to load channel patterns: {e}")

    def save_patterns(self):
        """Save learned patterns to database."""

        if not self.db_manager:
            return

        try:
            # This would save to database in a real implementation
            logger.info(f"Saving patterns for {len(self.channel_patterns)} channels")

        except Exception as e:
            logger.error(f"Failed to save channel patterns: {e}")

    def get_statistics(self) -> Dict:
        """Get statistics for the channel template pass."""

        total_channels = len(self.channel_stats)
        total_patterns = sum(len(patterns) for patterns in self.channel_patterns.values())

        channels_with_drift = sum(
            1 for stats in self.channel_stats.values() if stats.drift_detected
        )

        return {
            "total_channels": total_channels,
            "total_learned_patterns": total_patterns,
            "channels_with_drift": channels_with_drift,
            "avg_patterns_per_channel": total_patterns / max(total_channels, 1),
            "channel_success_rates": {
                channel_id: stats.successful_parses / max(stats.total_videos, 1)
                for channel_id, stats in self.channel_stats.items()
            },
        }
