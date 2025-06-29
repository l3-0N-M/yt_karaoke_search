"""Pass 0: Enhanced channel-template matching with dynamic pattern learning."""

import logging
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

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
            # Step 1: Try channel-specific learned patterns first
            if channel_id in self.channel_patterns:
                result = self._try_channel_patterns(
                    channel_id, title, description, tags, channel_name
                )
                if result and result.confidence > 0.8:
                    self._update_pattern_success(channel_id, result.pattern_used, title)
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

        # Sort patterns by success rate and recency
        patterns.sort(
            key=lambda p: (p.success_count / max(p.total_attempts, 1), p.last_used), reverse=True
        )

        for pattern in patterns:
            try:
                match = re.search(pattern.pattern, title, re.IGNORECASE | re.UNICODE)
                if match:
                    result = self._create_result_from_pattern_match(
                        match, pattern, f"learned_channel_{channel_id}", title
                    )
                    if result and result.confidence > 0:
                        pattern.total_attempts += 1
                        pattern.last_used = datetime.now()
                        return result

            except re.error as e:
                logger.warning(f"Invalid regex pattern for channel {channel_id}: {e}")
                continue

        return None

    def _enhanced_channel_detection(
        self, title: str, description: str, tags: str, channel_name: str, channel_id: str
    ) -> Optional[ParseResult]:
        """Enhanced channel-specific pattern detection."""

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

        # Try enhanced patterns
        for pattern, artist_group, title_group, confidence, pattern_name in enhanced_patterns:
            try:
                match = re.search(pattern, title, re.IGNORECASE | re.UNICODE)
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
            (r"^([^-–—]+)\s*[-–—]\s*([^(\[]+)", 2, 1, 0.65, "simple_dash"),
            (r'^"([^"]+)"\s*[-–—]\s*"([^"]+)"', 2, 1, 0.7, "simple_quoted"),
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
        self, match, pattern: ChannelPattern, method: str, original_title: str
    ) -> Optional[ParseResult]:
        """Create a ParseResult from a pattern match."""

        try:
            result = ParseResult(method=method, pattern_used=pattern.pattern)

            if pattern.artist_group and pattern.artist_group <= len(match.groups()):
                artist = self.advanced_parser._clean_extracted_text(
                    match.group(pattern.artist_group)
                )
                if self.advanced_parser._is_valid_artist_name(artist):
                    result.artist = artist

            if pattern.title_group and pattern.title_group <= len(match.groups()):
                song_title = self.advanced_parser._clean_extracted_text(
                    match.group(pattern.title_group)
                )
                if self.advanced_parser._is_valid_song_title(song_title):
                    result.song_title = song_title

            # Calculate confidence based on pattern history and extraction success
            base_confidence = pattern.confidence
            if result.artist and result.song_title:
                result.confidence = base_confidence
            elif result.artist or result.song_title:
                result.confidence = base_confidence * 0.7
            else:
                result.confidence = 0

            # Boost confidence based on pattern success history
            if pattern.total_attempts > 0:
                success_rate = pattern.success_count / pattern.total_attempts
                result.confidence *= 0.8 + 0.2 * success_rate  # 80-100% multiplier

            result.metadata = {
                "pattern_success_rate": pattern.success_count / max(pattern.total_attempts, 1),
                "pattern_age_days": (datetime.now() - pattern.created).days,
                "pattern_last_used": pattern.last_used.isoformat(),
            }

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

        if not self.db_manager:
            return

        try:
            # This would load from database in a real implementation
            # For now, we'll initialize empty
            logger.info("Channel pattern loading not yet implemented")

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
