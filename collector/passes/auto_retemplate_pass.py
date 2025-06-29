"""Pass 1: Auto-re-template on recent uploads with temporal pattern analysis."""

import logging
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from ..advanced_parser import AdvancedTitleParser, ParseResult
from .base import ParsingPass, PassType

logger = logging.getLogger(__name__)


@dataclass
class TemporalPattern:
    """A pattern with temporal information."""

    pattern: str
    artist_group: Optional[int]
    title_group: Optional[int]
    confidence: float
    first_seen: datetime
    last_seen: datetime
    video_count: int = 0
    success_count: int = 0
    recent_examples: deque = field(default_factory=lambda: deque(maxlen=10))


@dataclass
class ChannelTrend:
    """Trending patterns for a channel over time."""

    channel_id: str
    channel_name: str
    active_patterns: List[TemporalPattern] = field(default_factory=list)
    deprecated_patterns: List[TemporalPattern] = field(default_factory=list)
    pattern_change_detected: Optional[datetime] = None
    last_analysis: datetime = field(default_factory=datetime.now)
    total_recent_videos: int = 0
    successful_recent_parses: int = 0


class AutoRetemplatePass(ParsingPass):
    """Pass 1: Auto-re-template on recent uploads with intelligent pattern evolution."""

    def __init__(self, advanced_parser: AdvancedTitleParser, db_manager=None):
        self.advanced_parser = advanced_parser
        self.db_manager = db_manager

        # Temporal pattern analysis
        self.channel_trends: Dict[str, ChannelTrend] = {}
        self.recent_window = timedelta(days=30)  # Look at last 30 days
        self.pattern_change_window = timedelta(days=7)  # Detect changes in last 7 days
        self.min_videos_for_analysis = 5

        # Pattern evolution detection
        self.pattern_similarity_threshold = 0.8
        self.success_rate_threshold = 0.6
        self.pattern_confidence_decay = 0.95  # Daily decay factor

        # Load existing trends
        self._load_channel_trends()

    @property
    def pass_type(self) -> PassType:
        return PassType.AUTO_RETEMPLATE

    async def parse(
        self,
        title: str,
        description: str = "",
        tags: str = "",
        channel_name: str = "",
        channel_id: str = "",
        metadata: Optional[Dict] = None,
    ) -> Optional[ParseResult]:
        """Execute auto-retemplate parsing based on recent upload patterns."""

        if not channel_id:
            return None

        start_time = time.time()
        trend: ChannelTrend = self._get_or_create_trend(channel_id, channel_name)

        try:

            # Step 2: Try current active patterns first
            result = self._try_active_patterns(trend, title, description, tags)
            if result and result.confidence > 0.75:
                self._record_pattern_success(trend, result.pattern_used, title)
                return result

            # Step 3: Analyze recent videos to detect new patterns
            if self._should_analyze_patterns(trend):
                await self._analyze_recent_patterns(trend, title)

                # Try newly discovered patterns
                result = self._try_active_patterns(trend, title, description, tags)
                if result and result.confidence > 0.7:
                    self._record_pattern_success(trend, result.pattern_used, title)
                    return result

            # Step 4: Try deprecated patterns with reduced confidence
            result = self._try_deprecated_patterns(trend, title, description, tags)
            if result and result.confidence > 0.6:
                # Mark this pattern as potentially active again
                self._revive_pattern(trend, result.pattern_used)
                return result

            # Step 5: Attempt to learn from this title
            learned_result = self._attempt_learning(trend, title, description, tags)
            if learned_result:
                return learned_result

            return None

        except Exception as e:
            logger.error(f"Auto-retemplate pass failed: {e}")
            return None
        finally:
            processing_time = time.time() - start_time
            self._update_trend_stats(trend, processing_time)

    def _get_or_create_trend(self, channel_id: str, channel_name: str) -> ChannelTrend:
        """Get or create a channel trend tracker."""

        if channel_id not in self.channel_trends:
            self.channel_trends[channel_id] = ChannelTrend(
                channel_id=channel_id, channel_name=channel_name
            )

        return self.channel_trends[channel_id]

    def _try_active_patterns(
        self, trend: ChannelTrend, title: str, description: str, tags: str
    ) -> Optional[ParseResult]:
        """Try active patterns for the channel."""

        # Sort patterns by recency and success rate
        sorted_patterns = sorted(
            trend.active_patterns,
            key=lambda p: (p.last_seen, p.success_count / max(p.video_count, 1)),
            reverse=True,
        )

        for pattern in sorted_patterns:
            try:
                match = re.search(pattern.pattern, title, re.IGNORECASE | re.UNICODE)
                if match:
                    result = self._create_result_from_temporal_pattern(
                        match, pattern, "auto_retemplate_active", title
                    )
                    if result and result.confidence > 0:
                        pattern.video_count += 1
                        pattern.last_seen = datetime.now()
                        return result

            except re.error as e:
                logger.warning(f"Invalid temporal pattern: {e}")
                continue

        return None

    async def _analyze_recent_patterns(self, trend: ChannelTrend, current_title: str):
        """Analyze recent uploads to detect new patterns."""

        # This would typically query the database for recent videos
        # For now, we'll simulate with pattern detection on the current title

        if not self.db_manager:
            # Fallback: try to learn from current title structure
            self._learn_from_title_structure(trend, current_title)
            return

        try:
            # Query recent videos from this channel
            recent_videos = await self._get_recent_channel_videos(trend.channel_id)

            if len(recent_videos) < self.min_videos_for_analysis:
                return

            # Analyze patterns in recent videos
            self._detect_new_patterns(trend, recent_videos)

            # Detect pattern changes
            self._detect_pattern_changes(trend, recent_videos)

        except Exception as e:
            logger.warning(f"Failed to analyze recent patterns: {e}")

    async def _get_recent_channel_videos(self, channel_id: str) -> List[Dict]:
        """Get recent videos from the channel (placeholder for database query)."""

        # This would be a real database query in production
        # For now, return empty list
        return []

    def _detect_new_patterns(self, trend: ChannelTrend, recent_videos: List[Dict]):
        """Detect new patterns from recent videos."""

        # Group videos by similar title structures
        structure_groups = defaultdict(list)

        for video in recent_videos:
            title = video.get("title", "")
            structure = self._extract_title_structure(title)
            if structure:
                structure_groups[structure].append(video)

        # Look for new recurring patterns
        for structure, videos in structure_groups.items():
            if len(videos) >= 3:  # Need at least 3 examples
                pattern = self._structure_to_pattern(structure)
                if pattern and not self._pattern_exists(trend, pattern):
                    self._add_new_pattern(trend, pattern, videos)

    def _extract_title_structure(self, title: str) -> Optional[str]:
        """Extract the structural pattern from a title."""

        # Common karaoke title patterns
        structures = [
            # Artist - Song (Karaoke)
            (r"^([^-–—]+)\s*[-–—]\s*([^(]+)\s*\([^)]*[Kk]araoke[^)]*\)", "ARTIST-SONG-KARAOKE"),
            # "Artist" - "Song" Karaoke
            (r'^"([^"]+)"\s*[-–—]\s*"([^"]+)"\s*[Kk]araoke', "QUOTED-ARTIST-SONG-KARAOKE"),
            # [Channel] Artist - Song
            (r"^\[[^\]]+\]\s*([^-–—]+)[-–—](.+)", "CHANNEL-ARTIST-SONG"),
            # Song by Artist (Karaoke)
            (r"^([^(]+?)\s*by\s+([^(]+?)\s*\([^)]*[Kk]araoke[^)]*\)", "SONG-BY-ARTIST-KARAOKE"),
        ]

        for pattern, structure_name in structures:
            if re.search(pattern, title, re.IGNORECASE):
                return structure_name

        return None

    def _structure_to_pattern(self, structure: str) -> Optional[str]:
        """Convert a structure identifier to a regex pattern."""

        pattern_map = {
            "ARTIST-SONG-KARAOKE": r"^([^-–—]+)\s*[-–—]\s*([^(]+)\s*\([^)]*[Kk]araoke[^)]*\)",
            "QUOTED-ARTIST-SONG-KARAOKE": r'^"([^"]+)"\s*[-–—]\s*"([^"]+)"\s*[Kk]araoke',
            "CHANNEL-ARTIST-SONG": r"^\[[^\]]+\]\s*([^-–—]+)[-–—](.+)",
            "SONG-BY-ARTIST-KARAOKE": r"^([^(]+?)\s*by\s+([^(]+?)\s*\([^)]*[Kk]araoke[^)]*\)",
        }

        return pattern_map.get(structure)

    def _pattern_exists(self, trend: ChannelTrend, pattern: str) -> bool:
        """Check if a pattern already exists for the channel."""

        for existing_pattern in trend.active_patterns + trend.deprecated_patterns:
            if existing_pattern.pattern == pattern:
                return True

        return False

    def _add_new_pattern(self, trend: ChannelTrend, pattern: str, examples: List[Dict]):
        """Add a new pattern to the channel's active patterns."""

        temporal_pattern = TemporalPattern(
            pattern=pattern,
            artist_group=1,  # Assuming first group is artist
            title_group=2,  # Assuming second group is title
            confidence=0.8,  # Start with high confidence for detected patterns
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            video_count=len(examples),
            success_count=len(examples),  # All examples are successful by definition
            recent_examples=deque([ex.get("title", "") for ex in examples], maxlen=10),
        )

        trend.active_patterns.append(temporal_pattern)

        logger.info(
            f"Added new pattern for channel {trend.channel_id}: {pattern} "
            f"(based on {len(examples)} examples)"
        )

    def _detect_pattern_changes(self, trend: ChannelTrend, recent_videos: List[Dict]):
        """Detect if channel has changed its pattern recently."""

        if not trend.active_patterns:
            return

        # Check if recent videos match current active patterns
        recent_matches = 0
        recent_count = min(len(recent_videos), 10)  # Check last 10 videos

        for video in recent_videos[-recent_count:]:
            title = video.get("title", "")
            for pattern in trend.active_patterns:
                try:
                    if re.search(pattern.pattern, title, re.IGNORECASE):
                        recent_matches += 1
                        break
                except re.error:
                    continue

        # If match rate is low, mark pattern change
        match_rate = recent_matches / max(recent_count, 1)
        if match_rate < 0.5:  # Less than 50% match rate
            trend.pattern_change_detected = datetime.now()
            self._deprecate_old_patterns(trend)

            logger.info(
                f"Pattern change detected for channel {trend.channel_id}: "
                f"only {match_rate:.1%} of recent videos match current patterns"
            )

    def _deprecate_old_patterns(self, trend: ChannelTrend):
        """Move old patterns to deprecated status."""

        cutoff_date = datetime.now() - self.pattern_change_window

        patterns_to_deprecate = []
        for pattern in trend.active_patterns:
            if pattern.last_seen < cutoff_date:
                patterns_to_deprecate.append(pattern)

        for pattern in patterns_to_deprecate:
            trend.active_patterns.remove(pattern)
            trend.deprecated_patterns.append(pattern)
            pattern.confidence *= 0.7  # Reduce confidence for deprecated patterns

    def _learn_from_title_structure(self, trend: ChannelTrend, title: str):
        """Learn patterns from title structure when database is not available."""

        structure = self._extract_title_structure(title)
        if not structure:
            return

        pattern = self._structure_to_pattern(structure)
        if pattern and not self._pattern_exists(trend, pattern):
            # Create a new pattern based on this single example
            temporal_pattern = TemporalPattern(
                pattern=pattern,
                artist_group=1,
                title_group=2,
                confidence=0.6,  # Lower confidence for single example
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                video_count=1,
                success_count=0,  # Haven't validated yet
                recent_examples=deque([title], maxlen=10),
            )

            trend.active_patterns.append(temporal_pattern)

    def _try_deprecated_patterns(
        self, trend: ChannelTrend, title: str, description: str, tags: str
    ) -> Optional[ParseResult]:
        """Try deprecated patterns with reduced confidence."""

        for pattern in trend.deprecated_patterns:
            try:
                match = re.search(pattern.pattern, title, re.IGNORECASE | re.UNICODE)
                if match:
                    result = self._create_result_from_temporal_pattern(
                        match, pattern, "auto_retemplate_deprecated", title
                    )
                    if result and result.confidence > 0:
                        # Reduce confidence since it's a deprecated pattern
                        result.confidence *= 0.8
                        pattern.video_count += 1
                        return result

            except re.error:
                continue

        return None

    def _revive_pattern(self, trend: ChannelTrend, pattern_used: str):
        """Move a deprecated pattern back to active if it's working again."""

        for pattern in trend.deprecated_patterns:
            if pattern.pattern == pattern_used:
                trend.deprecated_patterns.remove(pattern)
                trend.active_patterns.append(pattern)
                pattern.confidence = min(pattern.confidence * 1.2, 0.9)  # Boost confidence
                pattern.last_seen = datetime.now()

                logger.info(f"Revived pattern for channel {trend.channel_id}: {pattern_used}")
                break

    def _attempt_learning(
        self, trend: ChannelTrend, title: str, description: str, tags: str
    ) -> Optional[ParseResult]:
        """Attempt to learn a new pattern from the current title."""

        # Try basic parsing first to see if we can extract artist/song
        basic_result = self.advanced_parser.parse_title(
            title, description, tags, trend.channel_name
        )

        if basic_result and basic_result.confidence > 0.5:
            # Try to create a new pattern based on this successful parse
            if basic_result.artist and basic_result.song_title:
                new_pattern = self._create_pattern_from_parse(title, basic_result)
                if new_pattern:
                    self._add_learned_pattern(trend, new_pattern, title)
                    return basic_result

        return None

    def _create_pattern_from_parse(self, title: str, result: ParseResult) -> Optional[str]:
        """Create a pattern from a successful parse."""

        try:
            # Escape the extracted parts
            # Replace the parts with capturing groups
            pattern = title or ""
            if result.artist:
                pattern = pattern.replace(result.artist, "([^-–—\"']+?)")
            if result.song_title:
                pattern = pattern.replace(result.song_title, "([^(\\[]+?)")

            # Clean up and validate
            pattern = f"^{pattern}$"
            pattern = re.sub(r"\s+", r"\\s+", pattern)

            # Test the pattern
            re.compile(pattern)
            return pattern

        except (re.error, AttributeError):
            return None

    def _add_learned_pattern(self, trend: ChannelTrend, pattern: str, example: str):
        """Add a pattern learned from a single example."""

        temporal_pattern = TemporalPattern(
            pattern=pattern,
            artist_group=1,
            title_group=2,
            confidence=0.5,  # Low confidence for learned patterns
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            video_count=1,
            success_count=1,
            recent_examples=deque([example], maxlen=10),
        )

        trend.active_patterns.append(temporal_pattern)

    def _create_result_from_temporal_pattern(
        self, match, pattern: TemporalPattern, method: str, title: str
    ) -> Optional[ParseResult]:
        """Create a ParseResult from a temporal pattern match."""

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

            # Calculate confidence based on pattern history
            base_confidence = pattern.confidence

            # Adjust based on recency
            age_days = (datetime.now() - pattern.last_seen).days
            recency_factor = max(0.7, 1.0 - (age_days / 30.0))  # Decay over 30 days

            # Adjust based on success rate
            success_rate = pattern.success_count / max(pattern.video_count, 1)
            success_factor = 0.5 + 0.5 * success_rate

            if result.artist and result.song_title:
                result.confidence = base_confidence * recency_factor * success_factor
            elif result.artist or result.song_title:
                result.confidence = base_confidence * recency_factor * success_factor * 0.7
            else:
                result.confidence = 0

            result.metadata = {
                "pattern_age_days": age_days,
                "pattern_success_rate": success_rate,
                "pattern_video_count": pattern.video_count,
                "recency_factor": recency_factor,
                "success_factor": success_factor,
            }

            return result

        except Exception as e:
            logger.warning(f"Failed to create result from temporal pattern: {e}")
            return None

    def _record_pattern_success(self, trend: ChannelTrend, pattern_used: str, title: str):
        """Record a successful pattern match."""

        for pattern in trend.active_patterns:
            if pattern.pattern == pattern_used:
                pattern.success_count += 1
                pattern.recent_examples.append(title)
                break

    def _should_analyze_patterns(self, trend: ChannelTrend) -> bool:
        """Determine if we should analyze patterns for this channel."""

        # Analyze if we haven't done so recently
        time_since_analysis = datetime.now() - trend.last_analysis
        if time_since_analysis < timedelta(hours=1):
            return False

        # Analyze if we have few patterns or low success rate
        if (
            len(trend.active_patterns) < 2
            or trend.successful_recent_parses / max(trend.total_recent_videos, 1) < 0.6
        ):
            return True

        # Analyze if pattern change was detected
        if trend.pattern_change_detected:
            time_since_change = datetime.now() - trend.pattern_change_detected
            if time_since_change < timedelta(days=1):
                return True

        return False

    def _update_trend_stats(self, trend: ChannelTrend, processing_time: float):
        """Update trend statistics."""

        trend.total_recent_videos += 1
        trend.last_analysis = datetime.now()

        # Decay old pattern confidences
        self._apply_confidence_decay(trend)

    def _apply_confidence_decay(self, trend: ChannelTrend):
        """Apply daily confidence decay to patterns."""

        for pattern in trend.active_patterns + trend.deprecated_patterns:
            days_since_last_use = (datetime.now() - pattern.last_seen).days
            if days_since_last_use > 0:
                decay_factor = self.pattern_confidence_decay**days_since_last_use
                pattern.confidence *= decay_factor

    def _load_channel_trends(self):
        """Load existing channel trends from database."""

        if not self.db_manager:
            return

        try:
            # This would load from database in a real implementation
            logger.info("Channel trend loading not yet implemented")

        except Exception as e:
            logger.warning(f"Failed to load channel trends: {e}")

    def save_trends(self):
        """Save channel trends to database."""

        if not self.db_manager:
            return

        try:
            logger.info(f"Saving trends for {len(self.channel_trends)} channels")

        except Exception as e:
            logger.error(f"Failed to save channel trends: {e}")

    def get_statistics(self) -> Dict:
        """Get statistics for the auto-retemplate pass."""

        total_channels = len(self.channel_trends)
        total_active_patterns = sum(
            len(trend.active_patterns) for trend in self.channel_trends.values()
        )
        total_deprecated_patterns = sum(
            len(trend.deprecated_patterns) for trend in self.channel_trends.values()
        )

        channels_with_changes = sum(
            1 for trend in self.channel_trends.values() if trend.pattern_change_detected
        )

        return {
            "total_channels": total_channels,
            "total_active_patterns": total_active_patterns,
            "total_deprecated_patterns": total_deprecated_patterns,
            "channels_with_pattern_changes": channels_with_changes,
            "avg_patterns_per_channel": (total_active_patterns / max(total_channels, 1)),
            "pattern_change_rate": (channels_with_changes / max(total_channels, 1)),
        }
