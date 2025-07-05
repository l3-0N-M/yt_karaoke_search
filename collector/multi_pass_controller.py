"""Multi-pass parsing ladder controller with confidence-based progression and intelligent stopping."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from .advanced_parser import AdvancedTitleParser, ParseResult
from .config import MultiPassConfig, MultiPassPassConfig
from .enhanced_search import MultiStrategySearchEngine
from .passes.base import ParsingPass, PassType

logger = logging.getLogger(__name__)


@dataclass
class PassResult:
    """Result from a single parsing pass."""

    pass_type: PassType
    parse_result: Optional[ParseResult] = None
    confidence: float = 0.0
    processing_time: float = 0.0
    success: bool = False
    error_message: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class MultiPassResult:
    """Result from the complete multi-pass parsing process."""

    video_id: str
    original_title: str
    final_result: Optional[ParseResult] = None
    passes_attempted: List[PassResult] = field(default_factory=list)
    total_processing_time: float = 0.0
    stopped_at_pass: Optional[PassType] = None
    final_confidence: float = 0.0
    retry_count: int = 0
    budget_consumed: Dict[str, float] = field(default_factory=dict)


# Note: Configuration classes are now imported from config.py


class MultiPassParsingController:
    """Controller for multi-pass parsing ladder with intelligent progression."""

    def __init__(
        self,
        config: MultiPassConfig,
        passes: List[ParsingPass],
        advanced_parser: AdvancedTitleParser,
        search_engine: Optional[MultiStrategySearchEngine] = None,
        db_manager=None,
    ):
        self.config = config
        self.passes = passes
        self.advanced_parser = advanced_parser
        self.search_engine = search_engine
        self.db_manager = db_manager

        # Create pass type to config mapping in optimized order
        self.pass_configs = {
            PassType.CHANNEL_TEMPLATE: config.channel_template,
            PassType.MUSICBRAINZ_SEARCH: config.musicbrainz_search,
            PassType.DISCOGS_SEARCH: config.discogs_search,
            PassType.MUSICBRAINZ_VALIDATION: config.musicbrainz_validation,
            PassType.ML_EMBEDDING: config.ml_embedding,
            PassType.AUTO_RETEMPLATE: config.auto_retemplate,
        }

        # Store a reference to the channel template pass for learning
        self.channel_template_pass: Optional[ParsingPass] = next(
            (p for p in passes if p.pass_type == PassType.CHANNEL_TEMPLATE), None
        )

        # Statistics and monitoring
        self.statistics = {
            "total_videos_processed": 0,
            "passes_attempted": {pass_type: 0 for pass_type in PassType},
            "passes_successful": {pass_type: 0 for pass_type in PassType},
            "average_processing_time": 0.0,
            "budget_consumption": {
                "cpu_total": 0.0,
                "api_total": 0,
                "cpu_per_pass": {pass_type: 0.0 for pass_type in PassType},
                "api_per_pass": {pass_type: 0 for pass_type in PassType},
            },
        }

    def _initialize_pass_implementations(self):
        """Initialize the specific implementations for each pass."""

        self.pass_implementations = {p.pass_type: p.parse for p in self.passes}

    async def parse_video(
        self,
        video_id: str,
        title: str,
        description: str = "",
        tags: str = "",
        channel_name: str = "",
        channel_id: str = "",
        metadata: Optional[Dict] = None,
    ) -> MultiPassResult:
        """Execute multi-pass parsing for a video."""

        if not self.config.enabled:
            # Fall back to basic advanced parser
            basic_result = self.advanced_parser.parse_title(title, description, tags, channel_name)
            return self._create_fallback_result(video_id, title, basic_result)

        start_time = time.time()
        result = MultiPassResult(
            video_id=video_id, original_title=title, budget_consumed={"cpu": 0.0, "api": 0}
        )

        try:
            # Execute the parsing ladder
            await self._execute_parsing_ladder(
                result, title, description, tags, channel_name, channel_id, metadata or {}
            )

            # Update final statistics
            result.total_processing_time = time.time() - start_time
            result.final_confidence = result.final_result.confidence if result.final_result else 0.0

            # Update global statistics
            self._update_statistics(result)

            logger.info(
                f"Multi-pass parsing completed for {video_id}: "
                f"{len(result.passes_attempted)} passes, "
                f"confidence {result.final_confidence:.2f}, "
                f"time {result.total_processing_time:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Multi-pass parsing failed for {video_id}: {e}")
            result.final_result = None
            result.total_processing_time = time.time() - start_time
            return result

    def _has_complete_metadata(self, parse_result) -> bool:
        """Check if parse result has complete metadata (genre, release_year)."""
        if not parse_result or not parse_result.metadata:
            return False

        metadata = parse_result.metadata
        has_genre = metadata.get("genre") or metadata.get("genres")
        has_year = metadata.get("release_year") or metadata.get("year")

        return bool(has_genre and has_year)

    def _merge_parse_metadata(self, base_metadata: Dict, channel_metadata: Dict) -> Dict:
        """Merge channel template metadata with enrichment metadata.

        Preserves all channel template fields (pattern_used, swap_corrected, etc.)
        while adding enrichment fields without overwriting.
        """
        if not channel_metadata:
            return base_metadata or {}

        if not base_metadata:
            return channel_metadata.copy()

        # Start with a copy of base metadata (from enrichment pass)
        merged = base_metadata.copy()

        # Add all channel template fields that aren't already in base
        # These are typically parsing-specific fields we want to preserve
        channel_specific_fields = [
            "pattern_used",
            "pattern_success_rate",
            "pattern_age_days",
            "pattern_last_used",
            "swap_corrected",
            "swap_confidence",
            "karaoke_channel_boost",
            "source_type",
            "method",  # method might come from channel template
        ]

        for field_name in channel_specific_fields:
            if field_name in channel_metadata and field_name not in merged:
                merged[field_name] = channel_metadata[field_name]

        # Also preserve any other fields from channel template that don't conflict
        for key, value in channel_metadata.items():
            if key not in merged:
                merged[key] = value

        # Merge metadata_sources arrays if they exist
        sources = set()
        if "metadata_sources" in base_metadata:
            sources.update(base_metadata["metadata_sources"])
        if "metadata_sources" in channel_metadata:
            sources.update(channel_metadata["metadata_sources"])

        # Add sources based on specific fields present
        if any(
            f in merged for f in ["musicbrainz_recording_id", "musicbrainz_artist_id", "mb_score"]
        ):
            sources.add("musicbrainz")
        if any(f in merged for f in ["discogs_id", "discogs_url", "discogs_genre"]):
            sources.add("discogs")

        if sources:
            merged["metadata_sources"] = list(sources)

        logger.debug(
            f"Merged metadata: {len(channel_metadata)} channel fields + {len(base_metadata)} enrichment fields = {len(merged)} total fields"
        )

        return merged

    async def _execute_parsing_ladder(
        self,
        result: MultiPassResult,
        title: str,
        description: str,
        tags: str,
        channel_name: str,
        channel_id: str,
        metadata: Dict,
    ):
        """Execute the optimized parsing ladder with intelligent flow control."""

        # Intelligent execution flow based on your strategy
        mb_search_result = None

        # Pass 0: Channel Template (Pattern Extrapolation)
        channel_result = await self._execute_pass_if_enabled(
            PassType.CHANNEL_TEMPLATE,
            result,
            title,
            description,
            tags,
            channel_name,
            channel_id,
            metadata,
        )

        if (
            channel_result
            and channel_result.success
            and channel_result.confidence
            >= self.pass_configs[PassType.CHANNEL_TEMPLATE].confidence_threshold
        ):
            # Check if we should skip external searches for very high confidence results
            if channel_result.confidence >= 0.95 and not self.config.always_enrich_metadata:
                # For 95%+ confidence, skip external searches unless we absolutely need metadata
                if self._has_complete_metadata(channel_result.parse_result):
                    result.final_result = channel_result.parse_result
                    result.stopped_at_pass = PassType.CHANNEL_TEMPLATE
                    logger.info(
                        f"Early exit at channel template with confidence {channel_result.confidence:.2f} (complete metadata)"
                    )
                    return
                elif not self.config.require_metadata:
                    # If metadata is not required and confidence is very high, skip external searches
                    result.final_result = channel_result.parse_result
                    result.stopped_at_pass = PassType.CHANNEL_TEMPLATE
                    logger.info(
                        f"Early exit at channel template with high confidence {channel_result.confidence:.2f} (metadata not required)"
                    )
                    return
            elif (
                self._has_complete_metadata(channel_result.parse_result)
                and not self.config.always_enrich_metadata
            ):
                result.final_result = channel_result.parse_result
                result.stopped_at_pass = PassType.CHANNEL_TEMPLATE
                logger.info(
                    f"Early exit at channel template with confidence {channel_result.confidence:.2f} (complete metadata)"
                )
                return
            else:
                # Continue to enrichment but store the successful channel result
                if self._has_complete_metadata(channel_result.parse_result):
                    logger.info(
                        f"Channel template success {channel_result.confidence:.2f} with complete metadata, but always_enrich_metadata=True"
                    )
                else:
                    logger.info(
                        f"Channel template success {channel_result.confidence:.2f} but missing metadata, continuing to enrichment"
                    )
                metadata["channel_template_result"] = channel_result.parse_result
                # If channel template found artist/title, use that for subsequent searches
                if (
                    channel_result.parse_result
                    and channel_result.parse_result.artist
                    and channel_result.parse_result.song_title
                ):
                    metadata["parsed_artist"] = channel_result.parse_result.artist
                    metadata["parsed_title"] = channel_result.parse_result.song_title
        elif (
            channel_result and not channel_result.success
        ):  # Continue to other passes instead of early exit
            logger.debug(
                f"Channel template pass failed for title '{title}': {channel_result.error_message}. Continuing to other passes."
            )

        # Pass 1: MusicBrainz Search (Validation)
        mb_result = await self._execute_pass_if_enabled(
            PassType.MUSICBRAINZ_SEARCH,
            result,
            title,
            description,
            tags,
            channel_name,
            channel_id,
            metadata,
        )

        if mb_result:
            if not mb_result.success:  # Continue to other passes instead of early exit
                logger.debug(
                    f"MusicBrainz search pass failed for title '{title}': {mb_result.error_message}. Continuing to other passes."
                )
            else:
                mb_search_result = mb_result.parse_result
                mb_confidence = mb_result.confidence

                # If MusicBrainz confidence is high enough, try Discogs enrichment before skipping web search
                if (
                    mb_confidence
                    >= self.pass_configs[PassType.MUSICBRAINZ_SEARCH].confidence_threshold
                ):
                    logger.info(
                        f"High MB confidence {mb_confidence:.2f}, attempting Discogs enrichment"
                    )

                    # Try Discogs enrichment for additional metadata (genre, year)
                    discogs_metadata = metadata.copy()
                    discogs_metadata["musicbrainz_result"] = mb_search_result
                    discogs_metadata["musicbrainz_confidence"] = mb_confidence
                    discogs_metadata["enrichment_mode"] = True

                    discogs_result = await self._execute_pass_if_enabled(
                        PassType.DISCOGS_SEARCH,
                        result,
                        title,
                        description,
                        tags,
                        channel_name,
                        channel_id,
                        discogs_metadata,
                    )

                    # Merge results if Discogs provided additional metadata
                    if discogs_result and discogs_result.success and discogs_result.parse_result:
                        from .data_transformer import DataTransformer

                        # Pass metadata directly for merging, not nested
                        mb_data = mb_search_result.metadata if mb_search_result else {}
                        discogs_data = (
                            discogs_result.parse_result.metadata
                            if discogs_result.parse_result
                            else {}
                        )

                        # Merge metadata from both sources
                        merged_metadata = DataTransformer.merge_metadata_sources(
                            mb_data, discogs_data
                        )

                        # Update the MusicBrainz result with enriched metadata
                        if merged_metadata and mb_search_result:
                            if mb_search_result.metadata is None:
                                mb_search_result.metadata = {}
                            mb_search_result.metadata.update(merged_metadata)

                        # Check what we actually merged
                        genre = merged_metadata.get("genre") or merged_metadata.get("genres")
                        year = merged_metadata.get("release_year") or merged_metadata.get("year")
                        logger.info(
                            f"Enriched MB result with Discogs metadata: "
                            f"genre={genre}, year={year}"
                        )

                    # Merge channel template metadata with MB result metadata
                    if metadata.get("channel_template_result") and mb_search_result:
                        channel_template_result = metadata["channel_template_result"]
                        if hasattr(channel_template_result, "metadata") and channel_template_result:
                            channel_template_metadata = (
                                getattr(channel_template_result, "metadata", None) or {}
                            )
                            mb_search_result.metadata = self._merge_parse_metadata(
                                mb_search_result.metadata or {}, channel_template_metadata
                            )

                    result.final_result = mb_search_result
                    result.stopped_at_pass = PassType.MUSICBRAINZ_SEARCH
                    logger.info("Using enriched MB result")

                    # Learn from this successful pattern for future efficiency
                    if mb_search_result:
                        await self._learn_channel_pattern(
                            channel_name, channel_id, title, mb_search_result
                        )
                    return
                else:
                    logger.info(
                        f"MB confidence {mb_confidence:.2f} below threshold, proceeding to Discogs"
                    )

        # Pass 1.5: Discogs Search (if MusicBrainz failed or had low confidence)
        if not (
            mb_result
            and mb_result.success
            and mb_result.confidence
            >= self.pass_configs[PassType.MUSICBRAINZ_SEARCH].confidence_threshold
        ):
            discogs_metadata = metadata.copy()
            if mb_result and mb_result.parse_result:
                discogs_metadata["musicbrainz_result"] = mb_result.parse_result
                discogs_metadata["musicbrainz_confidence"] = mb_result.confidence

            discogs_result = await self._execute_pass_if_enabled(
                PassType.DISCOGS_SEARCH,
                result,
                title,
                description,
                tags,
                channel_name,
                channel_id,
                discogs_metadata,
            )

            if discogs_result and discogs_result.success:
                discogs_confidence = discogs_result.confidence

                # If Discogs confidence is high enough, use it and skip web search
                if (
                    discogs_confidence
                    >= self.pass_configs[PassType.DISCOGS_SEARCH].confidence_threshold
                ):
                    # Merge channel template metadata with Discogs result metadata
                    if metadata.get("channel_template_result") and discogs_result.parse_result:
                        channel_template_result = metadata["channel_template_result"]
                        if hasattr(channel_template_result, "metadata") and channel_template_result:
                            channel_template_metadata = (
                                getattr(channel_template_result, "metadata", None) or {}
                            )
                            discogs_result.parse_result.metadata = self._merge_parse_metadata(
                                discogs_result.parse_result.metadata or {},
                                channel_template_metadata,
                            )

                    result.final_result = discogs_result.parse_result
                    result.stopped_at_pass = PassType.DISCOGS_SEARCH
                    logger.info(f"High Discogs confidence {discogs_confidence:.2f}")
                    return

        # Check if we should use the Channel Template result with any enrichment data
        if metadata.get("channel_template_result") and not result.final_result:
            channel_template_result = metadata["channel_template_result"]

            # Try to enrich with any metadata we found from MusicBrainz/Discogs attempts
            enriched_result = channel_template_result
            metadata_enriched = False

            # Check if we got any metadata from MusicBrainz or Discogs attempts
            for pass_result in result.passes_attempted:
                if pass_result.parse_result and pass_result.parse_result.metadata:
                    metadata_found = pass_result.parse_result.metadata
                    if (
                        metadata_found.get("genre")
                        or metadata_found.get("genres")
                        or metadata_found.get("release_year")
                        or metadata_found.get("year")
                    ):
                        # Merge the metadata into the channel template result
                        if enriched_result:
                            if enriched_result.metadata is None:
                                enriched_result.metadata = {}
                            enriched_result.metadata.update(metadata_found)
                        metadata_enriched = True
                        logger.info(
                            f"Enriched channel template result with metadata from {pass_result.pass_type.value}"
                        )
                        break

            # If we still don't have complete metadata, try a final metadata enrichment attempt
            if (
                not metadata_enriched
                and enriched_result
                and enriched_result.artist
                and enriched_result.song_title
            ):
                final_metadata = await self._try_final_metadata_enrichment(
                    enriched_result.artist, enriched_result.song_title, metadata
                )
                if final_metadata:
                    if enriched_result.metadata is None:
                        enriched_result.metadata = {}
                    enriched_result.metadata.update(final_metadata)
                    metadata_enriched = True
                    logger.info("Final metadata enrichment successful")

            # Boost confidence if we successfully enriched the result
            if metadata_enriched and enriched_result and hasattr(enriched_result, "confidence"):
                enriched_result.confidence = min(enriched_result.confidence * 1.1, 0.95)

            result.final_result = enriched_result
            result.stopped_at_pass = PassType.CHANNEL_TEMPLATE
            status = "enriched" if metadata_enriched else "fallback"
            logger.info(
                f"Using {status} channel template result with confidence {enriched_result.confidence if enriched_result and hasattr(enriched_result, 'confidence') else 0.0:.2f}"
            )
            return

        # Fallback passes if everything above failed
        fallback_passes = [PassType.ML_EMBEDDING, PassType.AUTO_RETEMPLATE]

        for pass_type in fallback_passes:
            fallback_result = await self._execute_pass_if_enabled(
                pass_type, result, title, description, tags, channel_name, channel_id, metadata
            )

            if fallback_result:
                if (
                    not fallback_result.success
                ):  # Continue to next fallback pass instead of early exit
                    logger.debug(
                        f"Fallback pass {pass_type.value} failed for title '{title}': {fallback_result.error_message}. Continuing to next pass."
                    )
                    continue

                if fallback_result.confidence >= self.pass_configs[pass_type].confidence_threshold:
                    # Merge channel template metadata with fallback result metadata
                    if metadata.get("channel_template_result") and fallback_result.parse_result:
                        channel_template_result = metadata["channel_template_result"]
                        if hasattr(channel_template_result, "metadata") and channel_template_result:
                            channel_template_metadata = (
                                getattr(channel_template_result, "metadata", None) or {}
                            )
                            fallback_result.parse_result.metadata = self._merge_parse_metadata(
                                fallback_result.parse_result.metadata or {},
                                channel_template_metadata,
                            )

                    result.final_result = fallback_result.parse_result
                    result.stopped_at_pass = pass_type
                    logger.info(
                        f"Fallback success at {pass_type.value} with confidence {fallback_result.confidence:.2f}"
                    )
                    return

        # If we get here, use the best result from any successful pass
        if not result.final_result and result.passes_attempted:
            # Filter out passes with 0 confidence before finding the best
            successful_passes = [p for p in result.passes_attempted if p.confidence > 0]
            if successful_passes:
                best_pass = max(successful_passes, key=lambda p: p.confidence)
                if best_pass.parse_result:
                    # Merge channel template metadata if best pass is not channel template
                    if best_pass.pass_type != PassType.CHANNEL_TEMPLATE and metadata.get(
                        "channel_template_result"
                    ):
                        channel_template_result = metadata["channel_template_result"]
                        if hasattr(channel_template_result, "metadata") and channel_template_result:
                            channel_template_metadata = (
                                getattr(channel_template_result, "metadata", None) or {}
                            )
                            best_pass.parse_result.metadata = self._merge_parse_metadata(
                                best_pass.parse_result.metadata or {}, channel_template_metadata
                            )

                    result.final_result = best_pass.parse_result
                    result.stopped_at_pass = best_pass.pass_type
                    logger.info(f"Using best available result from {best_pass.pass_type.value}")

    async def _execute_single_pass(
        self,
        pass_type: PassType,
        pass_config: MultiPassPassConfig,
        title: str,
        description: str,
        tags: str,
        channel_name: str,
        channel_id: str,
        metadata: Dict,
    ) -> PassResult:
        """Execute a single parsing pass with retry logic."""

        @retry(
            stop=stop_after_attempt(pass_config.max_retries + 1),
            wait=wait_exponential(
                multiplier=pass_config.exponential_backoff_base,
                max=pass_config.exponential_backoff_max,
            ),
        )
        async def _execute_with_retry():
            start_time = time.time()

            try:
                # Find the pass instance
                pass_instance = next((p for p in self.passes if p.pass_type == pass_type), None)
                if not pass_instance:
                    raise ValueError(f"Pass implementation not found for {pass_type.value}")

                parse_result = await asyncio.wait_for(
                    pass_instance.parse(
                        title, description, tags, channel_name, channel_id, metadata
                    ),
                    timeout=pass_config.timeout_seconds,
                )

                processing_time = time.time() - start_time

                # Update statistics
                self.statistics["passes_attempted"][pass_type] += 1
                # A pass is considered successful if it returns a result with confidence >= its configured threshold
                if parse_result and parse_result.confidence >= pass_config.confidence_threshold:
                    self.statistics["passes_successful"][pass_type] += 1

                return PassResult(
                    pass_type=pass_type,
                    parse_result=parse_result,
                    confidence=parse_result.confidence if parse_result else 0.0,
                    processing_time=processing_time,
                    success=parse_result is not None
                    and parse_result.confidence >= pass_config.confidence_threshold,
                    metadata={
                        "cpu_time": processing_time,
                        "api_calls": 0,  # Will be updated by specific implementations
                    },
                )

            except asyncio.TimeoutError:
                processing_time = time.time() - start_time
                logger.warning(f"Pass {pass_type.value} timed out after {processing_time:.2f}s")

                return PassResult(
                    pass_type=pass_type,
                    processing_time=processing_time,
                    error_message=f"Timeout after {pass_config.timeout_seconds}s",
                )

            except Exception as e:
                processing_time = time.time() - start_time
                logger.error(f"Pass {pass_type.value} failed: {e}")

                return PassResult(
                    pass_type=pass_type, processing_time=processing_time, error_message=str(e)
                )

        return await _execute_with_retry()

    def _check_budget_exceeded(
        self,
        result: MultiPassResult,
        pass_config: MultiPassPassConfig,
        current_pass_type: PassType,  # Add current_pass_type to the signature
    ) -> bool:
        """Check if executing this pass would exceed budget limits."""

        # Calculate potential CPU consumption if this pass runs
        potential_cpu_consumption = result.budget_consumed["cpu"] + pass_config.cpu_budget_limit
        if potential_cpu_consumption > self.config.total_cpu_budget:
            logger.debug(
                f"CPU budget exceeded for {current_pass_type.value}: "
                f"Current {result.budget_consumed['cpu']:.2f} + Pass {pass_config.cpu_budget_limit:.2f} > Total {self.config.total_cpu_budget:.2f}"
            )
            return True

        # Calculate potential API consumption if this pass runs
        potential_api_consumption = result.budget_consumed["api"] + pass_config.api_budget_limit
        if potential_api_consumption > self.config.total_api_budget:
            logger.debug(
                f"API budget exceeded for {current_pass_type.value}: "
                f"Current {result.budget_consumed['api']} + Pass {pass_config.api_budget_limit} > Total {self.config.total_api_budget}"
            )
            return True

        return False

    def _create_fallback_result(
        self, video_id: str, title: str, parse_result: ParseResult
    ) -> MultiPassResult:
        """Create a fallback result when multi-pass is disabled."""

        return MultiPassResult(
            video_id=video_id,
            original_title=title,
            final_result=parse_result,
            final_confidence=parse_result.confidence,
            total_processing_time=0.0,
            budget_consumed={"cpu": 0.0, "api": 0},
        )

    def _update_statistics(self, result: MultiPassResult):
        """Update global processing statistics."""

        self.statistics["total_videos_processed"] += 1

        # Update average processing time
        total_videos = self.statistics["total_videos_processed"]
        if total_videos == 1:
            self.statistics["average_processing_time"] = result.total_processing_time
        else:
            alpha = 0.1  # Smoothing factor
            self.statistics["average_processing_time"] = (
                alpha * result.total_processing_time
                + (1 - alpha) * self.statistics["average_processing_time"]
            )

        # Update budget consumption
        self.statistics["budget_consumption"]["cpu_total"] += result.budget_consumed["cpu"]
        self.statistics["budget_consumption"]["api_total"] += result.budget_consumed["api"]

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics for the multi-pass system."""

        stats = dict(self.statistics)

        # Add success rates
        stats["success_rates"] = {}
        for pass_type in PassType:
            attempted = self.statistics["passes_attempted"][pass_type]
            successful = self.statistics["passes_successful"][pass_type]
            stats["success_rates"][pass_type.value] = (
                successful / attempted if attempted > 0 else 0.0
            )

        # Add budget efficiency
        total_videos = self.statistics["total_videos_processed"]
        if total_videos > 0:
            stats["budget_efficiency"] = {
                "avg_cpu_per_video": self.statistics["budget_consumption"]["cpu_total"]
                / total_videos,
                "avg_api_per_video": self.statistics["budget_consumption"]["api_total"]
                / total_videos,
            }

        return stats

    async def _execute_pass_if_enabled(
        self,
        pass_type: PassType,
        result: MultiPassResult,
        title: str,
        description: str,
        tags: str,
        channel_name: str,
        channel_id: str,
        metadata: Dict,
    ) -> Optional[PassResult]:
        """Execute a single pass if enabled and within budget."""

        pass_config = self.pass_configs.get(pass_type)

        # Skip disabled passes
        if not pass_config or not pass_config.enabled:
            return None

        # Check if we've exceeded global budgets
        if self._check_budget_exceeded(result, pass_config, pass_type):
            logger.info(f"Skipping {pass_type.value} due to budget constraints")
            return None

        # Execute the pass
        pass_result = await self._execute_single_pass(
            pass_type, pass_config, title, description, tags, channel_name, channel_id, metadata
        )

        result.passes_attempted.append(pass_result)

        # Update budget consumption
        result.budget_consumed["cpu"] += pass_result.metadata.get("cpu_time", 0.0)
        result.budget_consumed["api"] += pass_result.metadata.get("api_calls", 0)

        return pass_result

    async def _learn_channel_pattern(
        self,
        channel_name: str,
        channel_id: str,
        title: str,
        successful_result: ParseResult,
    ):
        """Learn channel-specific patterns from successful high-confidence parses."""

        if not channel_id or not successful_result:
            return

        try:
            # Use the existing channel template pass learning functionality
            # Note: This would need to be implemented in the channel template pass
            # For now, just log that we would learn from this success
            if self.channel_template_pass:
                # TODO: Implement learn_from_success in channel template pass
                # self.channel_template_pass.learn_from_success(
                #     channel_id=channel_id,
                #     channel_name=channel_name,
                #     title=title,
                #     result=successful_result,
                # )
                logger.info(
                    f"Would learn new pattern for channel {channel_name} from title: {title}"
                )

        except Exception as e:
            logger.warning(f"Failed to learn channel pattern: {e}")
            # Non-critical, continue processing

    async def _try_final_metadata_enrichment(
        self, artist: str, song_title: str, metadata: Dict
    ) -> Optional[Dict]:
        """Try final metadata enrichment using simplified approaches."""
        try:
            # Strategy 1: Try a simplified MusicBrainz search with just artist + song
            mb_pass = next(
                (p for p in self.passes if p.pass_type == PassType.MUSICBRAINZ_SEARCH), None
            )
            if mb_pass:
                simple_query = f"{artist} - {song_title}"
                logger.debug(f"Trying final MB enrichment with: {simple_query}")

                mb_result = await mb_pass.parse(simple_query, "", "", "", "", metadata)

                if mb_result and mb_result.metadata:
                    mb_metadata = mb_result.metadata
                    extracted_metadata = {}

                    # Extract year and genre if available
                    if mb_metadata.get("release_year") or mb_metadata.get("year"):
                        extracted_metadata["release_year"] = mb_metadata.get(
                            "release_year"
                        ) or mb_metadata.get("year")

                    if mb_metadata.get("genre") or mb_metadata.get("genres"):
                        extracted_metadata["genre"] = mb_metadata.get("genre") or mb_metadata.get(
                            "genres"
                        )

                    if extracted_metadata:
                        logger.info(f"Final MB enrichment found: {extracted_metadata}")
                        return extracted_metadata

            # Strategy 2: Try Discogs with simplified search
            discogs_pass = next(
                (p for p in self.passes if p.pass_type == PassType.DISCOGS_SEARCH), None
            )
            if discogs_pass:
                # Create a simple metadata context for Discogs
                discogs_metadata = {
                    "artist": artist,
                    "song_title": song_title,
                    "enrichment_mode": True,
                }

                logger.debug(f"Trying final Discogs enrichment for: {artist} - {song_title}")

                discogs_result = await discogs_pass.parse(
                    f"{artist} - {song_title}", "", "", "", "", discogs_metadata
                )

                if discogs_result and discogs_result.metadata:
                    discogs_data = discogs_result.metadata
                    extracted_metadata = {}

                    # Extract year and genre if available
                    if discogs_data.get("release_year") or discogs_data.get("year"):
                        extracted_metadata["release_year"] = discogs_data.get(
                            "release_year"
                        ) or discogs_data.get("year")

                    if discogs_data.get("genre") or discogs_data.get("genres"):
                        extracted_metadata["genre"] = discogs_data.get("genre") or discogs_data.get(
                            "genres"
                        )

                    if extracted_metadata:
                        logger.info(f"Final Discogs enrichment found: {extracted_metadata}")
                        return extracted_metadata

            # Strategy 3: Use heuristics based on artist name for genre estimation
            genre_estimate = self._estimate_genre_from_artist(artist)
            if genre_estimate:
                logger.info(f"Genre estimation for {artist}: {genre_estimate}")
                return {"genre": genre_estimate}

        except Exception as e:
            logger.warning(f"Final metadata enrichment failed: {e}")

        return None

    def _estimate_genre_from_artist(self, artist: str) -> Optional[str]:
        """Estimate genre based on known artist patterns."""
        if not artist:
            return None

        artist_lower = artist.lower()

        # Country/Folk indicators
        country_indicators = [
            "kenny rogers",
            "morgan wallen",
            "chris stapleton",
            "carrie underwood",
            "keith urban",
            "blake shelton",
            "miranda lambert",
            "luke bryan",
            "florida georgia line",
            "little big town",
            "lady antebellum",
        ]

        # Hip Hop indicators
        hip_hop_indicators = [
            "kendrick lamar",
            "drake",
            "j. cole",
            "future",
            "migos",
            "cardi b",
            "post malone",
            "travis scott",
            "lil",
            "big sean",
            "chance the rapper",
        ]

        # Rock indicators
        rock_indicators = [
            "linkin park",
            "foo fighters",
            "green day",
            "red hot chili peppers",
            "coldplay",
            "imagine dragons",
            "one republic",
            "maroon 5",
            "fall out boy",
        ]

        # Pop indicators
        pop_indicators = [
            "taylor swift",
            "ariana grande",
            "dua lipa",
            "olivia rodrigo",
            "billie eilish",
            "the weeknd",
            "ed sheeran",
            "justin bieber",
            "selena gomez",
        ]

        # R&B/Soul indicators
        rnb_indicators = [
            "beyonce",
            "rihanna",
            "alicia keys",
            "john legend",
            "usher",
            "bruno mars",
            "the weeknd",
            "sza",
            "frank ocean",
            "miguel",
        ]

        # Check for matches
        for indicators, genre in [
            (country_indicators, "Folk, World, & Country"),
            (hip_hop_indicators, "Hip Hop"),
            (rock_indicators, "Rock"),
            (pop_indicators, "Pop"),
            (rnb_indicators, "Funk / Soul"),
        ]:
            for indicator in indicators:
                if indicator in artist_lower:
                    return genre

        # Check for partial matches with common suffixes
        if any(word in artist_lower for word in ["country", "nashville"]):
            return "Folk, World, & Country"
        elif any(word in artist_lower for word in ["rap", "hip", "hop"]):
            return "Hip Hop"
        elif any(word in artist_lower for word in ["rock", "metal", "punk"]):
            return "Rock"

        return None
