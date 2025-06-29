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
            PassType.WEB_SEARCH: config.web_search,
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
        web_search_result = None

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
            result.final_result = channel_result.parse_result
            result.stopped_at_pass = PassType.CHANNEL_TEMPLATE
            logger.info(
                f"Early exit at channel template with confidence {channel_result.confidence:.2f}"
            )
            return
        elif channel_result and not channel_result.success:  # Early exit on failure
            logger.warning(
                f"Early exit due to failed channel template pass: {channel_result.error_message}"
            )
            return

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
            if not mb_result.success:  # Early exit on failure
                logger.warning(
                    f"Early exit due to failed MusicBrainz search pass: {mb_result.error_message}"
                )
                return
            mb_search_result = mb_result.parse_result
            mb_confidence = mb_result.confidence

            # If MusicBrainz confidence is high enough, skip web search
            if mb_confidence >= self.pass_configs[PassType.MUSICBRAINZ_SEARCH].confidence_threshold:
                result.final_result = mb_search_result
                result.stopped_at_pass = PassType.MUSICBRAINZ_SEARCH
                logger.info(f"High MB confidence {mb_confidence:.2f}, skipping web search")

                # Learn from this successful pattern for future efficiency
                if mb_search_result:
                    await self._learn_channel_pattern(
                        channel_name, channel_id, title, mb_search_result
                    )
                return
            else:
                logger.info(
                    f"MB confidence {mb_confidence:.2f} below threshold, proceeding to web search"
                )

        # Pass 2: Web Search (only if MusicBrainz confidence was low)
        web_result = await self._execute_pass_if_enabled(
            PassType.WEB_SEARCH,
            result,
            title,
            description,
            tags,
            channel_name,
            channel_id,
            metadata,
        )

        if web_result:
            if not web_result.success:  # Early exit on failure
                logger.warning(
                    f"Early exit due to failed Web search pass: {web_result.error_message}"
                )
                return
            web_search_result = web_result.parse_result

            # Pass 3: MusicBrainz Validation (enrich web search results)
            validation_metadata = metadata.copy()
            validation_metadata["web_search_result"] = web_search_result

            validation_result = await self._execute_pass_if_enabled(
                PassType.MUSICBRAINZ_VALIDATION,
                result,
                title,
                description,
                tags,
                channel_name,
                channel_id,
                validation_metadata,
            )

            if validation_result:
                if not validation_result.success:  # Early exit on failure
                    logger.warning(
                        f"Early exit due to failed MusicBrainz validation pass: {validation_result.error_message}"
                    )
                    # If validation failed, but web search was successful, use web search result
                    if web_search_result:
                        result.final_result = web_search_result
                        result.stopped_at_pass = PassType.WEB_SEARCH
                        logger.info("Using web search result (validation failed)")
                    return

                if validation_result.parse_result:
                    result.final_result = validation_result.parse_result
                    result.stopped_at_pass = PassType.MUSICBRAINZ_VALIDATION

                    # Learn from successful web search + validation pattern
                    if validation_result.confidence >= 0.8:
                        await self._learn_channel_pattern(
                            channel_name, channel_id, title, validation_result.parse_result
                        )

                    logger.info(
                        f"Web search + validation completed with confidence {validation_result.confidence:.2f}"
                    )
                    return
            else:
                # Use web search result if validation was not even attempted or failed to return a result
                result.final_result = web_search_result
                result.stopped_at_pass = PassType.WEB_SEARCH
                logger.info("Using web search result (validation failed)")
                return

        # Fallback passes if everything above failed
        fallback_passes = [PassType.ML_EMBEDDING, PassType.AUTO_RETEMPLATE]

        for pass_type in fallback_passes:
            fallback_result = await self._execute_pass_if_enabled(
                pass_type, result, title, description, tags, channel_name, channel_id, metadata
            )

            if fallback_result:
                if not fallback_result.success:  # Early exit on failure
                    logger.warning(
                        f"Early exit due to failed fallback pass {pass_type.value}: {fallback_result.error_message}"
                    )
                    return

                if fallback_result.confidence >= self.pass_configs[pass_type].confidence_threshold:
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
            if self.channel_template_pass and hasattr(
                self.channel_template_pass, "_learn_from_success"
            ):
                self.channel_template_pass._learn_from_success(
                    channel_id=channel_id,
                    channel_name=channel_name,
                    title=title,
                    result=successful_result,
                )

                logger.info(f"Learned new pattern for channel {channel_name} from title: {title}")

        except Exception as e:
            logger.warning(f"Failed to learn channel pattern: {e}")
            # Non-critical, continue processing
