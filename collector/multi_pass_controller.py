"""Multi-pass parsing ladder controller with confidence-based progression and intelligent stopping."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from .advanced_parser import AdvancedTitleParser, ParseResult
from .config import MultiPassConfig, MultiPassPassConfig
from .enhanced_search import MultiStrategySearchEngine
from .passes.acoustic_fingerprint_pass import AcousticFingerprintPass
from .passes.auto_retemplate_pass import AutoRetemplatePass
from .passes.channel_template_pass import EnhancedChannelTemplatePass
from .passes.ml_embedding_pass import EnhancedMLEmbeddingPass
from .passes.web_search_pass import EnhancedWebSearchPass

logger = logging.getLogger(__name__)


class PassType(Enum):
    """Multi-pass parsing ladder pass types."""

    CHANNEL_TEMPLATE = "channel_template"  # Pass 0: Existing channel-template match
    AUTO_RETEMPLATE = "auto_retemplate"  # Pass 1: Auto-re-template on recent uploads
    ML_EMBEDDING = "ml_embedding"  # Pass 2: Light ML / embedding nearest-neighbour
    WEB_SEARCH = "web_search"  # Pass 3: Web search with filler-stripped query
    ACOUSTIC_FINGERPRINT = "acoustic_fingerprint"  # Pass 4: Optional acoustic fingerprint batch


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
        advanced_parser: AdvancedTitleParser,
        search_engine: Optional[MultiStrategySearchEngine] = None,
        db_manager=None,
    ):
        self.config = config
        self.advanced_parser = advanced_parser
        self.search_engine = search_engine
        self.db_manager = db_manager

        # Create pass type to config mapping
        self.pass_configs = {
            PassType.CHANNEL_TEMPLATE: config.channel_template,
            PassType.AUTO_RETEMPLATE: config.auto_retemplate,
            PassType.ML_EMBEDDING: config.ml_embedding,
            PassType.WEB_SEARCH: config.web_search,
            PassType.ACOUSTIC_FINGERPRINT: config.acoustic_fingerprint,
        }

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

        # Initialize pass implementations
        self.channel_template_pass = EnhancedChannelTemplatePass(advanced_parser, db_manager)
        self.auto_retemplate_pass = AutoRetemplatePass(advanced_parser, db_manager)

        # Initialize fuzzy matcher for ML embedding pass
        from .search.fuzzy_matcher import FuzzyMatcher

        fuzzy_matcher = FuzzyMatcher(config.dict() if hasattr(config, "__dataclass_fields__") else {})

        self.ml_embedding_pass = EnhancedMLEmbeddingPass(advanced_parser, fuzzy_matcher, db_manager)

        if search_engine:
            self.web_search_pass = EnhancedWebSearchPass(advanced_parser, search_engine, db_manager)
        else:
            self.web_search_pass = None

        self.acoustic_fingerprint_pass = AcousticFingerprintPass(advanced_parser, db_manager)

        self._initialize_pass_implementations()

    def _initialize_pass_implementations(self):
        """Initialize the specific implementations for each pass."""

        # Pass implementations will be loaded here
        # For now, we'll use placeholders that delegate to existing systems
        self.pass_implementations = {
            PassType.CHANNEL_TEMPLATE: self._pass_0_channel_template,
            PassType.AUTO_RETEMPLATE: self._pass_1_auto_retemplate,
            PassType.ML_EMBEDDING: self._pass_2_ml_embedding,
            PassType.WEB_SEARCH: self._pass_3_web_search,
            PassType.ACOUSTIC_FINGERPRINT: self._pass_4_acoustic_fingerprint,
        }

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
        """Execute the parsing ladder with confidence-based stopping."""

        for pass_type in PassType:
            pass_config = self.pass_configs.get(pass_type)

            # Skip disabled passes
            if not pass_config or not pass_config.enabled:
                continue

            # Check if we've exceeded global budgets
            if self._check_budget_exceeded(result, pass_config):
                logger.info(f"Skipping {pass_type.value} due to budget constraints")
                break

            # Execute the pass
            pass_result = await self._execute_single_pass(
                pass_type, pass_config, title, description, tags, channel_name, channel_id, metadata
            )

            result.passes_attempted.append(pass_result)

            # Update budget consumption
            result.budget_consumed["cpu"] += pass_result.metadata.get("cpu_time", 0.0)
            result.budget_consumed["api"] += pass_result.metadata.get("api_calls", 0)

            # Check if this pass was successful and meets threshold
            if pass_result.success and pass_result.confidence >= pass_config.confidence_threshold:

                result.final_result = pass_result.parse_result
                result.stopped_at_pass = pass_type

                logger.info(
                    f"Parsing stopped at {pass_type.value} with confidence {pass_result.confidence:.2f}"
                )

                if self.config.stop_on_first_success:
                    break

        # If no pass succeeded, use the best result
        if not result.final_result and result.passes_attempted:
            best_pass = max(result.passes_attempted, key=lambda p: p.confidence)
            if best_pass.parse_result:
                result.final_result = best_pass.parse_result
                result.stopped_at_pass = best_pass.pass_type

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
                # Execute the specific pass implementation
                implementation = self.pass_implementations[pass_type]
                parse_result = await asyncio.wait_for(
                    implementation(title, description, tags, channel_name, channel_id, metadata),
                    timeout=pass_config.timeout_seconds,
                )

                processing_time = time.time() - start_time

                # Update statistics
                self.statistics["passes_attempted"][pass_type] += 1
                if parse_result and parse_result.confidence > 0:
                    self.statistics["passes_successful"][pass_type] += 1

                return PassResult(
                    pass_type=pass_type,
                    parse_result=parse_result,
                    confidence=parse_result.confidence if parse_result else 0.0,
                    processing_time=processing_time,
                    success=parse_result is not None and parse_result.confidence > 0,
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
        self, result: MultiPassResult, pass_config: MultiPassPassConfig
    ) -> bool:
        """Check if executing this pass would exceed budget limits."""

        # Check CPU budget
        if (
            result.budget_consumed["cpu"] + pass_config.cpu_budget_limit
            > self.config.total_cpu_budget
        ):
            return True

        # Check API budget
        if (
            result.budget_consumed["api"] + pass_config.api_budget_limit
            > self.config.total_api_budget
        ):
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

    # Pass implementation placeholders - will be implemented in subsequent phases

    async def _pass_0_channel_template(
        self,
        title: str,
        description: str,
        tags: str,
        channel_name: str,
        channel_id: str,
        metadata: Dict,
    ) -> Optional[ParseResult]:
        """Pass 0: Enhanced channel-template match."""

        return self.channel_template_pass.parse(
            title, description, tags, channel_name, channel_id, metadata
        )

    async def _pass_1_auto_retemplate(
        self,
        title: str,
        description: str,
        tags: str,
        channel_name: str,
        channel_id: str,
        metadata: Dict,
    ) -> Optional[ParseResult]:
        """Pass 1: Auto-re-template on recent uploads."""

        return await self.auto_retemplate_pass.parse(
            title, description, tags, channel_name, channel_id, metadata
        )

    async def _pass_2_ml_embedding(
        self,
        title: str,
        description: str,
        tags: str,
        channel_name: str,
        channel_id: str,
        metadata: Dict,
    ) -> Optional[ParseResult]:
        """Pass 2: Light ML / embedding nearest-neighbour."""

        return await self.ml_embedding_pass.parse(
            title, description, tags, channel_name, channel_id, metadata
        )

    async def _pass_3_web_search(
        self,
        title: str,
        description: str,
        tags: str,
        channel_name: str,
        channel_id: str,
        metadata: Dict,
    ) -> Optional[ParseResult]:
        """Pass 3: Web search with filler-stripped query."""

        if not self.web_search_pass:
            return None

        return await self.web_search_pass.parse(
            title, description, tags, channel_name, channel_id, metadata
        )

    async def _pass_4_acoustic_fingerprint(
        self,
        title: str,
        description: str,
        tags: str,
        channel_name: str,
        channel_id: str,
        metadata: Dict,
    ) -> Optional[ParseResult]:
        """Pass 4: Optional acoustic fingerprint batch."""

        return await self.acoustic_fingerprint_pass.parse(
            title, description, tags, channel_name, channel_id, metadata
        )
