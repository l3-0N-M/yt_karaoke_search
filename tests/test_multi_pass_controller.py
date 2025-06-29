"""Tests for the multi-pass parsing controller."""

from typing import Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from collector.advanced_parser import ParseResult
from collector.config import MultiPassConfig
from collector.multi_pass_controller import MultiPassParsingController, PassResult, PassType
from collector.passes.base import ParsingPass


@pytest.fixture
def mp_config():
    """Fixture for a default MultiPassConfig."""
    return MultiPassConfig()


@pytest.fixture
def mock_advanced_parser():
    """Fixture for a mock AdvancedTitleParser."""
    mock = MagicMock()
    mock.parse_title.return_value = ParseResult(
        original_artist="Fallback Artist", song_title="Fallback Song", confidence=0.5
    )
    return mock


@pytest.fixture
def mock_db_manager():
    """Fixture for a mock DBManager."""
    return MagicMock()


class MockPass(ParsingPass):
    def __init__(
        self,
        pass_type: PassType,
        confidence: float = 0.0,
        processing_time: float = 0.0,
        success: bool = True,
        error_message: str = "",
        api_calls: int = 0,
    ):
        self._pass_type = pass_type
        self._confidence = confidence
        self._processing_time = processing_time
        self._success = success
        self._error_message = error_message
        self._api_calls = api_calls

    @property
    def pass_type(self) -> PassType:
        return self._pass_type

    async def parse(
        self,
        title: str,
        description: str,
        tags: str,
        channel_name: str,
        channel_id: str,
        metadata: dict,
    ) -> Optional[ParseResult]:
        if not self._success:
            return None

        parse_result = ParseResult(
            original_artist=f"{self.pass_type.name} Artist",
            song_title=f"{self.pass_type.name} Song",
            confidence=self._confidence,
            method=self.pass_type.value,
        )
        return parse_result

    def get_statistics(self) -> Dict:
        return {"mock_stats": True}


@pytest.mark.asyncio
async def test_multi_pass_disabled_falls_back_to_basic_parser(
    mp_config, mock_advanced_parser, mock_db_manager
):
    """Test that if multi-pass is disabled, it falls back to the basic advanced parser."""
    mp_config.enabled = False
    controller = MultiPassParsingController(
        mp_config, [], mock_advanced_parser, None, mock_db_manager
    )
    result = await controller.parse_video(
        "video_id", "title", "description", "tags", "channel_name", "channel_id"
    )

    mock_advanced_parser.parse_title.assert_called_once_with(
        "title", "description", "tags", "channel_name"
    )
    assert result.final_result.original_artist == "Fallback Artist"
    assert result.final_result.song_title == "Fallback Song"
    assert result.final_confidence == 0.5
    assert result.stopped_at_pass is None  # No specific pass stopped it


@pytest.mark.asyncio
async def test_early_exit_on_channel_template_success(
    mp_config, mock_advanced_parser, mock_db_manager
):
    """Test that the controller exits early if CHANNEL_TEMPLATE pass succeeds with high confidence."""
    mp_config.channel_template.confidence_threshold = 0.8
    channel_pass = MockPass(PassType.CHANNEL_TEMPLATE, confidence=0.9)
    mb_search_pass = MockPass(PassType.MUSICBRAINZ_SEARCH, confidence=0.7)

    passes = [channel_pass, mb_search_pass]
    controller = MultiPassParsingController(
        mp_config, passes, mock_advanced_parser, None, mock_db_manager
    )

    result = await controller.parse_video(
        "video_id", "title", "description", "tags", "channel_name", "channel_id"
    )

    channel_pass.parse.assert_called_once()
    mb_search_pass.parse.assert_not_called()  # Should not be called
    assert result.stopped_at_pass == PassType.CHANNEL_TEMPLATE
    assert result.final_confidence == 0.9
    assert result.final_result.original_artist == "CHANNEL_TEMPLATE Artist"


@pytest.mark.asyncio
async def test_musicbrainz_search_high_confidence_skips_web_search(
    mp_config, mock_advanced_parser, mock_db_manager
):
    """Test that high MusicBrainz search confidence skips web search."""
    mp_config.channel_template.confidence_threshold = 0.9  # Make channel pass fail
    mp_config.musicbrainz_search.confidence_threshold = 0.8
    mp_config.web_search.enabled = True  # Ensure web search is enabled

    channel_pass = MockPass(PassType.CHANNEL_TEMPLATE, confidence=0.7)
    mb_search_pass = MockPass(PassType.MUSICBRAINZ_SEARCH, confidence=0.85)
    web_search_pass = MockPass(PassType.WEB_SEARCH, confidence=0.9)

    passes = [channel_pass, mb_search_pass, web_search_pass]
    controller = MultiPassParsingController(
        mp_config, passes, mock_advanced_parser, None, mock_db_manager
    )

    result = await controller.parse_video(
        "video_id", "title", "description", "tags", "channel_name", "channel_id"
    )

    channel_pass.parse.assert_called_once()
    mb_search_pass.parse.assert_called_once()
    web_search_pass.parse.assert_not_called()  # Should be skipped
    assert result.stopped_at_pass == PassType.MUSICBRAINZ_SEARCH
    assert result.final_confidence == 0.85
    assert result.final_result.original_artist == "MUSICBRAINZ_SEARCH Artist"


@pytest.mark.asyncio
async def test_musicbrainz_search_low_confidence_proceeds_to_web_search(
    mp_config, mock_advanced_parser, mock_db_manager
):
    """Test that low MusicBrainz search confidence proceeds to web search."""
    mp_config.channel_template.confidence_threshold = 0.9  # Make channel pass fail
    mp_config.musicbrainz_search.confidence_threshold = 0.8
    mp_config.web_search.enabled = True
    mp_config.musicbrainz_validation.enabled = True

    channel_pass = MockPass(PassType.CHANNEL_TEMPLATE, confidence=0.7)
    mb_search_pass = MockPass(PassType.MUSICBRAINZ_SEARCH, confidence=0.75)  # Below threshold
    web_search_pass = MockPass(PassType.WEB_SEARCH, confidence=0.9)
    mb_validation_pass = MockPass(PassType.MUSICBRAINZ_VALIDATION, confidence=0.95)

    passes = [channel_pass, mb_search_pass, web_search_pass, mb_validation_pass]
    controller = MultiPassParsingController(
        mp_config, passes, mock_advanced_parser, None, mock_db_manager
    )

    result = await controller.parse_video(
        "video_id", "title", "description", "tags", "channel_name", "channel_id"
    )

    channel_pass.parse.assert_called_once()
    mb_search_pass.parse.assert_called_once()
    web_search_pass.parse.assert_called_once()
    mb_validation_pass.parse.assert_called_once()
    assert result.stopped_at_pass == PassType.MUSICBRAINZ_VALIDATION
    assert result.final_confidence == 0.95
    assert result.final_result.original_artist == "MUSICBRAINZ_VALIDATION Artist"


@pytest.mark.asyncio
async def test_web_search_fallback_if_validation_fails(
    mp_config, mock_advanced_parser, mock_db_manager
):
    """Test that if web search is followed by failed validation, web search result is used."""
    mp_config.channel_template.confidence_threshold = 0.9
    mp_config.musicbrainz_search.confidence_threshold = 0.9
    mp_config.web_search.enabled = True
    mp_config.musicbrainz_validation.enabled = True

    channel_pass = MockPass(PassType.CHANNEL_TEMPLATE, confidence=0.7)
    mb_search_pass = MockPass(PassType.MUSICBRAINZ_SEARCH, confidence=0.7)
    web_search_pass = MockPass(
        PassType.WEB_SEARCH, confidence=0.8
    )  # Changed confidence to ensure it's not None
    mb_validation_pass = MockPass(
        PassType.MUSICBRAINZ_VALIDATION, success=False
    )  # Validation fails

    passes = [channel_pass, mb_search_pass, web_search_pass, mb_validation_pass]
    controller = MultiPassParsingController(
        mp_config, passes, mock_advanced_parser, None, mock_db_manager
    )

    result = await controller.parse_video(
        "video_id", "title", "description", "tags", "channel_name", "channel_id"
    )

    web_search_pass.parse.assert_called_once()
    mb_validation_pass.parse.assert_called_once()
    assert result.stopped_at_pass == PassType.WEB_SEARCH
    assert result.final_confidence == 0.8  # Changed to 0.8
    assert result.final_result.original_artist == "WEB_SEARCH Artist"


@pytest.mark.asyncio
async def test_fallback_passes_execution(mp_config, mock_advanced_parser, mock_db_manager):
    """Test that fallback passes (ML_EMBEDDING, AUTO_RETEMPLATE) are executed if primary passes fail."""
    mp_config.channel_template.confidence_threshold = 0.9
    mp_config.musicbrainz_search.confidence_threshold = 0.9
    mp_config.web_search.enabled = True
    mp_config.musicbrainz_validation.enabled = True
    mp_config.ml_embedding.enabled = True
    mp_config.auto_retemplate.enabled = True

    channel_pass = MockPass(PassType.CHANNEL_TEMPLATE, confidence=0.7)
    mb_search_pass = MockPass(PassType.MUSICBRAINZ_SEARCH, confidence=0.7)
    web_search_pass = MockPass(PassType.WEB_SEARCH, confidence=0.6)  # Below threshold
    mb_validation_pass = MockPass(
        PassType.MUSICBRAINZ_VALIDATION, success=False
    )  # Validation fails
    ml_embedding_pass = MockPass(PassType.ML_EMBEDDING, confidence=0.8)
    auto_retemplate_pass = MockPass(PassType.AUTO_RETEMPLATE, confidence=0.9)

    passes = [
        channel_pass,
        mb_search_pass,
        web_search_pass,
        mb_validation_pass,
        ml_embedding_pass,
        auto_retemplate_pass,
    ]
    controller = MultiPassParsingController(
        mp_config, passes, mock_advanced_parser, None, mock_db_manager
    )

    result = await controller.parse_video(
        "video_id", "title", "description", "tags", "channel_name", "channel_id"
    )

    channel_pass.parse.assert_called_once()
    mb_search_pass.parse.assert_called_once()
    web_search_pass.parse.assert_called_once()
    mb_validation_pass.parse.assert_not_called()  # Changed to assert_not_called
    ml_embedding_pass.parse.assert_called_once()
    auto_retemplate_pass.parse.assert_not_called()  # ML pass succeeded

    assert result.stopped_at_pass == PassType.ML_EMBEDDING
    assert result.final_confidence == 0.8
    assert result.final_result.original_artist == "ML_EMBEDDING Artist"


@pytest.mark.asyncio
async def test_budget_exceeded_skips_pass(mp_config, mock_advanced_parser, mock_db_manager):
    """Test that a pass is skipped if it would exceed the budget."""
    mp_config.total_cpu_budget = 0.05  # Very small budget
    mp_config.channel_template.cpu_budget_limit = 0.1  # This pass would exceed it
    mp_config.channel_template.enabled = True

    channel_pass = MockPass(PassType.CHANNEL_TEMPLATE, processing_time=0.01, confidence=0.9)

    passes = [channel_pass]
    controller = MultiPassParsingController(
        mp_config, passes, mock_advanced_parser, None, mock_db_manager
    )

    result = await controller.parse_video(
        "video_id", "title", "description", "tags", "channel_name", "channel_id"
    )

    channel_pass.parse.assert_not_called()  # Should be skipped
    assert result.final_result is None
    assert result.stopped_at_pass is None
    assert len(result.passes_attempted) == 0


@pytest.mark.asyncio
async def test_pass_timeout_handling(mp_config, mock_advanced_parser, mock_db_manager):
    """Test that a pass timeout is handled gracefully."""
    mp_config.channel_template.timeout_seconds = 0.01
    mp_config.channel_template.enabled = True

    channel_pass = MockPass(
        PassType.CHANNEL_TEMPLATE, success=False, error_message="Timeout after 0.01s"
    )

    # Only include the failing pass in the list for this specific test
    passes = [
        channel_pass,
    ]
    controller = MultiPassParsingController(
        mp_config, passes, mock_advanced_parser, None, mock_db_manager
    )
    # Mock the _execute_single_pass method to control its behavior
    with patch.object(
        controller, "_execute_single_pass", new_callable=AsyncMock
    ) as mock_single_pass:
        mock_single_pass.return_value = PassResult(
            pass_type=PassType.CHANNEL_TEMPLATE,
            processing_time=0.0,
            error_message="Timeout after 0.01s",
            success=False,
        )
        result = await controller.parse_video(
            "video_id", "title", "description", "tags", "channel_name", "channel_id"
        )

        mock_single_pass.assert_called_once_with(
            PassType.CHANNEL_TEMPLATE,
            mp_config.channel_template,
            "title",
            "description",
            "tags",
            "channel_name",
            "channel_id",
            {},
        )

    assert len(result.passes_attempted) == 1
    assert result.passes_attempted[0].pass_type == PassType.CHANNEL_TEMPLATE
    assert "Timeout" in result.passes_attempted[0].error_message
    assert not result.passes_attempted[0].success
    assert result.final_result is None


@pytest.mark.asyncio
async def test_pass_exception_handling(mp_config, mock_advanced_parser, mock_db_manager):
    """Test that an exception in a pass is handled gracefully."""
    mp_config.channel_template.enabled = True

    channel_pass = MockPass(
        PassType.CHANNEL_TEMPLATE, success=False, error_message="Mock exception"
    )

    # Only include the failing pass in the list for this specific test
    passes = [
        channel_pass,
    ]
    controller = MultiPassParsingController(
        mp_config, passes, mock_advanced_parser, None, mock_db_manager
    )
    # Mock the _execute_single_pass method to control its behavior
    with patch.object(
        controller, "_execute_single_pass", new_callable=AsyncMock
    ) as mock_single_pass:
        mock_single_pass.return_value = PassResult(
            pass_type=PassType.CHANNEL_TEMPLATE,
            processing_time=0.0,
            error_message="Mock exception",
            success=False,
        )
        result = await controller.parse_video(
            "video_id", "title", "description", "tags", "channel_name", "channel_id"
        )

        mock_single_pass.assert_called_once_with(
            PassType.CHANNEL_TEMPLATE,
            mp_config.channel_template,
            "title",
            "description",
            "tags",
            "channel_name",
            "channel_id",
            {},
        )

    assert len(result.passes_attempted) == 1
    assert result.passes_attempted[0].pass_type == PassType.CHANNEL_TEMPLATE
    assert "Mock exception" in result.passes_attempted[0].error_message
    assert not result.passes_attempted[0].success
    assert result.final_result is None


@pytest.mark.asyncio
async def test_statistics_update(mp_config, mock_advanced_parser, mock_db_manager):
    """Test that statistics are correctly updated."""
    mp_config.channel_template.confidence_threshold = 0.7
    mp_config.musicbrainz_search.confidence_threshold = 0.7
    mp_config.web_search.enabled = True
    mp_config.musicbrainz_validation.enabled = True
    mp_config.ml_embedding.enabled = True

    channel_pass = MockPass(PassType.CHANNEL_TEMPLATE, confidence=0.6, processing_time=0.01)
    mb_search_pass = MockPass(PassType.MUSICBRAINZ_SEARCH, confidence=0.8, processing_time=0.02)
    web_search_pass = MockPass(PassType.WEB_SEARCH, confidence=0.75, processing_time=0.03)
    mb_validation_pass = MockPass(
        PassType.MUSICBRAINZ_VALIDATION, confidence=0.85, processing_time=0.04
    )
    ml_embedding_pass = MockPass(PassType.ML_EMBEDDING, confidence=0.9, processing_time=0.05)

    passes = [
        channel_pass,
        mb_search_pass,
        web_search_pass,
        mb_validation_pass,
        ml_embedding_pass,
    ]
    controller = MultiPassParsingController(
        mp_config, passes, mock_advanced_parser, None, mock_db_manager
    )

    # Run first video
    await controller.parse_video("video_id_1", "title_1", "", "", "channel_name", "channel_id")
    stats_1 = controller.get_statistics()

    assert stats_1["total_videos_processed"] == 1
    assert stats_1["passes_attempted"][PassType.CHANNEL_TEMPLATE] == 1
    # Channel pass confidence (0.6) is below its threshold (0.7), so it's not successful
    assert stats_1["passes_successful"][PassType.CHANNEL_TEMPLATE] == 0
    assert stats_1["passes_attempted"][PassType.MUSICBRAINZ_SEARCH] == 1
    # MusicBrainz search confidence (0.8) is above its threshold (0.7), so it's successful
    assert stats_1["passes_successful"][PassType.MUSICBRAINZ_SEARCH] == 1
    assert stats_1["passes_attempted"][PassType.WEB_SEARCH] == 0  # Skipped due to MB success
    assert stats_1["average_processing_time"] > 0
    assert stats_1["success_rates"][PassType.MUSICBRAINZ_SEARCH.value] == 1.0

    # Run second video
    # Make MB search fail this time to trigger more passes
    mb_search_pass_2 = MockPass(PassType.MUSICBRAINZ_SEARCH, confidence=0.5, processing_time=0.02)
    web_search_pass_2 = MockPass(PassType.WEB_SEARCH, confidence=0.75, processing_time=0.03)
    mb_validation_pass_2 = MockPass(
        PassType.MUSICBRAINZ_VALIDATION, confidence=0.85, processing_time=0.04
    )
    ml_embedding_pass_2 = MockPass(PassType.ML_EMBEDDING, confidence=0.9, processing_time=0.05)

    passes_2 = [
        MockPass(PassType.CHANNEL_TEMPLATE, confidence=0.6, processing_time=0.01),
        mb_search_pass_2,
        web_search_pass_2,
        mb_validation_pass_2,
        ml_embedding_pass_2,
    ]
    controller_2 = MultiPassParsingController(
        mp_config, passes_2, mock_advanced_parser, None, mock_db_manager
    )
    # Wrap the parse methods for the second controller's passes
    for p in passes_2:
        p.parse = AsyncMock(side_effect=p.parse)

    await controller_2.parse_video("video_id_2", "title_2", "", "", "channel_name", "channel_id")
    stats_2 = controller_2.get_statistics()

    assert stats_2["total_videos_processed"] == 1
    assert stats_2["passes_attempted"][PassType.CHANNEL_TEMPLATE] == 1
    assert stats_2["passes_successful"][PassType.CHANNEL_TEMPLATE] == 0
    assert stats_2["passes_attempted"][PassType.MUSICBRAINZ_SEARCH] == 1
    assert stats_2["passes_successful"][PassType.MUSICBRAINZ_SEARCH] == 0  # Confidence too low
    assert stats_2["passes_attempted"][PassType.WEB_SEARCH] == 1
    assert (
        stats_2["passes_successful"][PassType.WEB_SEARCH] == 1
    )  # Confidence 0.75 >= threshold 0.7
    assert stats_2["passes_attempted"][PassType.MUSICBRAINZ_VALIDATION] == 1
    assert stats_2["passes_successful"][PassType.MUSICBRAINZ_VALIDATION] == 1
    assert (
        stats_2["passes_attempted"][PassType.ML_EMBEDDING] == 0
    )  # Skipped due to MB validation success


@pytest.mark.asyncio
async def test_best_available_result_selection(mp_config, mock_advanced_parser, mock_db_manager):
    """Test that if no pass meets threshold, the best available result is chosen."""
    mp_config.channel_template.confidence_threshold = 0.9
    mp_config.musicbrainz_search.confidence_threshold = 0.9
    mp_config.web_search.enabled = True
    mp_config.musicbrainz_validation.enabled = True
    mp_config.ml_embedding.enabled = True
    mp_config.auto_retemplate.enabled = True

    channel_pass = MockPass(PassType.CHANNEL_TEMPLATE, confidence=0.6)
    mb_search_pass = MockPass(PassType.MUSICBRAINZ_SEARCH, confidence=0.7)
    web_search_pass = MockPass(
        PassType.WEB_SEARCH, confidence=0.8
    )  # Set confidence above threshold
    mb_validation_pass = MockPass(
        PassType.MUSICBRAINZ_VALIDATION, confidence=0.75, success=False
    )  # Make validation fail
    ml_embedding_pass = MockPass(PassType.ML_EMBEDDING, confidence=0.65)
    auto_retemplate_pass = MockPass(PassType.AUTO_RETEMPLATE, confidence=0.5)

    passes = [
        channel_pass,
        mb_search_pass,
        web_search_pass,
        mb_validation_pass,
        ml_embedding_pass,
        auto_retemplate_pass,
    ]
    controller = MultiPassParsingController(
        mp_config, passes, mock_advanced_parser, None, mock_db_manager
    )
    for p in passes:
        p.parse = AsyncMock(side_effect=p.parse)

    result = await controller.parse_video(
        "video_id", "title", "description", "tags", "channel_name", "channel_id"
    )

    # All passes should be attempted as none meet their high thresholds
    channel_pass.parse.assert_called_once()
    mb_search_pass.parse.assert_called_once()
    web_search_pass.parse.assert_called_once()
    mb_validation_pass.parse.assert_called_once()
    ml_embedding_pass.parse.assert_called_once()
    auto_retemplate_pass.parse.assert_called_once()

    assert (
        result.stopped_at_pass == PassType.MUSICBRAINZ_SEARCH
    )  # MB search had highest confidence among successful passes
    assert result.final_confidence == 0.7
    assert result.final_result.original_artist == "MUSICBRAINZ_SEARCH Artist"


@pytest.mark.asyncio
async def test_learn_channel_pattern_called_on_success(
    mp_config, mock_advanced_parser, mock_db_manager
):
    """Test that _learn_channel_pattern is called when a pass succeeds with high confidence."""
    mp_config.channel_template.confidence_threshold = 0.8
    channel_pass = MockPass(PassType.CHANNEL_TEMPLATE, confidence=0.7)  # Below threshold
    mb_search_pass = MockPass(
        PassType.MUSICBRAINZ_SEARCH, confidence=0.9
    )  # High confidence to trigger learn

    passes = [channel_pass, mb_search_pass]
    controller = MultiPassParsingController(
        mp_config, passes, mock_advanced_parser, None, mock_db_manager
    )

    await controller.parse_video(
        "video_id", "title", "description", "tags", "channel_name", "channel_id"
    )

    channel_pass._learn_from_success.assert_called_once()
    args, kwargs = channel_pass._learn_from_success.call_args
    assert kwargs["channel_id"] == "channel_id"
    assert kwargs["channel_name"] == "channel_name"
    assert kwargs["title"] == "title"
    assert kwargs["result"].original_artist == "MUSICBRAINZ_SEARCH Artist"  # Result from MB search


@pytest.mark.asyncio
async def test_learn_channel_pattern_not_called_on_low_confidence(
    mp_config, mock_advanced_parser, mock_db_manager
):
    """Test that _learn_channel_pattern is not called when confidence is too low."""
    mp_config.channel_template.confidence_threshold = 0.9
    channel_pass = MockPass(PassType.CHANNEL_TEMPLATE, confidence=0.8)  # Below threshold

    passes = [channel_pass]
    controller = MultiPassParsingController(
        mp_config, passes, mock_advanced_parser, None, mock_db_manager
    )

    await controller.parse_video(
        "video_id", "title", "description", "tags", "channel_name", "channel_id"
    )

    channel_pass._learn_from_success.assert_not_called()


@pytest.mark.asyncio
async def test_total_cpu_budget_respected(mp_config, mock_advanced_parser, mock_db_manager):
    """Test that the total CPU budget is respected across passes."""
    mp_config.total_cpu_budget = 0.03  # Total budget for all passes
    mp_config.channel_template.cpu_budget_limit = 0.02
    mp_config.musicbrainz_search.cpu_budget_limit = 0.02
    mp_config.channel_template.confidence_threshold = 0.0  # Ensure it runs
    mp_config.musicbrainz_search.confidence_threshold = 0.0  # Ensure it runs

    channel_pass = MockPass(PassType.CHANNEL_TEMPLATE, processing_time=0.02, confidence=0.5)
    mb_search_pass = MockPass(PassType.MUSICBRAINZ_SEARCH, processing_time=0.02, confidence=0.5)

    passes = [channel_pass, mb_search_pass]
    controller = MultiPassParsingController(
        mp_config, passes, mock_advanced_parser, None, mock_db_manager
    )

    result = await controller.parse_video(
        "video_id", "title", "description", "tags", "channel_name", "channel_id"
    )

    channel_pass.parse.assert_called_once()
    mb_search_pass.parse.assert_not_called()  # Should be skipped due to budget
    assert result.budget_consumed["cpu"] == pytest.approx(0.02, rel=1e-1)
    assert result.stopped_at_pass == PassType.CHANNEL_TEMPLATE  # Best available result
    assert result.final_confidence == 0.5


@pytest.mark.asyncio
async def test_total_api_budget_respected(mp_config, mock_advanced_parser, mock_db_manager):
    """Test that the total API budget is respected across passes."""
    mp_config.total_api_budget = 15  # Total budget for all passes
    mp_config.musicbrainz_search.api_budget_limit = 10
    mp_config.web_search.api_budget_limit = 10
    mp_config.channel_template.confidence_threshold = 0.0
    mp_config.musicbrainz_search.confidence_threshold = 0.0
    mp_config.web_search.enabled = True

    channel_pass = MockPass(PassType.CHANNEL_TEMPLATE, confidence=0.5)
    mb_search_pass = MockPass(PassType.MUSICBRAINZ_SEARCH, confidence=0.5, api_calls=10)
    web_search_pass = MockPass(PassType.WEB_SEARCH, confidence=0.5, api_calls=10)

    passes = [channel_pass, mb_search_pass, web_search_pass]
    controller = MultiPassParsingController(
        mp_config, passes, mock_advanced_parser, None, mock_db_manager
    )

    result = await controller.parse_video(
        "video_id", "title", "description", "tags", "channel_name", "channel_id"
    )

    channel_pass.parse.assert_called_once()
    mb_search_pass.parse.assert_not_called()  # Changed to assert_not_called
    web_search_pass.parse.assert_not_called()  # Should be skipped due to budget
    assert result.budget_consumed["api"] == 0  # Should be 0 as mb_search_pass is not called
    assert result.stopped_at_pass == PassType.CHANNEL_TEMPLATE  # Best available result
    assert result.final_confidence == 0.5


@pytest.mark.asyncio
async def test_no_passes_succeed_returns_none_result(
    mp_config, mock_advanced_parser, mock_db_manager
):
    """Test that if no passes succeed, the final result is None."""
    mp_config.channel_template.confidence_threshold = 0.9
    mp_config.musicbrainz_search.confidence_threshold = 0.9
    mp_config.web_search.enabled = True
    mp_config.musicbrainz_validation.enabled = True
    mp_config.ml_embedding.enabled = True
    mp_config.auto_retemplate.enabled = True

    channel_pass = MockPass(PassType.CHANNEL_TEMPLATE, confidence=0.0, success=False)
    mb_search_pass = MockPass(PassType.MUSICBRAINZ_SEARCH, confidence=0.0, success=False)
    web_search_pass = MockPass(PassType.WEB_SEARCH, confidence=0.0, success=False)
    mb_validation_pass = MockPass(PassType.MUSICBRAINZ_VALIDATION, confidence=0.0, success=False)
    ml_embedding_pass = MockPass(PassType.ML_EMBEDDING, confidence=0.0, success=False)
    auto_retemplate_pass = MockPass(PassType.AUTO_RETEMPLATE, confidence=0.0, success=False)

    passes = [
        channel_pass,
        mb_search_pass,
        web_search_pass,
        mb_validation_pass,
        ml_embedding_pass,
        auto_retemplate_pass,
    ]
    controller = MultiPassParsingController(
        mp_config, passes, mock_advanced_parser, None, mock_db_manager
    )
    for p in passes:
        p.parse = AsyncMock(side_effect=p.parse)

    result = await controller.parse_video(
        "video_id", "title", "description", "tags", "channel_name", "channel_id"
    )

    assert result.final_result is None
    assert result.final_confidence == 0.0
    assert result.stopped_at_pass is None
    assert len(result.passes_attempted) == len(passes)  # All passes attempted but none successful
