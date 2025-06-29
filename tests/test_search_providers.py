from collector.config import ScrapingConfig
from collector.enhanced_search import MultiStrategySearchEngine
from collector.search.providers.youtube import YouTubeSearchProvider


def _youtube_provider():
    return YouTubeSearchProvider(ScrapingConfig())


def _multi_strategy_engine():
    return MultiStrategySearchEngine([_youtube_provider()])


def test_is_likely_karaoke():
    provider = _youtube_provider()
    assert provider._is_likely_karaoke("great song karaoke version")
    assert not provider._is_likely_karaoke("Tutorial reaction video")


def test_calculate_relevance_score():
    provider = _youtube_provider()
    score = provider._calculate_relevance_score("HD Karaoke with lyrics", "karaoke")
    assert score > 1


def test_parse_and_after_date():
    provider = _youtube_provider()
    d = provider._parse_upload_date("20240101")
    assert d is not None and d.year == 2024
    assert provider._is_video_after_date("20240102", "2023-12-31")
    assert not provider._is_video_after_date("20220101", "2023-12-31")
