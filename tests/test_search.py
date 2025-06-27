from collector.config import ScrapingConfig, SearchConfig
from collector.config import SearchConfig, ScrapingConfig
from collector.search import SearchEngine


def _engine():
    return SearchEngine(SearchConfig(), ScrapingConfig())


def test_is_likely_karaoke():
    engine = _engine()
    assert engine._is_likely_karaoke("great song karaoke version")
    assert not engine._is_likely_karaoke("Tutorial reaction video")


def test_calculate_relevance_score():
    engine = _engine()
    score = engine._calculate_relevance_score("HD Karaoke with lyrics", "karaoke")
    assert score > 1


def test_parse_and_after_date():
    engine = _engine()
    d = engine._parse_upload_date("20240101")
    assert d.year == 2024
    assert engine._is_video_after_date("20240102", "2023-12-31")
    assert not engine._is_video_after_date("20220101", "2023-12-31")

