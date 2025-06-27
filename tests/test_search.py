from collector.config import ScrapingConfig, SearchConfig
from collector.search_engine import SearchEngine


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
    assert d is not None and d.year == 2024
    assert engine._is_video_after_date("20240102", "2023-12-31")
    assert not engine._is_video_after_date("20220101", "2023-12-31")


def test_import_without_tenacity(monkeypatch):
    import builtins
    import importlib
    import sys

    for mod in [
        "collector.search_engine",
        "collector.search.providers.youtube",
    ]:
        sys.modules.pop(mod, None)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("tenacity"):
            raise ImportError
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    se = importlib.import_module("collector.search_engine")
    yp = importlib.import_module("collector.search.providers.youtube")

    se.SearchEngine(SearchConfig(), ScrapingConfig())
    yp.YouTubeSearchProvider(ScrapingConfig())
