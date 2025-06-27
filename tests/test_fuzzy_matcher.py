import importlib

from collector.search import fuzzy_matcher


class DummyConfig:
    def __init__(self):
        self.fuzzy_matching = {"min_similarity": 0.5}


def test_config_handling_object():
    fm = fuzzy_matcher.FuzzyMatcher(DummyConfig())
    assert fm.min_similarity_threshold == 0.5


def test_normalization_cache_key_by_type():
    fm = fuzzy_matcher.FuzzyMatcher()
    fm.clear_caches()
    fm.normalize_text("The Beatles", text_type="artist")
    fm.normalize_text("The Beatles", text_type="song")
    assert len(fm._normalization_cache) == 2


def test_phonetic_fallback_without_jellyfish(monkeypatch):
    monkeypatch.setattr(fuzzy_matcher, "HAS_JELLYFISH", False)
    fm = fuzzy_matcher.FuzzyMatcher()
    sim = fm.phonetic_similarity("Catherine", "Katherine")
    assert sim > 0
    importlib.reload(fuzzy_matcher)
