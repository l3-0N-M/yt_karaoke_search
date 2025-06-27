from collector.validation_corrector import ValidationCorrector


def test_validation_corrector_valid():
    vc = ValidationCorrector()
    recording = {"title": "Song", "artist-credit": [{"name": "Artist"}]}
    res, suggestion = vc.validate("Artist", "Song", recording)
    assert res.artist_valid
    assert res.song_valid
    assert res.validation_score > 0.8
    assert suggestion is None


def test_validation_corrector_suggests():
    vc = ValidationCorrector()
    recording = {"title": "Real Song", "artist-credit": [{"name": "Real Artist"}]}
    res, suggestion = vc.validate("Wrong", "Track", recording)
    assert not res.artist_valid
    assert not res.song_valid
    assert suggestion.suggested_artist == "Real Artist"
    assert suggestion.suggested_title == "Real Song"
