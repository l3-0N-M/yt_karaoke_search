"""Validation and correction utilities.

Design note: ValidationCorrector validates extracted artist and song titles
against MusicBrainz recordings. It combines fuzzy string comparison with a
lightweight phonetic heuristic. A validation score in ``[0, 1]`` is computed as
the average of fuzzy and phonetic similarities for both artist and title. When
confidence scores are computed, the processor reweights the base confidence
using ``new = base * (0.7 + 0.3 * validation_score)`` to emphasise high quality
matches while keeping results bounded.
"""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, Optional, Tuple


@dataclass
class ValidationResult:
    """Result of validating artist and title against MusicBrainz."""

    artist_valid: bool
    song_valid: bool
    validation_score: float


@dataclass
class CorrectionSuggestion:
    """Suggested corrections when validation fails."""

    suggested_artist: Optional[str] = None
    suggested_title: Optional[str] = None
    reason: str = ""


class ValidationCorrector:
    """Validate extracted metadata and propose corrections."""

    def _phonetic(self, text: str) -> str:
        """Very small phonetic helper removing vowels."""
        cleaned = re.sub(r"[^a-z]", "", text.lower())
        if not cleaned:
            return ""
        first = cleaned[0]
        body = re.sub(r"[aeiou]", "", cleaned[1:])
        return first + body

    def _similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def validate(
        self, artist: str, title: str, recording: Dict
    ) -> Tuple[ValidationResult, Optional[CorrectionSuggestion]]:
        """Validate artist and title against a MusicBrainz recording."""
        rec_artist = ""
        credits = recording.get("artist-credit", [])
        if credits:
            rec_artist = credits[0].get("name", "")
        rec_title = recording.get("title", "")

        fuzzy_artist = self._similarity(artist, rec_artist)
        fuzzy_title = self._similarity(title, rec_title)
        phon_artist = self._similarity(self._phonetic(artist), self._phonetic(rec_artist))
        phon_title = self._similarity(self._phonetic(title), self._phonetic(rec_title))

        score = (fuzzy_artist + fuzzy_title + phon_artist + phon_title) / 4

        artist_valid = fuzzy_artist > 0.8 or phon_artist > 0.8
        song_valid = fuzzy_title > 0.8 or phon_title > 0.8

        suggestion: Optional[CorrectionSuggestion] = None
        if not artist_valid or not song_valid:
            suggestion = CorrectionSuggestion(
                suggested_artist=None if artist_valid else rec_artist or None,
                suggested_title=None if song_valid else rec_title or None,
                reason="musicbrainz-best-match",
            )

        return ValidationResult(artist_valid, song_valid, score), suggestion
