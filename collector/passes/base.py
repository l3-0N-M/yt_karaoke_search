"""Base class for all parsing passes."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional

from ..advanced_parser import ParseResult


class PassType(Enum):
    """Multi-pass parsing ladder pass types in optimized priority order."""

    CHANNEL_TEMPLATE = (
        "channel_template"  # Pass 0: Enhanced channel pattern extrapolation (EFFICIENCY)
    )
    MUSICBRAINZ_SEARCH = "musicbrainz_search"  # Pass 1: Direct MusicBrainz API lookup (VALIDATION)
    WEB_SEARCH = "web_search"  # Pass 2: Enhanced web search if MB confidence low (FALLBACK)
    MUSICBRAINZ_VALIDATION = (
        "musicbrainz_validation"  # Pass 3: Post-web-search MB validation (ENRICHMENT)
    )
    ML_EMBEDDING = "ml_embedding"
    AUTO_RETEMPLATE = "auto_retemplate"


class ParsingPass(ABC):
    """Abstract base class for a single parsing pass."""

    @property
    @abstractmethod
    def pass_type(self) -> PassType:
        """Return the type of the pass."""
        raise NotImplementedError

    @abstractmethod
    async def parse(
        self,
        title: str,
        description: str,
        tags: str,
        channel_name: str,
        channel_id: str,
        metadata: Dict,
    ) -> Optional[ParseResult]:
        """
        Execute the parsing logic for this pass.

        Args:
            title: The video title.
            description: The video description.
            tags: The video tags.
            channel_name: The channel name.
            channel_id: The channel ID.
            metadata: Additional metadata for the pass.

        Returns:
            A ParseResult if successful, otherwise None.
        """
        raise NotImplementedError

    def get_statistics(self) -> Dict:
        """Return statistics for the pass."""
        return {}
