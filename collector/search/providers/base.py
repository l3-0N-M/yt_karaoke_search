"""Base interface for search providers."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Standardized search result across all providers."""
    
    video_id: str
    url: str
    title: str
    channel: str
    channel_id: str
    duration: Optional[int] = None
    view_count: int = 0
    upload_date: Optional[str] = None
    thumbnail_url: Optional[str] = None
    description: Optional[str] = None
    
    # Provider-specific metadata
    provider: str = ""
    search_query: str = ""
    search_method: str = ""
    
    # Quality scoring
    relevance_score: float = 0.0
    quality_score: float = 0.0
    popularity_score: float = 0.0
    metadata_score: float = 0.0
    final_score: float = 0.0
    
    # Additional metadata
    metadata: Dict = field(default_factory=dict)
    extraction_timestamp: datetime = field(default_factory=datetime.now)


class SearchProvider(ABC):
    """Abstract base class for all search providers."""
    
    def __init__(self, config=None):
        self.config = config
        self.provider_name = self.__class__.__name__.replace("SearchProvider", "").lower()
        self.logger = logging.getLogger(f"{__name__}.{self.provider_name}")
        
        # Provider statistics
        self.total_searches = 0
        self.successful_searches = 0
        self.total_results = 0
        self.average_response_time = 0.0
        
    @abstractmethod
    async def search_videos(self, query: str, max_results: int = 100) -> List[SearchResult]:
        """Search for videos using this provider."""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if this provider is currently available."""
        pass
    
    @abstractmethod
    def get_provider_weight(self) -> float:
        """Get the weight/priority of this provider (0.0-1.0)."""
        pass
    
    def get_statistics(self) -> Dict:
        """Get provider performance statistics."""
        success_rate = (
            self.successful_searches / self.total_searches 
            if self.total_searches > 0 else 0.0
        )
        
        avg_results = (
            self.total_results / self.successful_searches 
            if self.successful_searches > 0 else 0.0
        )
        
        return {
            "provider": self.provider_name,
            "total_searches": self.total_searches,
            "successful_searches": self.successful_searches,
            "success_rate": success_rate,
            "total_results": self.total_results,
            "average_results_per_search": avg_results,
            "average_response_time": self.average_response_time,
        }
    
    def update_statistics(self, search_successful: bool, result_count: int, response_time: float):
        """Update provider statistics."""
        self.total_searches += 1
        if search_successful:
            self.successful_searches += 1
            self.total_results += result_count
        
        # Update average response time with moving average
        if self.total_searches == 1:
            self.average_response_time = response_time
        else:
            alpha = 0.1  # Smoothing factor
            self.average_response_time = (
                alpha * response_time + (1 - alpha) * self.average_response_time
            )
    
    def normalize_result(self, raw_result: Dict, query: str) -> SearchResult:
        """Convert provider-specific result to standardized SearchResult."""
        return SearchResult(
            video_id=raw_result.get("video_id", ""),
            url=raw_result.get("url", ""),
            title=raw_result.get("title", ""),
            channel=raw_result.get("channel", ""),
            channel_id=raw_result.get("channel_id", ""),
            duration=raw_result.get("duration"),
            view_count=raw_result.get("view_count", 0),
            upload_date=raw_result.get("upload_date"),
            thumbnail_url=raw_result.get("thumbnail_url"),
            description=raw_result.get("description"),
            provider=self.provider_name,
            search_query=query,
            search_method=raw_result.get("search_method", "api"),
            relevance_score=raw_result.get("relevance_score", 0.0),
            metadata=raw_result.get("metadata", {}),
        )
    
    def filter_karaoke_content(self, results: List[SearchResult]) -> List[SearchResult]:
        """Filter results to identify likely karaoke content."""
        filtered = []
        
        for result in results:
            if self._is_likely_karaoke(result.title, result.description or ""):
                # Apply duration filtering
                if result.duration and not (45 <= result.duration <= 900):
                    continue
                filtered.append(result)
        
        return filtered
    
    def _is_likely_karaoke(self, title: str, description: str = "") -> bool:
        """Determine if content is likely karaoke-related."""
        text = f"{title} {description}".lower()
        
        karaoke_indicators = [
            "karaoke", "backing track", "instrumental", "sing along",
            "minus one", "playback", "accompaniment", "with lyrics",
            "guide vocals", "piano version", "acoustic version"
        ]
        
        exclusions = [
            "reaction", "review", "tutorial", "lesson", "how to",
            "analysis", "behind the scenes", "interview", "documentary",
            "live performance", "concert", "official video"
        ]
        
        has_indicator = any(indicator in text for indicator in karaoke_indicators)
        has_exclusion = any(exclusion in text for exclusion in exclusions)
        
        return has_indicator and not has_exclusion