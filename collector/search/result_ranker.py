"""Intelligent result ranking system with multi-dimensional scoring."""

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from .providers.base import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class RankingWeights:
    """Configurable weights for different ranking factors."""

    relevance: float = 0.35
    quality: float = 0.25
    popularity: float = 0.20
    metadata: float = 0.20

    def normalize(self):
        """Normalize weights to sum to 1.0."""
        total = self.relevance + self.quality + self.popularity + self.metadata
        if total > 0:
            self.relevance /= total
            self.quality /= total
            self.popularity /= total
            self.metadata /= total


@dataclass
class RankingResult:
    """Result with detailed ranking breakdown."""

    result: SearchResult
    relevance_score: float = 0.0
    quality_score: float = 0.0
    popularity_score: float = 0.0
    metadata_score: float = 0.0
    final_score: float = 0.0
    ranking_factors: Dict = None

    def __post_init__(self):
        if self.ranking_factors is None:
            self.ranking_factors = {}


class ResultRanker:
    """Advanced result ranking with multi-dimensional scoring."""

    def __init__(self, config=None):
        self.config = config or {}
        self.weights = RankingWeights()

        # Load custom weights if provided
        if "ranking_weights" in self.config:
            weight_config = self.config["ranking_weights"]
            self.weights.relevance = weight_config.get("relevance", 0.35)
            self.weights.quality = weight_config.get("quality", 0.25)
            self.weights.popularity = weight_config.get("popularity", 0.20)
            self.weights.metadata = weight_config.get("metadata", 0.20)

        self.weights.normalize()

        # Quality indicators with weights
        self.quality_indicators = {
            "video_quality": {
                "4k": 1.0, "2160p": 1.0,
                "hd": 0.8, "1080p": 0.8, "720p": 0.6,
                "high quality": 0.7, "studio": 0.8,
                "professional": 0.7, "remastered": 0.6,
            },
            "audio_quality": {
                "flac": 1.0, "lossless": 1.0,
                "320kbps": 0.9, "high bitrate": 0.8,
                "studio quality": 0.8, "cd quality": 0.7,
            },
            "karaoke_features": {
                "with lyrics": 0.9, "lyrics on screen": 0.9,
                "guide vocals": 0.7, "backing vocals": 0.6,
                "instrumental": 0.8, "minus one": 0.8,
                "key change": 0.5, "pitch control": 0.5,
            },
            "negative_indicators": {
                "low quality": -0.5, "poor audio": -0.6,
                "out of sync": -0.8, "broken": -1.0,
                "corrupted": -1.0, "incomplete": -0.7,
            }
        }

        # Channel reputation patterns
        self.channel_reputation = {
            "verified_channels": {
                "sing king": 0.9, "karaoke mugen": 0.8,
                "zoom karaoke": 0.7, "karafun": 0.8,
            },
            "professional_indicators": [
                "official", "records", "music", "entertainment",
                "studio", "productions", "media"
            ],
            "amateur_indicators": [
                "home", "bedroom", "cover", "personal", "my"
            ]
        }

        # Statistics for learning
        self.ranking_stats = {
            "total_ranked": 0,
            "score_distribution": {},
            "factor_importance": {},
        }

    def rank_results(
        self,
        results: List[SearchResult],
        query: str = "",
        context: Dict = None
    ) -> List[RankingResult]:
        """Rank search results using multi-dimensional scoring."""
        if not results:
            return []

        context = context or {}
        ranked_results = []

        # Calculate scores for each result
        for result in results:
            ranking_result = self._calculate_comprehensive_score(result, query, context)
            ranked_results.append(ranking_result)

        # Sort by final score descending
        ranked_results.sort(key=lambda x: x.final_score, reverse=True)

        # Update statistics
        self._update_ranking_stats(ranked_results)

        return ranked_results

    def _calculate_comprehensive_score(
        self,
        result: SearchResult,
        query: str,
        context: Dict
    ) -> RankingResult:
        """Calculate comprehensive score for a single result."""

        # Calculate individual dimension scores
        relevance_score = self._calculate_relevance_score(result, query, context)
        quality_score = self._calculate_quality_score(result)
        popularity_score = self._calculate_popularity_score(result)
        metadata_score = self._calculate_metadata_score(result)

        # Weighted combination
        final_score = (
            relevance_score * self.weights.relevance +
            quality_score * self.weights.quality +
            popularity_score * self.weights.popularity +
            metadata_score * self.weights.metadata
        )

        # Apply bonuses and penalties
        final_score = self._apply_contextual_adjustments(final_score, result, context)

        # Ensure score is in valid range
        final_score = max(0.0, min(1.0, final_score))

        return RankingResult(
            result=result,
            relevance_score=relevance_score,
            quality_score=quality_score,
            popularity_score=popularity_score,
            metadata_score=metadata_score,
            final_score=final_score,
            ranking_factors={
                "query_match": self._get_query_match_details(result, query),
                "quality_indicators": self._get_quality_details(result),
                "popularity_metrics": self._get_popularity_details(result),
                "metadata_completeness": self._get_metadata_details(result),
            }
        )

    def _calculate_relevance_score(
        self,
        result: SearchResult,
        query: str,
        context: Dict
    ) -> float:
        """Calculate relevance score based on query matching."""
        if not query:
            return 0.5  # Default score when no query

        title = result.title.lower()
        query_lower = query.lower()
        description = (result.description or "").lower()

        score = 0.0

        # Exact query match in title (highest weight)
        if query_lower in title:
            score += 0.4

        # Exact query match in description
        if query_lower in description:
            score += 0.1

        # Individual term matching
        query_terms = query_lower.split()
        if query_terms:
            title_matches = sum(1 for term in query_terms if term in title)
            desc_matches = sum(1 for term in query_terms if term in description)

            title_ratio = title_matches / len(query_terms)
            desc_ratio = desc_matches / len(query_terms)

            score += title_ratio * 0.3  # Title matches weighted higher
            score += desc_ratio * 0.1   # Description matches weighted lower

        # Bonus for exact artist/song matching if available
        if hasattr(result, 'parsed_artist') and hasattr(result, 'parsed_song'):
            if result.parsed_artist and result.parsed_song:
                artist_in_query = any(
                    result.parsed_artist.lower() in term
                    for term in query_terms
                )
                song_in_query = any(
                    result.parsed_song.lower() in term
                    for term in query_terms
                )
                if artist_in_query and song_in_query:
                    score += 0.2

        # Penalty for very long titles (likely spam)
        if len(title) > 100:
            score *= 0.9

        return min(score, 1.0)

    def _calculate_quality_score(self, result: SearchResult) -> float:
        """Calculate quality score based on technical and content indicators."""
        text_to_analyze = f"{result.title} {result.description or ''}".lower()
        score = 0.5  # Base score

        # Check for quality indicators
        for category, indicators in self.quality_indicators.items():
            if category == "negative_indicators":
                continue  # Handle separately

            for indicator, weight in indicators.items():
                if indicator in text_to_analyze:
                    score += weight * 0.1  # Scale down individual contributions

        # Apply negative indicators
        for indicator, penalty in self.quality_indicators["negative_indicators"].items():
            if indicator in text_to_analyze:
                score += penalty * 0.1

        # Duration-based quality scoring
        if result.duration:
            duration_mins = result.duration / 60
            if 2 <= duration_mins <= 6:  # Typical song length
                score += 0.1
            elif duration_mins < 1 or duration_mins > 10:
                score -= 0.2

        # Channel reputation bonus
        channel_name = result.channel.lower()

        # Check verified channels
        for verified_channel, bonus in self.channel_reputation["verified_channels"].items():
            if verified_channel in channel_name:
                score += bonus * 0.1
                break

        # Professional vs amateur indicators
        for indicator in self.channel_reputation["professional_indicators"]:
            if indicator in channel_name:
                score += 0.05
                break

        for indicator in self.channel_reputation["amateur_indicators"]:
            if indicator in channel_name:
                score -= 0.05
                break

        return max(0.0, min(1.0, score))

    def _calculate_popularity_score(self, result: SearchResult) -> float:
        """Calculate popularity score based on engagement metrics."""
        if not result.view_count:
            return 0.3  # Default for missing data

        # Logarithmic scaling for view count
        log_views = math.log10(max(1, result.view_count))

        # Normalize based on typical karaoke video view ranges
        # Assuming 1K to 10M views as typical range
        if log_views <= 3:  # < 1K views
            view_score = 0.1
        elif log_views <= 4:  # 1K-10K views
            view_score = 0.3
        elif log_views <= 5:  # 10K-100K views
            view_score = 0.5
        elif log_views <= 6:  # 100K-1M views
            view_score = 0.7
        elif log_views <= 7:  # 1M-10M views
            view_score = 0.9
        else:  # > 10M views
            view_score = 1.0

        # Age factor - newer content might be more relevant
        age_score = 0.5  # Default
        if result.upload_date:
            try:
                # Assume upload_date is in YYYYMMDD format
                upload_date = datetime.strptime(result.upload_date, "%Y%m%d")
                days_old = (datetime.now() - upload_date).days

                if days_old <= 30:  # Very recent
                    age_score = 0.9
                elif days_old <= 180:  # Recent
                    age_score = 0.7
                elif days_old <= 365:  # Less than a year
                    age_score = 0.5
                elif days_old <= 365 * 3:  # Less than 3 years
                    age_score = 0.3
                else:  # Older content
                    age_score = 0.2
            except Exception:
                pass

        # Combine scores
        popularity_score = view_score * 0.7 + age_score * 0.3

        return max(0.0, min(1.0, popularity_score))

    def _calculate_metadata_score(self, result: SearchResult) -> float:
        """Calculate score based on metadata completeness and quality."""
        score = 0.0
        max_score = 0.0

        # Check presence and quality of various metadata fields
        metadata_checks = [
            ("title", result.title, 0.3),
            ("channel", result.channel, 0.2),
            ("description", result.description, 0.2),
            ("duration", result.duration, 0.1),
            ("view_count", result.view_count, 0.1),
            ("upload_date", result.upload_date, 0.1),
        ]

        for field_name, field_value, weight in metadata_checks:
            max_score += weight

            if field_value:
                if field_name == "title":
                    # Title quality assessment
                    if len(str(field_value)) > 10:
                        score += weight
                elif field_name == "description":
                    # Description quality assessment
                    if len(str(field_value)) > 50:
                        score += weight
                elif field_name in ["duration", "view_count"]:
                    # Numeric fields
                    if field_value > 0:
                        score += weight
                else:
                    # Other fields
                    score += weight

        # Normalize by maximum possible score
        if max_score > 0:
            score = score / max_score

        # Bonus for having parsed artist/song information
        if hasattr(result, 'parsed_artist') and hasattr(result, 'parsed_song'):
            if result.parsed_artist and result.parsed_song:
                score += 0.2

        return min(1.0, score)

    def _apply_contextual_adjustments(
        self,
        base_score: float,
        result: SearchResult,
        context: Dict
    ) -> float:
        """Apply contextual bonuses and penalties."""
        adjusted_score = base_score

        # Duplicate penalty
        if context.get("check_duplicates", False):
            # This would be implemented with actual duplicate detection
            pass

        # Provider-specific adjustments
        provider_weights = context.get("provider_weights", {})
        if result.provider in provider_weights:
            provider_weight = provider_weights[result.provider]
            adjusted_score *= provider_weight

        # Search method bonuses
        if result.search_method == "exact_match":
            adjusted_score *= 1.1
        elif result.search_method == "fuzzy_match":
            adjusted_score *= 0.95

        # Freshness bonus for recent content
        if context.get("prefer_recent", False) and result.upload_date:
            try:
                upload_date = datetime.strptime(result.upload_date, "%Y%m%d")
                days_old = (datetime.now() - upload_date).days
                if days_old <= 30:
                    adjusted_score *= 1.05
            except Exception:
                pass

        return adjusted_score

    def _get_query_match_details(self, result: SearchResult, query: str) -> Dict:
        """Get detailed query matching information."""
        if not query:
            return {}

        query_terms = query.lower().split()

        return {
            "exact_title_match": query.lower() in result.title.lower(),
            "term_matches": sum(1 for term in query_terms if term in result.title.lower()),
            "total_query_terms": len(query_terms),
            "match_ratio": (
                sum(1 for term in query_terms if term in result.title.lower()) /
                len(query_terms) if query_terms else 0
            ),
        }

    def _get_quality_details(self, result: SearchResult) -> Dict:
        """Get detailed quality assessment information."""
        text = f"{result.title} {result.description or ''}".lower()

        found_indicators = {}
        for category, indicators in self.quality_indicators.items():
            found_indicators[category] = [
                indicator for indicator in indicators.keys()
                if indicator in text
            ]

        return {
            "found_indicators": found_indicators,
            "duration_minutes": result.duration / 60 if result.duration else None,
            "channel_type": self._classify_channel(result.channel),
        }

    def _get_popularity_details(self, result: SearchResult) -> Dict:
        """Get detailed popularity metrics information."""
        return {
            "view_count": result.view_count,
            "log_view_score": math.log10(max(1, result.view_count)) if result.view_count else 0,
            "upload_date": result.upload_date,
            "estimated_age_days": self._calculate_age_days(result.upload_date),
        }

    def _get_metadata_details(self, result: SearchResult) -> Dict:
        """Get detailed metadata completeness information."""
        return {
            "has_title": bool(result.title),
            "has_description": bool(result.description),
            "has_duration": bool(result.duration),
            "has_view_count": bool(result.view_count),
            "has_upload_date": bool(result.upload_date),
            "title_length": len(result.title) if result.title else 0,
            "description_length": len(result.description) if result.description else 0,
        }

    def _classify_channel(self, channel_name: str) -> str:
        """Classify channel type based on name patterns."""
        channel_lower = channel_name.lower()

        if any(verified in channel_lower for verified in self.channel_reputation["verified_channels"]):
            return "verified_karaoke"
        elif any(prof in channel_lower for prof in self.channel_reputation["professional_indicators"]):
            return "professional"
        elif any(amateur in channel_lower for amateur in self.channel_reputation["amateur_indicators"]):
            return "amateur"
        else:
            return "unknown"

    def _calculate_age_days(self, upload_date: str) -> Optional[int]:
        """Calculate age in days from upload date."""
        if not upload_date:
            return None

        try:
            upload_dt = datetime.strptime(upload_date, "%Y%m%d")
            return (datetime.now() - upload_dt).days
        except Exception:
            return None

    def _update_ranking_stats(self, ranked_results: List[RankingResult]):
        """Update ranking statistics for analysis."""
        self.ranking_stats["total_ranked"] += len(ranked_results)

        # Update score distribution
        for result in ranked_results:
            score_bucket = int(result.final_score * 10) / 10
            if score_bucket not in self.ranking_stats["score_distribution"]:
                self.ranking_stats["score_distribution"][score_bucket] = 0
            self.ranking_stats["score_distribution"][score_bucket] += 1

    def get_statistics(self) -> Dict:
        """Get ranking system statistics."""
        return {
            "weights": {
                "relevance": self.weights.relevance,
                "quality": self.weights.quality,
                "popularity": self.weights.popularity,
                "metadata": self.weights.metadata,
            },
            "statistics": self.ranking_stats,
            "quality_indicators_count": sum(len(indicators) for indicators in self.quality_indicators.values()),
            "channel_reputation_entries": len(self.channel_reputation["verified_channels"]),
        }
