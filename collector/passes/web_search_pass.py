"""Pass 3: Enhanced web search with filler-stripped query and SERP caching."""

import hashlib
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from ..advanced_parser import AdvancedTitleParser, ParseResult
from ..enhanced_search import MultiStrategySearchEngine

logger = logging.getLogger(__name__)


@dataclass
class SERPCacheEntry:
    """Cached SERP result entry."""

    query: str
    query_hash: str
    results: List[Dict]
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_hours: int = 168  # 1 week default


@dataclass
class QueryCleaningResult:
    """Result of query cleaning process."""

    original_query: str
    cleaned_query: str
    removed_terms: List[str]
    confidence_boost: float = 1.0
    cleaning_method: str = ""


class FillerWordProcessor:
    """Language-aware filler word removal."""

    def __init__(self):
        self.filler_words = {
            "english": {
                "karaoke_terms": {
                    "karaoke",
                    "karaokê",
                    "karaoké",
                    "sing along",
                    "singalong",
                    "backing track",
                    "instrumental",
                    "minus one",
                    "minusone",
                    "minus-one",
                    "mr",
                    "inst",
                    "playback",
                    "accompaniment",
                },
                "quality_terms": {
                    "hd",
                    "hq",
                    "4k",
                    "1080p",
                    "720p",
                    "480p",
                    "high quality",
                    "high def",
                    "ultra hd",
                    "uhd",
                    "full hd",
                    "low quality",
                },
                "video_terms": {
                    "official",
                    "music video",
                    "video",
                    "audio",
                    "clip",
                    "music",
                    "song",
                    "track",
                    "single",
                    "album",
                    "ep",
                },
                "performance_terms": {
                    "live",
                    "acoustic",
                    "unplugged",
                    "studio",
                    "demo",
                    "rehearsal",
                    "soundcheck",
                    "concert",
                    "performance",
                },
                "generic_terms": {
                    "version",
                    "cover",
                    "tribute",
                    "style",
                    "like",
                    "type",
                    "remake",
                    "rendition",
                    "interpretation",
                },
            },
            "spanish": {
                "karaoke_terms": {
                    "karaoke",
                    "pista",
                    "instrumental",
                    "acompañamiento",
                    "sin voz",
                    "solo música",
                }
            },
            "french": {
                "karaoke_terms": {"karaoké", "playback", "version instrumentale", "sans voix"}
            },
            "german": {"karaoke_terms": {"karaoke", "playback", "instrumental", "ohne gesang"}},
            "portuguese": {"karaoke_terms": {"karaokê", "playback", "instrumental", "sem voz"}},
        }

        # Compile regex patterns for efficiency
        self.compiled_patterns = {}
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for all languages."""

        for language, categories in self.filler_words.items():
            self.compiled_patterns[language] = {}
            for category, terms in categories.items():
                # Create pattern that matches whole words
                pattern = r"\b(?:" + "|".join(re.escape(term) for term in terms) + r")\b"
                self.compiled_patterns[language][category] = re.compile(pattern, re.IGNORECASE)

    def clean_query(self, query: str, target_language: str = "english") -> QueryCleaningResult:
        """Clean query by removing filler words."""

        original_query = query
        cleaned = query
        removed_terms = []
        confidence_boost = 1.0

        # Get patterns for target language, fallback to English
        language_patterns = self.compiled_patterns.get(
            target_language, self.compiled_patterns["english"]
        )

        # Remove filler words by category, tracking what's removed
        for category, pattern in language_patterns.items():
            matches = pattern.findall(cleaned)
            if matches:
                removed_terms.extend(matches)
                cleaned = pattern.sub(" ", cleaned)

                # Boost confidence based on category
                if category == "karaoke_terms":
                    confidence_boost *= 1.2  # Strong boost for karaoke terms
                elif category == "quality_terms":
                    confidence_boost *= 1.1  # Medium boost for quality terms
                else:
                    confidence_boost *= 1.05  # Small boost for other terms

        # Clean up whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Additional cleaning
        cleaned = self._additional_cleaning(cleaned, removed_terms)

        return QueryCleaningResult(
            original_query=original_query,
            cleaned_query=cleaned,
            removed_terms=removed_terms,
            confidence_boost=min(confidence_boost, 1.5),  # Cap boost at 50%
            cleaning_method=f"{target_language}_filler_removal",
        )

    def _additional_cleaning(self, query: str, removed_terms: List[str]) -> str:
        """Additional query cleaning operations."""

        # Remove excessive punctuation
        query = re.sub(r"[^\w\s\-\'\"&]", " ", query)

        # Remove standalone numbers (often video IDs)
        query = re.sub(r"\b\d+\b", " ", query)

        # Remove very short words (often acronyms or noise)
        words = query.split()
        words = [word for word in words if len(word) >= 2 or word.lower() in {"a", "i"}]

        # Remove duplicate words
        seen = set()
        unique_words = []
        for word in words:
            word_lower = word.lower()
            if word_lower not in seen:
                seen.add(word_lower)
                unique_words.append(word)

        return " ".join(unique_words)


class SERPCache:
    """SERP result caching with TTL support."""

    def __init__(self, max_entries: int = 10000, default_ttl_hours: int = 168):
        self.cache: Dict[str, SERPCacheEntry] = {}
        self.max_entries = max_entries
        self.default_ttl_hours = default_ttl_hours

    def _get_query_hash(self, query: str) -> str:
        """Generate hash for query caching."""
        return hashlib.md5(query.lower().encode("utf-8")).hexdigest()

    def get(self, query: str) -> Optional[List[Dict]]:
        """Get cached results for a query."""

        query_hash = self._get_query_hash(query)

        if query_hash not in self.cache:
            return None

        entry = self.cache[query_hash]

        # Check if entry has expired
        age = datetime.now() - entry.created_at
        if age > timedelta(hours=entry.ttl_hours):
            del self.cache[query_hash]
            return None

        # Update access statistics
        entry.last_accessed = datetime.now()
        entry.access_count += 1

        return entry.results

    def put(self, query: str, results: List[Dict], ttl_hours: Optional[int] = None) -> None:
        """Cache results for a query."""

        query_hash = self._get_query_hash(query)
        ttl = ttl_hours or self.default_ttl_hours

        # Clean up cache if too large
        if len(self.cache) >= self.max_entries:
            self._cleanup_cache()

        entry = SERPCacheEntry(
            query=query,
            query_hash=query_hash,
            results=results,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            ttl_hours=ttl,
        )

        self.cache[query_hash] = entry

    def _cleanup_cache(self):
        """Remove old/unused entries to make space."""

        # Remove expired entries first
        now = datetime.now()
        expired_keys = []

        for key, entry in self.cache.items():
            age = now - entry.created_at
            if age > timedelta(hours=entry.ttl_hours):
                expired_keys.append(key)

        for key in expired_keys:
            del self.cache[key]

        # If still too large, remove least recently accessed
        if len(self.cache) >= self.max_entries:
            sorted_entries = sorted(
                self.cache.items(), key=lambda x: (x[1].last_accessed, x[1].access_count)
            )

            # Remove oldest 25% of entries
            remove_count = len(self.cache) // 4
            for key, _ in sorted_entries[:remove_count]:
                del self.cache[key]

    def get_statistics(self) -> Dict:
        """Get cache statistics."""

        now = datetime.now()
        total_entries = len(self.cache)
        expired_count = 0

        for entry in self.cache.values():
            age = now - entry.created_at
            if age > timedelta(hours=entry.ttl_hours):
                expired_count += 1

        return {
            "total_entries": total_entries,
            "expired_entries": expired_count,
            "active_entries": total_entries - expired_count,
            "hit_rate": self._calculate_hit_rate(),
            "avg_age_hours": self._calculate_avg_age_hours(),
            "most_accessed_queries": self._get_most_accessed_queries(5),
        }

    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""

        if not hasattr(self, "_total_requests"):
            return 0.0

        hits = sum(entry.access_count for entry in self.cache.values())
        return hits / max(self._total_requests, 1)

    def _calculate_avg_age_hours(self) -> float:
        """Calculate average age of cache entries."""

        if not self.cache:
            return 0.0

        now = datetime.now()
        total_age = sum(
            (now - entry.created_at).total_seconds() / 3600 for entry in self.cache.values()
        )

        return total_age / len(self.cache)

    def _get_most_accessed_queries(self, limit: int) -> List[Tuple[str, int]]:
        """Get most frequently accessed queries."""

        sorted_entries = sorted(self.cache.items(), key=lambda x: x[1].access_count, reverse=True)

        return [(entry.query, entry.access_count) for _, entry in sorted_entries[:limit]]


class EnhancedWebSearchPass:
    """Pass 3: Enhanced web search with intelligent query processing."""

    def __init__(
        self,
        advanced_parser: AdvancedTitleParser,
        search_engine: MultiStrategySearchEngine,
        db_manager=None,
    ):
        self.advanced_parser = advanced_parser
        self.search_engine = search_engine
        self.db_manager = db_manager

        # Initialize components
        self.filler_processor = FillerWordProcessor()
        self.serp_cache = SERPCache()

        # Configuration
        self.max_search_results = 50
        self.min_results_threshold = 5
        self.query_expansion_enabled = True
        self.parallel_search_enabled = True

        # Statistics
        self.search_stats = {
            "total_searches": 0,
            "cache_hits": 0,
            "successful_parses": 0,
            "query_cleaning_improvements": 0,
        }

    async def parse(
        self,
        title: str,
        description: str = "",
        tags: str = "",
        channel_name: str = "",
        channel_id: str = "",
        metadata: Optional[Dict] = None,
    ) -> Optional[ParseResult]:
        """Execute enhanced web search parsing."""

        start_time = time.time()

        try:
            self.search_stats["total_searches"] += 1

            # Step 1: Generate search queries from title
            search_queries = self._generate_search_queries(title, description, tags)
            if not search_queries:
                return None

            # Step 2: Try each query with caching
            best_result = None
            best_confidence = 0.0

            for query_info in search_queries:
                # Check cache first
                cached_results = self.serp_cache.get(query_info.cleaned_query)
                if cached_results:
                    self.search_stats["cache_hits"] += 1
                    results = cached_results
                else:
                    # Perform search
                    results = await self._search_with_engine(query_info.cleaned_query)
                    if results:
                        # Cache results
                        self.serp_cache.put(query_info.cleaned_query, results)

                if not results:
                    continue

                # Parse search results
                parse_result = await self._parse_search_results(results, query_info, title)

                if parse_result and parse_result.confidence > best_confidence:
                    best_confidence = parse_result.confidence
                    best_result = parse_result

                    # Early exit if confidence is high enough
                    if best_confidence > 0.85:
                        break

            if best_result:
                self.search_stats["successful_parses"] += 1

            return best_result

        except Exception as e:
            logger.error(f"Enhanced web search pass failed: {e}")
            return None
        finally:
            processing_time = time.time() - start_time
            if processing_time > 10.0:  # Log slow searches
                logger.warning(f"Web search pass took {processing_time:.2f}s")

    def _generate_search_queries(
        self, title: str, description: str, tags: str
    ) -> List[QueryCleaningResult]:
        """Generate multiple search queries with different cleaning strategies."""

        queries = []

        # Strategy 1: Clean the original title
        base_cleaning = self.filler_processor.clean_query(title, "english")
        if base_cleaning.cleaned_query:
            queries.append(base_cleaning)

        # Strategy 2: Try with channel context if available
        if description:
            combined_text = f"{title} {description[:200]}"  # Limit description length
            desc_cleaning = self.filler_processor.clean_query(combined_text, "english")
            if desc_cleaning.cleaned_query != base_cleaning.cleaned_query:
                queries.append(desc_cleaning)

        # Strategy 3: Extract quoted parts (often song titles)
        quoted_parts = re.findall(r'["\u201c\u201d]([^"\u201c\u201d]+)["\u201c\u201d]', title)
        for quoted in quoted_parts:
            quoted_cleaning = self.filler_processor.clean_query(quoted, "english")
            if quoted_cleaning.cleaned_query and len(quoted_cleaning.cleaned_query) > 3:
                quoted_cleaning.cleaning_method = "quoted_extraction"
                quoted_cleaning.confidence_boost *= 1.1  # Boost for quoted text
                queries.append(quoted_cleaning)

        # Strategy 4: Try language-specific cleaning if non-English characters detected
        if re.search(r"[^\x00-\x7F]", title):  # Non-ASCII characters
            for lang in ["spanish", "french", "german", "portuguese"]:
                lang_cleaning = self.filler_processor.clean_query(title, lang)
                if lang_cleaning.cleaned_query != base_cleaning.cleaned_query:
                    lang_cleaning.cleaning_method = f"{lang}_specific"
                    queries.append(lang_cleaning)

        # Strategy 5: Minimal cleaning (keep more context)
        minimal_cleaning = QueryCleaningResult(
            original_query=title,
            cleaned_query=re.sub(
                r"\b(?:karaoke|instrumental)\b", "", title, flags=re.IGNORECASE
            ).strip(),
            removed_terms=["karaoke", "instrumental"],
            confidence_boost=0.9,  # Lower boost for minimal cleaning
            cleaning_method="minimal",
        )
        if minimal_cleaning.cleaned_query != base_cleaning.cleaned_query:
            queries.append(minimal_cleaning)

        # Remove duplicates and sort by confidence boost
        unique_queries = []
        seen_queries = set()

        for query in queries:
            if query.cleaned_query not in seen_queries and len(query.cleaned_query) > 3:
                seen_queries.add(query.cleaned_query)
                unique_queries.append(query)

        # Sort by confidence boost (higher first)
        unique_queries.sort(key=lambda q: q.confidence_boost, reverse=True)

        return unique_queries[:5]  # Limit to top 5 queries

    async def _search_with_engine(self, query: str) -> List[Dict]:
        """Perform search using the enhanced search engine."""

        try:
            # Use the existing enhanced search engine
            search_results = await self.search_engine.search_videos(
                query, max_results=self.max_search_results, use_cache=True, enable_fallback=True
            )

            # Convert SearchResult objects to dictionaries
            results = []
            for result in search_results:
                result_dict = {
                    "video_id": result.video_id,
                    "title": result.title,
                    "description": getattr(result, "description", ""),
                    "channel_name": getattr(result, "channel_name", ""),
                    "channel_id": getattr(result, "channel_id", ""),
                    "duration": getattr(result, "duration", 0),
                    "view_count": getattr(result, "view_count", 0),
                    "relevance_score": getattr(result, "relevance_score", 0.0),
                    "final_score": getattr(result, "final_score", 0.0),
                }
                results.append(result_dict)

            return results

        except Exception as e:
            logger.warning(f"Search engine query failed for '{query}': {e}")
            return []

    async def _parse_search_results(
        self, results: List[Dict], query_info: QueryCleaningResult, original_title: str
    ) -> Optional[ParseResult]:
        """Parse search results to extract artist/song information."""

        if not results:
            return None

        # Strategy 1: Parse titles of top results
        parsed_results = []

        for i, result in enumerate(results[:10]):  # Top 10 results
            result_title = result.get("title", "")
            if not result_title:
                continue

            # Parse the result title
            parse_result = self.advanced_parser.parse_title(
                result_title, result.get("description", ""), "", result.get("channel_name", "")
            )

            if parse_result and parse_result.confidence > 0.5:
                # Adjust confidence based on search ranking and relevance
                ranking_factor = max(0.5, 1.0 - (i * 0.05))  # Decrease by position
                relevance_factor = result.get("relevance_score", 0.5)

                adjusted_confidence = (
                    parse_result.confidence
                    * ranking_factor
                    * relevance_factor
                    * query_info.confidence_boost
                )

                parse_result.confidence = min(adjusted_confidence, 0.95)
                parse_result.method = "web_search_parsing"
                parse_result.metadata = parse_result.metadata or {}
                parse_result.metadata.update(
                    {
                        "search_ranking": i + 1,
                        "relevance_score": relevance_factor,
                        "query_cleaning_boost": query_info.confidence_boost,
                        "cleaning_method": query_info.cleaning_method,
                        "removed_terms": query_info.removed_terms,
                        "source_video_id": result.get("video_id"),
                        "source_channel": result.get("channel_name"),
                    }
                )

                parsed_results.append(parse_result)

        if not parsed_results:
            return None

        # Strategy 2: Consensus-based result selection
        best_result = self._select_consensus_result(parsed_results, query_info)

        return best_result

    def _select_consensus_result(
        self, parsed_results: List[ParseResult], query_info: QueryCleaningResult
    ) -> Optional[ParseResult]:
        """Select the best result using consensus from multiple parses."""

        if not parsed_results:
            return None

        # Group results by artist/song combinations
        result_groups = defaultdict(list)

        for result in parsed_results:
            key = (
                result.original_artist.lower() if result.original_artist else "",
                result.song_title.lower() if result.song_title else "",
            )
            result_groups[key].append(result)

        # Find the group with highest combined confidence
        best_group = None
        best_score = 0.0

        for key, group in result_groups.items():
            if not key[0] and not key[1]:  # Skip empty results
                continue

            # Calculate group score
            avg_confidence = sum(r.confidence for r in group) / len(group)
            consensus_bonus = min(len(group) * 0.1, 0.3)  # Bonus for multiple agreements
            group_score = avg_confidence + consensus_bonus

            if group_score > best_score:
                best_score = group_score
                best_group = group

        if not best_group:
            # Fallback to highest individual confidence
            return max(parsed_results, key=lambda r: r.confidence)

        # Return the highest confidence result from the best group
        best_result = max(best_group, key=lambda r: r.confidence)

        # Update metadata with consensus information
        best_result.metadata = best_result.metadata or {}
        best_result.metadata.update(
            {
                "consensus_group_size": len(best_group),
                "consensus_bonus": min(len(best_group) * 0.1, 0.3),
                "total_candidates": len(parsed_results),
            }
        )

        return best_result

    def get_statistics(self) -> Dict:
        """Get statistics for the web search pass."""

        cache_stats = self.serp_cache.get_statistics()

        return {
            "search_performance": self.search_stats,
            "cache_statistics": cache_stats,
            "success_rate": (
                self.search_stats["successful_parses"] / max(self.search_stats["total_searches"], 1)
            ),
            "cache_hit_rate": (
                self.search_stats["cache_hits"] / max(self.search_stats["total_searches"], 1)
            ),
            "configuration": {
                "max_search_results": self.max_search_results,
                "query_expansion_enabled": self.query_expansion_enabled,
                "parallel_search_enabled": self.parallel_search_enabled,
            },
        }

    def clear_cache(self):
        """Clear the SERP cache."""
        self.serp_cache.cache.clear()
        logger.info("SERP cache cleared")

    def get_cache_size(self) -> int:
        """Get current cache size."""
        return len(self.serp_cache.cache)
