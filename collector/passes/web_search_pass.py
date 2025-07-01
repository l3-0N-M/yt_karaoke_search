"""Pass 3: Enhanced web search with filler-stripped query and SERP caching."""

import asyncio
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
from .base import ParsingPass, PassType

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
                    "melody",
                    "노래방",  # Korean: karaoke
                    "반주",  # Korean: accompaniment
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
                "language_prefixes": {
                    "de",
                    "en",
                    "fr",
                    "es",
                    "it",
                    "pt",
                    "nl",
                    "pl",
                    "ru",
                    "jp",
                    "kr",
                    "cn",
                },
                "noise_terms": {
                    "video",
                    "lyrics",
                    "with",
                    "without",
                    "original",
                    "new",
                    "best",
                    "top",
                    "hit",
                    "song",
                    "music",
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

        # Ensure query is a string with robust error handling
        if query is None:
            logger.warning("Query is None, converting to empty string")
            query = ""
        elif not isinstance(query, str):
            try:
                query = str(query)
            except Exception as e:
                logger.warning(f"Failed to convert query to string: {e}")
                query = ""

        # Handle potential unicode issues
        try:
            # Ensure the string is properly encoded/decoded
            if isinstance(query, bytes):
                query = query.decode("utf-8", errors="ignore")
            # Normalize the string to handle different unicode representations
            import unicodedata

            query = unicodedata.normalize("NFKC", query)
        except Exception as e:
            logger.warning(f"Unicode normalization failed for query: {e}")
            # Fallback to basic string conversion
            query = str(query) if query else ""

        # Final validation - ensure we have a valid string
        if not query or not isinstance(query, str):
            logger.warning("Query validation failed, returning empty result")
            return QueryCleaningResult("", "", [], 1.0, target_language)

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
            try:
                # Ensure cleaned is always a string before regex operations
                if not isinstance(cleaned, str):
                    cleaned = str(cleaned) if cleaned is not None else ""

                # Ensure pattern is properly compiled
                if not hasattr(pattern, "findall"):
                    logger.warning(f"Invalid pattern object for category {category}")
                    continue

                # Additional safety check for string content
                if not isinstance(cleaned, str):
                    logger.warning(
                        f"Non-string value passed to regex for category {category}: {type(cleaned)}"
                    )
                    cleaned = str(cleaned) if cleaned is not None else ""

                # Ensure the string is properly encoded
                try:
                    cleaned.encode("utf-8")
                except UnicodeEncodeError:
                    logger.warning(f"Unicode encoding issue in category {category}, normalizing")
                    cleaned = cleaned.encode("utf-8", errors="ignore").decode("utf-8")

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
            except (
                TypeError,
                re.error,
                AttributeError,
                UnicodeDecodeError,
                UnicodeEncodeError,
            ) as e:
                logger.warning(
                    f"Regex operation failed for category {category} with query type {type(cleaned)} '{str(cleaned)[:50]}...': {e}"
                )
                # Try to recover by ensuring we have a clean string
                try:
                    if cleaned is not None:
                        cleaned = str(cleaned).encode("utf-8", errors="ignore").decode("utf-8")
                    else:
                        cleaned = ""
                except Exception:
                    cleaned = ""
                continue

        # Clean up whitespace
        try:
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
        except (TypeError, re.error) as e:
            logger.warning(f"Whitespace cleanup failed: {e}")
            cleaned = " ".join(cleaned.split()) if cleaned else ""

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
        """Additional query cleaning operations with karaoke-specific enhancements."""

        # Ensure query is a string and handle edge cases
        if not isinstance(query, str):
            query = str(query) if query is not None else ""

        try:
            # Remove excessive punctuation with error handling
            query = re.sub(r"[^\w\s\-\'\"&]", " ", query)
        except (TypeError, re.error) as e:
            logger.warning(f"Punctuation removal failed: {e}")
            # Fallback to basic character filtering
            query = "".join(c if c.isalnum() or c.isspace() or c in "-'\"&" else " " for c in query)

        try:
            # Remove large ID numbers (> 1001) that are likely video IDs
            query = re.sub(r"\b\d{4,}\b", " ", query)

            # Remove smaller standalone numbers but keep years (1900-2099)
            query = re.sub(r"\b(?:(?:19|20)\d{2})\b", " YEAR ", query)  # Temporarily replace years
            query = re.sub(r"\b\d{1,3}\b", " ", query)  # Remove other small numbers
            query = re.sub(r"\bYEAR\b", "", query)  # Remove year placeholder (optional for search)
        except (TypeError, re.error) as e:
            logger.warning(f"Number removal failed: {e}")
            # Continue without number removal

        # Remove language prefixes at the beginning
        query = re.sub(
            r"^(?:DE|EN|FR|ES|IT|PT|NL|PL|RU|JP|KR|CN)\s+", "", query, flags=re.IGNORECASE
        )

        # Remove quality/format indicators
        quality_patterns = [
            r"\b(?:HD|HQ|4K|1080p|720p|480p|360p|UHD|FHD)\b",
            r"\b(?:high|low)\s+(?:quality|def|definition)\b",
            r"\b(?:ultra|full)\s+hd\b",
            r"\b(?:mp3|mp4|wav|flac|video|audio)\b",
        ]
        for pattern in quality_patterns:
            query = re.sub(pattern, " ", query, flags=re.IGNORECASE)

        # Remove timestamps and duration indicators
        query = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", " ", query)

        # Remove very short words (often acronyms or noise) but keep important short words
        # Ensure query is a string before splitting
        if not isinstance(query, str):
            query = str(query) if query is not None else ""
        words = query.split()
        important_short_words = {
            "a",
            "i",
            "am",
            "is",
            "be",
            "to",
            "of",
            "in",
            "on",
            "at",
            "by",
            "me",
            "my",
            "we",
            "up",
            "go",
            "no",
        }
        # Filter words with null safety
        words = [
            word
            for word in words
            if word and (len(word) >= 2 or word.lower() in important_short_words)
        ]

        # Remove duplicate words while preserving order
        seen = set()
        unique_words = []
        for word in words:
            if word:  # Null safety check
                word_lower = word.lower()
                if word_lower not in seen:
                    seen.add(word_lower)
                    unique_words.append(word)

        # Final cleanup - remove excessive whitespace
        cleaned = " ".join(unique_words)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned


class SERPCache:
    """SERP result caching with TTL support."""

    def __init__(self, max_entries: int = 10000, default_ttl_hours: int = 168):
        self.cache: Dict[str, SERPCacheEntry] = {}
        self.max_entries = max_entries
        self.default_ttl_hours = default_ttl_hours

    def _get_query_hash(self, query: str) -> str:
        """Generate hash for query caching."""
        # Ensure query is not None before hashing
        if not query:
            query = ""
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

        if not hasattr(self, "total_requests"):
            return 0.0

        hits = sum(entry.access_count for entry in self.cache.values())
        return hits / max(self.total_requests, 1)

    def set_total_requests(self, total_requests: int):
        self.total_requests = total_requests

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


class EnhancedWebSearchPass(ParsingPass):
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

    @property
    def pass_type(self) -> PassType:
        return PassType.WEB_SEARCH

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

            # Early input validation and sanitization
            if not title:
                logger.debug("Web search pass: empty title provided")
                return None

            # Ensure all inputs are properly converted to strings
            title = self._safe_string_convert(title, "title")
            description = self._safe_string_convert(description, "description")
            tags = self._safe_string_convert(tags, "tags")
            channel_name = self._safe_string_convert(channel_name, "channel_name")

            if not title or len(title.strip()) < 3:
                logger.debug(
                    f"Web search pass: title too short or invalid after conversion: '{title}'"
                )
                return None

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

        # Use the shared safe string conversion method
        title = self._safe_string_convert(title, "title")
        description = self._safe_string_convert(description, "description")
        tags = self._safe_string_convert(tags, "tags")

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
        try:
            quoted_parts = re.findall(r'["\u201c\u201d]([^"\u201c\u201d]+)["\u201c\u201d]', title)
            for quoted in quoted_parts:
                quoted = self._safe_string_convert(quoted, "quoted_part")
                if quoted:
                    quoted_cleaning = self.filler_processor.clean_query(quoted, "english")
                    if quoted_cleaning.cleaned_query and len(quoted_cleaning.cleaned_query) > 3:
                        quoted_cleaning.cleaning_method = "quoted_extraction"
                        quoted_cleaning.confidence_boost *= 1.1  # Boost for quoted text
                        queries.append(quoted_cleaning)
        except (TypeError, re.error) as e:
            logger.debug(f"Quote extraction failed for title '{title}': {e}")

        # Strategy 4: Try language-specific cleaning if non-English characters detected
        try:
            if re.search(r"[^\x00-\x7F]", title):  # Non-ASCII characters
                for lang in ["spanish", "french", "german", "portuguese"]:
                    lang_cleaning = self.filler_processor.clean_query(title, lang)
                    if lang_cleaning.cleaned_query != base_cleaning.cleaned_query:
                        lang_cleaning.cleaning_method = f"{lang}_specific"
                        queries.append(lang_cleaning)
        except (TypeError, re.error) as e:
            logger.debug(f"Language-specific cleaning failed for title '{title}': {e}")

        # Strategy 5: Minimal cleaning (keep more context)
        try:
            minimal_cleaned = re.sub(
                r"\b(?:karaoke|instrumental)\b", "", title, flags=re.IGNORECASE
            ).strip()
            minimal_cleaning = QueryCleaningResult(
                original_query=title,
                cleaned_query=minimal_cleaned,
                removed_terms=["karaoke", "instrumental"],
                confidence_boost=0.9,  # Lower boost for minimal cleaning
                cleaning_method="minimal",
            )
            if minimal_cleaning.cleaned_query != base_cleaning.cleaned_query:
                queries.append(minimal_cleaning)
        except (TypeError, re.error) as e:
            logger.debug(f"Minimal cleaning failed for title '{title}': {e}")

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

    async def _search_with_engine(self, query: str, max_retries: int = 3) -> List[Dict]:
        """Perform search using the enhanced search engine with retry logic and timeout."""

        retry_delay = 1.0  # Start with 1 second delay
        search_timeout = 8.0  # Timeout for individual search to prevent >12s operations

        for attempt in range(max_retries):
            try:
                # Use the existing enhanced search engine with timeout
                search_results = await asyncio.wait_for(
                    self.search_engine.search_videos(
                        query,
                        max_results=min(self.max_search_results, 15),  # Limit results for speed
                        use_cache=True,
                        enable_fallback=False,  # Disable fallback for speed
                    ),
                    timeout=search_timeout,
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

            except asyncio.TimeoutError:
                logger.warning(
                    f"Search attempt {attempt + 1} timed out after {search_timeout}s for '{query}'"
                )
                if attempt < max_retries - 1:
                    search_timeout += 2.0  # Increase timeout for next attempt
                    continue
                else:
                    logger.error(f"All {max_retries} search attempts timed out for '{query}'")
                    return []
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Search attempt {attempt + 1} failed for '{query}': {e}. Retrying in {retry_delay}s..."
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"All {max_retries} search attempts failed for '{query}': {e}")
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
                result.artist.lower() if result.artist else "",
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

    def _safe_string_convert(self, value, field_name: str) -> str:
        """Safely convert any value to a string with comprehensive error handling."""
        if value is None:
            return ""

        # If already a string, normalize it
        if isinstance(value, str):
            try:
                # Normalize unicode and handle mixed encodings
                import unicodedata

                normalized = unicodedata.normalize("NFKC", value)
                # Ensure it's valid UTF-8
                normalized.encode("utf-8")
                return normalized
            except (UnicodeError, UnicodeDecodeError, UnicodeEncodeError) as e:
                logger.debug(f"Unicode normalization/encoding failed for {field_name}: {e}")
                # Fallback to robust encoding
                try:
                    return value.encode("utf-8", errors="ignore").decode("utf-8")
                except Exception:
                    return str(value)
            except Exception as e:
                logger.debug(f"Unicode normalization failed for {field_name}: {e}")
                return str(value)

        # Handle bytes
        if isinstance(value, (bytes, bytearray)):
            try:
                return value.decode("utf-8", errors="ignore")
            except Exception:
                return str(value, errors="ignore")

        # Handle complex data types
        if isinstance(value, (list, dict, tuple, set)):
            try:
                import json

                return json.dumps(value, ensure_ascii=False, default=str)
            except Exception:
                return str(value)

        # Handle other types
        try:
            converted = str(value)
            # Ensure the converted string is properly encoded
            converted.encode("utf-8")
            return converted
        except UnicodeEncodeError:
            # Force UTF-8 encoding with error handling
            try:
                return str(value).encode("utf-8", errors="ignore").decode("utf-8")
            except Exception:
                return ""
        except Exception as e:
            logger.warning(f"Failed to convert {field_name} to string: {e}")
            return ""
