"""Enhanced search engine with multi-strategy, fuzzy matching, and intelligent ranking."""

import asyncio
import logging
import time
from typing import Dict, List, Optional

from .config import ScrapingConfig, SearchConfig
from .search.cache_manager import CacheManager
from .search.fuzzy_matcher import FuzzyMatcher
from .search.providers.base import SearchResult
from .search.providers.youtube import YouTubeSearchProvider
from .search.result_ranker import ResultRanker

try:
    from .search.providers.bing import BingSearchProvider

    HAS_BING = True
except ImportError:
    HAS_BING = False

try:
    from .search.providers.duckduckgo import DuckDuckGoSearchProvider

    HAS_DUCKDUCKGO = True
except ImportError:
    HAS_DUCKDUCKGO = False

logger = logging.getLogger(__name__)


class MultiStrategySearchEngine:
    """Enhanced search engine with multiple strategies and intelligent optimization."""

    def __init__(
        self, search_config: SearchConfig, scraping_config: ScrapingConfig, db_manager=None
    ):
        self.search_config = search_config
        self.scraping_config = scraping_config
        self.db_manager = db_manager

        # Initialize core components
        self.cache_manager = CacheManager(
            search_config.dict() if hasattr(search_config, "dict") else {}
        )
        self.result_ranker = ResultRanker(
            search_config.dict() if hasattr(search_config, "dict") else {}
        )
        self.fuzzy_matcher = FuzzyMatcher(
            search_config.dict() if hasattr(search_config, "dict") else {}
        )

        # Initialize search providers
        self.providers = {}
        self._initialize_providers()

        # Search strategy configuration
        self.fallback_threshold = getattr(search_config, "fallback_threshold", 10)
        self.max_fallback_providers = getattr(search_config, "max_fallback_providers", 2)
        self.enable_parallel_search = getattr(search_config, "enable_parallel_search", False)

        # Statistics and monitoring
        self.search_stats = {
            "total_searches": 0,
            "cache_hits": 0,
            "fallback_used": 0,
            "provider_usage": {},
            "average_response_time": 0.0,
        }

    def _initialize_providers(self):
        """Initialize available search providers."""
        # Primary provider (YouTube)
        self.providers["youtube"] = YouTubeSearchProvider(self.scraping_config)

        # Fallback providers
        if HAS_BING:
            self.providers["bing"] = BingSearchProvider()
        else:
            logger.info("Bing search provider not available")

        if HAS_DUCKDUCKGO:
            self.providers["duckduckgo"] = DuckDuckGoSearchProvider()
        else:
            logger.info("DuckDuckGo search provider not available")

        logger.info(
            f"Initialized {len(self.providers)} search providers: {list(self.providers.keys())}"
        )

    async def search_videos(
        self,
        query: str,
        max_results: int = 100,
        use_cache: bool = True,
        enable_fallback: bool = True,
    ) -> List[SearchResult]:
        """Enhanced search with multi-strategy approach."""
        start_time = time.time()
        self.search_stats["total_searches"] += 1

        try:
            # Step 1: Check cache first
            if use_cache:
                cached_results = await self._get_cached_results(query, max_results)
                if cached_results:
                    self.search_stats["cache_hits"] += 1
                    logger.info(f"Cache hit for query: '{query}' ({len(cached_results)} results)")
                    return await self._rank_and_return_results(cached_results, query)

            # Step 2: Primary search (YouTube)
            primary_results = await self._search_with_provider("youtube", query, max_results)

            # Step 3: Check if fallback is needed
            fallback_results = []
            if enable_fallback and len(primary_results) < self.fallback_threshold:
                fallback_results = await self._search_with_fallback_providers(
                    query, max_results, exclude=["youtube"]
                )
                self.search_stats["fallback_used"] += 1
                logger.info(f"Fallback search triggered for query: '{query}'")

            # Step 4: Combine and deduplicate results
            all_results = await self._combine_and_deduplicate_results(
                primary_results, fallback_results
            )

            # Step 5: Enhance with fuzzy matching if enabled
            if getattr(self.search_config, "enable_fuzzy_matching", True):
                all_results = await self._enhance_with_fuzzy_matching(all_results, query)

            # Step 6: Cache results for future use
            if use_cache and all_results:
                await self._cache_search_results(query, all_results, max_results)

            # Step 7: Rank and return final results
            final_results = await self._rank_and_return_results(all_results, query)

            # Update statistics
            response_time = time.time() - start_time
            self._update_search_statistics(response_time)

            logger.info(
                f"Search completed for '{query}': {len(final_results)} results in {response_time:.2f}s"
            )
            return final_results

        except Exception as e:
            logger.error(f"Search failed for '{query}': {e}")
            return []

    async def _get_cached_results(
        self, query: str, max_results: int
    ) -> Optional[List[SearchResult]]:
        """Get cached search results."""
        try:
            cached_data = await self.cache_manager.get_search_results(
                query, provider="multi", max_results=max_results
            )

            if cached_data:
                # Convert back to SearchResult objects
                return [SearchResult(**result_dict) for result_dict in cached_data]

        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")

        return None

    async def _search_with_provider(
        self, provider_name: str, query: str, max_results: int
    ) -> List[SearchResult]:
        """Search with a specific provider."""
        if provider_name not in self.providers:
            logger.warning(f"Provider '{provider_name}' not available")
            return []

        provider = self.providers[provider_name]

        try:
            # Check if provider is available
            if not await provider.is_available():
                logger.warning(f"Provider '{provider_name}' is currently unavailable")
                return []

            # Perform search
            results = await provider.search_videos(query, max_results)

            # Update usage statistics
            if provider_name not in self.search_stats["provider_usage"]:
                self.search_stats["provider_usage"][provider_name] = 0
            self.search_stats["provider_usage"][provider_name] += 1

            logger.info(f"Provider '{provider_name}' returned {len(results)} results for '{query}'")
            return results

        except Exception as e:
            logger.error(f"Search failed with provider '{provider_name}': {e}")
            return []

    async def _search_with_fallback_providers(
        self, query: str, max_results: int, exclude: List[str] = None
    ) -> List[SearchResult]:
        """Search with fallback providers."""
        exclude = exclude or []
        fallback_providers = [name for name in self.providers.keys() if name not in exclude]

        if not fallback_providers:
            return []

        # Sort providers by weight (descending)
        fallback_providers.sort(
            key=lambda name: self.providers[name].get_provider_weight(), reverse=True
        )

        # Limit number of fallback providers
        fallback_providers = fallback_providers[: self.max_fallback_providers]

        all_fallback_results = []

        if self.enable_parallel_search:
            # Parallel search with all fallback providers
            tasks = [
                self._search_with_provider(
                    provider_name, query, max_results // len(fallback_providers)
                )
                for provider_name in fallback_providers
            ]

            results_list = await asyncio.gather(*tasks, return_exceptions=True)

            for results in results_list:
                if isinstance(results, list):
                    all_fallback_results.extend(results)

        else:
            # Sequential search with fallback providers
            for provider_name in fallback_providers:
                results = await self._search_with_provider(
                    provider_name, query, max_results // len(fallback_providers)
                )
                all_fallback_results.extend(results)

                # Stop if we have enough results
                if len(all_fallback_results) >= max_results:
                    break

        return all_fallback_results

    async def _combine_and_deduplicate_results(
        self, primary_results: List[SearchResult], fallback_results: List[SearchResult]
    ) -> List[SearchResult]:
        """Combine results from multiple providers and remove duplicates."""
        # Use video_id as the deduplication key
        seen_video_ids = set()
        combined_results = []

        # Add primary results first (they have higher priority)
        for result in primary_results:
            if result.video_id not in seen_video_ids:
                seen_video_ids.add(result.video_id)
                combined_results.append(result)

        # Add fallback results, avoiding duplicates
        for result in fallback_results:
            if result.video_id not in seen_video_ids:
                seen_video_ids.add(result.video_id)
                combined_results.append(result)

        logger.info(
            f"Combined results: {len(primary_results)} primary + {len(fallback_results)} fallback = {len(combined_results)} unique"
        )
        return combined_results

    async def _enhance_with_fuzzy_matching(
        self, results: List[SearchResult], query: str
    ) -> List[SearchResult]:
        """Enhance results using fuzzy matching for better artist/song identification."""
        if not self.fuzzy_matcher:
            return results

        enhanced_results = []

        for result in results:
            try:
                # Try to extract better artist/song information using fuzzy matching
                # This would integrate with the advanced parser's fuzzy matching capabilities

                # For now, just add the original result
                # In a full implementation, you'd extract artist/song from the title
                # and use fuzzy matching to improve the extraction
                enhanced_results.append(result)

            except Exception as e:
                logger.warning(f"Fuzzy enhancement failed for result {result.video_id}: {e}")
                enhanced_results.append(result)

        return enhanced_results

    async def _cache_search_results(
        self, query: str, results: List[SearchResult], max_results: int
    ):
        """Cache search results for future use."""
        try:
            await self.cache_manager.cache_search_results(
                query, results, provider="multi", max_results=max_results
            )
            logger.debug(f"Cached {len(results)} results for query: '{query}'")

        except Exception as e:
            logger.warning(f"Failed to cache results: {e}")

    async def _rank_and_return_results(
        self, results: List[SearchResult], query: str
    ) -> List[SearchResult]:
        """Rank results and return the final list."""
        if not results:
            return []

        try:
            # Use the result ranker to score and sort results
            ranking_results = self.result_ranker.rank_results(results, query)

            # Extract the ranked SearchResult objects
            ranked_results = [ranking_result.result for ranking_result in ranking_results]

            # Update final scores in the SearchResult objects
            for ranking_result in ranking_results:
                ranking_result.result.final_score = ranking_result.final_score
                ranking_result.result.relevance_score = ranking_result.relevance_score
                ranking_result.result.quality_score = ranking_result.quality_score
                ranking_result.result.popularity_score = ranking_result.popularity_score
                ranking_result.result.metadata_score = ranking_result.metadata_score

            return ranked_results

        except Exception as e:
            logger.error(f"Ranking failed: {e}")
            # Return original results sorted by relevance score
            return sorted(results, key=lambda x: x.relevance_score, reverse=True)

    def _update_search_statistics(self, response_time: float):
        """Update search statistics."""
        # Update average response time with moving average
        if self.search_stats["total_searches"] == 1:
            self.search_stats["average_response_time"] = response_time
        else:
            alpha = 0.1  # Smoothing factor
            self.search_stats["average_response_time"] = (
                alpha * response_time + (1 - alpha) * self.search_stats["average_response_time"]
            )

    async def get_comprehensive_statistics(self) -> Dict:
        """Get comprehensive statistics from all components."""
        try:
            cache_stats = await self.cache_manager.get_comprehensive_stats()
            ranker_stats = self.result_ranker.get_statistics()
            fuzzy_stats = self.fuzzy_matcher.get_statistics() if self.fuzzy_matcher else {}

            provider_stats = {}
            for name, provider in self.providers.items():
                provider_stats[name] = provider.get_statistics()

            return {
                "search_engine": self.search_stats,
                "cache": cache_stats,
                "ranking": ranker_stats,
                "fuzzy_matching": fuzzy_stats,
                "providers": provider_stats,
            }

        except Exception as e:
            logger.error(f"Failed to get comprehensive statistics: {e}")
            return {"error": str(e)}

    async def warm_cache_for_popular_queries(self, popular_queries: List[str]):
        """Pre-warm cache with popular queries."""
        logger.info(f"Warming cache for {len(popular_queries)} popular queries")

        for query in popular_queries:
            try:
                # Perform search to populate cache
                await self.search_videos(query, max_results=50, use_cache=False)
                await asyncio.sleep(0.1)  # Small delay to prevent overwhelming providers

            except Exception as e:
                logger.warning(f"Cache warming failed for query '{query}': {e}")

    async def optimize_performance(self):
        """Optimize performance by cleaning up caches and analyzing patterns."""
        try:
            # Optimize caches
            cleanup_stats = await self.cache_manager.optimize_caches()

            # Log optimization results
            logger.info(f"Performance optimization completed: {cleanup_stats}")

            return cleanup_stats

        except Exception as e:
            logger.error(f"Performance optimization failed: {e}")
            return {"error": str(e)}

    async def search_with_fallback_strategies(
        self, query: str, max_results: int = 100
    ) -> List[SearchResult]:
        """Search with multiple fallback strategies for difficult queries."""
        strategies = [
            # Strategy 1: Original query
            query,
            # Strategy 2: Add "karaoke" if not present
            f"{query} karaoke" if "karaoke" not in query.lower() else None,
            # Strategy 3: Add "instrumental" as alternative
            f"{query} instrumental" if "instrumental" not in query.lower() else None,
            # Strategy 4: Remove artist if query seems to contain both artist and song
            self._extract_song_only(query) if " - " in query or " by " in query else None,
        ]

        # Filter out None strategies
        strategies = [s for s in strategies if s and s != query]

        best_results = []

        for strategy_query in [query] + strategies:
            try:
                results = await self.search_videos(
                    strategy_query, max_results, use_cache=True, enable_fallback=True
                )

                if len(results) > len(best_results):
                    best_results = results
                    logger.info(f"Better results found with strategy query: '{strategy_query}'")

                # If we have enough good results, stop trying strategies
                if len(results) >= max_results * 0.8:
                    break

            except Exception as e:
                logger.warning(f"Strategy query '{strategy_query}' failed: {e}")

        return best_results

    def _extract_song_only(self, query: str) -> Optional[str]:
        """Extract song title from artist - song format."""
        if " - " in query:
            parts = query.split(" - ", 1)
            if len(parts) == 2:
                return parts[1].strip()
        elif " by " in query:
            parts = query.split(" by ", 1)
            if len(parts) == 2:
                return parts[0].strip()

        return None
