"""Main collector orchestration."""

import asyncio
import logging
import signal
import threading
import time
from typing import Any, Dict, List, Optional

from .config import CollectorConfig
from .db import DatabaseManager
from .processor import VideoProcessor
from .search import SearchEngine

try:
    from tqdm.asyncio import tqdm_asyncio

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm_asyncio: Optional[Any] = None

logger = logging.getLogger(__name__)


class KaraokeCollector:
    """Main collector orchestrating search, processing, and storage."""

    def __init__(self, config: CollectorConfig):
        self.config = config
        self.db_manager = DatabaseManager(config.database)
        self.search_engine = SearchEngine(config.search, config.scraping)
        self.video_processor = VideoProcessor(config)
        self.shutdown_requested = False
        self._memory_cache_limit = 50000  # Limit in-memory cache size
        self._memory_cache_cleanup_threshold = 60000  # Clean when this size reached
        self.processed_video_ids = set()  # Start with empty set, load as needed
        self._processed_ids_lock = threading.RLock()  # Use RLock for nested locking
        self._last_cache_cleanup = time.time()

        # Reduce lock contention with worker-local state
        self._worker_local_cache = {}  # Worker-specific caches
        self._global_stats_lock = threading.Lock()  # Separate lock for stats

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True

    def _cleanup_memory_cache(self, force: bool = False):
        """Cleanup memory cache to prevent unbounded growth."""
        current_time = time.time()

        # Check if cleanup is needed (lock-free check first)
        with self._processed_ids_lock:
            cache_age = current_time - self._last_cache_cleanup
            current_size = len(self.processed_video_ids)

            # Clean if cache is too large, or periodically (every 30 minutes), or forced
            should_clean = (
                current_size > self._memory_cache_cleanup_threshold
                or cache_age > 1800  # 30 minutes
                or force
            )

            if not should_clean:
                return

            # Prevent concurrent cleanup operations
            if hasattr(self, "_cleanup_in_progress") and self._cleanup_in_progress:
                return
            self._cleanup_in_progress = True

        try:
            # Perform cleanup outside of the main lock to reduce contention
            if current_size > self._memory_cache_limit:
                # Get recent video IDs from database (outside lock)
                recent_ids = self.db_manager.get_recent_video_ids(days=7)

                # Update cache atomically
                with self._processed_ids_lock:
                    old_size = len(self.processed_video_ids)
                    self.processed_video_ids = recent_ids
                    self._last_cache_cleanup = current_time
                    logger.info(
                        f"Memory cache cleaned: {old_size} -> {len(self.processed_video_ids)} video IDs"
                    )
            else:
                # Just update timestamp
                with self._processed_ids_lock:
                    self._last_cache_cleanup = current_time
        finally:
            # Always clear the cleanup flag
            with self._processed_ids_lock:
                self._cleanup_in_progress = False

    def _is_video_processed(self, video_id: str, worker_id: Optional[str] = None) -> bool:
        """Check if video is processed using worker-local cache + global cache + database fallback."""
        # Use worker-local cache to reduce lock contention
        if worker_id:
            worker_cache = self._worker_local_cache.get(worker_id)
            if worker_cache and video_id in worker_cache:
                return True

        # Check global memory cache with minimal lock time
        with self._processed_ids_lock:
            if video_id in self.processed_video_ids:
                # Update worker-local cache
                if worker_id:
                    if worker_id not in self._worker_local_cache:
                        self._worker_local_cache[worker_id] = set()
                    self._worker_local_cache[worker_id].add(video_id)
                    # Limit worker cache size
                    if len(self._worker_local_cache[worker_id]) > 1000:
                        self._worker_local_cache[worker_id].clear()
                return True

        # If not in cache, check database (expensive operation outside lock)
        if self.db_manager.video_exists(video_id):
            # Add to both caches
            with self._processed_ids_lock:
                self.processed_video_ids.add(video_id)
                # Periodic cleanup check
                if len(self.processed_video_ids) > self._memory_cache_cleanup_threshold:
                    self._cleanup_memory_cache()

            if worker_id:
                if worker_id not in self._worker_local_cache:
                    self._worker_local_cache[worker_id] = set()
                self._worker_local_cache[worker_id].add(video_id)

            return True

        return False

    async def collect_videos(self, queries: List[str], max_videos_per_query: int = 100) -> int:
        """Main collection method."""
        start_time = time.time()
        total_processed = 0

        for query in queries:
            if self.shutdown_requested:
                break

            logger.info(f"Processing query: '{query}'")

            try:
                # Search for videos
                video_urls = await self.search_engine.search_videos(query, max_videos_per_query)
                logger.info(f"Found {len(video_urls)} videos for '{query}'")

                # Process videos
                processed_count = await self._process_video_batch(video_urls)
                total_processed += processed_count

            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")

        collection_time = time.time() - start_time
        logger.info(f"Collection completed: {total_processed} videos in {collection_time:.1f}s")
        return total_processed

    async def _process_video_batch(self, video_rows: List[Dict]) -> int:
        """
        Fully-async batch runner with optional progress bar:
          • uses one asyncio.Semaphore to cap concurrency
          • no ThreadPool / no new event loops per video
          • shows progress bar if tqdm available and UI config enabled
        """
        sem = asyncio.Semaphore(self.config.scraping.max_concurrent_workers)
        processed = 0

        async def _worker(vrow: Dict):
            nonlocal processed
            vid = vrow["video_id"]
            video_title = vrow.get("title", "Unknown")
            current_task = asyncio.current_task()
            if current_task is not None:
                worker_id = f"{current_task.get_name()}_{id(current_task)}"
            else:
                worker_id = "unknown"

            # Use optimized video processed check with worker-local cache
            if self._is_video_processed(vid, worker_id):
                return

            async with sem:
                try:
                    result = await self.video_processor.process_video(vrow["url"])
                    if result.is_success:
                        if self.db_manager.save_video_data(result):
                            # Thread-safe updates to shared state (minimized lock time)
                            with self._processed_ids_lock:
                                self.processed_video_ids.add(vid)

                            # Update worker-local cache
                            if worker_id not in self._worker_local_cache:
                                self._worker_local_cache[worker_id] = set()
                            self._worker_local_cache[worker_id].add(vid)

                            # Update processed count with separate lock
                            with self._global_stats_lock:
                                processed += 1

                            logger.debug(
                                f"Processed: {result.video_data.get('title', video_title)} "
                                f"(confidence: {result.confidence_score:.2f})"
                            )
                        else:
                            logger.warning(f"Failed to save video data: {video_title} ({vid})")
                    else:
                        logger.warning(
                            f"Failed to process video: {video_title} ({vid}) - {result.errors}"
                        )
                except Exception as e:
                    logger.error(f"Unexpected error processing video: {video_title} ({vid}) - {e}")
                    # Continue with next video instead of crashing

        # Get video IDs that might be new
        potential_video_ids = [v["video_id"] for v in video_rows]

        # Check database for existing videos (batch operation for efficiency)
        existing_in_db = self.db_manager.get_existing_video_ids_batch(potential_video_ids)

        # Combined filter: duration + not processed + not in database (optimized with batch checking)
        filtered_videos = []
        for v in video_rows:
            if (
                v["video_id"] not in existing_in_db
                and (v.get("duration") or 0) >= 45
                and (v.get("duration") or 0) <= 900
                and not self._is_video_processed(v["video_id"])
            ):
                filtered_videos.append(v)

        new_rows = filtered_videos

        if existing_in_db:
            logger.info(f"Skipped {len(existing_in_db)} videos already in database")

        if not new_rows:
            logger.info("All videos already processed or filtered out, skipping batch")
            return 0

        logger.info(f"Processing {len(new_rows)} new videos (after duration filter)")

        # Use progress bar if available and enabled
        if (
            HAS_TQDM and tqdm_asyncio and self.config.ui.show_progress_bar and len(new_rows) > 10
        ):  # Only show for larger batches

            tasks = [_worker(v) for v in new_rows]
            desc = (
                f"Processing videos (max {self.config.scraping.max_concurrent_workers} concurrent)"
            )
            await tqdm_asyncio.gather(*tasks, desc=desc, unit="video")
        else:
            await asyncio.gather(*[_worker(v) for v in new_rows])

        # Cleanup memory periodically and clear worker caches
        if len(new_rows) > 1000:  # Only for large batches
            self._cleanup_memory_cache()
            # Clear worker-local caches to prevent memory leaks
            self._worker_local_cache.clear()

        return processed

    async def collect_from_channel(
        self, channel_url: str, max_videos: Optional[int] = None, incremental: bool = True
    ) -> int:
        """Collect videos from a specific YouTube channel."""
        start_time = time.time()
        total_processed = 0

        try:
            # Extract channel information first
            logger.info(f"Processing channel: {channel_url}")
            channel_data = await self.search_engine.extract_channel_info(channel_url)

            if not channel_data or not channel_data.get("channel_id"):
                logger.error(f"Failed to extract channel info from {channel_url}")
                return 0

            # Save channel data
            self.db_manager.save_channel_data(channel_data)
            channel_id = channel_data["channel_id"]

            # Check if incremental and when last processed
            after_date = None
            if incremental:
                last_processed = self.db_manager.get_channel_last_processed(channel_id)
                if last_processed:
                    logger.info(f"Channel last processed: {last_processed}")
                    # Convert timestamp to date string for filtering
                    after_date = (
                        last_processed.split(" ")[0] if " " in last_processed else last_processed
                    )

            # Extract videos from channel (with date filtering for incremental)
            video_list = await self.search_engine.extract_channel_videos(
                channel_url, max_videos, after_date
            )

            if not video_list:
                logger.warning(f"No videos found in channel: {channel_data.get('channel_name')}")
                return 0

            logger.info(
                f"Found {len(video_list)} videos in channel: {channel_data.get('channel_name')}"
            )

            # Process videos using existing batch processing
            processed_count = await self._process_video_batch(video_list)
            total_processed += processed_count

            # Update channel processed timestamp
            self.db_manager.update_channel_processed(channel_id)

            collection_time = time.time() - start_time
            logger.info(
                f"Channel collection completed: {total_processed} videos in {collection_time:.1f}s"
            )
            return total_processed

        except Exception as e:
            logger.error(f"Error processing channel '{channel_url}': {e}")
            return 0

    async def collect_from_channels(
        self, channel_urls: List[str], max_videos_per_channel: Optional[int] = None
    ) -> int:
        """Collect videos from multiple YouTube channels with error recovery."""
        start_time = time.time()
        total_processed = 0
        successful_channels = 0
        failed_channels = []

        for i, channel_url in enumerate(channel_urls, 1):
            if self.shutdown_requested:
                logger.info("Shutdown requested, stopping channel processing")
                break

            logger.info(f"Processing channel {i}/{len(channel_urls)}: {channel_url}")

            try:
                processed_count = await self.collect_from_channel(
                    channel_url, max_videos_per_channel, incremental=True
                )
                total_processed += processed_count
                successful_channels += 1

                if processed_count > 0:
                    logger.info(f"✓ Channel completed: {processed_count} videos processed")
                else:
                    logger.info("✓ Channel completed: no new videos found")

            except Exception as e:
                failed_channels.append(channel_url)
                logger.error(f"✗ Channel failed: {channel_url} - {e}")

                # Continue with next channel after error
                logger.info("Continuing with next channel...")

            # Add delay between channels to be respectful
            if i < len(channel_urls):
                await asyncio.sleep(2)

        collection_time = time.time() - start_time

        # Summary statistics
        logger.info("=" * 60)
        logger.info("MULTI-CHANNEL COLLECTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total videos processed: {total_processed}")
        logger.info(f"Successful channels: {successful_channels}/{len(channel_urls)}")
        logger.info(f"Failed channels: {len(failed_channels)}")
        logger.info(f"Collection time: {collection_time:.1f}s")

        if failed_channels:
            logger.warning("Failed channel URLs:")
            for failed_url in failed_channels:
                logger.warning(f"  - {failed_url}")

        return total_processed

    async def get_channel_statistics(self) -> Dict:
        """Get statistics about processed channels."""
        channels = self.db_manager.get_processed_channels()

        stats = {
            "total_channels": len(channels),
            "karaoke_focused_channels": sum(1 for c in channels if c["is_karaoke_focused"]),
            "total_videos_from_channels": sum(c["collected_videos"] for c in channels),
            "channels": channels,
        }

        return stats

    async def get_statistics(self) -> Dict:
        """Get comprehensive collection statistics."""
        return self.db_manager.get_statistics()

    async def cleanup(self):
        """Cleanup resources."""
        try:
            await self.video_processor.cleanup()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
