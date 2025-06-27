"""Main collector orchestration."""

import asyncio
import logging
import signal
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
        self.processed_video_ids = self.db_manager.get_existing_video_ids()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True

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
            if vid in self.processed_video_ids:
                return
            async with sem:
                result = await self.video_processor.process_video(vrow["url"])
                if result.is_success:
                    self.db_manager.save_video_data(result)
                    self.processed_video_ids.add(vid)
                    processed += 1
                    logger.debug(
                        f"Processed: {result.video_data.get('title', 'Unknown')} "
                        f"(confidence: {result.confidence_score:.2f})"
                    )

        # duration filter (quick heuristic)
        new_rows = [
            v
            for v in video_rows
            if v["video_id"] not in self.processed_video_ids
            and (v.get("duration") or 0) >= 45
            and (v.get("duration") or 0) <= 900
        ]

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

        return processed

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
