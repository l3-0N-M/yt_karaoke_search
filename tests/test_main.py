import asyncio
import collector.main as main
from collector.config import CollectorConfig
from collector.processor import ProcessingResult
from collector.config import CollectorConfig
import collector.main as main
from collector.processor import ProcessingResult


class DummySearch:
    async def search_videos(self, query, max_results):
        return [
            {"video_id": "a1", "url": "u1", "title": "t1", "duration": 60},
            {"video_id": "a2", "url": "u2", "title": "t2", "duration": 80},
        ]

class DummyProcessor:
    async def process_video(self, url):
        return ProcessingResult(
            video_data={"video_id": url[-1], "url": url, "title": "x", "features": {}, "view_count": 0},
            confidence_score=1.0,
            processing_time=0,
            errors=[],
            warnings=[],
        )
    async def cleanup(self):
        pass

class DummyDB:
    def __init__(self):
        self.saved = []
    def get_recent_video_ids(self, days=7):
        return set()
    def video_exists(self, vid):
        return False
    def get_existing_video_ids_batch(self, vids):
        return set()
    def save_video_data(self, result):
        self.saved.append(result.video_data["video_id"])
        return True
    def get_statistics(self):
        return {"total_videos": len(self.saved)}


def test_collect_videos(monkeypatch):
    config = CollectorConfig()
    monkeypatch.setattr(main, "DatabaseManager", lambda cfg: DummyDB())
    collector = main.KaraokeCollector(config)
    collector.search_engine = DummySearch()
    collector.video_processor = DummyProcessor()

    count = asyncio.run(collector.collect_videos(["test"], 2))
    assert count == 2
    assert collector.db_manager.saved == ["1", "2"]

