import asyncio
from typing import cast

import pytest

import collector.main as main
from collector.config import CollectorConfig
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
            video_data={
                "video_id": url[-1],
                "url": url,
                "title": "x",
                "features": {},
                "view_count": 0,
            },
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


class DummyChannelSearch:
    def __init__(self):
        self.videos = [
            {"video_id": "c1", "url": "cu1", "title": "ct1", "duration": 70},
            {"video_id": "c2", "url": "cu2", "title": "ct2", "duration": 90},
        ]
        self.after_date = None

    async def extract_channel_info(self, url):
        return {"channel_id": "cid", "channel_name": "Test", "channel_url": url}

    async def extract_channel_videos(self, url, max_videos=None, after_date=None):
        self.after_date = after_date
        return self.videos


class DummyChannelDB(DummyDB):
    def __init__(self):
        super().__init__()
        self.channel_saved = []
        self.last_processed = None
        self.last_processed_called = False
        self.updated = False

    def save_channel_data(self, data):
        self.channel_saved.append(data)
        return True

    def get_channel_last_processed(self, channel_id):
        self.last_processed_called = True
        return self.last_processed

    def update_channel_processed(self, channel_id):
        self.updated = True
        return True


def test_collect_videos(monkeypatch):
    config = CollectorConfig()
    monkeypatch.setattr(main, "DatabaseManager", lambda cfg: DummyDB())
    collector = main.KaraokeCollector(config)
    collector.search_engine = cast(main.SearchEngine, DummySearch())
    collector.video_processor = cast(main.VideoProcessor, DummyProcessor())

    count = asyncio.run(collector.collect_videos(["test"], 2))
    assert count == 2
    assert cast(DummyDB, collector.db_manager).saved == ["1", "2"]


@pytest.mark.parametrize("incremental", [True, False])
def test_collect_from_channel_batch(monkeypatch, incremental):
    config = CollectorConfig()
    db = DummyChannelDB()
    db.last_processed = "2024-01-01"
    monkeypatch.setattr(main, "DatabaseManager", lambda cfg: db)
    collector = main.KaraokeCollector(config)
    search = DummyChannelSearch()
    collector.search_engine = cast(main.SearchEngine, search)

    async def fake_process(url):
        assert collector.video_processor.advanced_parser is not None
        return ProcessingResult(
            video_data={"video_id": url, "url": url, "title": "x", "features": {}, "view_count": 0},
            confidence_score=1.0,
            processing_time=0,
            errors=[],
            warnings=[],
        )

    collector.video_processor.process_video = fake_process

    captured = {}

    async def fake_batch(rows):
        captured["rows"] = rows
        for r in rows:
            await collector.video_processor.process_video(r["url"])
        return len(rows)

    monkeypatch.setattr(collector, "_process_video_batch", fake_batch)

    count = asyncio.run(
        collector.collect_from_channel("http://channel", max_videos=2, incremental=incremental)
    )

    assert count == 2
    assert captured["rows"] == search.videos
    if incremental:
        assert search.after_date == "2024-01-01"
        assert db.last_processed_called
    else:
        assert search.after_date is None
        assert not db.last_processed_called
    assert db.updated
