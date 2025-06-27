"""Unit tests for video processing functionality."""

import asyncio
import sqlite3
from pathlib import Path

from collector.config import CollectorConfig
from collector.db import DatabaseConfig, DatabaseManager
from collector.processor import ProcessingResult, VideoProcessor


def test_extract_karaoke_features():
    """Test karaoke feature extraction with confidence scoring."""
    config = CollectorConfig()
    processor = VideoProcessor(config)

    video_data = {
        'title': 'Amazing Song - Artist Name (Karaoke with Guide Vocals and Scrolling Lyrics)',
        'description': 'Piano only acoustic version with backing vocals',
        'tags': ['karaoke', 'instrumental', 'piano']
    }

    features = processor._extract_karaoke_features(video_data)

    # Check that features are detected
    assert features['has_guide_vocals']
    assert features['has_scrolling_lyrics']
    assert features['is_piano_only']
    assert features['is_acoustic']

def test_extract_featured_artists():
    """Test featured artists extraction."""
    config = CollectorConfig()
    processor = VideoProcessor(config)

    # Test various patterns
    test_cases = [
        ("Song Title feat. Artist Name (Karaoke)", "Artist Name"),
        ("Amazing Song featuring John Doe & Jane Smith", "John Doe, Jane Smith"),
        ("Track ft. Featured Artist", "Featured Artist"),
        ("Song with Collaboration", "Collaboration"),
        ("Normal Song (Karaoke)", None),
    ]

    for title, expected in test_cases:
        result = processor._extract_featured_artists(title, "", "")
        if expected:
            assert result == expected, f"Failed for '{title}': got '{result}', expected '{expected}'"
        else:
            assert result is None, f"Failed for '{title}': got '{result}', expected None"

def test_like_dislike_ratio_calculation():
    """Test like/dislike ratio calculation during save."""
    import tempfile

    db_path = Path(tempfile.gettempdir()) / "test_ratio.db"

    try:
        db_manager = DatabaseManager(DatabaseConfig(
            path=str(db_path),
            backup_enabled=False
        ))

        # Mock result with engagement data
        mock_result = ProcessingResult(
            video_data={
                'video_id': 'test123',
                'url': 'https://example.com',
                'title': 'Test Video',
                'view_count': 1000,
                'like_count': 800,
                'estimated_dislikes': 50,  # 8% dislike ratio
                'features': {}
            },
            confidence_score=0.8,
            processing_time=1.0,
            errors=[],
            warnings=[]
        )

        db_manager.save_video_data(mock_result)

        # Check that the video record was created correctly and the ratio calculated
        with sqlite3.connect(db_path) as con:
            cursor = con.cursor()

            # Exactly one video row should exist
            count = cursor.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
            assert count == 1, f"Expected one video record, found {count}"

            # Fetch the stored video data
            row = cursor.execute(
                "SELECT video_id, url, title, view_count, like_count, like_dislike_to_views_ratio "
                "FROM videos WHERE video_id='test123'"
            ).fetchone()

            assert row is not None, "Video record was not saved"
            vid, url, title, views, likes, ratio = row

            assert vid == 'test123'
            assert url == 'https://example.com'
            assert title == 'Test Video'
            assert views == 1000
            assert likes == 800

            expected_ratio = (800 - 50) / 1000  # 0.75
            assert abs(ratio - expected_ratio) < 0.001, (
                f"Ratio calculation error: {ratio} vs {expected_ratio}"
            )

            # RYD data should also have been stored
            ryd = cursor.execute(
                "SELECT estimated_dislikes FROM ryd_data WHERE video_id='test123'"
            ).fetchone()
            assert ryd is not None and ryd[0] == 50

    finally:
        if db_path.exists():
            db_path.unlink()

def test_calculate_quality_scores():
    """Test quality score calculation including dislike penalty."""
    config = CollectorConfig()
    processor = VideoProcessor(config)

    # Test video with high dislike ratio
    video_data = {
        'duration_seconds': 240,  # Good duration
        'view_count': 10000,
        'like_count': 100,
        'estimated_dislikes': 200,  # High dislike ratio (66%)
        'ryd_confidence': 0.8,  # High confidence
        'formats': [{'height': 1080, 'abr': 192}],  # Good quality
        'title': 'Test Video',
        'description': 'Test description',
        'tags': ['test'],
        'features': {'original_artist': 'Test Artist'}
    }

    scores = processor._calculate_quality_scores(video_data)

    # Should have technical score but reduced engagement due to dislike penalty
    assert scores['technical_score'] > 0
    assert scores['engagement_score'] < scores['technical_score']  # Penalty applied
    assert scores['metadata_completeness'] > 0

def test_process_video_uses_music_metadata(monkeypatch):
    """Ensure MusicBrainz lookup occurs after feature extraction."""
    config = CollectorConfig()
    config.data_sources.ryd_api_enabled = False  # Avoid network in tests

    processor = VideoProcessor(config)

    async def dummy_extract_basic_metadata(url):
        return {
            'video_id': 'dummy',
            'title': 'Artist One - Song One (Karaoke)',
            'description': '',
            'tags': []
        }

    async def dummy_get_ryd_data(video_id):
        return {}

    called = {'count': 0}

    async def dummy_get_music_metadata(artist, song):
        called['count'] += 1
        return {'estimated_release_year': 2020, 'release_year_confidence': 0.9}

    monkeypatch.setattr(processor, '_extract_basic_metadata', dummy_extract_basic_metadata)
    monkeypatch.setattr(processor, '_get_ryd_data', dummy_get_ryd_data)
    monkeypatch.setattr(processor, '_get_music_metadata', dummy_get_music_metadata)

    async def run():
        result = await processor.process_video('http://example.com')
        await processor.cleanup()
        return result

    result = asyncio.run(run())

    assert called['count'] == 1
    assert result.video_data.get('estimated_release_year') == 2020

    scores = result.video_data.get('quality_scores', {})

    # Validate overall score weighting and range
    expected_overall = (
        scores['technical_score'] * 0.3 +
        scores['engagement_score'] * 0.4 +
        scores['metadata_completeness'] * 0.3
    )

    assert 0 <= scores['overall_score'] <= 1
    assert abs(scores['overall_score'] - expected_overall) < 1e-6
