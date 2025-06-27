"""Unit tests for video processing functionality."""

from pathlib import Path
import sqlite3
from collector.processor import VideoProcessor, ProcessingResult
from collector.config import CollectorConfig
from collector.db import DatabaseManager, DatabaseConfig

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
        
        # Check that ratio was calculated correctly
        with sqlite3.connect(db_path) as con:
            ratio = con.execute(
                "SELECT like_dislike_to_views_ratio FROM videos WHERE video_id='test123'"
            ).fetchone()[0]
            
            expected_ratio = (800 - 50) / 1000  # 0.75
            assert abs(ratio - expected_ratio) < 0.001, f"Ratio calculation error: {ratio} vs {expected_ratio}"
    
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