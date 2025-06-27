"""Unit tests for database functionality."""

import pytest
import sqlite3
import time
from pathlib import Path
from collector.db import DatabaseManager, DatabaseConfig

def test_trigger_migration(tmp_path):
    """Test that the updated_at trigger works correctly without recursion."""
    db_path = tmp_path / "test.db"
    db_manager = DatabaseManager(DatabaseConfig(
        path=str(db_path), 
        backup_enabled=False,
        vacuum_on_startup=False
    ))
    
    with sqlite3.connect(db_path) as con:
        # Insert a test video
        con.execute("INSERT INTO videos(video_id,url,title) VALUES('test123','https://example.com','Test Video')")
        before = con.execute("SELECT updated_at FROM videos WHERE video_id='test123'").fetchone()[0]
        
        # Wait a moment to ensure timestamp difference
        time.sleep(1)
        
        # Update the video (should trigger updated_at change)
        con.execute("UPDATE videos SET title='Updated Title' WHERE video_id='test123'")
        after = con.execute("SELECT updated_at FROM videos WHERE video_id='test123'").fetchone()[0]
    
    # Verify the trigger fired and updated the timestamp
    assert before != after, "updated_at should change when other columns are modified"

def test_database_schema_version():
    """Test that database initializes with correct schema version."""
    import tempfile
    db_path = Path(tempfile.gettempdir()) / "test_schema.db"
    
    try:
        db_manager = DatabaseManager(DatabaseConfig(
            path=str(db_path),
            backup_enabled=False
        ))
        
        with sqlite3.connect(db_path) as con:
            version = con.execute("SELECT version FROM schema_info ORDER BY version DESC LIMIT 1").fetchone()[0]
            assert version == 3, f"Expected schema version 3, got {version}"
    finally:
        if db_path.exists():
            db_path.unlink()

def test_new_columns_exist():
    """Test that new columns exist in schema."""
    import tempfile
    db_path = Path(tempfile.gettempdir()) / "test_new_columns.db"
    
    try:
        db_manager = DatabaseManager(DatabaseConfig(
            path=str(db_path),
            backup_enabled=False
        ))
        
        with sqlite3.connect(db_path) as con:
            # Check that new columns exist in videos table
            cursor = con.execute("PRAGMA table_info(videos)")
            columns = [row[1] for row in cursor.fetchall()]
            
            expected_new_columns = ['featured_artists', 'like_dislike_to_views_ratio']
            
            for col in expected_new_columns:
                assert col in columns, f"Missing new column: {col}"
    finally:
        if db_path.exists():
            db_path.unlink()

def test_feature_confidence_columns():
    """Test that feature confidence columns exist in schema."""
    import tempfile
    db_path = Path(tempfile.gettempdir()) / "test_features.db"
    
    try:
        db_manager = DatabaseManager(DatabaseConfig(
            path=str(db_path),
            backup_enabled=False
        ))
        
        with sqlite3.connect(db_path) as con:
            # Check that confidence columns exist
            cursor = con.execute("PRAGMA table_info(video_features)")
            columns = [row[1] for row in cursor.fetchall()]
            
            expected_confidence_columns = [
                'has_guide_vocals_confidence',
                'has_scrolling_lyrics_confidence', 
                'has_backing_vocals_confidence',
                'is_instrumental_only_confidence',
                'is_piano_only_confidence',
                'is_acoustic_confidence'
            ]
            
            for col in expected_confidence_columns:
                assert col in columns, f"Missing confidence column: {col}"
    finally:
        if db_path.exists():
            db_path.unlink()