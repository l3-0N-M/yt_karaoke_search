"""Unit tests for database functionality."""

import logging
import sqlite3
from pathlib import Path

from collector.config import DatabaseConfig
from collector.db_optimized import OptimizedDatabaseManager
from collector.processor import ProcessingResult


def test_trigger_migration(tmp_path):
    """Test basic database creation with optimized schema."""
    db_path = tmp_path / "test.db"
    OptimizedDatabaseManager(
        DatabaseConfig(
            path=str(db_path),
            backup_enabled=False,
            vacuum_on_startup=False,
        )
    )

    with sqlite3.connect(db_path) as con:
        # Insert a test video
        con.execute(
            "INSERT INTO videos(video_id,url,title) VALUES('test123','https://example.com','Test Video')"
        )

        # Check the video was inserted
        result = con.execute("SELECT video_id FROM videos WHERE video_id='test123'").fetchone()
        assert result is not None, "Video should be inserted successfully"


def test_database_schema_version():
    """Test that database initializes with correct schema version."""
    import tempfile

    db_path = Path(tempfile.gettempdir()) / "test_schema.db"

    try:
        OptimizedDatabaseManager(
            DatabaseConfig(
                path=str(db_path),
                backup_enabled=False,
            )
        )

        with sqlite3.connect(db_path) as con:
            version = con.execute(
                "SELECT version FROM schema_info ORDER BY version DESC LIMIT 1"
            ).fetchone()[0]
            assert version == 2, f"Expected schema version 2, got {version}"
    finally:
        if db_path.exists():
            db_path.unlink()


def test_channel_indexes_created(tmp_path, caplog):
    """Ensure channel-related indexes exist without warnings."""
    db_path = tmp_path / "channels.db"

    with caplog.at_level(logging.WARNING, logger="collector.db_optimized"):
        OptimizedDatabaseManager(DatabaseConfig(path=str(db_path), backup_enabled=False))

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert not warnings, f"Unexpected warnings: {[w.message for w in warnings]}"

    with sqlite3.connect(db_path) as con:
        idxs = {row[1] for row in con.execute("PRAGMA index_list('channels')")}
        # Check for indexes that exist in optimized schema
        assert "idx_channels_name" in idxs


def test_new_columns_exist():
    """Test that new columns exist in schema."""
    import tempfile

    db_path = Path(tempfile.gettempdir()) / "test_new_columns.db"

    try:
        OptimizedDatabaseManager(
            DatabaseConfig(
                path=str(db_path),
                backup_enabled=False,
            )
        )

        with sqlite3.connect(db_path) as con:
            # Check that new columns exist in videos table
            cursor = con.execute("PRAGMA table_info(videos)")
            columns = [row[1] for row in cursor.fetchall()]

            expected_new_columns = ["featured_artists", "engagement_ratio"]

            for col in expected_new_columns:
                assert col in columns, f"Missing new column: {col}"
    finally:
        if db_path.exists():
            db_path.unlink()


def test_feature_columns():
    """Test that feature columns exist in optimized schema."""
    import tempfile

    db_path = Path(tempfile.gettempdir()) / "test_features.db"

    try:
        OptimizedDatabaseManager(
            DatabaseConfig(
                path=str(db_path),
                backup_enabled=False,
            )
        )

        with sqlite3.connect(db_path) as con:
            # Check that basic feature columns exist
            cursor = con.execute("PRAGMA table_info(video_features)")
            columns = [row[1] for row in cursor.fetchall()]

            expected_basic_columns = [
                "has_guide_vocals",
                "has_scrolling_lyrics",
                "has_backing_vocals",
                "is_instrumental",
                "is_piano_only",
                "is_acoustic",
                "overall_confidence",
            ]

            for col in expected_basic_columns:
                assert col in columns, f"Missing feature column: {col}"
    finally:
        if db_path.exists():
            db_path.unlink()


def test_video_data_insert(tmp_path):
    """Test basic video data insert into optimized schema."""
    db_path = tmp_path / "val.db"
    dbm = OptimizedDatabaseManager(DatabaseConfig(path=str(db_path), backup_enabled=False))

    result = ProcessingResult(
        video_data={
            "video_id": "vid1",
            "url": "http://e",
            "title": "T",
            "view_count": 1000,
            "like_count": 50,
            "comment_count": 5,
        },
        confidence_score=0.5,
        processing_time=0.0,
        errors=[],
        warnings=[],
    )

    saved = dbm.save_video_data(result)
    assert saved is not False  # Should not fail to save

    with sqlite3.connect(db_path) as con:
        row = con.execute(
            "SELECT video_id, url, title, view_count FROM videos WHERE video_id='vid1'"
        ).fetchone()
        assert row is not None
        assert row[0] == "vid1"


def test_pool_settings_applied(tmp_path):
    db_path = tmp_path / "pool.db"
    cfg = DatabaseConfig(
        path=str(db_path),
        backup_enabled=False,
        connection_pool_size=3,
        connection_timeout=5.0,
    )

    dbm = OptimizedDatabaseManager(cfg)
    assert dbm._max_pool_size == 3
    assert dbm._pool_timeout == 5.0


def test_optimized_schema_applied(tmp_path):
    """Ensure optimized schema is applied correctly."""
    db_path = tmp_path / "migrations.db"

    OptimizedDatabaseManager(DatabaseConfig(path=str(db_path), backup_enabled=False))

    with sqlite3.connect(db_path) as con:
        versions = {row[0] for row in con.execute("SELECT version FROM schema_info").fetchall()}
        assert {2}.issubset(versions)

        # Check optimized schema columns
        columns = {row[1] for row in con.execute("PRAGMA table_info(videos)")}
        assert "engagement_ratio" in columns
        assert "artist" in columns
        assert "parse_confidence" in columns

        # Check core tables exist
        tables = {
            row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "videos" in tables
        assert "channels" in tables
        assert "video_features" in tables
