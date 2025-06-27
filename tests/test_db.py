"""Unit tests for database functionality."""

import logging
import sqlite3
import time
from pathlib import Path

from collector.db import DatabaseConfig, DatabaseManager
from collector.processor import ProcessingResult


def test_trigger_migration(tmp_path):
    """Test that the updated_at trigger works correctly without recursion."""
    db_path = tmp_path / "test.db"
    DatabaseManager(
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
        DatabaseManager(
            DatabaseConfig(
                path=str(db_path),
                backup_enabled=False,
            )
        )

        with sqlite3.connect(db_path) as con:
            version = con.execute(
                "SELECT version FROM schema_info ORDER BY version DESC LIMIT 1"
            ).fetchone()[0]
            assert version == 7, f"Expected schema version 7, got {version}"
    finally:
        if db_path.exists():
            db_path.unlink()


def test_channel_indexes_created(tmp_path, caplog):
    """Ensure channel-related indexes exist without warnings."""
    db_path = tmp_path / "channels.db"

    with caplog.at_level(logging.WARNING, logger="collector.db"):
        DatabaseManager(DatabaseConfig(path=str(db_path), backup_enabled=False))

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert not warnings, f"Unexpected warnings: {[w.message for w in warnings]}"

    with sqlite3.connect(db_path) as con:
        idxs = {row[1] for row in con.execute("PRAGMA index_list('channels')")}
        assert "idx_channels_processed_karaoke" in idxs
        assert "idx_channels_subscriber_count" in idxs


def test_new_columns_exist():
    """Test that new columns exist in schema."""
    import tempfile

    db_path = Path(tempfile.gettempdir()) / "test_new_columns.db"

    try:
        DatabaseManager(
            DatabaseConfig(
                path=str(db_path),
                backup_enabled=False,
            )
        )

        with sqlite3.connect(db_path) as con:
            # Check that new columns exist in videos table
            cursor = con.execute("PRAGMA table_info(videos)")
            columns = [row[1] for row in cursor.fetchall()]

            expected_new_columns = ["featured_artists", "like_dislike_to_views_ratio"]

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
        DatabaseManager(
            DatabaseConfig(
                path=str(db_path),
                backup_enabled=False,
            )
        )

        with sqlite3.connect(db_path) as con:
            # Check that confidence columns exist
            cursor = con.execute("PRAGMA table_info(video_features)")
            columns = [row[1] for row in cursor.fetchall()]

            expected_confidence_columns = [
                "has_guide_vocals_confidence",
                "has_scrolling_lyrics_confidence",
                "has_backing_vocals_confidence",
                "is_instrumental_only_confidence",
                "is_piano_only_confidence",
                "is_acoustic_confidence",
            ]

            for col in expected_confidence_columns:
                assert col in columns, f"Missing confidence column: {col}"
    finally:
        if db_path.exists():
            db_path.unlink()


def test_validation_table_and_insert(tmp_path):
    db_path = tmp_path / "val.db"
    dbm = DatabaseManager(DatabaseConfig(path=str(db_path), backup_enabled=False))

    result = ProcessingResult(
        video_data={
            "video_id": "vid1",
            "url": "http://e",
            "title": "T",
            "features": {},
            "validation": {"artist_valid": True, "song_valid": False, "validation_score": 0.6},
            "correction_suggestion": {"suggested_title": "X", "reason": "r"},
        },
        confidence_score=0.5,
        processing_time=0.0,
        errors=[],
        warnings=[],
    )

    dbm.save_video_data(result)

    with sqlite3.connect(db_path) as con:
        row = con.execute(
            "SELECT artist_valid, song_valid, validation_score, suggested_title FROM validation_results WHERE video_id='vid1'"
        ).fetchone()
        assert row == (1, 0, 0.6, "X")


def test_pool_settings_applied(tmp_path):
    db_path = tmp_path / "pool.db"
    cfg = DatabaseConfig(
        path=str(db_path),
        backup_enabled=False,
        connection_pool_size=3,
        connection_timeout=5.0,
    )

    dbm = DatabaseManager(cfg)
    assert dbm._max_pool_size == 3
    assert dbm._pool_timeout == 5.0


def test_migrations_applied(tmp_path):
    """Ensure migrations 5-7 apply correctly on new setup."""
    db_path = tmp_path / "migrations.db"

    DatabaseManager(DatabaseConfig(path=str(db_path), backup_enabled=False))

    with sqlite3.connect(db_path) as con:
        versions = {row[0] for row in con.execute("SELECT version FROM schema_info").fetchall()}
        assert {5, 6, 7}.issubset(versions)

        # Migration 6 adds additional MusicBrainz columns
        columns = {row[1] for row in con.execute("PRAGMA table_info(videos)")}
        assert "record_label" in columns
        assert "recording_length_ms" in columns

        # Migration 7 introduces search cache table
        tables = {
            row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "search_cache" in tables
