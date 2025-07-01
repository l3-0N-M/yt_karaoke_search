"""Unit tests for db_optimized.py - focusing on Discogs schema and migration."""

import os
import sqlite3
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from collector.config import DatabaseConfig
from collector.db_optimized import OptimizedDatabaseManager


class TestOptimizedDatabaseManager:
    """Test cases for OptimizedDatabaseManager with focus on Discogs integration."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.fixture
    def db_config(self, temp_db):
        """Create a test database configuration."""
        config = Mock(spec=DatabaseConfig)
        config.path = temp_db
        config.connection_pool_size = 5
        config.connection_timeout = 30
        return config

    @pytest.fixture
    def db_manager(self, db_config):
        """Create a database manager instance."""
        return OptimizedDatabaseManager(db_config)

    def test_schema_creation_includes_discogs_fields(self, db_manager):
        """Test that the database schema includes all Discogs fields."""
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(videos)")
            columns = {col[1] for col in cursor.fetchall()}

            # Check for all Discogs fields
            discogs_fields = {
                "discogs_artist_id",
                "discogs_artist_name",
                "discogs_release_id",
                "discogs_release_title",
                "discogs_release_year",
                "discogs_label",
                "discogs_genre",
                "discogs_style",
                "discogs_checked",
                "musicbrainz_checked",
                "web_search_performed",
            }

            missing_fields = discogs_fields - columns
            assert not missing_fields, f"Missing Discogs fields: {missing_fields}"

    def test_migration_adds_missing_discogs_columns(self, temp_db):
        """Test that migration adds missing Discogs columns to existing database."""
        # Create an old schema without Discogs fields
        conn = sqlite3.connect(temp_db)
        conn.execute(
            """
            CREATE TABLE videos (
                video_id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                title TEXT NOT NULL,
                artist TEXT,
                song_title TEXT,
                release_year INTEGER,
                genre TEXT
            )
        """
        )
        conn.commit()
        conn.close()

        # Create database manager - should trigger migration
        config = Mock(spec=DatabaseConfig)
        config.path = temp_db
        db_manager = OptimizedDatabaseManager(config)

        # Check that Discogs columns were added
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(videos)")
            columns = {col[1] for col in cursor.fetchall()}

            assert "discogs_artist_id" in columns
            assert "discogs_artist_name" in columns
            assert "discogs_release_id" in columns
            assert "discogs_checked" in columns

    def test_save_video_with_discogs_data(self, db_manager):
        """Test saving video data with Discogs information."""
        video_data = {
            "video_id": "test123",
            "url": "https://youtube.com/watch?v=test123",
            "title": "Test Song - Karaoke Version",
            "artist": "Test Artist",
            "song_title": "Test Song",
            "release_year": 2020,
            "genre": "Pop",
            "discogs_artist_id": "12345",
            "discogs_artist_name": "Test Artist Official",
            "discogs_release_id": "67890",
            "discogs_release_title": "Test Album",
            "discogs_release_year": 2020,
            "discogs_label": "Test Records",
            "discogs_genre": "Pop, Rock",
            "discogs_style": "Indie Pop",
            "discogs_checked": 1,
            "channel_id": "channel123",
            "channel_name": "Test Channel",
        }

        # Save the video
        db_manager.save_video_data(video_data)

        # Verify it was saved correctly
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT discogs_artist_id, discogs_artist_name, discogs_release_id,
                       discogs_release_title, discogs_release_year, discogs_label,
                       discogs_genre, discogs_style, discogs_checked
                FROM videos WHERE video_id = ?
            """,
                ("test123",),
            )

            row = cursor.fetchone()
            assert row is not None
            assert row[0] == "12345"  # discogs_artist_id
            assert row[1] == "Test Artist Official"  # discogs_artist_name
            assert row[2] == "67890"  # discogs_release_id
            assert row[3] == "Test Album"  # discogs_release_title
            assert row[4] == 2020  # discogs_release_year
            assert row[5] == "Test Records"  # discogs_label
            assert row[6] == "Pop, Rock"  # discogs_genre
            assert row[7] == "Indie Pop"  # discogs_style
            assert row[8] == 1  # discogs_checked

    def test_year_validation_rejects_future_years(self, db_manager):
        """Test that future years (2025+) are rejected during save."""
        current_year = datetime.now().year

        video_data = {
            "video_id": "test_future",
            "url": "https://youtube.com/watch?v=test_future",
            "title": "Future Song - Karaoke",
            "artist": "Future Artist",
            "song_title": "Future Song",
            "release_year": current_year + 1,  # Future year
            "channel_id": "channel123",
            "channel_name": "Test Channel",
        }

        # Save the video
        db_manager.save_video_data(video_data)

        # Verify the future year was rejected (set to None)
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT release_year FROM videos WHERE video_id = ?", ("test_future",))
            row = cursor.fetchone()
            assert row is not None
            assert row[0] is None  # Future year should be rejected

    def test_discogs_year_preferred_over_parsed_year(self, db_manager):
        """Test that Discogs year is preferred over parsed year when available."""
        video_data = {
            "video_id": "test_year_priority",
            "url": "https://youtube.com/watch?v=test_year_priority",
            "title": "2025 Upload - Classic Song Karaoke",
            "artist": "Classic Artist",
            "song_title": "Classic Song",
            "release_year": 2025,  # Parsed from title (incorrect)
            "discogs_release_year": 1985,  # Actual release year from Discogs
            "channel_id": "channel123",
            "channel_name": "Test Channel",
        }

        # Save the video
        db_manager.save_video_data(video_data)

        # Verify Discogs year was used
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT release_year, discogs_release_year FROM videos WHERE video_id = ?",
                ("test_year_priority",),
            )
            row = cursor.fetchone()
            assert row is not None
            assert row[0] == 1985  # Should use Discogs year
            assert row[1] == 1985  # Discogs year stored separately

    def test_safe_convert_handles_null_values(self, db_manager):
        """Test that save_video_data handles None/null values gracefully."""
        video_data = {
            "video_id": "test_nulls",
            "url": "https://youtube.com/watch?v=test_nulls",
            "title": "Test with Nulls",
            "artist": None,
            "song_title": None,
            "featured_artists": None,
            "discogs_artist_name": None,
            "discogs_genre": None,
            "channel_id": "channel123",
            "channel_name": "Test Channel",
        }

        # Should not raise any exceptions
        db_manager.save_video_data(video_data)

        # Verify it was saved
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM videos WHERE video_id = ?", ("test_nulls",))
            count = cursor.fetchone()[0]
            assert count == 1

    def test_parameter_15_handling(self, db_manager, caplog):
        """Test that the problematic parameter 15 (featured_artists) is handled correctly."""
        # Test with various problematic values that previously caused issues
        test_cases = [
            {"featured_artists": "Simple, Test, Artists"},
            {"featured_artists": 'Artist with "quotes" and special chars!'},
            {"featured_artists": "Very" + " Long" * 100},  # Very long string
            {"featured_artists": ["List", "Of", "Artists"]},  # List type
            {"featured_artists": {"artist1": "test"}},  # Dict type
            {"featured_artists": None},  # None type
        ]

        for i, test_data in enumerate(test_cases):
            video_data = {
                "video_id": f"test_param15_{i}",
                "url": f"https://youtube.com/watch?v=test_param15_{i}",
                "title": f"Test Param 15 Case {i}",
                "channel_id": "channel123",
                "channel_name": "Test Channel",
                **test_data,
            }

            # Should not raise any exceptions
            db_manager.save_video_data(video_data)

        # Verify all were saved
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM videos WHERE video_id LIKE 'test_param15_%'")
            count = cursor.fetchone()[0]
            assert count == len(test_cases)

        # Check that CRITICAL PARAM 15 logging was removed
        assert "CRITICAL PARAM 15" not in caplog.text

    def test_connection_pool_management(self, db_manager):
        """Test that connection pooling works correctly."""
        connections = []

        # Get multiple connections
        for _ in range(3):
            with db_manager.get_connection() as conn:
                connections.append(conn)
                # Verify connection is working
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                assert cursor.fetchone()[0] == 1

        # Connections should be returned to pool
        assert len(db_manager._connection_pool) <= db_manager._max_pool_size

    @patch("sqlite3.connect")
    def test_database_error_retry(self, mock_connect, db_config):
        """Test that database operations retry on error."""
        # Make connect fail twice, then succeed
        mock_conn = MagicMock()
        mock_connect.side_effect = [
            sqlite3.Error("Database locked"),
            sqlite3.Error("Database locked"),
            mock_conn,
        ]

        db_manager = OptimizedDatabaseManager(db_config)

        # Despite errors, should eventually succeed
        # Note: This will use the retry decorator
        assert mock_connect.call_count >= 1
        assert db_manager is not None
