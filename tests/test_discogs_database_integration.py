"""Tests for Discogs database integration."""

import json
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from collector.config import DatabaseConfig
from collector.db_optimized import OptimizedDatabaseManager


class TestDiscogsDatabase:
    """Test Discogs database integration."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        config = DatabaseConfig(path=db_path)
        db_manager = OptimizedDatabaseManager(config)
        
        yield db_manager
        
        # Cleanup
        Path(db_path).unlink(missing_ok=True)

    def test_discogs_table_creation(self, temp_db):
        """Test that discogs_data table is created with correct schema."""
        with temp_db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='discogs_data'"
            )
            assert cursor.fetchone() is not None
            
            # Check schema
            cursor.execute("PRAGMA table_info(discogs_data)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}
            
            expected_columns = {
                'video_id': 'TEXT',
                'release_id': 'TEXT',
                'master_id': 'TEXT',
                'artist_name': 'TEXT',
                'song_title': 'TEXT',
                'year': 'INTEGER',
                'genres': 'TEXT',
                'styles': 'TEXT',
                'label': 'TEXT',
                'country': 'TEXT',
                'format': 'TEXT',
                'confidence': 'REAL',
                'discogs_url': 'TEXT',
                'community_have': 'INTEGER',
                'community_want': 'INTEGER',
                'barcode': 'TEXT',
                'catno': 'TEXT',
                'fetched_at': 'TIMESTAMP'
            }
            
            for col_name, col_type in expected_columns.items():
                assert col_name in columns
                assert columns[col_name] == col_type

    def test_discogs_indexes_creation(self, temp_db):
        """Test that Discogs indexes are created."""
        with temp_db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check indexes exist
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_discogs_%'"
            )
            indexes = [row[0] for row in cursor.fetchall()]
            
            assert 'idx_discogs_release' in indexes
            assert 'idx_discogs_artist' in indexes

    def test_save_discogs_data_basic(self, temp_db):
        """Test saving basic Discogs data."""
        video_data = {
            "video_id": "test_video_123",
            "url": "https://youtube.com/watch?v=test_video_123",
            "title": "Test Video",
            "discogs_release_id": "12345",
            "metadata": {
                "discogs_master_id": "67890",
                "artist_name": "Test Artist",
                "song_title": "Test Song",
                "year": 2020,
                "genres": ["Pop", "Rock"],
                "styles": ["Alternative"],
                "label": "Test Label",
                "country": "US",
                "format": "CD",
                "confidence": 0.85,
                "discogs_url": "https://www.discogs.com/release/12345",
                "community": {"have": 150, "want": 50},
                "barcode": "123456789",
                "catno": "TL001"
            }
        }
        
        success = temp_db.save_result(video_data)
        assert success
        
        # Verify Discogs data was saved
        with temp_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM discogs_data WHERE video_id = ?", ("test_video_123",))
            row = cursor.fetchone()
            
            assert row is not None
            assert row["release_id"] == "12345"
            assert row["master_id"] == "67890"
            assert row["artist_name"] == "Test Artist"
            assert row["song_title"] == "Test Song"
            assert row["year"] == 2020
            assert json.loads(row["genres"]) == ["Pop", "Rock"]
            assert json.loads(row["styles"]) == ["Alternative"]
            assert row["label"] == "Test Label"
            assert row["country"] == "US"
            assert row["format"] == "CD"
            assert row["confidence"] == 0.85
            assert row["discogs_url"] == "https://www.discogs.com/release/12345"
            assert row["community_have"] == 150
            assert row["community_want"] == 50
            assert row["barcode"] == "123456789"
            assert row["catno"] == "TL001"

    def test_save_discogs_data_minimal(self, temp_db):
        """Test saving minimal Discogs data."""
        video_data = {
            "video_id": "test_video_456",
            "url": "https://youtube.com/watch?v=test_video_456",
            "title": "Test Video 2",
            "discogs_release_id": "54321",
            "artist": "Minimal Artist",
            "song_title": "Minimal Song"
        }
        
        success = temp_db.save_result(video_data)
        assert success
        
        # Verify minimal Discogs data was saved
        with temp_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM discogs_data WHERE video_id = ?", ("test_video_456",))
            row = cursor.fetchone()
            
            assert row is not None
            assert row["release_id"] == "54321"
            assert row["artist_name"] == "Minimal Artist"
            assert row["song_title"] == "Minimal Song"
            assert row["master_id"] is None
            assert row["year"] is None
            assert row["genres"] is None
            assert row["confidence"] == 0.0  # Default value

    def test_save_discogs_data_no_release_id(self, temp_db):
        """Test that no Discogs data is saved without release_id."""
        video_data = {
            "video_id": "test_video_789",
            "url": "https://youtube.com/watch?v=test_video_789",
            "title": "Test Video 3",
            "artist": "No Discogs Artist",
            "song_title": "No Discogs Song"
        }
        
        success = temp_db.save_result(video_data)
        assert success
        
        # Verify no Discogs data was saved
        with temp_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM discogs_data WHERE video_id = ?", ("test_video_789",))
            row = cursor.fetchone()
            
            assert row is None

    def test_save_discogs_data_replace(self, temp_db):
        """Test that Discogs data can be replaced."""
        video_id = "test_video_replace"
        
        # Save initial data
        video_data1 = {
            "video_id": video_id,
            "url": "https://youtube.com/watch?v=" + video_id,
            "title": "Test Video",
            "discogs_release_id": "11111",
            "metadata": {
                "artist_name": "Original Artist",
                "confidence": 0.5
            }
        }
        
        success = temp_db.save_result(video_data1)
        assert success
        
        # Save updated data
        video_data2 = {
            "video_id": video_id,
            "url": "https://youtube.com/watch?v=" + video_id,
            "title": "Test Video Updated",
            "discogs_release_id": "22222",
            "metadata": {
                "artist_name": "Updated Artist",
                "confidence": 0.9
            }
        }
        
        success = temp_db.save_result(video_data2)
        assert success
        
        # Verify data was replaced
        with temp_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM discogs_data WHERE video_id = ?", (video_id,))
            rows = cursor.fetchall()
            
            assert len(rows) == 1  # Should only have one record
            row = rows[0]
            assert row["release_id"] == "22222"
            assert row["artist_name"] == "Updated Artist"
            assert row["confidence"] == 0.9

    def test_discogs_foreign_key_constraint(self, temp_db):
        """Test foreign key constraint with videos table."""
        # Try to insert Discogs data without corresponding video
        with temp_db.get_connection() as conn:
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    """
                    INSERT INTO discogs_data (video_id, release_id)
                    VALUES (?, ?)
                    """,
                    ("nonexistent_video", "12345")
                )
                conn.commit()

    def test_discogs_data_cascade_delete(self, temp_db):
        """Test that Discogs data is deleted when video is deleted."""
        video_data = {
            "video_id": "test_cascade",
            "url": "https://youtube.com/watch?v=test_cascade",
            "title": "Test Cascade",
            "discogs_release_id": "cascade_release",
            "artist": "Cascade Artist"
        }
        
        # Save video with Discogs data
        success = temp_db.save_result(video_data)
        assert success
        
        # Verify Discogs data exists
        with temp_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM discogs_data WHERE video_id = ?", ("test_cascade",))
            count = cursor.fetchone()[0]
            assert count == 1
            
            # Delete video
            cursor.execute("DELETE FROM videos WHERE video_id = ?", ("test_cascade",))
            conn.commit()
            
            # Verify Discogs data was cascade deleted
            cursor.execute("SELECT COUNT(*) FROM discogs_data WHERE video_id = ?", ("test_cascade",))
            count = cursor.fetchone()[0]
            assert count == 0

    def test_schema_version_updated(self, temp_db):
        """Test that schema version is updated to include Discogs support."""
        with temp_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT version, description FROM schema_info ORDER BY version DESC LIMIT 1")
            row = cursor.fetchone()
            
            assert row is not None
            assert row["version"] == 3
            assert "Discogs" in row["description"]

    def test_statistics_include_discogs(self, temp_db):
        """Test that statistics include Discogs record count."""
        # Add some test data
        video_data = {
            "video_id": "test_stats",
            "url": "https://youtube.com/watch?v=test_stats",
            "title": "Test Stats",
            "discogs_release_id": "stats_release",
            "artist": "Stats Artist"
        }
        
        temp_db.save_result(video_data)
        
        stats = temp_db.get_statistics()
        assert "discogs_records" in stats
        assert stats["discogs_records"] >= 1

    def test_json_array_handling(self, temp_db):
        """Test proper JSON serialization of genres and styles arrays."""
        video_data = {
            "video_id": "test_json",
            "url": "https://youtube.com/watch?v=test_json",
            "title": "Test JSON",
            "discogs_release_id": "json_release",
            "metadata": {
                "genres": ["Electronic", "House", "Techno"],
                "styles": ["Deep House", "Minimal"]
            }
        }
        
        success = temp_db.save_result(video_data)
        assert success
        
        with temp_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT genres, styles FROM discogs_data WHERE video_id = ?", ("test_json",))
            row = cursor.fetchone()
            
            assert row is not None
            
            # Verify JSON can be parsed back to arrays
            genres = json.loads(row["genres"])
            styles = json.loads(row["styles"])
            
            assert genres == ["Electronic", "House", "Techno"]
            assert styles == ["Deep House", "Minimal"]

    def test_empty_arrays_handling(self, temp_db):
        """Test handling of empty genres and styles arrays."""
        video_data = {
            "video_id": "test_empty",
            "url": "https://youtube.com/watch?v=test_empty",
            "title": "Test Empty",
            "discogs_release_id": "empty_release",
            "metadata": {
                "genres": [],
                "styles": []
            }
        }
        
        success = temp_db.save_result(video_data)
        assert success
        
        with temp_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT genres, styles FROM discogs_data WHERE video_id = ?", ("test_empty",))
            row = cursor.fetchone()
            
            assert row is not None
            assert row["genres"] == "[]"
            assert row["styles"] == "[]"