"""Database management with proper connection handling and migrations."""

import contextlib
import logging
import shutil
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from .config import DatabaseConfig

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Enhanced database management with migrations and backups."""

    SCHEMA_VERSION = 7  # Schema version expected by tests

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.db_path = Path(config.path)
        self.backup_dir = self.db_path.parent / "backups"
        self.migrations_dir = self.db_path.parent / "migrations"

        # Connection pool to prevent connection exhaustion
        self._connection_pool = []
        self._pool_lock = threading.Lock()
        self._max_pool_size = config.connection_pool_size
        self._pool_timeout = config.connection_timeout

        self.setup_database()

    def _get_pooled_connection(self) -> sqlite3.Connection:
        """Get a connection from the pool or create a new one."""
        with self._pool_lock:
            if self._connection_pool:
                return self._connection_pool.pop()

        # Create new connection if pool is empty
        conn = sqlite3.connect(str(self.db_path), timeout=self._pool_timeout)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _return_pooled_connection(self, conn: sqlite3.Connection):
        """Return a connection to the pool or close it if pool is full."""
        if conn:
            with self._pool_lock:
                if len(self._connection_pool) < self._max_pool_size:
                    self._connection_pool.append(conn)
                    return
            # Pool is full, close the connection
            try:
                conn.close()
            except Exception as e:
                logger.warning(f"Failed to close database connection: {e}")

    @contextlib.contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections with connection pooling."""
        conn = None
        try:
            conn = self._get_pooled_connection()
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                self._return_pooled_connection(conn)

    def setup_database(self):
        """Create database schema with migrations support."""
        if self.config.backup_enabled:
            self.backup_dir.mkdir(exist_ok=True)

        # Create migrations directory and files if they don't exist
        self.migrations_dir.mkdir(exist_ok=True)
        self._create_migration_files()

        if self.db_path.exists() and self.config.backup_enabled:
            self._create_backup()

        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_info (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            cursor.execute("SELECT version FROM schema_info ORDER BY version DESC LIMIT 1")
            version_row = cursor.fetchone()
            start_version = version_row[0] if version_row else 0

            self._apply_migrations(cursor, start_version)

            cursor.execute("SELECT version FROM schema_info ORDER BY version DESC LIMIT 1")
            version_row = cursor.fetchone()
            end_version = version_row[0] if version_row else start_version

            if start_version < 4 <= end_version:
                self._create_performance_indexes(cursor)

            if self.config.vacuum_on_startup:
                self._conditional_vacuum(cursor)

        logger.info(f"Database initialized: {self.db_path}")

    def _create_migration_files(self):
        """Create migration files if they don't exist."""
        migration_002_path = self.migrations_dir / "002_updated_at_trigger.sql"
        if not migration_002_path.exists():
            migration_002_content = """-- Migration 002: Fix updated_at trigger (recursion-safe)
DROP TRIGGER IF EXISTS videos_updated_at;
CREATE TRIGGER trg_videos_updated
AFTER UPDATE ON videos
FOR EACH ROW
WHEN OLD.updated_at = NEW.updated_at
BEGIN
  UPDATE videos SET updated_at = CURRENT_TIMESTAMP
  WHERE id = NEW.id;
END;
INSERT OR REPLACE INTO schema_info(version) VALUES (2);
"""
            with open(migration_002_path, "w") as f:
                f.write(migration_002_content)

        migration_003_path = self.migrations_dir / "003_featured_artists_and_ratios.sql"
        if not migration_003_path.exists():
            migration_003_content = """-- Migration 003: Add featured artists and engagement ratios
ALTER TABLE videos ADD COLUMN featured_artists TEXT;
ALTER TABLE videos ADD COLUMN engagement_ratio REAL;

-- Update existing records with computed ratios
UPDATE videos
SET engagement_ratio =
  CASE
    WHEN view_count > 0 THEN
      (like_count - COALESCE((SELECT estimated_dislikes FROM ryd_data WHERE ryd_data.video_id = videos.video_id), 0)) * 1.0 / view_count
    ELSE NULL
  END
WHERE view_count > 0;

INSERT OR REPLACE INTO schema_info(version) VALUES (3);
"""
            with open(migration_003_path, "w") as f:
                f.write(migration_003_content)

        migration_004_path = self.migrations_dir / "004_channels_support.sql"
        if not migration_004_path.exists():
            migration_004_content = """-- Migration 004: Add channels support
CREATE TABLE IF NOT EXISTS channels (
    channel_id TEXT PRIMARY KEY,
    channel_url TEXT NOT NULL,
    channel_name TEXT,
    subscriber_count INTEGER DEFAULT 0,
    video_count INTEGER DEFAULT 0,
    description TEXT,
    is_karaoke_focused BOOLEAN DEFAULT 1,
    last_processed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add index for faster channel lookups
CREATE INDEX IF NOT EXISTS idx_channels_processed_at ON channels(last_processed_at);
CREATE INDEX IF NOT EXISTS idx_videos_channel_id ON videos(channel_id);

INSERT OR REPLACE INTO schema_info(version) VALUES (4);
"""
            with open(migration_004_path, "w") as f:
                f.write(migration_004_content)

        migration_005_path = self.migrations_dir / "005_add_foreign_keys.sql"
        if not migration_005_path.exists():
            migration_005_content = """-- Migration 005: Add foreign key constraints
PRAGMA foreign_keys = ON;
INSERT OR REPLACE INTO schema_info(version) VALUES (5);
"""
            with open(migration_005_path, "w") as f:
                f.write(migration_005_content)

        migration_006_path = self.migrations_dir / "006_enhanced_musicbrainz.sql"
        if not migration_006_path.exists():
            migration_006_content = """-- Enhanced MusicBrainz metadata support
-- Migration 006: Add comprehensive MusicBrainz fields to videos table
ALTER TABLE videos ADD COLUMN musicbrainz_recording_id TEXT;
ALTER TABLE videos ADD COLUMN musicbrainz_artist_id TEXT;
ALTER TABLE videos ADD COLUMN musicbrainz_genre TEXT;
ALTER TABLE videos ADD COLUMN musicbrainz_tags TEXT;
ALTER TABLE videos ADD COLUMN parse_confidence REAL;
ALTER TABLE videos ADD COLUMN record_label TEXT;
ALTER TABLE videos ADD COLUMN recording_length_ms INTEGER;

CREATE INDEX IF NOT EXISTS idx_videos_musicbrainz_recording_id ON videos(musicbrainz_recording_id);
CREATE INDEX IF NOT EXISTS idx_videos_musicbrainz_artist_id ON videos(musicbrainz_artist_id);
CREATE INDEX IF NOT EXISTS idx_videos_musicbrainz_genre ON videos(musicbrainz_genre);
CREATE INDEX IF NOT EXISTS idx_videos_record_label ON videos(record_label);
INSERT OR REPLACE INTO schema_info(version) VALUES (6);
"""
            with open(migration_006_path, "w") as f:
                f.write(migration_006_content)

        # Create migration 007 for cache tables
        migration_007_path = self.migrations_dir / "007_cache_tables.sql"
        if not migration_007_path.exists():
            migration_007_content = """-- Migration 007: Add cache tables for search optimization
CREATE TABLE IF NOT EXISTS search_cache (
    key TEXT PRIMARY KEY,
    value BLOB NOT NULL,
    created_at TIMESTAMP NOT NULL,
    last_accessed TIMESTAMP NOT NULL,
    access_count INTEGER DEFAULT 0,
    ttl_seconds INTEGER,
    size_bytes INTEGER DEFAULT 0,
    metadata TEXT,
    namespace TEXT,
    query_hash TEXT
);

CREATE INDEX IF NOT EXISTS idx_search_cache_created ON search_cache(created_at);
CREATE INDEX IF NOT EXISTS idx_search_cache_accessed ON search_cache(last_accessed);
CREATE INDEX IF NOT EXISTS idx_search_cache_namespace ON search_cache(namespace);
CREATE INDEX IF NOT EXISTS idx_search_cache_query_hash ON search_cache(query_hash);

-- Table for tracking search query performance
CREATE TABLE IF NOT EXISTS search_analytics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    query_normalized TEXT,
    provider TEXT,
    result_count INTEGER DEFAULT 0,
    response_time_ms INTEGER,
    cache_hit BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_search_analytics_query ON search_analytics(query);
CREATE INDEX IF NOT EXISTS idx_search_analytics_provider ON search_analytics(provider);
CREATE INDEX IF NOT EXISTS idx_search_analytics_created ON search_analytics(created_at);

-- Table for fuzzy matching reference data
CREATE TABLE IF NOT EXISTS fuzzy_reference_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    data_type TEXT NOT NULL, -- 'artist' or 'song'
    original_text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    popularity_score REAL DEFAULT 0.0,
    confidence REAL DEFAULT 1.0,
    source TEXT, -- 'musicbrainz', 'manual', 'extracted'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_fuzzy_ref_type ON fuzzy_reference_data(data_type);
CREATE INDEX IF NOT EXISTS idx_fuzzy_ref_normalized ON fuzzy_reference_data(normalized_text);
CREATE INDEX IF NOT EXISTS idx_fuzzy_ref_popularity ON fuzzy_reference_data(popularity_score);

-- Trigger for fuzzy reference data updated_at
CREATE TRIGGER IF NOT EXISTS trg_fuzzy_ref_updated
AFTER UPDATE ON fuzzy_reference_data
FOR EACH ROW
WHEN OLD.updated_at = NEW.updated_at
BEGIN
  UPDATE fuzzy_reference_data SET updated_at = CURRENT_TIMESTAMP
  WHERE id = NEW.id;
END;

INSERT OR REPLACE INTO schema_info(version) VALUES (7);
"""
            with open(migration_007_path, "w") as f:
                f.write(migration_007_content)

    def _apply_migrations(self, cursor: sqlite3.Cursor, current_version: int):
        """Apply database migrations up to ``SCHEMA_VERSION``."""
        target_version = self.SCHEMA_VERSION
        if current_version < 1:
            self._create_initial_schema(cursor)
            cursor.execute("INSERT INTO schema_info (version) VALUES (1)")
            logger.info("Applied migration: Initial schema (v1)")

            current_version = 1
            if current_version >= target_version:
                return

        if current_version < 2:
            # Apply migration from file
            migration_002_path = self.migrations_dir / "002_updated_at_trigger.sql"
            if migration_002_path.exists():
                with open(migration_002_path, "r") as f:
                    migration_sql = f.read()
                cursor.executescript(migration_sql)
                logger.info("Applied migration: Updated trigger (v2)")

                current_version = 2
            if current_version >= target_version:
                return

        if current_version < 3:
            # Apply migration from file
            migration_003_path = self.migrations_dir / "003_featured_artists_and_ratios.sql"
            if migration_003_path.exists():
                # Skip if columns already exist
                cursor.execute("PRAGMA table_info(videos)")
                existing_cols = {row[1] for row in cursor.fetchall()}
                if "featured_artists" not in existing_cols:
                    with open(migration_003_path, "r") as f:
                        migration_sql = f.read()
                    cursor.executescript(migration_sql)
                    logger.info("Applied migration: Featured artists and ratios (v3)")
                else:
                    cursor.execute("INSERT OR REPLACE INTO schema_info(version) VALUES (3)")
                    logger.info("Skipped migration 003; columns already present")
                current_version = 3
            if current_version >= target_version:
                return

        if current_version < 4:
            # Apply migration from file
            migration_004_path = self.migrations_dir / "004_channels_support.sql"
            if migration_004_path.exists():
                # Check if channels table already exists
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='channels'"
                )
                if not cursor.fetchone():
                    with open(migration_004_path, "r") as f:
                        migration_sql = f.read()
                    cursor.executescript(migration_sql)
                    logger.info("Applied migration: Channels support (v4)")
                else:
                    cursor.execute("INSERT OR REPLACE INTO schema_info(version) VALUES (4)")
                    logger.info("Skipped migration 004; channels table already exists")
                current_version = 4
            if current_version >= target_version:
                return

        if current_version < 5:
            # Apply foreign key constraints migration
            migration_005_path = self.migrations_dir / "005_add_foreign_keys.sql"
            if migration_005_path.exists():
                try:
                    with open(migration_005_path, "r") as f:
                        migration_sql = f.read()
                    cursor.executescript(migration_sql)
                    logger.info("Applied migration: Foreign key constraints (v5)")
                except Exception as e:
                    logger.warning(f"Failed to apply migration 005: {e}")
                    # Update version even if migration partially failed to avoid retry loops
                    cursor.execute("INSERT OR REPLACE INTO schema_info(version) VALUES (5)")
                current_version = 5
            if current_version >= target_version:
                return

        if current_version < 6:
            # Apply enhanced MusicBrainz metadata migration
            migration_006_path = self.migrations_dir / "006_enhanced_musicbrainz.sql"
            if migration_006_path.exists():
                # Check if new columns already exist
                cursor.execute("PRAGMA table_info(videos)")
                existing_cols = {row[1] for row in cursor.fetchall()}
                if "musicbrainz_recording_id" not in existing_cols:
                    try:
                        with open(migration_006_path, "r") as f:
                            migration_sql = f.read()
                        cursor.executescript(migration_sql)
                        logger.info("Applied migration: Enhanced MusicBrainz metadata (v6)")
                    except Exception as e:
                        logger.warning(f"Failed to apply migration 006: {e}")
                        cursor.execute("INSERT OR REPLACE INTO schema_info(version) VALUES (6)")
                else:
                    cursor.execute("INSERT OR REPLACE INTO schema_info(version) VALUES (6)")
                    logger.info("Skipped migration 006; MusicBrainz columns already present")
                current_version = 6
            if current_version >= target_version:
                return

        if current_version < 7:
            # Apply cache tables migration
            migration_007_path = self.migrations_dir / "007_cache_tables.sql"
            if migration_007_path.exists():
                # Check if cache tables already exist
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='search_cache'"
                )
                if not cursor.fetchone():
                    try:
                        with open(migration_007_path, "r") as f:
                            migration_sql = f.read()
                        cursor.executescript(migration_sql)
                        logger.info("Applied migration: Cache tables (v7)")
                    except Exception as e:
                        logger.warning(f"Failed to apply migration 007: {e}")
                        cursor.execute("INSERT OR REPLACE INTO schema_info(version) VALUES (7)")
                else:
                    cursor.execute("INSERT OR REPLACE INTO schema_info(version) VALUES (7)")
                    logger.info("Skipped migration 007; cache tables already exist")
                current_version = 7
            if current_version >= target_version:
                return

    def _create_initial_schema(self, cursor: sqlite3.Cursor):
        """Create the initial database schema."""

        # Main videos table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT UNIQUE NOT NULL,
                url TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                duration_seconds INTEGER,
                view_count INTEGER DEFAULT 0,
                like_count INTEGER DEFAULT 0,
                comment_count INTEGER DEFAULT 0,
                upload_date TEXT,
                thumbnail_url TEXT,
                channel_name TEXT,
                channel_id TEXT,
                original_artist TEXT,
                featured_artists TEXT,
                song_title TEXT,
                release_year INTEGER,
                musicbrainz_recording_id TEXT,
                musicbrainz_artist_id TEXT,
                musicbrainz_genre TEXT,
                musicbrainz_tags TEXT,
                parse_confidence REAL,
                record_label TEXT,
                recording_length_ms INTEGER,
                genre TEXT,
                language TEXT,
                engagement_ratio REAL,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Video features table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS video_features (
                video_id TEXT PRIMARY KEY,
                has_guide_vocals BOOLEAN DEFAULT 0,
                has_scrolling_lyrics BOOLEAN DEFAULT 0,
                has_backing_vocals BOOLEAN DEFAULT 0,
                is_instrumental_only BOOLEAN DEFAULT 0,
                is_piano_only BOOLEAN DEFAULT 0,
                is_acoustic BOOLEAN DEFAULT 0,
                has_guide_vocals_confidence REAL DEFAULT 0.0,
                has_scrolling_lyrics_confidence REAL DEFAULT 0.0,
                has_backing_vocals_confidence REAL DEFAULT 0.0,
                is_instrumental_only_confidence REAL DEFAULT 0.0,
                is_piano_only_confidence REAL DEFAULT 0.0,
                is_acoustic_confidence REAL DEFAULT 0.0,
                video_style TEXT,
                difficulty_level TEXT,
                special_features TEXT,
                confidence_score REAL DEFAULT 0.0,
                FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
            )
        """
        )

        # Quality scores table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS quality_scores (
                video_id TEXT PRIMARY KEY,
                overall_score REAL DEFAULT 0.0,
                technical_score REAL DEFAULT 0.0,
                engagement_score REAL DEFAULT 0.0,
                metadata_completeness REAL DEFAULT 0.0,
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
            )
        """
        )

        # Return YouTube Dislike data table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS ryd_data (
                video_id TEXT PRIMARY KEY,
                estimated_dislikes INTEGER DEFAULT 0,
                ryd_likes INTEGER DEFAULT 0,
                ryd_rating REAL DEFAULT 0.0,
                ryd_confidence REAL DEFAULT 0.0,
                ryd_deleted BOOLEAN DEFAULT 0,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
            )
        """
        )

        # Validation results table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS validation_results (
                video_id TEXT PRIMARY KEY,
                artist_valid BOOLEAN,
                song_valid BOOLEAN,
                validation_score REAL DEFAULT 0.0,
                suggested_artist TEXT,
                suggested_title TEXT,
                suggestion_reason TEXT,
                FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
            )
        """
        )

        # Search history table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                search_method TEXT NOT NULL,
                videos_found INTEGER DEFAULT 0,
                videos_processed INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0,
                processing_time_seconds REAL DEFAULT 0.0,
                search_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Error log table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS error_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT,
                error_type TEXT NOT NULL,
                error_message TEXT,
                retry_count INTEGER DEFAULT 0,
                resolved BOOLEAN DEFAULT 0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create indexes
        self._create_indexes(cursor)

    def _create_indexes(self, cursor: sqlite3.Cursor):
        """Create database indexes for performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_videos_artist ON videos(original_artist)",
            "CREATE INDEX IF NOT EXISTS idx_videos_upload_date ON videos(upload_date)",
            "CREATE INDEX IF NOT EXISTS idx_videos_views ON videos(view_count)",
            "CREATE INDEX IF NOT EXISTS idx_features_confidence ON video_features(confidence_score)",
            "CREATE INDEX IF NOT EXISTS idx_quality_overall ON quality_scores(overall_score)",
        ]

        for index in indexes:
            cursor.execute(index)

    def _create_performance_indexes(self, cursor: sqlite3.Cursor):
        """Create additional indexes optimized for large-scale queries."""
        performance_indexes = [
            # Critical indexes for scalability
            "CREATE INDEX IF NOT EXISTS idx_videos_channel_upload ON videos(channel_id, upload_date)",
            "CREATE INDEX IF NOT EXISTS idx_videos_scraped_updated ON videos(scraped_at, updated_at)",
            "CREATE INDEX IF NOT EXISTS idx_videos_duration_views ON videos(duration_seconds, view_count)",
            "CREATE INDEX IF NOT EXISTS idx_videos_channel_scraped ON videos(channel_id, scraped_at)",
            # Video features performance indexes
            "CREATE INDEX IF NOT EXISTS idx_features_karaoke_confidence ON video_features(has_guide_vocals, is_instrumental_only, confidence_score)",
            "CREATE INDEX IF NOT EXISTS idx_features_video_style ON video_features(video_style, difficulty_level)",
            # Quality and engagement indexes
            "CREATE INDEX IF NOT EXISTS idx_quality_overall_technical ON quality_scores(overall_score, technical_score)",
            "CREATE INDEX IF NOT EXISTS idx_quality_engagement ON quality_scores(engagement_score, calculated_at)",
            # Channel processing indexes
            "CREATE INDEX IF NOT EXISTS idx_channels_processed_karaoke ON channels(last_processed_at, is_karaoke_focused)",
            "CREATE INDEX IF NOT EXISTS idx_channels_subscriber_count ON channels(subscriber_count, video_count)",
            # Search and error tracking indexes
            "CREATE INDEX IF NOT EXISTS idx_search_history_date_method ON search_history(search_date, search_method)",
            "CREATE INDEX IF NOT EXISTS idx_error_log_timestamp_resolved ON error_log(timestamp, resolved)",
            "CREATE INDEX IF NOT EXISTS idx_error_log_video_type ON error_log(video_id, error_type)",
            # Composite indexes for common query patterns
            "CREATE INDEX IF NOT EXISTS idx_videos_artist_views_date ON videos(artist, view_count, upload_date)",
            "CREATE INDEX IF NOT EXISTS idx_videos_channel_artist_title ON videos(channel_id, artist, song_title)",
        ]

        for index in performance_indexes:
            try:
                cursor.execute(index)
            except Exception as e:
                logger.warning(f"Failed to create performance index: {e}")
                # Continue with other indexes even if one fails

    def _conditional_vacuum(self, cursor: sqlite3.Cursor):
        """Only run VACUUM if database has grown significantly, with WAL checkpoint."""
        try:
            # First checkpoint WAL
            cursor.execute("PRAGMA wal_checkpoint(FULL);")

            cursor.execute("PRAGMA page_count")
            page_count = cursor.fetchone()[0]
            cursor.execute("PRAGMA page_size")
            page_size = cursor.fetchone()[0]

            db_size_mb = (page_count * page_size) / (1024 * 1024)

            if db_size_mb > self.config.vacuum_threshold_mb:
                logger.info(f"Running VACUUM on {db_size_mb:.1f}MB database...")
                cursor.execute("VACUUM")
                logger.info("VACUUM completed")
        except Exception as e:
            logger.warning(f"VACUUM failed: {e}")

    def _create_backup(self):
        """Create a backup of the database with timestamp check."""
        if not self.db_path.exists():
            return

        try:
            # Check if backup is needed based on interval
            backup_files = list(self.backup_dir.glob(f"{self.db_path.stem}_*.db"))
            if backup_files:
                latest_backup = max(backup_files, key=lambda x: x.stat().st_mtime)
                age_hours = (datetime.now().timestamp() - latest_backup.stat().st_mtime) / 3600
                if age_hours < self.config.backup_interval_hours:
                    logger.debug(f"Backup not needed, latest is {age_hours:.1f} hours old")
                    return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"{self.db_path.stem}_{timestamp}.db"

            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backed up to: {backup_path}")
            self._cleanup_old_backups()
        except Exception as e:
            logger.error(f"Backup failed: {e}")

    def _cleanup_old_backups(self):
        """Remove old backup files based on retention policy."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config.backup_retention_days)
            for backup_file in self.backup_dir.glob("*.db"):
                if backup_file.stat().st_mtime < cutoff_date.timestamp():
                    backup_file.unlink()
                    logger.debug(f"Removed old backup: {backup_file}")
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")

    def get_existing_video_ids(self) -> set:
        """Get set of existing video IDs for duplicate checking."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT video_id FROM videos")
                return {row[0] for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"Failed to get existing video IDs: {e}")
            return set()

    def get_recent_video_ids(self, days: int = 7) -> set:
        """Get video IDs from recent days to limit memory usage."""
        try:
            # Validate input parameter
            if not isinstance(days, int) or days < 0:
                raise ValueError(f"Days must be a non-negative integer, got: {days}")

            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT video_id FROM videos WHERE scraped_at >= date('now', '-' || ? || ' days')",
                    (days,),
                )
                return {row[0] for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"Failed to get recent video IDs: {e}")
            return set()

    def save_video_data(self, result):
        """Save processed video data to database with atomic transactions."""
        try:
            with self.get_connection() as conn:
                # Start explicit transaction for atomicity
                conn.execute("BEGIN IMMEDIATE")
                try:
                    cursor = conn.cursor()

                    video_data = result.video_data
                    features = video_data.get("features", {})
                    quality_scores = video_data.get("quality_scores", {})

                    # Calculate like/dislike to views ratio
                    like_dislike_ratio = None
                    views = video_data.get("view_count", 0)
                    likes = video_data.get("like_count", 0)
                    dislikes = video_data.get("estimated_dislikes")
                    if dislikes is None:
                        dislikes = 0

                    if views > 0:
                        like_dislike_ratio = (likes - dislikes) / views

                    # Ensure channel exists before inserting video
                    if not self.ensure_channel_exists(cursor, video_data):
                        raise Exception("Failed to create required channel record")

                    # Insert main video record
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO videos (
                            video_id, url, title, description, duration_seconds,
                            view_count, like_count, comment_count, upload_date,
                            thumbnail_url, channel_name, channel_id, artist,
                            featured_artists, song_title, release_year,
                            genre, language, engagement_ratio,
                            musicbrainz_recording_id, musicbrainz_artist_id, musicbrainz_genre,
                            musicbrainz_tags, parse_confidence, record_label,
                            recording_length_ms
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            video_data.get("video_id"),
                            video_data.get("url"),
                            video_data.get("title"),
                            video_data.get("description", "")[:2000],
                            video_data.get("duration_seconds"),
                            video_data.get("view_count"),
                            video_data.get("like_count"),
                            video_data.get("comment_count"),
                            video_data.get("upload_date"),
                            video_data.get("thumbnail"),
                            video_data.get("uploader"),
                            video_data.get("uploader_id") or None,
                            features.get("artist"),
                            features.get("featured_artists"),
                            features.get("song_title"),
                            video_data.get("release_year"),
                            features.get("genre"),
                            video_data.get("language"),
                            like_dislike_ratio,
                            video_data.get("musicbrainz_recording_id"),
                            video_data.get("musicbrainz_artist_id"),
                            video_data.get("musicbrainz_genre"),
                            video_data.get("musicbrainz_tags"),
                            video_data.get("parse_confidence"),
                            video_data.get("record_label"),
                            video_data.get("recording_length_ms"),
                        ),
                    )

                    # Insert features with confidence scores
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO video_features (
                            video_id, has_guide_vocals, has_scrolling_lyrics,
                            has_backing_vocals, is_instrumental_only, is_piano_only,
                            is_acoustic, has_guide_vocals_confidence, has_scrolling_lyrics_confidence,
                            has_backing_vocals_confidence, is_instrumental_only_confidence,
                            is_piano_only_confidence, is_acoustic_confidence, confidence_score
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            video_data.get("video_id"),
                            features.get("has_guide_vocals", False),
                            features.get("has_scrolling_lyrics", False),
                            features.get("has_backing_vocals", False),
                            features.get("is_instrumental_only", False),
                            features.get("is_piano_only", False),
                            features.get("is_acoustic", False),
                            features.get("has_guide_vocals_confidence", 0.0),
                            features.get("has_scrolling_lyrics_confidence", 0.0),
                            features.get("has_backing_vocals_confidence", 0.0),
                            features.get("is_instrumental_only_confidence", 0.0),
                            features.get("is_piano_only_confidence", 0.0),
                            features.get("is_acoustic_confidence", 0.0),
                            result.confidence_score,
                        ),
                    )

                    # Insert quality scores
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO quality_scores (
                            video_id, overall_score, technical_score,
                            engagement_score, metadata_completeness
                        ) VALUES (?, ?, ?, ?, ?)
                    """,
                        (
                            video_data.get("video_id"),
                            quality_scores.get("overall_score", 0),
                            quality_scores.get("technical_score", 0),
                            quality_scores.get("engagement_score", 0),
                            quality_scores.get("metadata_completeness", 0),
                        ),
                    )

                    # Insert RYD data if available
                    if video_data.get("estimated_dislikes") is not None:
                        cursor.execute(
                            """
                            INSERT OR REPLACE INTO ryd_data (
                                video_id, estimated_dislikes, ryd_likes, ryd_rating, ryd_confidence
                            ) VALUES (?, ?, ?, ?, ?)
                        """,
                            (
                                video_data.get("video_id"),
                                video_data.get("estimated_dislikes"),
                                video_data.get("ryd_likes"),
                                video_data.get("ryd_rating"),
                                video_data.get("ryd_confidence"),
                            ),
                        )

                    validation = video_data.get("validation", {})
                    suggestion = video_data.get("correction_suggestion", {})
                    if validation:
                        cursor.execute(
                            """
                            INSERT OR REPLACE INTO validation_results (
                                video_id, artist_valid, song_valid, validation_score,
                                suggested_artist, suggested_title, suggestion_reason
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                video_data.get("video_id"),
                                bool(validation.get("artist_valid")),
                                bool(validation.get("song_valid")),
                                validation.get("validation_score", 0.0),
                                suggestion.get("suggested_artist"),
                                suggestion.get("suggested_title"),
                                suggestion.get("reason"),
                            ),
                        )

                    # Commit the transaction if all operations succeeded
                    conn.execute("COMMIT")
                    return True

                except Exception as e:
                    # Rollback on any failure
                    conn.execute("ROLLBACK")
                    logger.error(f"Database transaction failed, rolled back: {e}")
                    raise

        except Exception as e:
            logger.error(f"Database save failed: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                stats = {}

                cursor.execute("SELECT COUNT(*) FROM videos")
                stats["total_videos"] = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM videos WHERE original_artist IS NOT NULL")
                stats["videos_with_artist"] = cursor.fetchone()[0]

                cursor.execute(
                    "SELECT AVG(confidence_score) FROM video_features WHERE confidence_score > 0"
                )
                result = cursor.fetchone()
                stats["avg_confidence"] = result[0] if result[0] else 0.0

                cursor.execute(
                    "SELECT AVG(overall_score) FROM quality_scores WHERE overall_score > 0"
                )
                result = cursor.fetchone()
                stats["avg_quality"] = result[0] if result[0] else 0.0

                cursor.execute(
                    """
                    SELECT artist, COUNT(*) as count, AVG(view_count) as avg_views
                    FROM videos
                    WHERE original_artist IS NOT NULL
                    GROUP BY original_artist
                    ORDER BY count DESC
                    LIMIT 10
                """
                )
                stats["top_artists"] = cursor.fetchall()

                return stats
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}

    def save_channel_data(self, channel_data: Dict) -> bool:
        """Save or update channel information."""
        try:
            with self.get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO channels (
                        channel_id, channel_url, channel_name, subscriber_count,
                        video_count, description, is_karaoke_focused, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (
                        channel_data.get("channel_id"),
                        channel_data.get("channel_url"),
                        channel_data.get("channel_name"),
                        channel_data.get("subscriber_count", 0),
                        channel_data.get("video_count", 0),
                        channel_data.get("description"),
                        channel_data.get("is_karaoke_focused", True),
                    ),
                )
                logger.debug(f"Saved channel: {channel_data.get('channel_name')}")
                return True
        except Exception as e:
            logger.error(f"Failed to save channel data: {e}")
            return False

    def ensure_channel_exists(self, cursor, video_data: Dict) -> bool:
        """Ensure a channel record exists before saving video data."""
        channel_id = video_data.get("uploader_id")
        if not channel_id:
            return True  # No channel_id to create

        try:
            # Check if channel already exists
            cursor.execute("SELECT 1 FROM channels WHERE channel_id = ?", (channel_id,))
            if cursor.fetchone():
                return True  # Channel already exists

            # Create channel record from video data
            channel_url = f"https://www.youtube.com/channel/{channel_id}"
            channel_name = video_data.get("uploader", "Unknown Channel")

            cursor.execute(
                """
                INSERT INTO channels (
                    channel_id, channel_url, channel_name, subscriber_count,
                    video_count, description, is_karaoke_focused, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (
                    channel_id,
                    channel_url,
                    channel_name,
                    0,  # subscriber_count (unknown)
                    0,  # video_count (unknown)
                    None,  # description (unknown)
                    True,  # is_karaoke_focused (assume true for karaoke collector)
                ),
            )
            logger.debug(f"Created channel record: {channel_name} ({channel_id})")
            return True
        except Exception as e:
            logger.error(f"Failed to create channel record for {channel_id}: {e}")
            return False

    def get_channel_last_processed(self, channel_id: str) -> Optional[str]:
        """Get the last processed timestamp for a channel."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT last_processed_at FROM channels WHERE channel_id = ?", (channel_id,)
                )
                result = cursor.fetchone()
                return result[0] if result and result[0] else None
        except Exception as e:
            logger.error(f"Failed to get last processed time for channel {channel_id}: {e}")
            return None

    def update_channel_processed(self, channel_id: str) -> bool:
        """Update the last processed timestamp for a channel."""
        try:
            with self.get_connection() as conn:
                conn.execute(
                    "UPDATE channels SET last_processed_at = CURRENT_TIMESTAMP WHERE channel_id = ?",
                    (channel_id,),
                )
                return True
        except Exception as e:
            logger.error(f"Failed to update processed time for channel {channel_id}: {e}")
            return False

    def get_channel_videos_count(self, channel_id: str) -> int:
        """Get the count of videos for a specific channel."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM videos WHERE channel_id = ?", (channel_id,)
                )
                result = cursor.fetchone()
                return result[0] if result else 0
        except Exception as e:
            logger.error(f"Failed to get video count for channel {channel_id}: {e}")
            return 0

    def get_processed_channels(self) -> List[Dict]:
        """Get list of all processed channels with their stats."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT
                        c.channel_id, c.channel_name, c.channel_url,
                        c.subscriber_count, c.is_karaoke_focused,
                        c.last_processed_at, COUNT(v.video_id) as collected_videos
                    FROM channels c
                    LEFT JOIN videos v ON c.channel_id = v.channel_id
                    GROUP BY c.channel_id
                    ORDER BY c.channel_name
                    """
                )
                return [
                    {
                        "channel_id": row[0],
                        "channel_name": row[1],
                        "channel_url": row[2],
                        "subscriber_count": row[3],
                        "is_karaoke_focused": bool(row[4]),
                        "last_processed_at": row[5],
                        "collected_videos": row[6],
                    }
                    for row in cursor.fetchall()
                ]
        except Exception as e:
            logger.error(f"Failed to get processed channels: {e}")
            return []

    def video_exists(self, video_id: str) -> bool:
        """Check if a video already exists in the database."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT 1 FROM videos WHERE video_id = ? LIMIT 1", (video_id,)
                )
                return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Failed to check if video exists {video_id}: {e}")
            return False  # Assume doesn't exist to avoid skipping on error

    def get_existing_video_ids_batch(self, video_ids: List[str]) -> set:
        """Get set of video IDs that already exist in database (batch operation)."""
        if not video_ids:
            return set()

        try:
            with self.get_connection() as conn:
                # Create placeholders for the IN clause
                placeholders = ",".join(["?"] * len(video_ids))
                cursor = conn.execute(
                    f"SELECT video_id FROM videos WHERE video_id IN ({placeholders})", video_ids
                )
                return {row[0] for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"Failed to get existing video IDs: {e}")
            return set()  # Return empty set to avoid skipping on error

    def close_all_connections(self):
        """Close all pooled connections for clean shutdown."""
        with self._pool_lock:
            connections_closed = 0
            while self._connection_pool:
                conn = self._connection_pool.pop()
                try:
                    conn.close()
                    connections_closed += 1
                except Exception as e:
                    logger.warning(f"Failed to close pooled connection: {e}")
            if connections_closed > 0:
                logger.info(f"Closed {connections_closed} pooled database connections")

    def __del__(self):
        """Ensure connections are closed when object is destroyed."""
        try:
            self.close_all_connections()
        except Exception:
            pass  # Best effort cleanup
