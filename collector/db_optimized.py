"""Optimized database manager for streamlined schema."""

import contextlib
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator

from .config import DatabaseConfig
from .data_transformer import DataTransformer

logger = logging.getLogger(__name__)


class OptimizedDatabaseManager:
    """Database manager for the optimized streamlined schema."""

    SCHEMA_VERSION = 2  # Optimized schema version

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.db_path = Path(config.path)

        # Use optimized database if it exists
        optimized_path = self.db_path.parent / (self.db_path.stem + "_optimized.db")
        if optimized_path.exists():
            self.db_path = optimized_path
            logger.info(f"Using optimized database: {optimized_path}")

        # Connection pool
        self._connection_pool = []
        self._pool_lock = threading.Lock()
        self._max_pool_size = getattr(config, "connection_pool_size", 5)
        self._pool_timeout = getattr(config, "connection_timeout", 30)

        self.setup_database()

    def _get_pooled_connection(self) -> sqlite3.Connection:
        """Get a connection from the pool or create a new one."""
        with self._pool_lock:
            if self._connection_pool:
                return self._connection_pool.pop()

        conn = sqlite3.connect(str(self.db_path), timeout=self._pool_timeout)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _return_pooled_connection(self, conn: sqlite3.Connection):
        """Return a connection to the pool or close it if pool is full."""
        with self._pool_lock:
            if len(self._connection_pool) < self._max_pool_size:
                self._connection_pool.append(conn)
                return
        conn.close()

    @contextlib.contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections."""
        conn = self._get_pooled_connection()
        try:
            yield conn
        finally:
            self._return_pooled_connection(conn)

    def setup_database(self):
        """Set up the optimized database schema."""
        if not self.db_path.exists():
            logger.info("Creating new optimized database")
            self.create_tables()
        else:
            logger.info(f"Using existing database: {self.db_path}")

    def create_tables(self):
        """Create the optimized database schema."""
        with self.get_connection() as conn:
            # Core video data (streamlined)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS videos (
                    video_id TEXT PRIMARY KEY,
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
                    -- Parsed metadata
                    artist TEXT,
                    song_title TEXT,
                    featured_artists TEXT,
                    release_year INTEGER,
                    -- Quality metrics (rounded)
                    parse_confidence REAL,
                    quality_score REAL,
                    engagement_ratio REAL, -- as percentage
                    -- Single timestamp
                    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
                )
            """
            )

            # Simplified channels
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS channels (
                    channel_id TEXT PRIMARY KEY,
                    channel_url TEXT,
                    channel_name TEXT,
                    video_count INTEGER DEFAULT 0,
                    description TEXT,
                    is_karaoke_focused BOOLEAN DEFAULT 1,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Simplified video features
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS video_features (
                    video_id TEXT PRIMARY KEY,
                    has_guide_vocals BOOLEAN DEFAULT 0,
                    has_scrolling_lyrics BOOLEAN DEFAULT 0,
                    has_backing_vocals BOOLEAN DEFAULT 0,
                    is_instrumental BOOLEAN DEFAULT 0,
                    is_piano_only BOOLEAN DEFAULT 0,
                    is_acoustic BOOLEAN DEFAULT 0,
                    overall_confidence REAL DEFAULT 0.0,
                    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
                )
            """
            )

            # Simplified MusicBrainz data
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS musicbrainz_data (
                    video_id TEXT PRIMARY KEY,
                    recording_id TEXT,
                    artist_id TEXT,
                    genre TEXT,
                    confidence REAL,
                    recording_length_ms INTEGER,
                    tags TEXT,
                    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
                )
            """
            )

            # Simplified validation results
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS validation_results (
                    video_id TEXT PRIMARY KEY,
                    is_valid BOOLEAN DEFAULT 0,
                    validation_score REAL DEFAULT 0.0,
                    alt_artist TEXT,
                    alt_title TEXT,
                    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
                )
            """
            )

            # Quality scores (simplified)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS quality_scores (
                    video_id TEXT PRIMARY KEY,
                    overall_score REAL DEFAULT 0.0,
                    technical_score REAL DEFAULT 0.0,
                    engagement_score REAL DEFAULT 0.0,
                    metadata_score REAL DEFAULT 0.0,
                    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
                )
            """
            )

            # RYD data (simplified)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ryd_data (
                    video_id TEXT PRIMARY KEY,
                    estimated_dislikes INTEGER DEFAULT 0,
                    ryd_likes INTEGER DEFAULT 0,
                    ryd_rating REAL DEFAULT 0.0,
                    ryd_confidence REAL DEFAULT 0.0,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
                )
            """
            )

            # Schema version tracking
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_info (
                    version INTEGER PRIMARY KEY,
                    description TEXT,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create indexes for performance
            self._create_indexes(conn)

            # Insert schema version
            conn.execute(
                """
                INSERT OR REPLACE INTO schema_info (version, description)
                VALUES (?, ?)
            """,
                (self.SCHEMA_VERSION, "Optimized schema - streamlined and rounded values"),
            )

            conn.commit()

    def _create_indexes(self, conn: sqlite3.Connection):
        """Create performance indexes for the optimized schema."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_videos_channel_id ON videos(channel_id)",
            "CREATE INDEX IF NOT EXISTS idx_videos_artist ON videos(artist)",
            "CREATE INDEX IF NOT EXISTS idx_videos_scraped_at ON videos(scraped_at)",
            "CREATE INDEX IF NOT EXISTS idx_musicbrainz_recording ON musicbrainz_data(recording_id)",
            "CREATE INDEX IF NOT EXISTS idx_channels_name ON channels(channel_name)",
            "CREATE INDEX IF NOT EXISTS idx_quality_overall ON quality_scores(overall_score)",
        ]

        for index_sql in indexes:
            conn.execute(index_sql)

    def save_video_data(self, result):
        """Save video data using the optimized schema."""
        try:
            # Handle ProcessingResult objects
            if hasattr(result, "video_data"):
                video_data = result.video_data
            else:
                video_data = result

            # Transform data to optimized format
            optimized_result = DataTransformer.transform_video_data_to_optimized(video_data)

            with self.get_connection() as conn:
                # Round values for the optimized schema
                parse_confidence = optimized_result.get("parse_confidence")
                engagement_ratio = optimized_result.get("engagement_ratio")

                # Insert into videos table
                conn.execute(
                    """
                    INSERT OR REPLACE INTO videos (
                        video_id, url, title, description, duration_seconds,
                        view_count, like_count, comment_count, upload_date,
                        thumbnail_url, channel_name, channel_id,
                        artist, song_title, featured_artists, release_year,
                        parse_confidence, engagement_ratio, scraped_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        optimized_result["video_id"],
                        optimized_result["url"],
                        optimized_result["title"],
                        optimized_result.get("description", ""),
                        optimized_result.get("duration_seconds"),
                        optimized_result.get("view_count", 0),
                        optimized_result.get("like_count", 0),
                        optimized_result.get("comment_count", 0),
                        optimized_result.get("upload_date"),
                        optimized_result.get("thumbnail_url"),
                        optimized_result.get("channel_name"),
                        optimized_result.get("channel_id"),
                        optimized_result.get("artist"),
                        optimized_result.get("song_title"),
                        optimized_result.get("featured_artists"),
                        optimized_result.get("release_year"),
                        parse_confidence,
                        engagement_ratio,
                        datetime.now(),
                    ),
                )

                # Save related data if available
                if optimized_result.get("musicbrainz_recording_id"):
                    self._save_musicbrainz_data(conn, optimized_result)

                if optimized_result.get("video_features"):
                    self._save_video_features(conn, optimized_result)

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Failed to save video data: {e}")
            return False

    def _save_musicbrainz_data(self, conn: sqlite3.Connection, result: Dict):
        """Save MusicBrainz data to optimized table."""
        conn.execute(
            """
            INSERT OR REPLACE INTO musicbrainz_data (
                video_id, recording_id, artist_id, genre,
                confidence, recording_length_ms, tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                result["video_id"],
                result.get("musicbrainz_recording_id"),
                result.get("musicbrainz_artist_id"),
                result.get("musicbrainz_genre"),
                round(result.get("parse_confidence", 0), 2),
                result.get("recording_length_ms"),
                result.get("musicbrainz_tags"),
            ),
        )

    def _save_video_features(self, conn: sqlite3.Connection, result: Dict):
        """Save video features to optimized table."""
        features = result.get("video_features", {})

        conn.execute(
            """
            INSERT OR REPLACE INTO video_features (
                video_id, has_guide_vocals, has_scrolling_lyrics,
                has_backing_vocals, is_instrumental, is_piano_only,
                is_acoustic, overall_confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                result["video_id"],
                features.get("has_guide_vocals", False),
                features.get("has_scrolling_lyrics", False),
                features.get("has_backing_vocals", False),
                features.get("is_instrumental_only", False),
                features.get("is_piano_only", False),
                features.get("is_acoustic", False),
                round(features.get("confidence_score", 0), 2),
            ),
        )

    def get_existing_video_ids(self) -> set:
        """Get all existing video IDs."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT video_id FROM videos")
                return {row[0] for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"Failed to get existing video IDs: {e}")
            return set()

    def get_recent_video_ids(self, days: int = 7) -> set:
        """Get video IDs from recent days."""
        try:
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

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            with self.get_connection() as conn:
                stats = {}

                # Basic counts
                stats["total_videos"] = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
                stats["total_channels"] = conn.execute("SELECT COUNT(*) FROM channels").fetchone()[
                    0
                ]

                # Parsing success
                parsed_videos = conn.execute(
                    "SELECT COUNT(*) FROM videos WHERE artist IS NOT NULL AND song_title IS NOT NULL"
                ).fetchone()[0]
                stats["parsing_success_rate"] = (
                    (parsed_videos / stats["total_videos"]) if stats["total_videos"] > 0 else 0
                )

                # MusicBrainz integration
                stats["musicbrainz_records"] = conn.execute(
                    "SELECT COUNT(*) FROM musicbrainz_data"
                ).fetchone()[0]

                # Average scores
                avg_quality = conn.execute(
                    "SELECT AVG(overall_score) FROM quality_scores"
                ).fetchone()[0]
                stats["avg_quality_score"] = round(avg_quality, 2) if avg_quality else 0

                return stats

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}

    def cleanup(self):
        """Clean up database connections."""
        with self._pool_lock:
            for conn in self._connection_pool:
                conn.close()
            self._connection_pool.clear()
