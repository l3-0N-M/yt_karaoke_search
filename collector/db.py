"""Database management with proper connection handling and migrations."""

import sqlite3
import contextlib
import shutil
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Generator, Dict, List, Any, Optional
from .config import DatabaseConfig

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Enhanced database management with migrations and backups."""
    
    SCHEMA_VERSION = 3  # Updated to version 3 for new columns
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.db_path = Path(config.path)
        self.backup_dir = self.db_path.parent / "backups"
        self.migrations_dir = self.db_path.parent / "migrations"
        self.setup_database()
    
    @contextlib.contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
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
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS schema_info (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute("SELECT version FROM schema_info ORDER BY version DESC LIMIT 1")
            current_version = cursor.fetchone()
            current_version = current_version[0] if current_version else 0
            
            self._apply_migrations(cursor, current_version)
            
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
            with open(migration_002_path, 'w') as f:
                f.write(migration_002_content)
        
        migration_003_path = self.migrations_dir / "003_featured_artists_and_ratios.sql"
        if not migration_003_path.exists():
            migration_003_content = """-- Migration 003: Add featured artists and engagement ratios
ALTER TABLE videos ADD COLUMN featured_artists TEXT;
ALTER TABLE videos ADD COLUMN like_dislike_to_views_ratio REAL;

-- Update existing records with computed ratios
UPDATE videos 
SET like_dislike_to_views_ratio = 
  CASE 
    WHEN view_count > 0 THEN
      (like_count - COALESCE((SELECT estimated_dislikes FROM ryd_data WHERE ryd_data.video_id = videos.video_id), 0)) * 1.0 / view_count
    ELSE NULL 
  END
WHERE view_count > 0;

INSERT OR REPLACE INTO schema_info(version) VALUES (3);
"""
            with open(migration_003_path, 'w') as f:
                f.write(migration_003_content)
    
    def _apply_migrations(self, cursor: sqlite3.Cursor, current_version: int):
        """Apply database migrations."""
        if current_version < 1:
            self._create_initial_schema(cursor)
            cursor.execute("INSERT INTO schema_info (version) VALUES (1)")
            logger.info("Applied migration: Initial schema (v1)")
        
        current_version = 1

        if current_version < 2:
            # Apply migration from file
            migration_002_path = self.migrations_dir / "002_updated_at_trigger.sql"
            if migration_002_path.exists():
                with open(migration_002_path, 'r') as f:
                    migration_sql = f.read()
                cursor.executescript(migration_sql)
                logger.info("Applied migration: Updated trigger (v2)")

                current_version = 2
        
        if current_version < 3:
            # Apply migration from file
            migration_003_path = self.migrations_dir / "003_featured_artists_and_ratios.sql"
            if migration_003_path.exists():
                # Skip if columns already exist
                cursor.execute("PRAGMA table_info(videos)")
                existing_cols = {row[1] for row in cursor.fetchall()}
                if 'featured_artists' not in existing_cols:
                    with open(migration_003_path, 'r') as f:
                        migration_sql = f.read()
                    cursor.executescript(migration_sql)
                    logger.info("Applied migration: Featured artists and ratios (v3)")
                else:
                    cursor.execute("INSERT OR REPLACE INTO schema_info(version) VALUES (3)")
                    logger.info("Skipped migration 003; columns already present")
                current_version = 3
    
    def _create_initial_schema(self, cursor: sqlite3.Cursor):
        """Create the initial database schema."""
        
        # Main videos table
        cursor.execute('''
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
                estimated_release_year INTEGER,
                genre TEXT,
                language TEXT,
                like_dislike_to_views_ratio REAL,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Video features table
        cursor.execute('''
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
        ''')
        
        # Quality scores table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_scores (
                video_id TEXT PRIMARY KEY,
                overall_score REAL DEFAULT 0.0,
                technical_score REAL DEFAULT 0.0,
                engagement_score REAL DEFAULT 0.0,
                metadata_completeness REAL DEFAULT 0.0,
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
            )
        ''')
        
        # Return YouTube Dislike data table
        cursor.execute('''
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
        ''')
        
        # Search history table
        cursor.execute('''
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
        ''')
        
        # Error log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS error_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT,
                error_type TEXT NOT NULL,
                error_message TEXT,
                retry_count INTEGER DEFAULT 0,
                resolved BOOLEAN DEFAULT 0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes
        self._create_indexes(cursor)
    
    def _create_indexes(self, cursor: sqlite3.Cursor):
        """Create database indexes for performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_videos_artist ON videos(original_artist)",
            "CREATE INDEX IF NOT EXISTS idx_videos_upload_date ON videos(upload_date)",
            "CREATE INDEX IF NOT EXISTS idx_videos_views ON videos(view_count)",
            "CREATE INDEX IF NOT EXISTS idx_features_confidence ON video_features(confidence_score)",
            "CREATE INDEX IF NOT EXISTS idx_quality_overall ON quality_scores(overall_score)"
        ]
        
        for index in indexes:
            cursor.execute(index)
    
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
    
    def save_video_data(self, result):
        """Save processed video data to database."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                video_data = result.video_data
                features = video_data.get('features', {})
                quality_scores = video_data.get('quality_scores', {})
                
                # Calculate like/dislike to views ratio
                like_dislike_ratio = None
                views = video_data.get('view_count', 0)
                likes = video_data.get('like_count', 0)
                dislikes = video_data.get('estimated_dislikes', 0)
                
                if views > 0:
                    like_dislike_ratio = (likes - dislikes) / views
                
                # Insert main video record
                cursor.execute('''
                    INSERT OR REPLACE INTO videos (
                        video_id, url, title, description, duration_seconds,
                        view_count, like_count, comment_count, upload_date,
                        thumbnail_url, channel_name, channel_id, original_artist, 
                        featured_artists, song_title, estimated_release_year,
                        like_dislike_to_views_ratio
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    video_data.get('video_id'),
                    video_data.get('url'),
                    video_data.get('title'),
                    video_data.get('description', '')[:2000],
                    video_data.get('duration_seconds'),
                    video_data.get('view_count'),
                    video_data.get('like_count'),
                    video_data.get('comment_count'),
                    video_data.get('upload_date'),
                    video_data.get('thumbnail'),
                    video_data.get('uploader'),
                    video_data.get('uploader_id'),
                    features.get('original_artist'),
                    features.get('featured_artists'),
                    features.get('song_title'),
                    video_data.get('estimated_release_year'),
                    like_dislike_ratio
                ))
                
                # Insert features with confidence scores
                cursor.execute('''
                    INSERT OR REPLACE INTO video_features (
                        video_id, has_guide_vocals, has_scrolling_lyrics,
                        has_backing_vocals, is_instrumental_only, is_piano_only,
                        is_acoustic, has_guide_vocals_confidence, has_scrolling_lyrics_confidence,
                        has_backing_vocals_confidence, is_instrumental_only_confidence,
                        is_piano_only_confidence, is_acoustic_confidence, confidence_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    video_data.get('video_id'),
                    features.get('has_guide_vocals', False),
                    features.get('has_scrolling_lyrics', False),
                    features.get('has_backing_vocals', False),
                    features.get('is_instrumental_only', False),
                    features.get('is_piano_only', False),
                    features.get('is_acoustic', False),
                    features.get('has_guide_vocals_confidence', 0.0),
                    features.get('has_scrolling_lyrics_confidence', 0.0),
                    features.get('has_backing_vocals_confidence', 0.0),
                    features.get('is_instrumental_only_confidence', 0.0),
                    features.get('is_piano_only_confidence', 0.0),
                    features.get('is_acoustic_confidence', 0.0),
                    result.confidence_score
                ))
                
                # Insert quality scores
                cursor.execute('''
                    INSERT OR REPLACE INTO quality_scores (
                        video_id, overall_score, technical_score,
                        engagement_score, metadata_completeness
                    ) VALUES (?, ?, ?, ?, ?)
                ''', (
                    video_data.get('video_id'),
                    quality_scores.get('overall_score', 0),
                    quality_scores.get('technical_score', 0),
                    quality_scores.get('engagement_score', 0),
                    quality_scores.get('metadata_completeness', 0)
                ))
                
                # Insert RYD data if available
                if video_data.get('estimated_dislikes') is not None:
                    cursor.execute('''
                        INSERT OR REPLACE INTO ryd_data (
                            video_id, estimated_dislikes, ryd_likes, ryd_rating, ryd_confidence
                        ) VALUES (?, ?, ?, ?, ?)
                    ''', (
                        video_data.get('video_id'),
                        video_data.get('estimated_dislikes'),
                        video_data.get('ryd_likes'),
                        video_data.get('ryd_rating'),
                        video_data.get('ryd_confidence')
                    ))
                
        except Exception as e:
            logger.error(f"Database save failed: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                stats = {}
                
                cursor.execute("SELECT COUNT(*) FROM videos")
                stats['total_videos'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM videos WHERE original_artist IS NOT NULL")
                stats['videos_with_artist'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT AVG(confidence_score) FROM video_features WHERE confidence_score > 0")
                result = cursor.fetchone()
                stats['avg_confidence'] = result[0] if result[0] else 0.0
                
                cursor.execute("SELECT AVG(overall_score) FROM quality_scores WHERE overall_score > 0")
                result = cursor.fetchone()
                stats['avg_quality'] = result[0] if result[0] else 0.0
                
                cursor.execute('''
                    SELECT original_artist, COUNT(*) as count, AVG(view_count) as avg_views
                    FROM videos 
                    WHERE original_artist IS NOT NULL 
                    GROUP BY original_artist 
                    ORDER BY count DESC 
                    LIMIT 10
                ''')
                stats['top_artists'] = cursor.fetchall()
                
                return stats
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}