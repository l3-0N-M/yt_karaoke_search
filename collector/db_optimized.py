"""Optimized database manager for streamlined schema."""

import contextlib
import logging
import sqlite3
import threading
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Generator, List

from .config import DatabaseConfig
from .data_transformer import DataTransformer

logger = logging.getLogger(__name__)


def retry_on_database_error(max_retries=3, delay=0.1):
    """Decorator to retry database operations on failure."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except sqlite3.Error as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Database error in {func.__name__} (attempt {attempt + 1}/{max_retries}): {e}"
                        )
                        time.sleep(delay * (2**attempt))  # Exponential backoff
                    else:
                        logger.error(
                            f"Database error in {func.__name__} after {max_retries} attempts: {e}"
                        )
            raise last_exception

        return wrapper

    return decorator


class OptimizedDatabaseManager:
    """Database manager for the optimized streamlined schema."""

    SCHEMA_VERSION = 3  # Optimized schema version with Discogs support

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
            # Run migrations on existing database
            with self.get_connection() as conn:
                self._add_missing_columns(conn)

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
                    genre TEXT,
                    -- Quality metrics (rounded)
                    parse_confidence REAL,
                    quality_score REAL,
                    engagement_ratio REAL, -- as percentage
                    -- Discogs metadata
                    discogs_artist_id TEXT,
                    discogs_artist_name TEXT,
                    discogs_release_id TEXT,
                    discogs_release_title TEXT,
                    discogs_release_year INTEGER,
                    discogs_label TEXT,
                    discogs_genre TEXT,
                    discogs_style TEXT,
                    discogs_checked BOOLEAN DEFAULT 0,
                    -- MusicBrainz metadata
                    musicbrainz_checked BOOLEAN DEFAULT 0,
                    -- Web search metadata
                    web_search_performed BOOLEAN DEFAULT 0,
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

            # Discogs metadata
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS discogs_data (
                    video_id TEXT PRIMARY KEY,
                    release_id TEXT,
                    master_id TEXT,
                    artist_name TEXT,
                    song_title TEXT,
                    year INTEGER,
                    genres TEXT, -- JSON array of genres
                    styles TEXT, -- JSON array of styles
                    label TEXT,
                    country TEXT,
                    format TEXT,
                    confidence REAL DEFAULT 0.0,
                    discogs_url TEXT,
                    community_have INTEGER DEFAULT 0,
                    community_want INTEGER DEFAULT 0,
                    barcode TEXT,
                    catno TEXT,
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

            # Add missing columns to existing tables (migration)
            self._add_missing_columns(conn)

            # Insert schema version
            conn.execute(
                """
                INSERT OR REPLACE INTO schema_info (version, description)
                VALUES (?, ?)
            """,
                (self.SCHEMA_VERSION, "Optimized schema with Discogs integration support"),
            )

            conn.commit()

    def _create_indexes(self, conn: sqlite3.Connection):
        """Create performance indexes for the optimized schema."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_videos_channel_id ON videos(channel_id)",
            "CREATE INDEX IF NOT EXISTS idx_videos_artist ON videos(artist)",
            "CREATE INDEX IF NOT EXISTS idx_videos_scraped_at ON videos(scraped_at)",
            "CREATE INDEX IF NOT EXISTS idx_videos_genre ON videos(genre)",
            "CREATE INDEX IF NOT EXISTS idx_videos_release_year ON videos(release_year)",
            "CREATE INDEX IF NOT EXISTS idx_videos_upload_date ON videos(upload_date)",
            "CREATE INDEX IF NOT EXISTS idx_musicbrainz_recording ON musicbrainz_data(recording_id)",
            "CREATE INDEX IF NOT EXISTS idx_discogs_release ON discogs_data(release_id)",
            "CREATE INDEX IF NOT EXISTS idx_discogs_artist ON discogs_data(artist_name)",
            "CREATE INDEX IF NOT EXISTS idx_channels_name ON channels(channel_name)",
            "CREATE INDEX IF NOT EXISTS idx_quality_overall ON quality_scores(overall_score)",
        ]

        for index_sql in indexes:
            conn.execute(index_sql)

    def _add_missing_columns(self, conn: sqlite3.Connection):
        """Add missing columns to existing tables for backwards compatibility."""
        try:
            # Check existing columns in videos table
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(videos)")
            columns = [col[1] for col in cursor.fetchall()]

            # Add missing columns one by one
            columns_to_add = [
                ("genre", "TEXT"),
                ("discogs_artist_id", "TEXT"),
                ("discogs_artist_name", "TEXT"),
                ("discogs_release_id", "TEXT"),
                ("discogs_release_title", "TEXT"),
                ("discogs_release_year", "INTEGER"),
                ("discogs_label", "TEXT"),
                ("discogs_genre", "TEXT"),
                ("discogs_style", "TEXT"),
                ("discogs_checked", "BOOLEAN DEFAULT 0"),
                ("musicbrainz_checked", "BOOLEAN DEFAULT 0"),
                ("web_search_performed", "BOOLEAN DEFAULT 0"),
            ]

            for col_name, col_type in columns_to_add:
                if col_name not in columns:
                    conn.execute(f"ALTER TABLE videos ADD COLUMN {col_name} {col_type}")
                    logger.info(f"Added {col_name} column to videos table")

        except Exception as e:
            logger.warning(f"Failed to add missing columns: {e}")

    @retry_on_database_error()
    def save_video_data(self, result):
        """Save video data using the optimized schema."""
        logger.debug(
            "ENHANCED_DB_FIX_V2: save_video_data called"
        )  # Version marker to confirm our code is running
        try:
            # Handle ProcessingResult objects
            if hasattr(result, "video_data"):
                video_data = result.video_data
            else:
                video_data = result

            # Transform data to optimized format
            optimized_result = DataTransformer.transform_video_data_to_optimized(video_data)

            with self.get_connection() as conn:
                # Ensure channel exists before saving video
                self._ensure_channel_exists(conn, optimized_result)

                # Round values for the optimized schema
                parse_confidence = optimized_result.get("parse_confidence")
                engagement_ratio = optimized_result.get("engagement_ratio")

                # Extract quality score from quality_scores data if available
                quality_score = None
                quality_scores = optimized_result.get("quality_scores", {})
                if quality_scores and quality_scores.get("overall_score") is not None:
                    quality_score = round(quality_scores["overall_score"], 2)

                # Prefer Discogs year over parsed year if available
                parsed_year = optimized_result.get("release_year")
                discogs_year = optimized_result.get("discogs_release_year")

                # Use Discogs year if available and valid, otherwise use parsed year
                release_year = discogs_year if discogs_year and discogs_year > 1900 else parsed_year

                # Additional validation - reject current/future years
                current_year = datetime.now().year
                if release_year and release_year >= current_year:
                    logger.warning(
                        f"Rejecting release year {release_year} for {optimized_result['video_id']} as it's current/future year"
                    )
                    release_year = None

                # Update the result with the validated year
                optimized_result["release_year"] = release_year

                # Log what genre/year values we're about to save
                genre = optimized_result.get("genre")
                video_id = optimized_result["video_id"]

                if release_year or genre:
                    logger.info(
                        f"Saving metadata for {video_id}: release_year={release_year} (parsed={parsed_year}, discogs={discogs_year}), genre={genre}"
                    )

                # Enhanced safe_convert function with string length limits and better validation
                def safe_convert(value, default=None, max_length=500):
                    """Convert value to database-compatible type with comprehensive error handling and length limits."""
                    try:
                        if value is None:
                            return default

                        # Handle complex data types that SQLite can't store
                        if isinstance(value, (list, dict, tuple, set)):
                            if not value:  # Empty containers
                                return default
                            # Convert to JSON string for complex types
                            import json

                            try:
                                json_str = json.dumps(value, ensure_ascii=False, default=str)
                                # Truncate if too long
                                if len(json_str) > max_length:
                                    logger.warning(
                                        f"Truncating JSON string from {len(json_str)} to {max_length} chars"
                                    )
                                    return json_str[: max_length - 3] + "..."
                                return json_str
                            except (TypeError, ValueError):
                                str_value = str(value)
                                if len(str_value) > max_length:
                                    logger.warning(
                                        f"Truncating complex type string from {len(str_value)} to {max_length} chars"
                                    )
                                    return str_value[: max_length - 3] + "..."
                                return str_value

                        # Handle boolean types
                        if isinstance(value, bool):
                            return int(value)

                        # Handle bytes/bytearray
                        if isinstance(value, (bytes, bytearray)):
                            try:
                                decoded = value.decode("utf-8", errors="ignore")
                                if len(decoded) > max_length:
                                    logger.warning(
                                        f"Truncating decoded bytes from {len(decoded)} to {max_length} chars"
                                    )
                                    return decoded[: max_length - 3] + "..."
                                return decoded
                            except (UnicodeDecodeError, AttributeError):
                                str_value = str(value)
                                if len(str_value) > max_length:
                                    return str_value[: max_length - 3] + "..."
                                return str_value

                        # Handle string types with length validation
                        if isinstance(value, str):
                            # Check if it's a numeric string that needs conversion
                            if value.strip() == "":
                                return default

                            # Critical fix: Truncate extremely long strings that cause binding errors
                            if len(value) > max_length:
                                logger.warning(
                                    f"Truncating string from {len(value)} to {max_length} chars: '{value[:50]}...'"
                                )
                                value = value[: max_length - 3] + "..."

                            # Ultra-aggressive character cleaning for SQLite compatibility
                            try:
                                # First, ensure UTF-8 compatibility
                                clean_value = value.encode("utf-8", errors="ignore").decode("utf-8")

                                # Remove any null bytes that cause SQLite binding issues
                                clean_value = clean_value.replace("\x00", "")

                                # Remove or replace other problematic characters
                                import re

                                # Remove control characters except common ones (tab, newline, carriage return)
                                clean_value = re.sub(
                                    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", clean_value
                                )

                                # Additional SQLite-specific cleaning
                                # Remove any potential SQL injection patterns (as extra safety)
                                clean_value = clean_value.replace("'", "''")  # Escape single quotes

                                # Final validation: ensure it's still a valid string
                                if clean_value != value:
                                    logger.warning(
                                        f"String cleaned for SQLite compatibility: '{value[:50]}...' -> '{clean_value[:50]}...'"
                                    )

                                return clean_value
                            except Exception as clean_error:
                                logger.warning(
                                    f"String cleaning failed: {clean_error}, using fallback"
                                )
                                # Fallback: create a completely safe string
                                safe_str = "".join(
                                    c for c in value if c.isprintable() and ord(c) < 128
                                )[:max_length]
                                logger.warning(f"Fallback string created: '{safe_str[:50]}...'")
                                return safe_str or default

                        # Handle numeric types
                        if isinstance(value, (int, float)):
                            # Check for invalid numeric values
                            if str(value).lower() in ("nan", "inf", "-inf"):
                                logger.warning(f"Invalid numeric value: {value}, using default")
                                return default
                            # Check for extremely large numbers that might cause issues
                            if isinstance(value, int) and abs(value) > 2**63 - 1:
                                logger.warning(f"Integer too large: {value}, truncating")
                                return 2**63 - 1 if value > 0 else -(2**63 - 1)
                            return value

                        # Handle objects with custom string representations
                        try:
                            # Check if it's a custom object/class that might not be SQLite-compatible
                            if hasattr(value, "__dict__") or hasattr(value, "__class__"):
                                logger.warning(f"Converting complex object {type(value)} to string")

                            str_value = str(value)
                            # Verify the string is valid and not too long
                            if str_value and str_value != "None":
                                if len(str_value) > max_length:
                                    logger.warning(
                                        f"Truncating object string from {len(str_value)} to {max_length} chars"
                                    )
                                    return str_value[: max_length - 3] + "..."
                                # Final validation: ensure it's a plain string
                                return str(str_value)  # Double conversion to ensure plain string
                            else:
                                return default
                        except Exception as e:
                            logger.warning(f"Failed to convert object to string: {e}")
                            return default

                    except Exception as e:
                        logger.error(
                            f"safe_convert failed for value {type(value)} '{str(value)[:50] if value else None}': {e}"
                        )
                        return default

                # Prepare and validate all parameters before database insertion
                # Special handling for problematic featured_artists field (parameter 15)
                featured_artists_raw = optimized_result.get("featured_artists")
                if isinstance(featured_artists_raw, str) and len(featured_artists_raw) > 500:
                    logger.warning(
                        f"Featured artists field extremely long ({len(featured_artists_raw)} chars), truncating: '{featured_artists_raw[:50]}...'"
                    )

                params = [
                    safe_convert(optimized_result["video_id"], ""),
                    safe_convert(optimized_result["url"], ""),
                    safe_convert(optimized_result["title"], ""),
                    safe_convert(
                        optimized_result.get("description"), "", 2000
                    ),  # Increased from 500
                    safe_convert(optimized_result.get("duration_seconds")),
                    safe_convert(optimized_result.get("view_count"), 0),
                    safe_convert(optimized_result.get("like_count"), 0),
                    safe_convert(optimized_result.get("comment_count"), 0),
                    safe_convert(optimized_result.get("upload_date")),
                    safe_convert(optimized_result.get("thumbnail_url")),
                    safe_convert(optimized_result.get("channel_name")),
                    safe_convert(optimized_result.get("channel_id")),
                    safe_convert(optimized_result.get("artist")),
                    safe_convert(optimized_result.get("song_title")),
                    safe_convert(
                        optimized_result.get("featured_artists"), "", 500
                    ),  # Increased from 200
                    safe_convert(optimized_result.get("release_year")),
                    safe_convert(optimized_result.get("genre")),
                    safe_convert(parse_confidence),
                    safe_convert(quality_score),
                    safe_convert(engagement_ratio),
                    # Discogs fields
                    safe_convert(optimized_result.get("discogs_artist_id")),
                    safe_convert(optimized_result.get("discogs_artist_name")),
                    safe_convert(optimized_result.get("discogs_release_id")),
                    safe_convert(optimized_result.get("discogs_release_title")),
                    safe_convert(optimized_result.get("discogs_release_year")),
                    safe_convert(optimized_result.get("discogs_label")),
                    safe_convert(optimized_result.get("discogs_genre")),
                    safe_convert(optimized_result.get("discogs_style")),
                    safe_convert(optimized_result.get("discogs_checked", 0)),
                    safe_convert(optimized_result.get("musicbrainz_checked", 0)),
                    safe_convert(optimized_result.get("web_search_performed", 0)),
                    datetime.now(),
                ]

                # Enhanced debug logging for all parameters before database insertion
                logger.debug(f"Attempting to save video {video_id} with {len(params)} parameters")
                for i, param in enumerate(params):
                    param_info = f"Param {i+1} ({type(param).__name__})"
                    if isinstance(param, str) and len(param) > 100:
                        param_info += f": length={len(param)}, preview='{param[:50]}...'"
                    elif param is not None:
                        param_info += f": {repr(param)}"
                    else:
                        param_info += ": None"

                    # Log parameters at debug level
                    if not isinstance(param, (str, int, float, type(datetime.now()), type(None))):
                        logger.warning(f"UNEXPECTED TYPE: {param_info}")
                        # Deep analysis for any unexpected types
                        if hasattr(param, "__dict__"):
                            logger.warning(f"UNEXPECTED TYPE __dict__: {param.__dict__}")
                    else:
                        logger.debug(param_info)

                # Critical test: Try parameter 15 individually before full INSERT
                try:
                    test_cursor = conn.cursor()
                    test_cursor.execute("SELECT ?", (params[14],))  # Test parameter 15 specifically
                    logger.debug("Parameter 15 individual test: PASSED")
                except Exception as param15_error:
                    logger.error(f"Parameter 15 individual test FAILED: {param15_error}")
                    logger.error(f"Parameter 15 problematic value: {repr(params[14])}")
                    # Ultra-aggressive fallback conversion
                    original_param15 = params[14]
                    if params[14] is not None:
                        # Create an ASCII-only, printable-only string
                        import re

                        ascii_only = "".join(
                            c for c in str(params[14]) if ord(c) < 128 and c.isprintable()
                        )
                        # Remove any remaining problematic patterns
                        ascii_only = re.sub(
                            r'[^\w\s\-.,!?()[\]:;"\'@#$%^&*+=<>/\\|`~]', "", ascii_only
                        )
                        params[14] = ascii_only[:200] if ascii_only else "CLEANED_STRING"
                    else:
                        params[14] = ""
                    logger.warning(
                        f"Parameter 15 ULTRA-AGGRESSIVE conversion: {repr(original_param15)} -> {repr(params[14])}"
                    )

                    # Test the cleaned parameter
                    try:
                        test_cursor.execute("SELECT ?", (params[14],))
                        logger.info("Parameter 15 post-cleaning test: PASSED")
                    except Exception as still_failing:
                        logger.error(f"Parameter 15 STILL FAILING after cleaning: {still_failing}")
                        params[14] = "FALLBACK_STRING"  # Ultimate fallback

                # Insert into videos table with enhanced error handling
                logger.debug("ENHANCED_DB_FIX_V2: About to execute main INSERT statement")
                try:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO videos (
                            video_id, url, title, description, duration_seconds,
                            view_count, like_count, comment_count, upload_date,
                            thumbnail_url, channel_name, channel_id,
                            artist, song_title, featured_artists, release_year, genre,
                            parse_confidence, quality_score, engagement_ratio,
                            discogs_artist_id, discogs_artist_name, discogs_release_id,
                            discogs_release_title, discogs_release_year, discogs_label,
                            discogs_genre, discogs_style, discogs_checked,
                            musicbrainz_checked, web_search_performed, scraped_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        params,
                    )
                    logger.debug(f"Successfully inserted video {video_id} into database")
                except Exception as sql_error:
                    logger.error(f"SQL INSERT failed for video {video_id}: {sql_error}")
                    param_summary = []
                    for i, p in enumerate(params):
                        if len(str(p)) < 100:
                            param_summary.append(f"P{i+1}: {type(p).__name__} = {repr(p)}")
                        else:
                            param_summary.append(
                                f"P{i+1}: {type(p).__name__} = ({len(str(p))} chars)"
                            )
                    logger.error(f"Failed parameters: {param_summary}")

                    # Try to identify the specific problematic parameter
                    try:
                        # Test each parameter individually by checking its SQL compatibility
                        import sqlite3

                        test_conn = sqlite3.connect(":memory:")
                        test_conn.execute("CREATE TABLE test (value TEXT)")

                        for i, param in enumerate(params):
                            try:
                                test_conn.execute("INSERT INTO test VALUES (?)", (param,))
                                test_conn.execute("DELETE FROM test")
                            except Exception as param_error:
                                param_display = (
                                    repr(param)
                                    if len(str(param)) < 200
                                    else f"({len(str(param))} chars)"
                                )
                                logger.error(
                                    f"Parameter {i+1} failed individual test: {param_error} | Value: {param_display}"
                                )

                        test_conn.close()
                    except Exception as test_error:
                        logger.error(f"Parameter testing failed: {test_error}")

                    raise sql_error

                # Save related data if available
                if optimized_result.get("musicbrainz_recording_id"):
                    self._save_musicbrainz_data(conn, optimized_result)

                if optimized_result.get("video_features"):
                    self._save_video_features(conn, optimized_result)

                if optimized_result.get("quality_scores"):
                    self._save_quality_scores(conn, optimized_result)

                if optimized_result.get("ryd_data") or optimized_result.get("estimated_dislikes"):
                    self._save_ryd_data(conn, optimized_result)

                if optimized_result.get("discogs_release_id"):
                    try:
                        discogs_success = self._save_discogs_data(conn, optimized_result)
                        # Record database save if we have access to monitor
                        if hasattr(self, "_discogs_monitor") and self._discogs_monitor:
                            self._discogs_monitor.record_database_save(discogs_success)
                    except Exception as discogs_error:
                        logger.error(
                            f"Discogs data save failed for video {optimized_result['video_id']}: {discogs_error}"
                        )
                        logger.warning("Continuing with main video save despite Discogs failure")
                        # Record the failure if we have access to monitor
                        if hasattr(self, "_discogs_monitor") and self._discogs_monitor:
                            self._discogs_monitor.record_database_save(False)

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Failed to save video data: {e}")
            return False

    def _ensure_channel_exists(self, conn: sqlite3.Connection, video_data: Dict):
        """Ensure a channel record exists before saving video data."""
        channel_id = video_data.get("channel_id")
        if not channel_id:
            return

        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM channels WHERE channel_id = ?", (channel_id,))
            if cursor.fetchone():
                return

            channel_url = f"https://www.youtube.com/channel/{channel_id}"
            channel_name = video_data.get("channel_name", "Unknown Channel")

            cursor.execute(
                """
                INSERT OR IGNORE INTO channels (
                    channel_id, channel_url, channel_name, updated_at
                ) VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (channel_id, channel_url, channel_name),
            )
            logger.debug(f"Created missing channel record: {channel_name} ({channel_id})")

        except Exception as e:
            logger.error(f"Failed to ensure channel exists for {channel_id}: {e}")

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

    def _save_quality_scores(self, conn: sqlite3.Connection, result: Dict):
        """Save quality scores to optimized table."""
        quality_scores = result.get("quality_scores", {})

        conn.execute(
            """
            INSERT OR REPLACE INTO quality_scores (
                video_id, overall_score, technical_score,
                engagement_score, metadata_score, calculated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                result["video_id"],
                round(quality_scores.get("overall_score", 0), 2),
                round(quality_scores.get("technical_score", 0), 2),
                round(quality_scores.get("engagement_score", 0), 2),
                round(quality_scores.get("metadata_score", 0), 2),
                datetime.now(),
            ),
        )

    def _save_ryd_data(self, conn: sqlite3.Connection, result: Dict):
        """Save Return YouTube Dislike data to optimized table."""
        ryd_data = result.get("ryd_data", {})

        # Handle both nested ryd_data and top-level fields
        estimated_dislikes = ryd_data.get("estimated_dislikes") or result.get(
            "estimated_dislikes", 0
        )
        ryd_likes = ryd_data.get("likes") or result.get("ryd_likes", 0)
        ryd_rating = ryd_data.get("rating") or result.get("ryd_rating", 0.0)
        ryd_confidence = ryd_data.get("confidence") or result.get("ryd_confidence", 0.0)

        conn.execute(
            """
            INSERT OR REPLACE INTO ryd_data (
                video_id, estimated_dislikes, ryd_likes,
                ryd_rating, ryd_confidence, fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                result["video_id"],
                int(estimated_dislikes),
                int(ryd_likes),
                round(float(ryd_rating), 2),
                round(float(ryd_confidence), 2),
                datetime.now(),
            ),
        )

    def _save_discogs_data(self, conn: sqlite3.Connection, result: Dict) -> bool:
        """Save Discogs data to optimized table."""
        import json

        # Extract Discogs data from metadata or top-level fields
        discogs_data = {}
        metadata = result.get("metadata", {})

        # Check for Discogs data in metadata or top-level
        for key in [
            "discogs_release_id",
            "discogs_master_id",
            "year",
            "genres",
            "styles",
            "label",
            "country",
            "format",
            "confidence",
            "discogs_url",
            "community",
            "barcode",
            "catno",
        ]:
            value = metadata.get(key) or result.get(key)
            if value is not None:
                discogs_data[key] = value

        # Also check for artist_name and song_title from Discogs
        discogs_artist = metadata.get("artist_name") or result.get("artist")
        discogs_title = metadata.get("song_title") or result.get("song_title")

        if not discogs_data.get("discogs_release_id"):
            return False  # No Discogs data to save

        # Handle community data with safe type conversion
        community = discogs_data.get("community", {})

        def safe_int_convert(value, default=0):
            """Safely convert value to integer."""
            try:
                if value is None:
                    return default
                if isinstance(value, (int, float)):
                    return int(value)
                if isinstance(value, str):
                    # Try to parse string numbers
                    if value.strip().isdigit():
                        return int(value)
                    return default
                # For complex objects, log and return default
                logger.warning(
                    f"Cannot convert community value to int: {type(value)} = {repr(value)}"
                )
                return default
            except (ValueError, TypeError, OverflowError) as e:
                logger.warning(f"Failed to convert community value to int: {e}")
                return default

        if isinstance(community, dict):
            community_have = safe_int_convert(community.get("have", 0))
            community_want = safe_int_convert(community.get("want", 0))
        else:
            logger.warning(f"Community data is not a dict: {type(community)} = {repr(community)}")
            community_have = 0
            community_want = 0

        # Convert arrays to JSON strings
        genres_json = (
            json.dumps(discogs_data.get("genres", [])) if discogs_data.get("genres") else None
        )
        styles_json = (
            json.dumps(discogs_data.get("styles", [])) if discogs_data.get("styles") else None
        )

        # Prepare Discogs parameters with enhanced validation
        discogs_params = [
            result["video_id"],
            discogs_data.get("discogs_release_id"),
            discogs_data.get("discogs_master_id"),
            discogs_artist,
            discogs_title,
            discogs_data.get("year"),
            genres_json,
            styles_json,
            discogs_data.get("label"),
            discogs_data.get("country"),
            discogs_data.get("format"),
            round(float(discogs_data.get("confidence", 0)), 2),
            discogs_data.get("discogs_url"),
            community_have,  # Parameter 14
            community_want,  # Parameter 15
            # Convert list fields to comma-separated strings
            (
                ", ".join(str(b) for b in discogs_data.get("barcode", []))
                if isinstance(discogs_data.get("barcode"), list)
                else str(discogs_data.get("barcode") or "")
            ),
            (
                ", ".join(str(c) for c in discogs_data.get("catno", []))
                if isinstance(discogs_data.get("catno"), list)
                else str(discogs_data.get("catno") or "")
            ),
            datetime.now(),
        ]

        # Enhanced Discogs parameter debugging
        logger.debug(
            f"DISCOGS: Saving data for video {result['video_id']} with {len(discogs_params)} parameters"
        )
        for i, param in enumerate(discogs_params):
            if i in [13, 14]:  # Focus on community parameters (14=have, 15=want)
                logger.info(f"DISCOGS PARAM {i+1}: {type(param).__name__} = {repr(param)}")
            elif isinstance(param, str) and len(str(param)) > 100:
                logger.debug(f"DISCOGS PARAM {i+1}: {type(param).__name__} (length={len(param)})")
            else:
                logger.debug(f"DISCOGS PARAM {i+1}: {type(param).__name__} = {repr(param)}")

        # Retry logic for Discogs save
        last_error = None
        for retry_attempt in range(3):
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO discogs_data (
                        video_id, release_id, master_id, artist_name, song_title,
                        year, genres, styles, label, country, format, confidence,
                        discogs_url, community_have, community_want, barcode, catno,
                        fetched_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    discogs_params,
                )
                logger.debug(f"DISCOGS: Successfully saved data for video {result['video_id']}")
                return True
            except Exception as discogs_error:
                last_error = discogs_error
                if retry_attempt < 2:
                    logger.warning(
                        f"DISCOGS: Retry {retry_attempt + 1}/3 for video {result['video_id']}: {discogs_error}"
                    )
                    time.sleep(0.1 * (2**retry_attempt))
                    continue

        # If all retries failed, handle the error
        if last_error:
            logger.error(
                f"DISCOGS: Failed to save data for video {result['video_id']}: {last_error}"
            )
            # Enhanced error recovery for parameter binding issues
            if "parameter" in str(last_error).lower() and "binding" in str(last_error).lower():
                logger.error(f"DISCOGS PARAMETER BINDING ERROR: {last_error}")
                logger.error("Problematic parameters summary:")
                for i, param in enumerate(discogs_params):
                    logger.error(
                        f"  Param {i+1}: {type(param).__name__} = {repr(param) if len(str(param)) < 100 else f'({len(str(param))} chars)'}"
                    )

                # Create ultra-safe fallback parameters
                logger.warning("DISCOGS: Creating ultra-safe fallback parameters")
                safe_params = [
                    str(result["video_id"]),  # video_id - force to string
                    str(discogs_data.get("discogs_release_id", "unknown")),  # release_id
                    None,  # master_id
                    str(discogs_artist or "unknown"),  # artist_name
                    str(discogs_title or "unknown"),  # song_title
                    int(discogs_data.get("year", 0)) if discogs_data.get("year") else None,  # year
                    genres_json,  # genres (already JSON)
                    styles_json,  # styles (already JSON)
                    (
                        str(discogs_data.get("label", "")) if discogs_data.get("label") else None
                    ),  # label
                    (
                        str(discogs_data.get("country", ""))
                        if discogs_data.get("country")
                        else None
                    ),  # country
                    (
                        str(discogs_data.get("format", "")) if discogs_data.get("format") else None
                    ),  # format
                    float(discogs_data.get("confidence", 0.0)),  # confidence
                    (
                        str(discogs_data.get("discogs_url", ""))
                        if discogs_data.get("discogs_url")
                        else None
                    ),  # discogs_url
                    0,  # community_have - force to 0
                    0,  # community_want - force to 0
                    (
                        str(discogs_data.get("barcode", ""))
                        if discogs_data.get("barcode")
                        else None
                    ),  # barcode
                    (
                        str(discogs_data.get("catno", "")) if discogs_data.get("catno") else None
                    ),  # catno
                    datetime.now(),  # fetched_at
                ]

                logger.warning("DISCOGS: Attempting retry with ultra-safe parameters")
                try:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO discogs_data (
                            video_id, release_id, master_id, artist_name, song_title,
                            year, genres, styles, label, country, format, confidence,
                            discogs_url, community_have, community_want, barcode, catno,
                            fetched_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        safe_params,
                    )
                    logger.info("DISCOGS: Ultra-safe retry successful")
                    return True
                except Exception as retry_error:
                    logger.error(f"DISCOGS: Ultra-safe retry also failed: {retry_error}")
                    # At this point, we give up and let the main video save continue
                    logger.error(
                        "DISCOGS: Giving up on Discogs save, but allowing main video save to proceed"
                    )
                    return False
            else:
                # Non-parameter binding errors should still be raised
                raise last_error
        return False

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

                # Discogs integration
                stats["discogs_records"] = conn.execute(
                    "SELECT COUNT(*) FROM discogs_data"
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

    def save_channel_data(self, channel_data: Dict) -> bool:
        """Save or update channel information."""
        try:
            with self.get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO channels (
                        channel_id, channel_url, channel_name, video_count,
                        description, is_karaoke_focused, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (
                        channel_data.get("channel_id"),
                        channel_data.get("channel_url"),
                        channel_data.get("channel_name"),
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

    def get_processed_channels(self) -> List[Dict[str, Any]]:
        """Get list of all processed channels with their stats."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT
                        c.channel_id, c.channel_name, c.channel_url,
                        c.video_count, c.is_karaoke_focused,
                        c.updated_at, COUNT(v.video_id) as collected_videos
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
                        "video_count": row[3],
                        "is_karaoke_focused": bool(row[4]),
                        "last_processed_at": row[5],
                        "collected_videos": row[6],
                    }
                    for row in cursor.fetchall()
                ]
        except Exception as e:
            logger.error(f"Failed to get processed channels: {e}")
            return []

    def get_channel_last_processed(self, channel_id: str) -> str | None:
        """Get the last processed timestamp for a channel."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT updated_at FROM channels WHERE channel_id = ?", (channel_id,)
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
                    "UPDATE channels SET updated_at = CURRENT_TIMESTAMP WHERE channel_id = ?",
                    (channel_id,),
                )
                return True
        except Exception as e:
            logger.error(f"Failed to update processed time for channel {channel_id}: {e}")
            return False

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
            return False

    def get_existing_video_ids_batch(self, video_ids: List[str]) -> set:
        """Get set of video IDs that already exist in database (batch operation)."""
        if not video_ids:
            return set()

        try:
            with self.get_connection() as conn:
                placeholders = ",".join(["?"] * len(video_ids))
                cursor = conn.execute(
                    f"SELECT video_id FROM videos WHERE video_id IN ({placeholders})", video_ids
                )
                return {row[0] for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"Failed to get existing video IDs: {e}")
            return set()

    def cleanup(self):
        """Clean up database connections."""
        with self._pool_lock:
            for conn in self._connection_pool:
                conn.close()
            self._connection_pool.clear()
