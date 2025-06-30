#!/usr/bin/env python3
"""
Database Optimization Script
Streamlines the karaoke database by removing unused fields, rounding values, and simplifying schema.
"""

import os
import sqlite3


def round_value(value, decimals=2):
    """Round a numeric value to specified decimal places."""
    if value is None:
        return None
    try:
        return round(float(value), decimals)
    except (ValueError, TypeError):
        return value


def optimize_database(db_path: str, output_path: str = None):
    """Optimize the database by streamlining schema and values."""

    if output_path is None:
        output_path = db_path.replace(".db", "_optimized.db")

    print(f"Optimizing database: {db_path} -> {output_path}")

    # Connect to source database
    source_conn = sqlite3.connect(db_path)
    source_conn.row_factory = sqlite3.Row

    # Create optimized database
    if os.path.exists(output_path):
        os.remove(output_path)

    opt_conn = sqlite3.connect(output_path)
    opt_conn.execute("PRAGMA foreign_keys = ON")

    try:
        # Create optimized schema
        create_optimized_schema(opt_conn)

        # Migrate data with transformations (channels first for FK constraints)
        migrate_channels_data(source_conn, opt_conn)
        migrate_videos_data(source_conn, opt_conn)
        migrate_video_features_data(source_conn, opt_conn)
        migrate_musicbrainz_data(source_conn, opt_conn)
        migrate_validation_data(source_conn, opt_conn)
        migrate_quality_scores(source_conn, opt_conn)
        migrate_ryd_data(source_conn, opt_conn)

        # Commit changes
        opt_conn.commit()

        # Verify migration
        verify_migration(source_conn, opt_conn)

        print(f"✅ Database optimization complete! Saved to: {output_path}")

    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        opt_conn.rollback()
        raise
    finally:
        source_conn.close()
        opt_conn.close()


def create_optimized_schema(conn):
    """Create the optimized database schema."""

    print("📋 Creating optimized schema...")

    # Core video data (streamlined)
    conn.execute(
        """
        CREATE TABLE videos (
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
        CREATE TABLE channels (
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
        CREATE TABLE video_features (
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
        CREATE TABLE musicbrainz_data (
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
        CREATE TABLE validation_results (
            video_id TEXT PRIMARY KEY,
            is_valid BOOLEAN DEFAULT 0,
            validation_score REAL DEFAULT 0.0,
            alt_artist TEXT,
            alt_title TEXT,
            FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
        )
    """
    )

    # Keep essential quality scores but simplified
    conn.execute(
        """
        CREATE TABLE quality_scores (
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

    # Keep RYD data but simplified
    conn.execute(
        """
        CREATE TABLE ryd_data (
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
        CREATE TABLE schema_info (
            version INTEGER PRIMARY KEY,
            description TEXT,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Insert schema version
    conn.execute(
        """
        INSERT INTO schema_info (version, description)
        VALUES (2, 'Optimized schema - streamlined fields and rounded values')
    """
    )


def migrate_videos_data(source_conn, opt_conn):
    """Migrate videos table with transformations."""

    print("🎬 Migrating videos data...")

    videos = source_conn.execute(
        """
        SELECT
            video_id, url, title, description, duration_seconds,
            view_count, like_count, comment_count, upload_date,
            thumbnail_url, channel_name, channel_id,
            original_artist, song_title, featured_artists,
            estimated_release_year, musicbrainz_confidence,
            like_dislike_to_views_ratio, scraped_at
        FROM videos
    """
    ).fetchall()

    for video in videos:
        # Calculate engagement ratio as percentage
        engagement_ratio = None
        if video["like_dislike_to_views_ratio"] is not None:
            engagement_ratio = round_value(video["like_dislike_to_views_ratio"] * 100, 3)

        # Round confidence
        parse_confidence = round_value(video["musicbrainz_confidence"], 2)

        opt_conn.execute(
            """
            INSERT INTO videos (
                video_id, url, title, description, duration_seconds,
                view_count, like_count, comment_count, upload_date,
                thumbnail_url, channel_name, channel_id,
                artist, song_title, featured_artists, release_year,
                parse_confidence, engagement_ratio, scraped_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                video["video_id"],
                video["url"],
                video["title"],
                video["description"],
                video["duration_seconds"],
                video["view_count"],
                video["like_count"],
                video["comment_count"],
                video["upload_date"],
                video["thumbnail_url"],
                video["channel_name"],
                video["channel_id"],
                video["original_artist"],
                video["song_title"],
                video["featured_artists"],
                video["estimated_release_year"],
                parse_confidence,
                engagement_ratio,
                video["scraped_at"],
            ),
        )

    print(f"   ✅ Migrated {len(videos)} videos")


def migrate_channels_data(source_conn, opt_conn):
    """Migrate channels table simplified."""

    print("📺 Migrating channels data...")

    channels = source_conn.execute(
        """
        SELECT channel_id, channel_url, channel_name, video_count,
               description, is_karaoke_focused, updated_at
        FROM channels
    """
    ).fetchall()

    for channel in channels:
        opt_conn.execute(
            """
            INSERT INTO channels (
                channel_id, channel_url, channel_name, video_count,
                description, is_karaoke_focused, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                channel["channel_id"],
                channel["channel_url"],
                channel["channel_name"],
                channel["video_count"],
                channel["description"],
                channel["is_karaoke_focused"],
                channel["updated_at"],
            ),
        )

    print(f"   ✅ Migrated {len(channels)} channels")


def migrate_video_features_data(source_conn, opt_conn):
    """Migrate video features with boolean simplification."""

    print("🎵 Migrating video features...")

    features = source_conn.execute(
        """
        SELECT video_id, has_guide_vocals, has_scrolling_lyrics,
               has_backing_vocals, is_instrumental_only, is_piano_only,
               is_acoustic, has_guide_vocals_confidence, confidence_score
        FROM video_features
    """
    ).fetchall()

    for feature in features:
        # Use confidence thresholds to set boolean values
        guide_vocals = (
            bool(feature["has_guide_vocals"]) or (feature["has_guide_vocals_confidence"] or 0) > 0.5
        )

        opt_conn.execute(
            """
            INSERT INTO video_features (
                video_id, has_guide_vocals, has_scrolling_lyrics,
                has_backing_vocals, is_instrumental, is_piano_only,
                is_acoustic, overall_confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                feature["video_id"],
                guide_vocals,
                feature["has_scrolling_lyrics"],
                feature["has_backing_vocals"],
                feature["is_instrumental_only"],
                feature["is_piano_only"],
                feature["is_acoustic"],
                round_value(feature["confidence_score"], 2),
            ),
        )

    print(f"   ✅ Migrated {len(features)} video features")


def migrate_musicbrainz_data(source_conn, opt_conn):
    """Migrate MusicBrainz data simplified."""

    print("🎼 Migrating MusicBrainz data...")

    mb_data = source_conn.execute(
        """
        SELECT video_id, musicbrainz_recording_id, musicbrainz_artist_id,
               musicbrainz_genre, musicbrainz_confidence, recording_length_ms,
               musicbrainz_tags
        FROM videos
        WHERE musicbrainz_recording_id IS NOT NULL
    """
    ).fetchall()

    for mb in mb_data:
        opt_conn.execute(
            """
            INSERT INTO musicbrainz_data (
                video_id, recording_id, artist_id, genre,
                confidence, recording_length_ms, tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                mb["video_id"],
                mb["musicbrainz_recording_id"],
                mb["musicbrainz_artist_id"],
                mb["musicbrainz_genre"],
                round_value(mb["musicbrainz_confidence"], 2),
                mb["recording_length_ms"],
                mb["musicbrainz_tags"],
            ),
        )

    print(f"   ✅ Migrated {len(mb_data)} MusicBrainz records")


def migrate_validation_data(source_conn, opt_conn):
    """Migrate validation results simplified."""

    print("✅ Migrating validation data...")

    validation = source_conn.execute(
        """
        SELECT video_id, artist_valid, song_valid, validation_score,
               suggested_artist, suggested_title
        FROM validation_results
    """
    ).fetchall()

    for val in validation:
        is_valid = bool(val["artist_valid"]) and bool(val["song_valid"])

        opt_conn.execute(
            """
            INSERT INTO validation_results (
                video_id, is_valid, validation_score,
                alt_artist, alt_title
            ) VALUES (?, ?, ?, ?, ?)
        """,
            (
                val["video_id"],
                is_valid,
                round_value(val["validation_score"], 2),
                val["suggested_artist"],
                val["suggested_title"],
            ),
        )

    print(f"   ✅ Migrated {len(validation)} validation records")


def migrate_quality_scores(source_conn, opt_conn):
    """Migrate quality scores with rounding."""

    print("⭐ Migrating quality scores...")

    scores = source_conn.execute(
        """
        SELECT video_id, overall_score, technical_score,
               engagement_score, metadata_completeness, calculated_at
        FROM quality_scores
    """
    ).fetchall()

    for score in scores:
        opt_conn.execute(
            """
            INSERT INTO quality_scores (
                video_id, overall_score, technical_score,
                engagement_score, metadata_score, calculated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                score["video_id"],
                round_value(score["overall_score"], 2),
                round_value(score["technical_score"], 2),
                round_value(score["engagement_score"], 2),
                round_value(score["metadata_completeness"], 2),
                score["calculated_at"],
            ),
        )

    print(f"   ✅ Migrated {len(scores)} quality scores")


def migrate_ryd_data(source_conn, opt_conn):
    """Migrate RYD data with rounding."""

    print("👍 Migrating RYD data...")

    ryd = source_conn.execute(
        """
        SELECT video_id, estimated_dislikes, ryd_likes,
               ryd_rating, ryd_confidence, fetched_at
        FROM ryd_data
    """
    ).fetchall()

    for r in ryd:
        opt_conn.execute(
            """
            INSERT INTO ryd_data (
                video_id, estimated_dislikes, ryd_likes,
                ryd_rating, ryd_confidence, fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                r["video_id"],
                r["estimated_dislikes"],
                r["ryd_likes"],
                round_value(r["ryd_rating"], 1),  # 1 decimal for ratings
                round_value(r["ryd_confidence"], 2),
                r["fetched_at"],
            ),
        )

    print(f"   ✅ Migrated {len(ryd)} RYD records")


def verify_migration(source_conn, opt_conn):
    """Verify the migration was successful."""

    print("🔍 Verifying migration...")

    # Count records in both databases
    source_videos = source_conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    opt_videos = opt_conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]

    source_channels = source_conn.execute("SELECT COUNT(*) FROM channels").fetchone()[0]
    opt_channels = opt_conn.execute("SELECT COUNT(*) FROM channels").fetchone()[0]

    print(f"   Videos: {source_videos} → {opt_videos}")
    print(f"   Channels: {source_channels} → {opt_channels}")

    if source_videos != opt_videos:
        raise ValueError(f"Video count mismatch: {source_videos} vs {opt_videos}")
    if source_channels != opt_channels:
        raise ValueError(f"Channel count mismatch: {source_channels} vs {opt_channels}")

    # Check data integrity
    sample_video = opt_conn.execute(
        """
        SELECT video_id, parse_confidence, engagement_ratio
        FROM videos WHERE parse_confidence IS NOT NULL LIMIT 1
    """
    ).fetchone()

    if sample_video:
        conf = sample_video[1]
        ratio = sample_video[2]
        if conf and (conf < 0 or conf > 1):
            raise ValueError(f"Invalid confidence value: {conf}")
        if ratio and (ratio < 0 or ratio > 100):
            raise ValueError(f"Invalid engagement ratio: {ratio}")

    print("   ✅ Migration verification passed!")


if __name__ == "__main__":
    optimize_database("karaoke_videos.db")
