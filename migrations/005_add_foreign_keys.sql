-- Migration 005: Add foreign key constraints and improve data integrity

-- Enable foreign key enforcement (SQLite specific)
PRAGMA foreign_keys = ON;

-- Create backup of existing data before adding constraints
CREATE TABLE IF NOT EXISTS videos_backup AS SELECT * FROM videos;

-- Clean up orphaned data before adding constraints
DELETE FROM videos WHERE channel_id IS NOT NULL AND channel_id NOT IN (SELECT channel_id FROM channels);
DELETE FROM video_features WHERE video_id NOT IN (SELECT video_id FROM videos);
DELETE FROM quality_scores WHERE video_id NOT IN (SELECT video_id FROM videos);
DELETE FROM ryd_data WHERE video_id NOT IN (SELECT video_id FROM videos);

-- Add foreign key constraints (SQLite doesn't support ALTER TABLE ADD CONSTRAINT)
-- We need to recreate tables with constraints

-- Create new videos table with foreign key
CREATE TABLE videos_new (
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
    musicbrainz_recording_id TEXT,
    musicbrainz_artist_id TEXT,
    musicbrainz_genre TEXT,
    musicbrainz_tags TEXT,
    musicbrainz_confidence REAL,
    record_label TEXT,
    recording_length_ms INTEGER,
    genre TEXT,
    language TEXT,
    like_dislike_to_views_ratio REAL,
    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (channel_id) REFERENCES channels(channel_id) ON DELETE SET NULL
);

-- Copy data to new table
INSERT INTO videos_new (
    video_id,
    url,
    title,
    description,
    duration_seconds,
    view_count,
    like_count,
    comment_count,
    upload_date,
    thumbnail_url,
    channel_name,
    channel_id,
    original_artist,
    featured_artists,
    song_title,
    estimated_release_year,
    musicbrainz_recording_id,
    musicbrainz_artist_id,
    musicbrainz_genre,
    musicbrainz_tags,
    musicbrainz_confidence,
    record_label,
    recording_length_ms,
    genre,
    language,
    like_dislike_to_views_ratio,
    scraped_at,
    updated_at
) SELECT
    video_id,
    url,
    title,
    description,
    duration_seconds,
    view_count,
    like_count,
    comment_count,
    upload_date,
    thumbnail_url,
    channel_name,
    channel_id,
    original_artist,
    featured_artists,
    song_title,
    estimated_release_year,
    musicbrainz_recording_id,
    musicbrainz_artist_id,
    musicbrainz_genre,
    musicbrainz_tags,
    musicbrainz_confidence,
    record_label,
    recording_length_ms,
    genre,
    language,
    like_dislike_to_views_ratio,
    scraped_at,
    updated_at
FROM videos;

-- Drop old table and rename new one
DROP TABLE videos;
ALTER TABLE videos_new RENAME TO videos;

-- Recreate indexes for videos table
CREATE INDEX IF NOT EXISTS idx_videos_video_id ON videos(video_id);
CREATE INDEX IF NOT EXISTS idx_videos_channel_id ON videos(channel_id);
CREATE INDEX IF NOT EXISTS idx_videos_upload_date ON videos(upload_date);
CREATE INDEX IF NOT EXISTS idx_videos_view_count ON videos(view_count);
CREATE INDEX IF NOT EXISTS idx_videos_artist_song ON videos(original_artist, song_title);

-- Recreate the trigger for the new table
DROP TRIGGER IF EXISTS trg_videos_updated;
CREATE TRIGGER trg_videos_updated
AFTER UPDATE OF title, description, view_count, like_count, channel_id, 
               original_artist, song_title, featured_artists, estimated_release_year
ON videos
FOR EACH ROW
WHEN OLD.updated_at = NEW.updated_at
BEGIN
  UPDATE videos SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Update schema version
INSERT OR REPLACE INTO schema_info(version) VALUES (5);
