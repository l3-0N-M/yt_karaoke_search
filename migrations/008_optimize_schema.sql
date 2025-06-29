-- Migration 008: Optimized schema with streamlined fields and rounded values
-- This migration creates the new optimized database structure

BEGIN TRANSACTION;

-- Drop old incompatible tables that are no longer needed
DROP TABLE IF EXISTS search_history;
DROP TABLE IF EXISTS error_log;
DROP TABLE IF EXISTS videos_backup;
DROP TABLE IF EXISTS search_cache;
DROP TABLE IF EXISTS search_analytics;
DROP TABLE IF EXISTS fuzzy_reference_data;

-- Create optimized videos table
CREATE TABLE IF NOT EXISTS videos_optimized (
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
    -- Parsed metadata (renamed fields)
    artist TEXT,  -- was original_artist
    song_title TEXT,
    featured_artists TEXT,
    release_year INTEGER,  -- was estimated_release_year
    -- Quality metrics (rounded to 2 decimals)
    parse_confidence REAL,  -- was musicbrainz_confidence rounded
    engagement_ratio REAL,  -- was like_dislike_to_views_ratio as percentage
    -- Single timestamp
    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
);

-- Migrate data from old videos table with transformations
INSERT INTO videos_optimized (
    video_id, url, title, description, duration_seconds,
    view_count, like_count, comment_count, upload_date,
    thumbnail_url, channel_name, channel_id,
    artist, song_title, featured_artists, release_year,
    parse_confidence, engagement_ratio, scraped_at
)
SELECT 
    video_id, url, title, description, duration_seconds,
    view_count, like_count, comment_count, upload_date,
    thumbnail_url, channel_name, channel_id,
    original_artist,  -- rename field
    song_title,
    featured_artists,
    estimated_release_year,  -- rename field
    ROUND(COALESCE(musicbrainz_confidence, 0), 2),  -- round confidence
    CASE 
        WHEN like_dislike_to_views_ratio IS NOT NULL THEN
            ROUND(like_dislike_to_views_ratio * 100, 3)  -- convert to percentage
        ELSE NULL 
    END,
    scraped_at
FROM videos;

-- Create optimized channels table (simplified)
CREATE TABLE IF NOT EXISTS channels_optimized (
    channel_id TEXT PRIMARY KEY,
    channel_url TEXT,
    channel_name TEXT,
    video_count INTEGER DEFAULT 0,
    description TEXT,
    is_karaoke_focused BOOLEAN DEFAULT 1,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Migrate channels data (keep most recent timestamp)
INSERT INTO channels_optimized (
    channel_id, channel_url, channel_name, video_count,
    description, is_karaoke_focused, updated_at
)
SELECT 
    channel_id, channel_url, channel_name, video_count,
    description, is_karaoke_focused,
    COALESCE(updated_at, last_processed_at, created_at)  -- use most recent timestamp
FROM channels;

-- Create optimized video_features table
CREATE TABLE IF NOT EXISTS video_features_optimized (
    video_id TEXT PRIMARY KEY,
    has_guide_vocals BOOLEAN DEFAULT 0,
    has_scrolling_lyrics BOOLEAN DEFAULT 0,
    has_backing_vocals BOOLEAN DEFAULT 0,
    is_instrumental BOOLEAN DEFAULT 0,
    is_piano_only BOOLEAN DEFAULT 0,
    is_acoustic BOOLEAN DEFAULT 0,
    overall_confidence REAL DEFAULT 0.0,
    FOREIGN KEY (video_id) REFERENCES videos_optimized(video_id) ON DELETE CASCADE
);

-- Migrate video features with smart boolean logic
INSERT INTO video_features_optimized (
    video_id, has_guide_vocals, has_scrolling_lyrics,
    has_backing_vocals, is_instrumental, is_piano_only,
    is_acoustic, overall_confidence
)
SELECT 
    video_id,
    CASE 
        WHEN has_guide_vocals = 1 OR has_guide_vocals_confidence > 0.5 THEN 1
        ELSE 0 
    END,
    has_scrolling_lyrics,
    has_backing_vocals,
    is_instrumental_only,
    is_piano_only,
    is_acoustic,
    ROUND(COALESCE(confidence_score, 0), 2)
FROM video_features;

-- Create optimized MusicBrainz data table
CREATE TABLE IF NOT EXISTS musicbrainz_data_optimized (
    video_id TEXT PRIMARY KEY,
    recording_id TEXT,
    artist_id TEXT,
    genre TEXT,
    confidence REAL,
    recording_length_ms INTEGER,
    tags TEXT,
    FOREIGN KEY (video_id) REFERENCES videos_optimized(video_id) ON DELETE CASCADE
);

-- Migrate MusicBrainz data from videos table
INSERT INTO musicbrainz_data_optimized (
    video_id, recording_id, artist_id, genre,
    confidence, recording_length_ms, tags
)
SELECT 
    video_id,
    musicbrainz_recording_id,
    musicbrainz_artist_id,
    musicbrainz_genre,
    ROUND(COALESCE(musicbrainz_confidence, 0), 2),
    recording_length_ms,
    musicbrainz_tags
FROM videos
WHERE musicbrainz_recording_id IS NOT NULL;

-- Create optimized validation_results table
CREATE TABLE IF NOT EXISTS validation_results_optimized (
    video_id TEXT PRIMARY KEY,
    is_valid BOOLEAN DEFAULT 0,
    validation_score REAL DEFAULT 0.0,
    alt_artist TEXT,
    alt_title TEXT,
    FOREIGN KEY (video_id) REFERENCES videos_optimized(video_id) ON DELETE CASCADE
);

-- Migrate validation results with simplified boolean
INSERT INTO validation_results_optimized (
    video_id, is_valid, validation_score, alt_artist, alt_title
)
SELECT 
    video_id,
    CASE WHEN artist_valid = 1 AND song_valid = 1 THEN 1 ELSE 0 END,
    ROUND(COALESCE(validation_score, 0), 2),
    suggested_artist,
    suggested_title
FROM validation_results;

-- Create optimized quality_scores table
CREATE TABLE IF NOT EXISTS quality_scores_optimized (
    video_id TEXT PRIMARY KEY,
    overall_score REAL DEFAULT 0.0,
    technical_score REAL DEFAULT 0.0,
    engagement_score REAL DEFAULT 0.0,
    metadata_score REAL DEFAULT 0.0,
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES videos_optimized(video_id) ON DELETE CASCADE
);

-- Migrate quality scores with rounding
INSERT INTO quality_scores_optimized (
    video_id, overall_score, technical_score,
    engagement_score, metadata_score, calculated_at
)
SELECT 
    video_id,
    ROUND(COALESCE(overall_score, 0), 2),
    ROUND(COALESCE(technical_score, 0), 2),
    ROUND(COALESCE(engagement_score, 0), 2),
    ROUND(COALESCE(metadata_completeness, 0), 2),
    calculated_at
FROM quality_scores;

-- Create optimized ryd_data table
CREATE TABLE IF NOT EXISTS ryd_data_optimized (
    video_id TEXT PRIMARY KEY,
    estimated_dislikes INTEGER DEFAULT 0,
    ryd_likes INTEGER DEFAULT 0,
    ryd_rating REAL DEFAULT 0.0,
    ryd_confidence REAL DEFAULT 0.0,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES videos_optimized(video_id) ON DELETE CASCADE
);

-- Migrate RYD data with rating rounding
INSERT INTO ryd_data_optimized (
    video_id, estimated_dislikes, ryd_likes,
    ryd_rating, ryd_confidence, fetched_at
)
SELECT 
    video_id, estimated_dislikes, ryd_likes,
    ROUND(COALESCE(ryd_rating, 0), 1),  -- 1 decimal for ratings
    ROUND(COALESCE(ryd_confidence, 0), 2),
    fetched_at
FROM ryd_data;

-- Replace old tables with optimized versions
DROP TABLE IF EXISTS videos;
DROP TABLE IF EXISTS channels;
DROP TABLE IF EXISTS video_features;
DROP TABLE IF EXISTS musicbrainz_data;
DROP TABLE IF EXISTS validation_results;
DROP TABLE IF EXISTS quality_scores;
DROP TABLE IF EXISTS ryd_data;

-- Rename optimized tables to original names
ALTER TABLE videos_optimized RENAME TO videos;
ALTER TABLE channels_optimized RENAME TO channels;
ALTER TABLE video_features_optimized RENAME TO video_features;
ALTER TABLE musicbrainz_data_optimized RENAME TO musicbrainz_data;
ALTER TABLE validation_results_optimized RENAME TO validation_results;
ALTER TABLE quality_scores_optimized RENAME TO quality_scores;
ALTER TABLE ryd_data_optimized RENAME TO ryd_data;

-- Create performance indexes for optimized schema
CREATE INDEX IF NOT EXISTS idx_videos_channel_id ON videos(channel_id);
CREATE INDEX IF NOT EXISTS idx_videos_artist ON videos(artist);
CREATE INDEX IF NOT EXISTS idx_videos_scraped_at ON videos(scraped_at);
CREATE INDEX IF NOT EXISTS idx_musicbrainz_recording ON musicbrainz_data(recording_id);
CREATE INDEX IF NOT EXISTS idx_channels_name ON channels(channel_name);
CREATE INDEX IF NOT EXISTS idx_quality_overall ON quality_scores(overall_score);

-- Update schema version to optimized version
INSERT OR REPLACE INTO schema_info (version, description, applied_at) 
VALUES (2, 'Optimized schema - streamlined fields and rounded values', CURRENT_TIMESTAMP);

COMMIT;