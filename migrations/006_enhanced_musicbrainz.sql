-- Enhanced MusicBrainz metadata support
-- Migration 006: Add comprehensive MusicBrainz fields to videos table

-- Add new MusicBrainz columns
ALTER TABLE videos ADD COLUMN musicbrainz_recording_id TEXT;
ALTER TABLE videos ADD COLUMN musicbrainz_artist_id TEXT;
ALTER TABLE videos ADD COLUMN musicbrainz_genre TEXT;
ALTER TABLE videos ADD COLUMN musicbrainz_tags TEXT;
ALTER TABLE videos ADD COLUMN musicbrainz_confidence REAL;
ALTER TABLE videos ADD COLUMN record_label TEXT;
ALTER TABLE videos ADD COLUMN recording_length_ms INTEGER;

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_videos_musicbrainz_recording_id ON videos(musicbrainz_recording_id);
CREATE INDEX IF NOT EXISTS idx_videos_musicbrainz_artist_id ON videos(musicbrainz_artist_id);
CREATE INDEX IF NOT EXISTS idx_videos_musicbrainz_genre ON videos(musicbrainz_genre);
CREATE INDEX IF NOT EXISTS idx_videos_record_label ON videos(record_label);