-- Migration 004: Add channels support
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