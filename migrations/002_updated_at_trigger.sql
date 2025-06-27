-- Migration 002: Fix updated_at trigger (recursion-safe)
DROP TRIGGER IF EXISTS videos_updated_at;
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

INSERT OR REPLACE INTO schema_info(version) VALUES (2);