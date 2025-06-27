"""
-- Migration 002: Fix updated_at trigger (recursion-safe)
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