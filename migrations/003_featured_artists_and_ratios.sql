-- Migration 003: Add featured artists and engagement ratios
BEGIN TRANSACTION;

-- Add new columns
ALTER TABLE videos ADD COLUMN featured_artists TEXT;
ALTER TABLE videos ADD COLUMN like_dislike_to_views_ratio REAL;

-- Update existing records with computed ratios in batches for safety
-- Use a more robust calculation with error handling
UPDATE videos 
SET like_dislike_to_views_ratio = 
  CASE 
    WHEN view_count > 0 AND like_count IS NOT NULL THEN
      CASE 
        WHEN EXISTS (SELECT 1 FROM ryd_data WHERE ryd_data.video_id = videos.video_id) THEN
          (like_count - COALESCE((SELECT estimated_dislikes FROM ryd_data WHERE ryd_data.video_id = videos.video_id LIMIT 1), 0)) * 1.0 / view_count
        ELSE 
          like_count * 1.0 / view_count
      END
    ELSE NULL 
  END
WHERE view_count > 0 AND like_count IS NOT NULL;

-- Verify the migration worked correctly
-- Check if any ratios are invalid (negative or too large)
SELECT CASE 
  WHEN EXISTS (SELECT 1 FROM videos WHERE like_dislike_to_views_ratio < -1 OR like_dislike_to_views_ratio > 1) THEN
    RAISE(ABORT, 'Migration produced invalid ratios')
  ELSE 1
END;

INSERT OR REPLACE INTO schema_info(version) VALUES (3);

COMMIT;