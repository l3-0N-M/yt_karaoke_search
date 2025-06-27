"""
-- Migration 003: Add featured artists and engagement ratios
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