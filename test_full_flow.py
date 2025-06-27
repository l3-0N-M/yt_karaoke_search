#!/usr/bin/env python3
"""Test the full flow from extraction to database save to identify truncation."""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from collector.config import CollectorConfig
from collector.processor import VideoProcessor


def test_full_flow():
    """Test the complete flow to see where truncation occurs."""
    config = CollectorConfig()
    processor = VideoProcessor(config)

    # Test title that we know works
    test_title = "Karaoke - What Child Is This - Christmas Traditional"

    print(f"Testing full flow for: '{test_title}'")
    print("=" * 80)

    # Step 1: Test the extraction
    print("Step 1: Artist/Song extraction")
    artist_song_result = processor._extract_artist_song_info(test_title, "", "")
    print(f"  Result: {artist_song_result}")

    # Step 2: Test the karaoke features extraction (this calls _extract_artist_song_info)
    print("\nStep 2: Karaoke features extraction")

    # Simulate video_data
    video_data = {
        "title": test_title,
        "description": "",
        "tags": [],
        "uploader": "Test Channel"
    }

    features = processor._extract_karaoke_features(video_data)
    print("  Features extracted:")
    for key, value in features.items():
        if 'artist' in key or 'song' in key or 'title' in key:
            print(f"    {key}: '{value}' (length: {len(str(value)) if value is not None else 'None'})")

    # Step 3: Test what would be saved to database
    print("\nStep 3: Database save simulation")

    # Create a mock ProcessingResult
    class MockResult:
        def __init__(self):
            self.video_data = {
                **video_data,
                "video_id": "test123",
                "url": "https://youtube.com/watch?v=test123",
                "duration_seconds": 180,
                "view_count": 1000,
                "like_count": 100,
                "comment_count": 10,
                "upload_date": "20231201",
                "thumbnail": "https://test.jpg",
                "uploader": "Test Channel",
                "uploader_id": "testchannel123",
                "features": features,
                "quality_scores": {"overall_score": 0.8}
            }
            self.confidence_score = 0.9

    mock_result = MockResult()

    # Check what would be inserted
    features_for_db = mock_result.video_data.get("features", {})
    print(f"  artist for DB: '{features_for_db.get('original_artist')}' (length: {len(str(features_for_db.get('original_artist', '')))}) ")
    print(f"  song_title for DB: '{features_for_db.get('song_title')}' (length: {len(str(features_for_db.get('song_title', '')))})")

    # Let's also check if there are any other places where song_title might get modified
    print("\nStep 4: Check for any string modifications")

    # Look at the actual values that would be passed to the SQL
    sql_values = (
        mock_result.video_data.get("video_id"),
        mock_result.video_data.get("url"),
        mock_result.video_data.get("title"),
        mock_result.video_data.get("description", "")[:2000],  # Note: description is truncated
        mock_result.video_data.get("duration_seconds"),
        mock_result.video_data.get("view_count"),
        mock_result.video_data.get("like_count"),
        mock_result.video_data.get("comment_count"),
        mock_result.video_data.get("upload_date"),
        mock_result.video_data.get("thumbnail"),
        mock_result.video_data.get("uploader"),
        mock_result.video_data.get("uploader_id"),
        features_for_db.get("original_artist"),
        features_for_db.get("featured_artists"),
        features_for_db.get("song_title"),  # This is the key one
        mock_result.video_data.get("estimated_release_year"),
        None,  # like_dislike_ratio
    )

    print(f"  SQL value for original_artist: '{sql_values[12]}' (length: {len(str(sql_values[12])) if sql_values[12] else 'None'})")
    print(f"  SQL value for song_title: '{sql_values[14]}' (length: {len(str(sql_values[14])) if sql_values[14] else 'None'})")

    # Check if there's any modification happening in the database manager
    print("\nStep 5: Manual database insertion test")

    # Let's manually check if there's an issue with the database schema or insertion
    import sqlite3

    try:
        conn = sqlite3.connect(':memory:')  # Use in-memory database for testing
        cursor = conn.cursor()

        # Create table with same schema as real database
        cursor.execute("""
            CREATE TABLE videos (
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
                like_dislike_to_views_ratio REAL,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Insert our test data
        cursor.execute("""
            INSERT INTO videos (
                video_id, url, title, description, duration_seconds,
                view_count, like_count, comment_count, upload_date,
                thumbnail_url, channel_name, channel_id, original_artist,
                featured_artists, song_title, estimated_release_year,
                like_dislike_to_views_ratio
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, sql_values)

        # Read it back
        cursor.execute("SELECT original_artist, song_title FROM videos WHERE video_id = ?", (sql_values[0],))
        row = cursor.fetchone()

        if row:
            print(f"  Retrieved from DB - artist: '{row[0]}' (length: {len(row[0]) if row[0] else 'None'})")
            print(f"  Retrieved from DB - song_title: '{row[1]}' (length: {len(row[1]) if row[1] else 'None'})")

        conn.close()

    except Exception as e:
        print(f"  Database test error: {e}")

if __name__ == "__main__":
    test_full_flow()
