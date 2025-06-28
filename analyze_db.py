#!/usr/bin/env python3
"""Database analysis script to investigate artist/title swapping issue."""

import sqlite3
import pandas as pd
from pathlib import Path

def analyze_database():
    """Analyze the karaoke database for artist/title swapping patterns."""
    
    db_path = Path("/home/leon/Karaoke/Karaoke_Search_Script/yt_karaoke_search/karaoke_videos.db")
    
    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    
    try:
        # Check database schema
        print("=== DATABASE SCHEMA ===")
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(videos)")
        columns = cursor.fetchall()
        
        relevant_columns = [col for col in columns if any(keyword in col[1].lower() 
                           for keyword in ['artist', 'title', 'musicbrainz'])]
        
        print("Relevant columns for artist/title analysis:")
        for col in relevant_columns:
            print(f"  {col[1]} ({col[2]}) - {'NOT NULL' if col[3] else 'NULL'}")
        
        # Get total record count
        cursor.execute("SELECT COUNT(*) FROM videos")
        total_records = cursor.fetchone()[0]
        print(f"\nTotal videos: {total_records}")
        
        # Check records with MusicBrainz data
        cursor.execute("""
            SELECT COUNT(*) FROM videos 
            WHERE musicbrainz_recording_id IS NOT NULL 
            AND musicbrainz_artist_id IS NOT NULL
        """)
        mb_records = cursor.fetchone()[0]
        print(f"Records with MusicBrainz data: {mb_records}")
        
        # Sample some records to analyze the data structure
        print("\n=== SAMPLE RECORDS ===")
        cursor.execute("""
            SELECT video_id, title, original_artist, song_title, 
                   musicbrainz_recording_id, musicbrainz_artist_id,
                   musicbrainz_confidence, channel_name
            FROM videos 
            LIMIT 10
        """)
        
        sample_records = cursor.fetchall()
        
        for i, record in enumerate(sample_records):
            print(f"\nRecord {i+1}:")
            print(f"  Video ID: {record[0]}")
            print(f"  Original Title: {record[1][:100]}...")
            print(f"  Parsed Artist: {record[2]}")
            print(f"  Parsed Song Title: {record[3]}")
            print(f"  MB Recording ID: {record[4]}")
            print(f"  MB Artist ID: {record[5]}")
            print(f"  MB Confidence: {record[6]}")
            print(f"  Channel: {record[7]}")
        
        # Look for potential swapping patterns
        print("\n=== POTENTIAL SWAPPING ANALYSIS ===")
        
        # Check for cases where artist looks like a song title and vice versa
        cursor.execute("""
            SELECT video_id, title, original_artist, song_title, channel_name
            FROM videos 
            WHERE original_artist IS NOT NULL 
            AND song_title IS NOT NULL
            AND (
                -- Artist looks like a song title (has quotes, common song patterns)
                original_artist LIKE '%"%' 
                OR original_artist LIKE '%''%'
                OR original_artist LIKE '%I %'
                OR original_artist LIKE '%You %'
                OR original_artist LIKE '%My %'
                OR original_artist LIKE '%Love%'
                -- Song title looks like artist (very short, common artist patterns)
                OR (LENGTH(song_title) < 15 AND song_title NOT LIKE '%I %' AND song_title NOT LIKE '%You %')
            )
            LIMIT 20
        """)
        
        suspicious_records = cursor.fetchall()
        
        print(f"Found {len(suspicious_records)} potentially suspicious records:")
        for record in suspicious_records:
            print(f"\nVideo: {record[0]}")
            print(f"  Title: {record[1][:80]}...")
            print(f"  Artist: '{record[2]}'")
            print(f"  Song: '{record[3]}'")
            print(f"  Channel: {record[4]}")
        
        # Check MusicBrainz validation results
        print("\n=== MUSICBRAINZ VALIDATION ANALYSIS ===")
        
        # Check if validation_results table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='validation_results'
        """)
        
        if cursor.fetchone():
            cursor.execute("""
                SELECT COUNT(*) FROM validation_results
                WHERE artist_valid = 0 OR song_valid = 0
            """)
            invalid_count = cursor.fetchone()[0]
            print(f"Records with validation issues: {invalid_count}")
            
            # Sample validation failures
            cursor.execute("""
                SELECT v.video_id, v.title, v.original_artist, v.song_title,
                       vr.suggested_artist, vr.suggested_title, vr.suggestion_reason
                FROM videos v
                JOIN validation_results vr ON v.video_id = vr.video_id
                WHERE vr.artist_valid = 0 OR vr.song_valid = 0
                LIMIT 10
            """)
            
            validation_failures = cursor.fetchall()
            
            print("\nSample validation failures:")
            for record in validation_failures:
                print(f"\nVideo: {record[0]}")
                print(f"  Original: {record[2]} - {record[3]}")
                print(f"  Suggested: {record[4]} - {record[5]}")
                print(f"  Reason: {record[6]}")
        else:
            print("No validation_results table found")
        
        # Check for obvious swapping patterns in channel names
        print("\n=== CHANNEL PATTERN ANALYSIS ===")
        cursor.execute("""
            SELECT channel_name, COUNT(*) as count,
                   AVG(CASE WHEN original_artist IS NOT NULL AND song_title IS NOT NULL THEN 1 ELSE 0 END) as parse_rate
            FROM videos 
            WHERE channel_name IS NOT NULL
            GROUP BY channel_name
            HAVING count > 5
            ORDER BY count DESC
            LIMIT 10
        """)
        
        channel_stats = cursor.fetchall()
        print("Top channels by video count:")
        for channel, count, parse_rate in channel_stats:
            print(f"  {channel}: {count} videos, {parse_rate:.2%} parse rate")
        
        # Look for specific problematic patterns in titles
        print("\n=== TITLE PATTERN ANALYSIS ===")
        cursor.execute("""
            SELECT title, original_artist, song_title
            FROM videos 
            WHERE title LIKE '%"%"%"%'  -- Multiple quoted sections
            OR title LIKE '%style of%'  -- Style of patterns
            OR title LIKE '%karaoke%-%-%'  -- Multiple dashes
            LIMIT 10
        """)
        
        pattern_examples = cursor.fetchall()
        print("Complex title patterns:")
        for title, artist, song in pattern_examples:
            print(f"  Title: {title[:100]}...")
            print(f"    Parsed: {artist} - {song}")
            print()
    
    finally:
        conn.close()

if __name__ == "__main__":
    analyze_database()