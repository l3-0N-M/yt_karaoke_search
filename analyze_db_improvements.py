#!/usr/bin/env python3
"""
Analyze karaoke database improvements focusing on:
1. The 2025 release year issue
2. String field lengths (truncation)
3. Overall data quality improvements
"""

import sqlite3
from datetime import datetime
from collections import Counter

def analyze_database(db_path):
    """Analyze the karaoke database for improvements"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print(f"\n{'='*80}")
    print(f"KARAOKE DATABASE ANALYSIS - {db_path}")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # 1. Check videos with release_year = 2025
    print("1. RELEASE YEAR 2025 ISSUE ANALYSIS")
    print("-" * 50)
    
    cursor.execute("""
        SELECT COUNT(*) FROM videos WHERE release_year = 2025
    """)
    count_2025 = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM videos")
    total_videos = cursor.fetchone()[0]
    
    print(f"Total videos in database: {total_videos}")
    print(f"Videos with release_year = 2025: {count_2025} ({count_2025/total_videos*100:.2f}%)")
    
    # Get some examples
    cursor.execute("""
        SELECT video_id, title, channel_name, release_year, upload_date, artist
        FROM videos 
        WHERE release_year = 2025
        ORDER BY upload_date DESC
        LIMIT 10
    """)
    examples_2025 = cursor.fetchall()
    
    if examples_2025:
        print("\nExamples of videos with 2025 release year:")
        print("(These are likely parsing errors from video titles)")
        for i, video in enumerate(examples_2025, 1):
            print(f"\n  {i}. Title: {video[1][:70]}...")
            print(f"     Channel: {video[2]}")
            print(f"     Upload Date: {video[4]}")
            print(f"     Parsed Artist: {video[5] if video[5] else 'N/A'}")
    
    # 2. Check string field lengths for truncation
    print("\n\n2. STRING FIELD TRUNCATION ANALYSIS")
    print("-" * 50)
    
    # Analyze description field
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            MIN(LENGTH(description)) as min_len,
            MAX(LENGTH(description)) as max_len,
            AVG(LENGTH(description)) as avg_len,
            COUNT(CASE WHEN LENGTH(description) >= 4990 THEN 1 END) as possibly_truncated,
            COUNT(CASE WHEN LENGTH(description) = 5000 THEN 1 END) as definitely_truncated
        FROM videos
        WHERE description IS NOT NULL AND description != ''
    """)
    desc_stats = cursor.fetchone()
    
    print(f"\nDescription field analysis:")
    print(f"  Total with descriptions: {desc_stats[0]}")
    print(f"  Length range: {desc_stats[1]} - {desc_stats[2]} characters")
    print(f"  Average length: {desc_stats[3]:.0f} characters")
    print(f"  Possibly truncated (≥4990 chars): {desc_stats[4]} ({desc_stats[4]/desc_stats[0]*100:.2f}%)")
    print(f"  Definitely truncated (=5000 chars): {desc_stats[5]} ({desc_stats[5]/desc_stats[0]*100:.2f}%)")
    
    # Analyze featured_artists field
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            MIN(LENGTH(featured_artists)) as min_len,
            MAX(LENGTH(featured_artists)) as max_len,
            AVG(LENGTH(featured_artists)) as avg_len,
            COUNT(CASE WHEN LENGTH(featured_artists) >= 490 THEN 1 END) as possibly_truncated,
            COUNT(CASE WHEN LENGTH(featured_artists) = 500 THEN 1 END) as definitely_truncated
        FROM videos
        WHERE featured_artists IS NOT NULL AND featured_artists != ''
    """)
    feat_stats = cursor.fetchone()
    
    if feat_stats and feat_stats[0] > 0:
        print(f"\nFeatured artists field analysis:")
        print(f"  Total with featured artists: {feat_stats[0]}")
        print(f"  Length range: {feat_stats[1]} - {feat_stats[2]} characters")
        print(f"  Average length: {feat_stats[3]:.0f} characters")
        print(f"  Possibly truncated (≥490 chars): {feat_stats[4]} ({feat_stats[4]/feat_stats[0]*100:.2f}%)")
        print(f"  Definitely truncated (=500 chars): {feat_stats[5]} ({feat_stats[5]/feat_stats[0]*100:.2f}%)")
    
    # 3. Overall data quality statistics
    print("\n\n3. DATA QUALITY AND PARSING SUCCESS")
    print("-" * 50)
    
    # Release year distribution
    cursor.execute("""
        SELECT release_year, COUNT(*) as count
        FROM videos
        WHERE release_year IS NOT NULL
        GROUP BY release_year
        ORDER BY count DESC
        LIMIT 15
    """)
    year_dist = cursor.fetchall()
    
    print("\nTop 15 release years by count:")
    for year, count in year_dist:
        bar = '█' * int(count / total_videos * 100)
        print(f"  {year}: {count:>5} videos ({count/total_videos*100:>5.1f}%) {bar}")
    
    # Check for valid years
    cursor.execute("""
        SELECT 
            COUNT(CASE WHEN release_year IS NULL THEN 1 END) as null_years,
            COUNT(CASE WHEN release_year < 1900 THEN 1 END) as too_old,
            COUNT(CASE WHEN release_year > 2024 THEN 1 END) as future,
            COUNT(CASE WHEN release_year BETWEEN 1900 AND 2024 THEN 1 END) as valid
        FROM videos
    """)
    year_validity = cursor.fetchone()
    
    print(f"\nRelease year validity:")
    print(f"  NULL/Missing: {year_validity[0]} ({year_validity[0]/total_videos*100:.1f}%)")
    print(f"  Too old (<1900): {year_validity[1]} ({year_validity[1]/total_videos*100:.1f}%)")
    print(f"  Future (>2024): {year_validity[2]} ({year_validity[2]/total_videos*100:.1f}%)")
    print(f"  Valid (1900-2024): {year_validity[3]} ({year_validity[3]/total_videos*100:.1f}%)")
    
    # Metadata extraction success
    cursor.execute("""
        SELECT 
            COUNT(CASE WHEN artist IS NOT NULL AND artist != '' THEN 1 END) as has_artist,
            COUNT(CASE WHEN song_title IS NOT NULL AND song_title != '' THEN 1 END) as has_song,
            COUNT(CASE WHEN genre IS NOT NULL AND genre != '' THEN 1 END) as has_genre,
            COUNT(CASE WHEN featured_artists IS NOT NULL AND featured_artists != '' THEN 1 END) as has_featured
        FROM videos
    """)
    metadata_stats = cursor.fetchone()
    
    print(f"\nMetadata extraction success:")
    print(f"  Has artist: {metadata_stats[0]} ({metadata_stats[0]/total_videos*100:.1f}%)")
    print(f"  Has song title: {metadata_stats[1]} ({metadata_stats[1]/total_videos*100:.1f}%)")
    print(f"  Has genre: {metadata_stats[2]} ({metadata_stats[2]/total_videos*100:.1f}%)")
    print(f"  Has featured artists: {metadata_stats[3]} ({metadata_stats[3]/total_videos*100:.1f}%)")
    
    # Parse confidence distribution
    cursor.execute("""
        SELECT 
            AVG(parse_confidence) as avg_conf,
            MIN(parse_confidence) as min_conf,
            MAX(parse_confidence) as max_conf,
            COUNT(CASE WHEN parse_confidence >= 0.8 THEN 1 END) as high_conf,
            COUNT(CASE WHEN parse_confidence >= 0.5 AND parse_confidence < 0.8 THEN 1 END) as med_conf,
            COUNT(CASE WHEN parse_confidence < 0.5 THEN 1 END) as low_conf
        FROM videos
        WHERE parse_confidence IS NOT NULL
    """)
    conf_stats = cursor.fetchone()
    
    if conf_stats and conf_stats[0]:
        print(f"\nParse confidence analysis:")
        print(f"  Average confidence: {conf_stats[0]:.3f}")
        print(f"  Range: {conf_stats[1]:.3f} - {conf_stats[2]:.3f}")
        print(f"  High confidence (≥0.8): {conf_stats[3]} ({conf_stats[3]/total_videos*100:.1f}%)")
        print(f"  Medium confidence (0.5-0.8): {conf_stats[4]} ({conf_stats[4]/total_videos*100:.1f}%)")
        print(f"  Low confidence (<0.5): {conf_stats[5]} ({conf_stats[5]/total_videos*100:.1f}%)")
    
    # 4. Channel analysis
    print("\n\n4. CHANNEL ANALYSIS")
    print("-" * 50)
    
    cursor.execute("SELECT COUNT(DISTINCT channel_id) FROM videos")
    total_channels = cursor.fetchone()[0]
    print(f"Total unique channels: {total_channels}")
    
    cursor.execute("""
        SELECT channel_name, COUNT(*) as video_count
        FROM videos
        GROUP BY channel_id
        ORDER BY video_count DESC
        LIMIT 10
    """)
    top_channels = cursor.fetchall()
    
    print("\nTop 10 channels by video count:")
    for channel, count in top_channels:
        print(f"  {channel}: {count} videos")
    
    # 5. Recent uploads vs claimed release years
    print("\n\n5. UPLOAD DATE VS RELEASE YEAR ANALYSIS")
    print("-" * 50)
    
    cursor.execute("""
        SELECT 
            COUNT(*) as count,
            AVG(CAST(SUBSTR(upload_date, 1, 4) AS INTEGER) - release_year) as avg_diff
        FROM videos
        WHERE release_year IS NOT NULL 
        AND release_year BETWEEN 1900 AND 2024
        AND upload_date IS NOT NULL
        AND LENGTH(upload_date) >= 4
    """)
    diff_stats = cursor.fetchone()
    
    if diff_stats and diff_stats[0] > 0:
        print(f"Videos with valid release year and upload date: {diff_stats[0]}")
        print(f"Average years between release and upload: {diff_stats[1]:.1f} years")
    
    # Check for suspiciously recent "releases"
    cursor.execute("""
        SELECT release_year, COUNT(*) as count
        FROM videos
        WHERE release_year >= 2020
        GROUP BY release_year
        ORDER BY release_year DESC
    """)
    recent_years = cursor.fetchall()
    
    print("\nVideos claiming recent release years:")
    for year, count in recent_years:
        print(f"  {year}: {count} videos")
    
    conn.close()
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        db_path = "karaoke_videos.db"
    
    analyze_database(db_path)