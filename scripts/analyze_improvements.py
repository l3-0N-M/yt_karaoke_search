#!/usr/bin/env python3
"""
Analyze karaoke database improvements:
1. Check videos with release_year = 2025
2. Verify Discogs data integration
3. Compare string field lengths
4. Show overall statistics
"""

import sqlite3


def analyze_database(db_path):
    """Analyze the karaoke database for improvements"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print(f"\n{'='*80}")
    print(f"KARAOKE DATABASE ANALYSIS - {db_path}")
    print(f"{'='*80}\n")

    # 1. Check videos with release_year = 2025
    print("1. VIDEOS WITH RELEASE_YEAR = 2025")
    print("-" * 50)

    cursor.execute(
        """
        SELECT COUNT(*) FROM videos WHERE release_year = 2025
    """
    )
    count_2025 = cursor.fetchone()[0]
    print(f"Total videos with release_year = 2025: {count_2025}")

    # Get some examples
    cursor.execute(
        """
        SELECT video_id, title, channel_name, release_year, upload_date
        FROM videos
        WHERE release_year = 2025
        LIMIT 10
    """
    )
    examples_2025 = cursor.fetchall()

    if examples_2025:
        print("\nExamples of videos with 2025 release year:")
        for video in examples_2025:
            print(f"  - {video[1][:60]}...")
            print(f"    Channel: {video[2]}, Upload: {video[4]}")

    # 2. Check Discogs data integration
    print("\n\n2. DISCOGS DATA INTEGRATION")
    print("-" * 50)

    # Count non-null Discogs fields
    discogs_fields = [
        "discogs_artist_id",
        "discogs_artist_name",
        "discogs_release_id",
        "discogs_release_title",
        "discogs_release_year",
        "discogs_label",
        "discogs_genre",
        "discogs_style",
    ]

    for field in discogs_fields:
        cursor.execute(
            f"""
            SELECT COUNT(*) FROM videos
            WHERE {field} IS NOT NULL AND {field} != ''
        """
        )
        count = cursor.fetchone()[0]
        print(f"{field}: {count} non-null entries")

    # Show some examples with Discogs data
    cursor.execute(
        """
        SELECT video_id, title, discogs_artist_name, discogs_release_title,
               discogs_release_year, discogs_genre
        FROM videos
        WHERE discogs_artist_name IS NOT NULL
        AND discogs_release_title IS NOT NULL
        LIMIT 5
    """
    )
    discogs_examples = cursor.fetchall()

    if discogs_examples:
        print("\nExamples of videos with Discogs data:")
        for video in discogs_examples:
            print(f"\n  Video: {video[1][:50]}...")
            print(f"  Discogs Artist: {video[2]}")
            print(f"  Discogs Release: {video[3]}")
            print(f"  Discogs Year: {video[4]}")
            print(f"  Discogs Genre: {video[5]}")

    # 3. Check string field lengths
    print("\n\n3. STRING FIELD LENGTH ANALYSIS")
    print("-" * 50)

    # Analyze description field
    cursor.execute(
        """
        SELECT
            MIN(LENGTH(description)) as min_len,
            MAX(LENGTH(description)) as max_len,
            AVG(LENGTH(description)) as avg_len,
            COUNT(CASE WHEN LENGTH(description) >= 4990 THEN 1 END) as truncated
        FROM videos
        WHERE description IS NOT NULL
    """
    )
    desc_stats = cursor.fetchone()
    print("\nDescription field:")
    print(f"  Min length: {desc_stats[0]}")
    print(f"  Max length: {desc_stats[1]}")
    print(f"  Avg length: {desc_stats[2]:.1f}")
    print(f"  Likely truncated (>= 4990 chars): {desc_stats[3]}")

    # Analyze featured_artists field
    cursor.execute(
        """
        SELECT
            MIN(LENGTH(featured_artists)) as min_len,
            MAX(LENGTH(featured_artists)) as max_len,
            AVG(LENGTH(featured_artists)) as avg_len,
            COUNT(CASE WHEN LENGTH(featured_artists) >= 490 THEN 1 END) as truncated
        FROM videos
        WHERE featured_artists IS NOT NULL AND featured_artists != ''
    """
    )
    feat_stats = cursor.fetchone()
    print("\nFeatured artists field:")
    print(f"  Min length: {feat_stats[0] if feat_stats[0] else 0}")
    print(f"  Max length: {feat_stats[1] if feat_stats[1] else 0}")
    print(f"  Avg length: {feat_stats[2]:.1f if feat_stats[2] else 0}")
    print(f"  Likely truncated (>= 490 chars): {feat_stats[3] if feat_stats[3] else 0}")

    # 4. Overall statistics
    print("\n\n4. OVERALL DATABASE STATISTICS")
    print("-" * 50)

    cursor.execute("SELECT COUNT(*) FROM videos")
    total_videos = cursor.fetchone()[0]
    print(f"Total videos: {total_videos}")

    cursor.execute("SELECT COUNT(DISTINCT channel_id) FROM videos")
    total_channels = cursor.fetchone()[0]
    print(f"Total unique channels: {total_channels}")

    # Release year distribution
    cursor.execute(
        """
        SELECT release_year, COUNT(*) as count
        FROM videos
        WHERE release_year IS NOT NULL
        GROUP BY release_year
        ORDER BY count DESC
        LIMIT 10
    """
    )
    year_dist = cursor.fetchall()
    print("\nTop 10 release years by count:")
    for year, count in year_dist:
        print(f"  {year}: {count} videos ({count/total_videos*100:.1f}%)")

    # Check for improved year detection
    cursor.execute(
        """
        SELECT COUNT(*) FROM videos
        WHERE release_year IS NOT NULL
        AND release_year >= 1900
        AND release_year <= 2024
    """
    )
    valid_years = cursor.fetchone()[0]
    print(
        f"\nVideos with valid release years (1900-2024): {valid_years} ({valid_years/total_videos*100:.1f}%)"
    )

    # Check multi-pass success
    cursor.execute(
        """
        SELECT
            COUNT(CASE WHEN web_search_performed = 1 THEN 1 END) as web_searched,
            COUNT(CASE WHEN musicbrainz_checked = 1 THEN 1 END) as mb_checked,
            COUNT(CASE WHEN discogs_checked = 1 THEN 1 END) as discogs_checked
        FROM videos
    """
    )
    pass_stats = cursor.fetchone()
    print("\nMulti-pass processing statistics:")
    print(f"  Web searched: {pass_stats[0]} ({pass_stats[0]/total_videos*100:.1f}%)")
    print(f"  MusicBrainz checked: {pass_stats[1]} ({pass_stats[1]/total_videos*100:.1f}%)")
    print(f"  Discogs checked: {pass_stats[2]} ({pass_stats[2]/total_videos*100:.1f}%)")

    # Check quality improvements
    cursor.execute(
        """
        SELECT
            COUNT(CASE WHEN original_artist IS NOT NULL AND original_artist != '' THEN 1 END) as has_artist,
            COUNT(CASE WHEN song_title IS NOT NULL AND song_title != '' THEN 1 END) as has_song,
            COUNT(CASE WHEN genre IS NOT NULL AND genre != '' THEN 1 END) as has_genre,
            COUNT(CASE WHEN language IS NOT NULL AND language != '' THEN 1 END) as has_language
        FROM videos
    """
    )
    quality_stats = cursor.fetchone()
    print("\nData quality statistics:")
    print(f"  Has original artist: {quality_stats[0]} ({quality_stats[0]/total_videos*100:.1f}%)")
    print(f"  Has song title: {quality_stats[1]} ({quality_stats[1]/total_videos*100:.1f}%)")
    print(f"  Has genre: {quality_stats[2]} ({quality_stats[2]/total_videos*100:.1f}%)")
    print(f"  Has language: {quality_stats[3]} ({quality_stats[3]/total_videos*100:.1f}%)")

    conn.close()
    print(f"\n{'='*80}\n")


def compare_databases(old_db, new_db):
    """Compare two databases to show improvements"""
    print(f"\n{'='*80}")
    print("DATABASE COMPARISON")
    print(f"{'='*80}\n")

    try:
        old_conn = sqlite3.connect(old_db)
        new_conn = sqlite3.connect(new_db)

        # Compare total counts
        old_cursor = old_conn.cursor()
        new_cursor = new_conn.cursor()

        old_cursor.execute("SELECT COUNT(*) FROM videos")
        old_count = old_cursor.fetchone()[0]

        new_cursor.execute("SELECT COUNT(*) FROM videos")
        new_count = new_cursor.fetchone()[0]

        print(f"Old database: {old_count} videos")
        print(f"New database: {new_count} videos")
        print(
            f"Difference: {new_count - old_count} ({(new_count - old_count)/old_count*100:.1f}% change)"
        )

        # Compare 2025 issue
        old_cursor.execute("SELECT COUNT(*) FROM videos WHERE release_year = 2025")
        old_2025 = old_cursor.fetchone()[0]

        new_cursor.execute("SELECT COUNT(*) FROM videos WHERE release_year = 2025")
        new_2025 = new_cursor.fetchone()[0]

        print("\nVideos with 2025 release year:")
        print(f"  Old: {old_2025}")
        print(f"  New: {new_2025}")
        print(f"  Improvement: {old_2025 - new_2025} fewer incorrect years")

        old_conn.close()
        new_conn.close()

    except Exception as e:
        print(f"Error comparing databases: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        # Default to test database
        db_path = "test_karaoke.db"

    analyze_database(db_path)

    # If you want to compare with an older database, uncomment:
    # if len(sys.argv) > 2:
    #     old_db = sys.argv[2]
    #     compare_databases(old_db, db_path)
