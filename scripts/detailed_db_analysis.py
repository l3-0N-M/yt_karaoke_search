#!/usr/bin/env python3
"""Detailed database analysis focusing on specific issues."""

import sqlite3
from datetime import datetime


def detailed_analysis():
    conn = sqlite3.connect("karaoke_videos.db")
    cursor = conn.cursor()

    print("=== DETAILED DATABASE ANALYSIS ===\n")

    # 1. Analyze year anomalies
    print("1. YEAR ANOMALIES")
    print("-" * 50)
    cursor.execute(
        """
        SELECT release_year, COUNT(*) as count
        FROM videos
        WHERE release_year IS NOT NULL
        GROUP BY release_year
        ORDER BY release_year DESC
    """
    )
    year_data = cursor.fetchall()

    current_year = datetime.now().year
    future_years = [(y, c) for y, c in year_data if y > current_year]
    if future_years:
        print(f"⚠️  Found {sum(c for _, c in future_years)} videos with future years:")
        for year, count in future_years[:5]:
            print(f"   - Year {year}: {count} videos")

    # Check sample videos from year 2025
    cursor.execute(
        """
        SELECT video_id, title, artist, song_title, upload_date
        FROM videos
        WHERE release_year = 2025
        LIMIT 5
    """
    )
    print("\nSample videos with year 2025:")
    for row in cursor.fetchall():
        print(f"   - {row[1][:50]}... (artist: {row[2]}, upload: {row[4]})")

    # 2. Analyze empty/null patterns
    print("\n2. DATA COMPLETENESS ANALYSIS")
    print("-" * 50)

    columns_to_check = [
        "description",
        "artist",
        "song_title",
        "featured_artists",
        "genre",
        "thumbnail_url",
        "upload_date",
    ]

    for col in columns_to_check:
        cursor.execute(
            f"""
            SELECT
                COUNT(CASE WHEN {col} IS NULL THEN 1 END) as null_count,
                COUNT(CASE WHEN {col} = '' THEN 1 END) as empty_count,
                COUNT(CASE WHEN {col} IS NOT NULL AND {col} != '' THEN 1 END) as filled_count
            FROM videos
        """
        )
        null_c, empty_c, filled_c = cursor.fetchone()
        total = null_c + empty_c + filled_c
        print(
            f"{col:20} - Null: {null_c:3} ({null_c/total*100:5.1f}%), "
            f"Empty: {empty_c:3} ({empty_c/total*100:5.1f}%), "
            f"Filled: {filled_c:3} ({filled_c/total*100:5.1f}%)"
        )

    # 3. Check parsing confidence distribution
    print("\n3. PARSING CONFIDENCE DISTRIBUTION")
    print("-" * 50)
    cursor.execute(
        """
        SELECT
            CASE
                WHEN parse_confidence IS NULL THEN 'NULL'
                WHEN parse_confidence = 0 THEN '0.0'
                WHEN parse_confidence < 0.5 THEN '< 0.5'
                WHEN parse_confidence < 0.8 THEN '0.5-0.8'
                ELSE '>= 0.8'
            END as conf_range,
            COUNT(*) as count
        FROM videos
        GROUP BY conf_range
        ORDER BY conf_range
    """
    )
    for conf_range, count in cursor.fetchall():
        print(f"   {conf_range:10} : {count:3} videos ({count/200*100:5.1f}%)")

    # 4. Check for data inconsistencies
    print("\n4. DATA INCONSISTENCIES")
    print("-" * 50)

    # Videos with quality scores but no quality_score in main table
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM quality_scores qs
        LEFT JOIN videos v ON qs.video_id = v.video_id
        WHERE v.quality_score IS NULL OR v.quality_score = 0
    """
    )
    inconsistent_quality = cursor.fetchone()[0]
    print(f"Videos with quality_scores records but null/0 in main table: {inconsistent_quality}")

    # Check engagement ratio validity
    cursor.execute(
        """
        SELECT COUNT(*), MIN(engagement_ratio), MAX(engagement_ratio), AVG(engagement_ratio)
        FROM videos
        WHERE engagement_ratio IS NOT NULL
    """
    )
    er_count, er_min, er_max, er_avg = cursor.fetchone()
    print(
        f"Engagement ratio - Count: {er_count}, Min: {er_min:.2f}, Max: {er_max:.2f}, Avg: {er_avg:.2f}"
    )

    # 5. Check related tables coverage
    print("\n5. RELATED TABLES COVERAGE")
    print("-" * 50)

    tables_to_check = [
        ("musicbrainz_data", "MusicBrainz"),
        ("discogs_data", "Discogs"),
        ("quality_scores", "Quality Scores"),
        ("ryd_data", "Return YouTube Dislike"),
        ("video_features", "Video Features"),
        ("validation_results", "Validation Results"),
    ]

    for table, name in tables_to_check:
        cursor.execute(
            f"""
            SELECT COUNT(DISTINCT vd.video_id)
            FROM videos vd
            LEFT JOIN {table} t ON vd.video_id = t.video_id
            WHERE t.video_id IS NOT NULL
        """
        )
        covered = cursor.fetchone()[0]
        coverage = (covered / 200) * 100
        print(f"{name:25} : {covered:3} videos ({coverage:5.1f}% coverage)")

    # 6. Check for long strings that might cause issues
    print("\n6. STRING LENGTH ANALYSIS")
    print("-" * 50)

    string_columns = ["title", "description", "artist", "song_title", "featured_artists"]
    for col in string_columns:
        cursor.execute(
            f"""
            SELECT MAX(LENGTH({col})) as max_len,
                   AVG(LENGTH({col})) as avg_len,
                   COUNT(CASE WHEN LENGTH({col}) > 200 THEN 1 END) as long_count
            FROM videos
            WHERE {col} IS NOT NULL AND {col} != ''
        """
        )
        max_len, avg_len, long_count = cursor.fetchone()
        if max_len:
            print(f"{col:20} - Max: {max_len:5}, Avg: {avg_len:5.1f}, >200 chars: {long_count}")

            # Show longest example
            if max_len > 200:
                cursor.execute(
                    f"""
                    SELECT video_id, {col}
                    FROM videos
                    WHERE LENGTH({col}) = ?
                    LIMIT 1
                """,
                    (max_len,),
                )
                video_id, long_text = cursor.fetchone()
                print(f"   Longest ({max_len} chars) in {video_id}: {long_text[:100]}...")

    # 7. Performance optimization opportunities
    print("\n7. PERFORMANCE OPTIMIZATION OPPORTUNITIES")
    print("-" * 50)

    # Check for columns that might benefit from indexes
    cursor.execute(
        """
        SELECT artist, COUNT(*) as count
        FROM videos
        WHERE artist IS NOT NULL
        GROUP BY artist
        ORDER BY count DESC
        LIMIT 5
    """
    )
    print("Top artists (already indexed):")
    for artist, count in cursor.fetchall():
        print(f"   {artist}: {count} videos")

    # Check for missing foreign key indexes
    cursor.execute("PRAGMA foreign_key_list(videos)")
    fks = cursor.fetchall()
    print(f"\nForeign keys in videos table: {len(fks)}")

    # Check table sizes
    print("\n8. TABLE SIZE ANALYSIS")
    print("-" * 50)
    tables = [
        "videos",
        "channels",
        "musicbrainz_data",
        "discogs_data",
        "quality_scores",
        "ryd_data",
        "video_features",
        "validation_results",
    ]

    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]

        # Estimate table size (rough approximation)
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        est_row_size = sum(50 if "TEXT" in col[2] else 8 for col in columns)
        est_size_kb = (count * est_row_size) / 1024

        print(f"{table:20} : {count:6} rows, ~{est_size_kb:8.1f} KB")

    # 9. Check for potential duplicates based on content
    print("\n9. POTENTIAL DUPLICATE ANALYSIS")
    print("-" * 50)

    cursor.execute(
        """
        SELECT artist, song_title, COUNT(*) as count
        FROM videos
        WHERE artist IS NOT NULL AND song_title IS NOT NULL
        GROUP BY artist, song_title
        HAVING COUNT(*) > 1
        ORDER BY count DESC
        LIMIT 10
    """
    )
    duplicates = cursor.fetchall()
    if duplicates:
        print("Potential duplicate songs (same artist + title):")
        for artist, title, count in duplicates:
            print(f"   {artist} - {title}: {count} versions")
    else:
        print("No duplicate artist/title combinations found")

    # 10. Data freshness
    print("\n10. DATA FRESHNESS")
    print("-" * 50)

    cursor.execute(
        """
        SELECT
            DATE(scraped_at) as scrape_date,
            COUNT(*) as count
        FROM videos
        GROUP BY DATE(scraped_at)
        ORDER BY scrape_date DESC
        LIMIT 10
    """
    )
    print("Recent scraping activity:")
    for date, count in cursor.fetchall():
        print(f"   {date}: {count} videos")

    conn.close()


if __name__ == "__main__":
    detailed_analysis()
