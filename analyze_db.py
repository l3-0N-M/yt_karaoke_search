#!/usr/bin/env python3
"""Database analysis script for karaoke search database."""

import sqlite3
import pandas as pd
from pathlib import Path
from collections import defaultdict
import json

def analyze_database(db_path):
    """Analyze the karaoke database structure and content."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    analysis = {
        "schema": {},
        "statistics": {},
        "data_quality": {},
        "performance": {},
        "anomalies": []
    }
    
    # 1. Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"\n=== Database Analysis for {db_path} ===")
    print(f"\nTables found: {len(tables)}")
    for table in tables:
        print(f"  - {table}")
    
    # 2. Analyze each table
    for table in tables:
        print(f"\n=== Table: {table} ===")
        
        # Get table schema
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        analysis["schema"][table] = {
            "columns": [
                {
                    "name": col[1],
                    "type": col[2],
                    "not_null": bool(col[3]),
                    "default": col[4],
                    "primary_key": bool(col[5])
                }
                for col in columns
            ]
        }
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        row_count = cursor.fetchone()[0]
        print(f"Row count: {row_count:,}")
        
        # Get indexes
        cursor.execute(f"PRAGMA index_list({table})")
        indexes = cursor.fetchall()
        print(f"Indexes: {len(indexes)}")
        for idx in indexes:
            print(f"  - {idx[1]} (unique: {bool(idx[2])})")
        
        # Store statistics
        analysis["statistics"][table] = {
            "row_count": row_count,
            "column_count": len(columns),
            "index_count": len(indexes)
        }
        
        # Analyze data quality for tables with data
        if row_count > 0:
            print(f"\nData quality analysis for {table}:")
            quality_issues = []
            
            for col in columns:
                col_name = col[1]
                col_type = col[2]
                
                # Check for nulls
                cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {col_name} IS NULL")
                null_count = cursor.fetchone()[0]
                null_percentage = (null_count / row_count) * 100
                
                if null_count > 0 and col[3]:  # NOT NULL constraint violated
                    quality_issues.append(f"Column {col_name} has {null_count} nulls despite NOT NULL constraint")
                elif null_percentage > 50:
                    quality_issues.append(f"Column {col_name} has {null_percentage:.1f}% nulls")
                
                # Check for empty strings in text fields
                if 'TEXT' in col_type.upper():
                    cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {col_name} = ''")
                    empty_count = cursor.fetchone()[0]
                    if empty_count > 0:
                        empty_percentage = (empty_count / row_count) * 100
                        print(f"  - {col_name}: {null_count} nulls ({null_percentage:.1f}%), {empty_count} empty strings ({empty_percentage:.1f}%)")
                else:
                    if null_count > 0:
                        print(f"  - {col_name}: {null_count} nulls ({null_percentage:.1f}%)")
            
            analysis["data_quality"][table] = quality_issues
    
    # 3. Check for duplicates in key tables
    print("\n=== Duplicate Analysis ===")
    if 'videos' in tables:
        # Check for duplicate video_ids
        cursor.execute("""
            SELECT video_id, COUNT(*) as count
            FROM videos
            GROUP BY video_id
            HAVING COUNT(*) > 1
        """)
        duplicates = cursor.fetchall()
        if duplicates:
            print(f"Found {len(duplicates)} duplicate video_ids!")
            analysis["anomalies"].append(f"Found {len(duplicates)} duplicate video_ids")
        else:
            print("No duplicate video_ids found")
    
    # 4. Check data relationships
    print("\n=== Data Relationships ===")
    if 'videos' in tables and 'channels' in tables:
        # Check for orphaned videos
        cursor.execute("""
            SELECT COUNT(*)
            FROM videos v
            WHERE v.channel_id IS NOT NULL
            AND NOT EXISTS (SELECT 1 FROM channels c WHERE c.channel_id = v.channel_id)
        """)
        orphaned_videos = cursor.fetchone()[0]
        if orphaned_videos > 0:
            print(f"Found {orphaned_videos} videos with non-existent channel_ids")
            analysis["anomalies"].append(f"Found {orphaned_videos} orphaned videos")
    
    # 5. Performance analysis
    print("\n=== Performance Analysis ===")
    
    # Check database size
    cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
    db_size = cursor.fetchone()[0]
    print(f"Database size: {db_size / 1024 / 1024:.2f} MB")
    
    # Check for missing indexes on foreign keys
    if 'videos' in tables:
        cursor.execute("PRAGMA foreign_key_list(videos)")
        foreign_keys = cursor.fetchall()
        for fk in foreign_keys:
            fk_column = fk[3]
            # Check if there's an index on this column
            cursor.execute(f"""
                SELECT COUNT(*) FROM pragma_index_list('videos') il
                JOIN pragma_index_info(il.name) ii ON ii.name = ?
            """, (fk_column,))
            if cursor.fetchone()[0] == 0:
                print(f"Missing index on foreign key: videos.{fk_column}")
                analysis["performance"]["missing_indexes"] = analysis["performance"].get("missing_indexes", [])
                analysis["performance"]["missing_indexes"].append(f"videos.{fk_column}")
    
    # 6. Specific data analysis
    if 'videos' in tables and row_count > 0:
        print("\n=== Video Data Analysis ===")
        
        # Parse success rate
        cursor.execute("SELECT COUNT(*) FROM videos WHERE artist IS NOT NULL AND song_title IS NOT NULL")
        parsed_count = cursor.fetchone()[0]
        parse_rate = (parsed_count / analysis["statistics"]["videos"]["row_count"]) * 100
        print(f"Parse success rate: {parse_rate:.1f}% ({parsed_count:,} videos)")
        
        # Videos by year
        cursor.execute("""
            SELECT release_year, COUNT(*) as count
            FROM videos
            WHERE release_year IS NOT NULL
            GROUP BY release_year
            ORDER BY count DESC
            LIMIT 10
        """)
        year_dist = cursor.fetchall()
        if year_dist:
            print("\nTop 10 years by video count:")
            for year, count in year_dist:
                print(f"  {year}: {count:,} videos")
        
        # Top channels
        cursor.execute("""
            SELECT channel_name, COUNT(*) as count
            FROM videos
            WHERE channel_name IS NOT NULL
            GROUP BY channel_name
            ORDER BY count DESC
            LIMIT 10
        """)
        top_channels = cursor.fetchall()
        print("\nTop 10 channels by video count:")
        for channel, count in top_channels:
            print(f"  {channel}: {count:,} videos")
        
        # Quality score distribution
        if 'quality_scores' in tables:
            cursor.execute("""
                SELECT
                    CASE
                        WHEN overall_score >= 0.9 THEN '0.9-1.0'
                        WHEN overall_score >= 0.8 THEN '0.8-0.9'
                        WHEN overall_score >= 0.7 THEN '0.7-0.8'
                        WHEN overall_score >= 0.6 THEN '0.6-0.7'
                        WHEN overall_score >= 0.5 THEN '0.5-0.6'
                        WHEN overall_score >= 0.4 THEN '0.4-0.5'
                        WHEN overall_score >= 0.3 THEN '0.3-0.4'
                        WHEN overall_score >= 0.2 THEN '0.2-0.3'
                        WHEN overall_score >= 0.1 THEN '0.1-0.2'
                        ELSE '0.0-0.1'
                    END as score_range,
                    COUNT(*) as count
                FROM quality_scores
                GROUP BY score_range
                ORDER BY score_range DESC
            """)
            quality_dist = cursor.fetchall()
            if quality_dist:
                print("\nQuality score distribution:")
                for range_label, count in quality_dist:
                    print(f"  {range_label}: {count:,} videos")
    
    # 7. Check for anomalies in data
    if 'videos' in tables:
        # Check for unusual durations
        cursor.execute("""
            SELECT COUNT(*) FROM videos
            WHERE duration_seconds IS NOT NULL
            AND (duration_seconds < 30 OR duration_seconds > 3600)
        """)
        unusual_durations = cursor.fetchone()[0]
        if unusual_durations > 0:
            print(f"\nFound {unusual_durations} videos with unusual durations (<30s or >1h)")
            analysis["anomalies"].append(f"{unusual_durations} videos with unusual durations")
        
        # Check for suspiciously high view counts
        cursor.execute("""
            SELECT COUNT(*) FROM videos
            WHERE view_count > 100000000
        """)
        high_views = cursor.fetchone()[0]
        if high_views > 0:
            print(f"Found {high_views} videos with >100M views")
    
    conn.close()
    
    # Save detailed analysis to JSON
    with open('database_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print("\n=== Analysis complete ===")
    print("Detailed analysis saved to database_analysis.json")
    
    return analysis

if __name__ == "__main__":
    # Analyze main database
    main_db = Path("karaoke_videos.db")
    if main_db.exists():
        analyze_database(main_db)
    
    # Also analyze cache database if it exists
    cache_db = Path("cache/search_cache.db")
    if cache_db.exists():
        print("\n" + "="*50 + "\n")
        analyze_database(cache_db)