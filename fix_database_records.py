#!/usr/bin/env python3
"""Database correction script to fix existing bad parsing data."""

import sqlite3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collector.advanced_parser import AdvancedTitleParser
from pathlib import Path

def identify_problematic_records(db_path):
    """Identify records that likely have parsing issues."""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Find records that match the problematic patterns we identified
    problematic_records = []
    
    # Query for records that likely have wrong data based on our analysis
    suspicious_queries = [
        # Records with very unusual artist names that don't match typical patterns
        """
        SELECT id, video_id, title, original_artist, song_title, channel_name
        FROM videos 
        WHERE original_artist IN (
            'Victims Family', 'Morbid Savouring', 'Hamish and Andy', 'Ahzumjot', 
            'Fatoni', 'odaxelagnia', 'Danju', 'DJ Envy'
        )
        """,
        
        # Let's Sing Karaoke records with comma format that might be parsed wrong
        """
        SELECT id, video_id, title, original_artist, song_title, channel_name
        FROM videos 
        WHERE channel_name = "Let's Sing Karaoke"
        AND title LIKE '%,%-%'
        AND title LIKE '%(Karaoke & Lyrics)'
        """,
        
        # Lugn records with bullet format that might be parsed wrong
        """
        SELECT id, video_id, title, original_artist, song_title, channel_name
        FROM videos 
        WHERE channel_name = 'Lugn'
        AND title LIKE '%•%•%'
        """,
        
        # KaraFun Deutschland records that might have swapped artist/song
        """
        SELECT id, video_id, title, original_artist, song_title, channel_name
        FROM videos 
        WHERE channel_name = 'KaraFun Deutschland - Karaoke'
        AND title LIKE 'Karaoke %-%*'
        """,
    ]
    
    for query in suspicious_queries:
        cursor.execute(query)
        records = cursor.fetchall()
        for record in records:
            if record not in problematic_records:
                problematic_records.append(record)
    
    conn.close()
    return problematic_records

def reparse_record(parser, record):
    """Re-parse a record using the fixed patterns."""
    
    record_id, video_id, title, old_artist, old_song, channel_name = record
    
    # Parse with our fixed patterns
    result = parser.parse_title(
        title=title,
        channel_name=channel_name
    )
    
    new_artist = result.original_artist
    new_song = result.song_title
    
    return {
        'id': record_id,
        'video_id': video_id,
        'title': title,
        'channel_name': channel_name,
        'old_artist': old_artist,
        'old_song': old_song,
        'new_artist': new_artist,
        'new_song': new_song,
        'confidence': result.confidence,
        'method': result.method,
        'changed': (old_artist != new_artist or old_song != new_song)
    }

def fix_database_records(db_path, dry_run=True):
    """Fix problematic records in the database."""
    
    print("=== DATABASE CORRECTION SCRIPT ===")
    print(f"Database: {db_path}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE UPDATE'}")
    print()
    
    # Initialize parser with our fixes
    parser = AdvancedTitleParser()
    
    # Find problematic records
    print("🔍 Identifying problematic records...")
    problematic_records = identify_problematic_records(db_path)
    print(f"Found {len(problematic_records)} potentially problematic records")
    print()
    
    if not problematic_records:
        print("No problematic records found!")
        return
    
    # Re-parse each record
    print("🔧 Re-parsing records with fixed patterns...")
    corrections = []
    
    for i, record in enumerate(problematic_records, 1):
        print(f"Processing {i}/{len(problematic_records)}: {record[1]}")
        
        correction = reparse_record(parser, record)
        corrections.append(correction)
        
        if correction['changed']:
            print(f"  📝 CORRECTION NEEDED:")
            print(f"    Title: {correction['title'][:80]}...")
            print(f"    Channel: {correction['channel_name']}")
            print(f"    OLD: Artist='{correction['old_artist']}', Song='{correction['old_song']}'")
            print(f"    NEW: Artist='{correction['new_artist']}', Song='{correction['new_song']}'")
            print(f"    Method: {correction['method']}, Confidence: {correction['confidence']:.3f}")
        else:
            print(f"  ✅ No change needed")
        print()
    
    # Summary
    changes_needed = [c for c in corrections if c['changed']]
    print("=== SUMMARY ===")
    print(f"Records analyzed: {len(corrections)}")
    print(f"Records needing correction: {len(changes_needed)}")
    print(f"Records already correct: {len(corrections) - len(changes_needed)}")
    print()
    
    if not changes_needed:
        print("🎉 No corrections needed! All records are already correctly parsed.")
        return
    
    # Apply changes if not dry run
    if not dry_run:
        print("💾 Applying corrections to database...")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        updates_applied = 0
        for correction in changes_needed:
            try:
                cursor.execute("""
                    UPDATE videos 
                    SET original_artist = ?, song_title = ?
                    WHERE id = ?
                """, (correction['new_artist'], correction['new_song'], correction['id']))
                updates_applied += 1
            except Exception as e:
                print(f"Error updating record {correction['id']}: {e}")
        
        conn.commit()
        conn.close()
        
        print(f"✅ Applied {updates_applied} corrections to database")
    else:
        print("ℹ️  This was a dry run. To apply changes, run with --apply flag")
    
    # Show detailed breakdown
    print("\n=== DETAILED BREAKDOWN ===")
    by_channel = {}
    for correction in changes_needed:
        channel = correction['channel_name']
        if channel not in by_channel:
            by_channel[channel] = []
        by_channel[channel].append(correction)
    
    for channel, channel_corrections in by_channel.items():
        print(f"\n{channel}: {len(channel_corrections)} corrections")
        for corr in channel_corrections[:3]:  # Show first 3 examples
            print(f"  • {corr['video_id']}: '{corr['old_artist']}' → '{corr['new_artist']}'")
        if len(channel_corrections) > 3:
            print(f"  ... and {len(channel_corrections) - 3} more")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix problematic parsing records in karaoke database')
    parser.add_argument('--apply', action='store_true', help='Apply fixes to database (default is dry run)')
    parser.add_argument('--db', default='karaoke_videos.db', help='Database file path')
    
    args = parser.parse_args()
    
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database file not found: {db_path}")
        return 1
    
    fix_database_records(str(db_path), dry_run=not args.apply)
    return 0

if __name__ == "__main__":
    sys.exit(main())