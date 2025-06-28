#!/usr/bin/env python3
"""Compare old and new databases to verify parsing fixes are working."""

import sqlite3
import sys
from collections import defaultdict

def analyze_database(db_path, label):
    """Analyze a database and return key statistics."""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Basic statistics
    cursor.execute("SELECT COUNT(*) FROM videos")
    total_videos = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM videos WHERE musicbrainz_recording_id IS NOT NULL")
    mb_count = cursor.fetchone()[0]
    
    print(f"\n=== {label} DATABASE ===")
    print(f"Total videos: {total_videos}")
    print(f"With MusicBrainz: {mb_count} ({mb_count/total_videos*100:.1f}%)")
    
    # Channel breakdown
    cursor.execute("SELECT channel_name, COUNT(*) FROM videos GROUP BY channel_name ORDER BY COUNT(*) DESC")
    channels = cursor.fetchall()
    print("\nChannels:")
    for ch, cnt in channels:
        print(f"  {ch}: {cnt} videos")
    
    # Check for problematic artists that we identified
    problematic_artists = [
        'Victims Family', 'Morbid Savouring', 'Hamish and Andy', 'Ahzumjot', 
        'Fatoni', 'odaxelagnia', 'Danju', 'DJ Envy'
    ]
    
    cursor.execute(f"""
        SELECT original_artist, COUNT(*) as count
        FROM videos 
        WHERE original_artist IN ({','.join(['?'] * len(problematic_artists))})
        GROUP BY original_artist
        ORDER BY count DESC
    """, problematic_artists)
    
    bad_artists = cursor.fetchall()
    print(f"\nProblematic artists found: {len(bad_artists)}")
    for artist, count in bad_artists:
        print(f"  {artist}: {count} records")
    
    # Sample records for each channel to check parsing quality
    channel_samples = {}
    for channel, _ in channels:
        cursor.execute("""
            SELECT video_id, title, original_artist, song_title, musicbrainz_confidence
            FROM videos 
            WHERE channel_name = ?
            ORDER BY id
            LIMIT 5
        """, (channel,))
        channel_samples[channel] = cursor.fetchall()
    
    conn.close()
    
    return {
        'total_videos': total_videos,
        'mb_count': mb_count,
        'channels': dict(channels),
        'bad_artists': dict(bad_artists),
        'channel_samples': channel_samples
    }

def compare_specific_records(old_db, new_db):
    """Compare specific problematic records we identified."""
    
    # Known problematic video IDs from our previous analysis
    problematic_ids = [
        '7jd-MJBbpjM',  # George Benson - This Masquerade
        'JascTL5g2yI',  # Jennifer Lopez - Ain't If Funny  
        'uSvyfF3UFc4',  # Whitney Houston - So Emotional
        'lyHsN2OL3QE',  # AYLIVA - Nein!
        'oxjKuYoDTPk',  # AYLIVA - Nein!
        'giAW3FwcilA',  # Kendrick Lamar - Not Like Us
        'B5qW2Sz4Klk',  # Helene Fischer - Feliz Navidad
        'O_bJfLVgyIM',  # Helene Fischer - Die Rose
        'L2hbjG_rs_M',  # 01099 feat. CRO - Glücklich
        'OojYQ_V5Tf4',  # P Diddy & Usher & Loon - I Need A Girl
    ]
    
    print("\n=== COMPARING SPECIFIC PROBLEMATIC RECORDS ===")
    
    old_conn = sqlite3.connect(old_db)
    new_conn = sqlite3.connect(new_db)
    
    improvements = 0
    total_compared = 0
    
    for video_id in problematic_ids:
        # Get old record
        old_cursor = old_conn.execute("""
            SELECT title, original_artist, song_title, channel_name, musicbrainz_confidence
            FROM videos WHERE video_id = ?
        """, (video_id,))
        old_record = old_cursor.fetchone()
        
        # Get new record  
        new_cursor = new_conn.execute("""
            SELECT title, original_artist, song_title, channel_name, musicbrainz_confidence
            FROM videos WHERE video_id = ?
        """, (video_id,))
        new_record = new_cursor.fetchone()
        
        if old_record and new_record:
            total_compared += 1
            print(f"\n📹 {video_id}")
            print(f"   Title: {old_record[0][:80]}...")
            print(f"   Channel: {old_record[3]}")
            print(f"   OLD: Artist='{old_record[1]}', Song='{old_record[2]}'")
            print(f"   NEW: Artist='{new_record[1]}', Song='{new_record[2]}'")
            
            # Check if it improved
            if old_record[1] != new_record[1] or old_record[2] != new_record[2]:
                improvements += 1
                
                # Determine if this looks like an improvement
                old_artist = old_record[1] or ""
                new_artist = new_record[1] or ""
                
                # Check if we fixed known bad artists
                bad_artists = ['Victims Family', 'Morbid Savouring', 'Hamish and Andy', 'Ahzumjot', 'Fatoni', 'odaxelagnia', 'Danju', 'DJ Envy']
                
                if old_artist in bad_artists:
                    print(f"   ✅ FIXED: Removed problematic artist '{old_artist}'")
                elif new_artist and len(new_artist) > len(old_artist):
                    print(f"   ✅ IMPROVED: Better artist extraction")
                else:
                    print(f"   🔄 CHANGED: Different parsing result")
            else:
                print(f"   ➡️  No change in parsing")
        elif new_record:
            print(f"\n📹 {video_id} (new record only)")
            print(f"   NEW: Artist='{new_record[1]}', Song='{new_record[2]}'")
        elif old_record:
            print(f"\n📹 {video_id} (missing in new database)")
    
    old_conn.close()
    new_conn.close()
    
    print(f"\n=== COMPARISON SUMMARY ===")
    print(f"Records compared: {total_compared}")
    print(f"Records with changes: {improvements}")
    if total_compared > 0:
        print(f"Change rate: {improvements/total_compared*100:.1f}%")

def main():
    """Main comparison function."""
    
    old_db = "old_database.db"
    new_db = "karaoke_videos.db"
    
    print("=== DATABASE COMPARISON: VERIFYING PARSING FIXES ===")
    print(f"Old database: {old_db}")
    print(f"New database: {new_db}")
    
    # Analyze both databases
    try:
        old_stats = analyze_database(old_db, "OLD")
        new_stats = analyze_database(new_db, "NEW")
    except Exception as e:
        print(f"Error analyzing databases: {e}")
        return 1
    
    # Overall comparison
    print(f"\n=== OVERALL COMPARISON ===")
    
    # Check if problematic artists are reduced
    old_bad_count = sum(old_stats['bad_artists'].values())
    new_bad_count = sum(new_stats['bad_artists'].values())
    
    print(f"Problematic artist records:")
    print(f"  OLD: {old_bad_count}")
    print(f"  NEW: {new_bad_count}")
    if old_bad_count > 0:
        reduction = (old_bad_count - new_bad_count) / old_bad_count * 100
        print(f"  REDUCTION: {reduction:.1f}%")
        
        if reduction > 80:
            print("  🎉 EXCELLENT: Major reduction in problematic records!")
        elif reduction > 50:
            print("  ✅ GOOD: Significant improvement")
        elif reduction > 20:
            print("  👍 BETTER: Some improvement")
        else:
            print("  ⚠️  LIMITED: Minor improvement")
    
    # Channel-specific improvements
    print(f"\nChannel-specific bad artist reduction:")
    for channel in ['Let\'s Sing Karaoke', 'Lugn', 'KaraFun Deutschland - Karaoke']:
        if channel in old_stats['channels'] and channel in new_stats['channels']:
            print(f"  {channel}:")
            print(f"    OLD videos: {old_stats['channels'][channel]}")
            print(f"    NEW videos: {new_stats['channels'][channel]}")
    
    # Compare specific problematic records
    compare_specific_records(old_db, new_db)
    
    print("\n=== CONCLUSION ===")
    if new_bad_count < old_bad_count * 0.2:  # 80%+ reduction
        print("🎉 FIXES WORKING EXCELLENTLY! Parsing quality dramatically improved.")
    elif new_bad_count < old_bad_count * 0.5:  # 50%+ reduction  
        print("✅ FIXES WORKING WELL! Significant parsing improvements.")
    elif new_bad_count < old_bad_count:
        print("👍 FIXES WORKING! Some parsing improvements detected.")
    else:
        print("⚠️  Fixes may need adjustment - limited improvement detected.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())