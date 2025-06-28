#!/usr/bin/env python3
"""Analyze the database to identify swapped artist/title records."""

import sqlite3
import re
from collections import defaultdict

def analyze_swapping_patterns():
    """Analyze the database for artist/title swapping patterns."""
    
    conn = sqlite3.connect('karaoke_videos.db')
    cursor = conn.cursor()
    
    # Get all records with artist and title data
    cursor.execute("""
        SELECT video_id, title, original_artist, song_title, 
               musicbrainz_recording_id, musicbrainz_confidence
        FROM videos 
        WHERE original_artist IS NOT NULL AND song_title IS NOT NULL
    """)
    
    results = cursor.fetchall()
    
    analysis = {
        'total_records': len(results),
        'karaoke_pattern_records': 0,
        'de_pattern_records': 0,
        'other_records': 0,
        'with_musicbrainz': 0,
        'high_confidence_mb': 0,
        'suspected_swapped': [],
        'pattern_breakdown': defaultdict(int)
    }
    
    print(f"Analyzing {len(results)} records...")
    print("=" * 80)
    
    for video_id, title, artist, song, mb_id, mb_conf in results:
        # Categorize by title pattern
        if title.lower().startswith('karaoke'):
            analysis['karaoke_pattern_records'] += 1
            pattern_type = 'karaoke'
        elif title.upper().startswith('DE '):
            analysis['de_pattern_records'] += 1
            pattern_type = 'german'
        else:
            analysis['other_records'] += 1
            pattern_type = 'other'
            
        analysis['pattern_breakdown'][pattern_type] += 1
        
        # Count MusicBrainz data
        if mb_id:
            analysis['with_musicbrainz'] += 1
            if mb_conf and mb_conf > 0.7:
                analysis['high_confidence_mb'] += 1
        
        # Check for potential swapping issues
        # For karaoke titles, check if artist looks like a song title
        if pattern_type == 'karaoke':
            # Common indicators that artist/title might be swapped
            artist_indicators = [
                len(artist.split()) > 4,  # Very long artist name
                any(word in artist.lower() for word in ['remix', 'version', 'mix', 'edit']),  # Song-like words in artist
                artist.lower().startswith(('the ', 'a ', 'an ')),  # Article at start (more common in songs)
            ]
            
            song_indicators = [
                len(song.split()) <= 2,  # Very short song title
                song.lower() in ['remix', 'version', 'mix', 'radio edit'],  # Artist-like words in song
            ]
            
            # If multiple indicators suggest swapping
            if sum(artist_indicators) >= 2 or sum(song_indicators) >= 1:
                analysis['suspected_swapped'].append({
                    'video_id': video_id,
                    'title': title,
                    'artist': artist,
                    'song': song,
                    'mb_confidence': mb_conf,
                    'indicators': {
                        'artist_issues': artist_indicators,
                        'song_issues': song_indicators
                    }
                })
    
    # Print analysis results
    print("ANALYSIS RESULTS:")
    print(f"Total records: {analysis['total_records']}")
    print(f"Karaoke pattern: {analysis['karaoke_pattern_records']}")
    print(f"German (DE) pattern: {analysis['de_pattern_records']}")
    print(f"Other patterns: {analysis['other_records']}")
    print(f"With MusicBrainz data: {analysis['with_musicbrainz']}")
    print(f"High confidence MB (>0.7): {analysis['high_confidence_mb']}")
    print(f"Suspected swapped records: {len(analysis['suspected_swapped'])}")
    
    print("\nPATTERN BREAKDOWN:")
    for pattern, count in analysis['pattern_breakdown'].items():
        percentage = (count / analysis['total_records']) * 100
        print(f"  {pattern}: {count} ({percentage:.1f}%)")
    
    print("\nSUSPECTED SWAPPED RECORDS (first 20):")
    print("-" * 80)
    for i, record in enumerate(analysis['suspected_swapped'][:20]):
        print(f"{i+1:2}. {record['title']}")
        print(f"    Artist: {record['artist']}")
        print(f"    Song:   {record['song']}")
        print(f"    MB Conf: {record['mb_confidence']}")
        print()
    
    # Sample correct records for comparison
    print("\nSAMPLE RECORDS BY PATTERN:")
    print("-" * 80)
    
    # Show examples of each pattern type
    cursor.execute("""
        SELECT title, original_artist, song_title, musicbrainz_confidence
        FROM videos 
        WHERE title LIKE 'Karaoke%' AND original_artist IS NOT NULL 
        LIMIT 5
    """)
    karaoke_samples = cursor.fetchall()
    
    print("Karaoke pattern samples:")
    for title, artist, song, mb_conf in karaoke_samples:
        print(f"  '{title}' -> Artist: '{artist}', Song: '{song}' (MB: {mb_conf})")
    
    cursor.execute("""
        SELECT title, original_artist, song_title, musicbrainz_confidence
        FROM videos 
        WHERE title LIKE 'DE %' AND original_artist IS NOT NULL 
        LIMIT 5
    """)
    de_samples = cursor.fetchall()
    
    print("\nGerman (DE) pattern samples:")
    for title, artist, song, mb_conf in de_samples:
        print(f"  '{title}' -> Artist: '{artist}', Song: '{song}' (MB: {mb_conf})")
    
    conn.close()
    return analysis

if __name__ == "__main__":
    analyze_swapping_patterns()