#!/usr/bin/env python3
import sqlite3

conn = sqlite3.connect('/home/leon/Karaoke/Karaoke_Search_Script/yt_karaoke_search/karaoke_videos.db')
cursor = conn.cursor()

print('=== CHANNEL-SPECIFIC ANALYSIS ===')

channels = ['KaraFun Deutschland - Karaoke', "Let's Sing Karaoke", 'Lugn']

for channel in channels:
    print(f'\n--- {channel} ---')
    
    # Get channel statistics
    cursor.execute('''
        SELECT COUNT(*) as total,
               COUNT(CASE WHEN musicbrainz_recording_id IS NOT NULL THEN 1 END) as with_mb,
               AVG(CASE WHEN musicbrainz_confidence IS NOT NULL THEN musicbrainz_confidence END) as avg_conf
        FROM videos 
        WHERE channel_name = ?
    ''', (channel,))
    
    stats = cursor.fetchone()
    print(f'Total videos: {stats[0]}')
    print(f'With MusicBrainz: {stats[1]} ({stats[1]/stats[0]*100:.1f}%)')
    print(f'Avg MB confidence: {stats[2]:.3f}' if stats[2] else 'N/A')
    
    # Sample problematic records for this channel
    cursor.execute('''
        SELECT video_id, title, original_artist, song_title, musicbrainz_confidence
        FROM videos 
        WHERE channel_name = ?
        AND musicbrainz_recording_id IS NOT NULL
        AND musicbrainz_confidence < 0.9
        ORDER BY musicbrainz_confidence ASC
        LIMIT 3
    ''', (channel,))
    
    problems = cursor.fetchall()
    if problems:
        print('Low confidence examples:')
        for p in problems:
            print(f'  {p[2]} - {p[3]} (conf: {p[4]})')
    
    # Check for artist mismatches in this channel
    cursor.execute('''
        SELECT video_id, title, original_artist, song_title
        FROM videos 
        WHERE channel_name = ?
        AND title LIKE '% - %'
        AND musicbrainz_confidence = 1.0
        LIMIT 5
    ''', (channel,))
    
    title_examples = cursor.fetchall()
    mismatches = []
    for record in title_examples:
        title_parts = record[1].split(' - ')
        if len(title_parts) >= 2:
            title_artist = title_parts[-1].split(' *')[0] if ' *' in title_parts[-1] else title_parts[-1]
            mb_artist = record[2]
            if title_artist.strip() != mb_artist.strip():
                mismatches.append((title_artist.strip(), mb_artist.strip(), record[3]))
    
    if mismatches:
        print('Artist mismatches:')
        for title_art, mb_art, song in mismatches:
            print(f'  Title: "{title_art}" vs MB: "{mb_art}" - {song}')

conn.close()