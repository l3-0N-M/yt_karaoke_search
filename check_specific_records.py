#!/usr/bin/env python3
import sqlite3

conn = sqlite3.connect('/home/leon/Karaoke/Karaoke_Search_Script/yt_karaoke_search/karaoke_videos.db')
cursor = conn.cursor()

ids = [236, 273, 276, 290, 8, 43, 67, 69, 126, 130, 159, 173, 201]

print('=== SPECIFIC VIDEO RECORDS ANALYSIS ===')

for video_id in ids:
    cursor.execute('''
        SELECT id, video_id, title, original_artist, song_title, 
               musicbrainz_recording_id, musicbrainz_artist_id, musicbrainz_confidence,
               channel_name, duration_seconds
        FROM videos 
        WHERE id = ?
    ''', (video_id,))
    
    record = cursor.fetchone()
    if record:
        print(f'\n--- Record ID: {record[0]} ---')
        print(f'Video ID: {record[1]}')
        print(f'Title: {record[2]}')
        print(f'Parsed Artist: "{record[3]}"')
        print(f'Parsed Song: "{record[4]}"')
        print(f'MB Recording ID: {record[5]}')
        print(f'MB Artist ID: {record[6]}')
        print(f'MB Confidence: {record[7]}')
        print(f'Channel: {record[8]}')
        print(f'Duration: {record[9]}')
        
        # Check if this looks like a potential issue
        if record[2] and ' - ' in record[2]:
            title_parts = record[2].split(' - ')
            if len(title_parts) >= 2:
                suggested_artist = title_parts[-1].split(' *')[0] if ' *' in title_parts[-1] else title_parts[-1]
                suggested_artist = suggested_artist.replace('(Karaoke & Lyrics)', '').strip()
                if record[3] and suggested_artist.lower() != record[3].lower():
                    print(f'🚨 MISMATCH: Title suggests "{suggested_artist}" but parsed as "{record[3]}"')
        
        # Check for other quality issues
        if record[7] and record[7] < 0.8:
            print(f'⚠️  LOW CONFIDENCE: MusicBrainz confidence {record[7]}')
        
        if record[4] and (record[4].startswith('Video ') or ':' in record[4] or len(record[4]) > 50):
            print(f'⚠️  UNUSUAL SONG TITLE: "{record[4]}"')
    else:
        print(f'❌ Record ID {video_id} not found')

conn.close()