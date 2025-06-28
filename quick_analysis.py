#!/usr/bin/env python3
import sqlite3

conn = sqlite3.connect('/home/leon/Karaoke/Karaoke_Search_Script/yt_karaoke_search/karaoke_videos.db')
cursor = conn.cursor()

print('=== ARTIST/TITLE MISMATCHES ===')
cursor.execute('''
    SELECT video_id, title, original_artist, song_title, 
           musicbrainz_confidence, channel_name
    FROM videos 
    WHERE musicbrainz_recording_id IS NOT NULL
    AND title LIKE '% - %'
    AND musicbrainz_confidence = 1.0
    LIMIT 10
''')

examples = cursor.fetchall()
for record in examples:
    title_parts = record[1].split(' - ')
    if len(title_parts) >= 2:
        title_artist = title_parts[-1].split(' *')[0] if ' *' in title_parts[-1] else title_parts[-1]
        mb_artist = record[2]
        if title_artist.strip() != mb_artist.strip():
            print(f'Video: {record[0]}')
            print(f'  Title suggests: "{title_artist}"')
            print(f'  MusicBrainz found: "{mb_artist}"')
            print(f'  Song: "{record[3]}"')
            print()

print('\n=== UNUSUAL SONG TITLES ===')
cursor.execute('''
    SELECT video_id, original_artist, song_title, musicbrainz_confidence
    FROM videos 
    WHERE musicbrainz_recording_id IS NOT NULL
    AND (song_title LIKE 'Video%' OR song_title LIKE '%:%' OR LENGTH(song_title) > 50)
    LIMIT 5
''')

for record in cursor.fetchall():
    print(f'Artist: {record[1]}')
    print(f'Unusual Song: "{record[2]}"')
    print(f'Confidence: {record[3]}')
    print()

print('\n=== LOW CONFIDENCE MATCHES ===')
cursor.execute('''
    SELECT video_id, original_artist, song_title, musicbrainz_confidence
    FROM videos 
    WHERE musicbrainz_recording_id IS NOT NULL
    AND musicbrainz_confidence < 0.8
    ORDER BY musicbrainz_confidence ASC
    LIMIT 5
''')

for record in cursor.fetchall():
    print(f'Video: {record[0]}')
    print(f'Artist: "{record[1]}"')
    print(f'Song: "{record[2]}"')
    print(f'Low Confidence: {record[3]}')
    print()

conn.close()