#!/usr/bin/env python3
"""Fix swapped artist/title data in the database."""

import sqlite3
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def backup_database():
    """Create a backup of the database before making changes."""
    import shutil
    backup_name = f"karaoke_videos_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
    shutil.copy2('karaoke_videos.db', backup_name)
    logger.info(f"Created backup: {backup_name}")
    return backup_name

def fix_swapped_karaoke_records():
    """Fix swapped artist/title in karaoke records."""
    
    # Create backup first
    backup_file = backup_database()
    
    conn = sqlite3.connect('karaoke_videos.db')
    cursor = conn.cursor()
    
    # Get all karaoke records that likely need swapping
    query = """
        SELECT video_id, title, original_artist, song_title, 
               musicbrainz_recording_id, musicbrainz_confidence
        FROM videos 
        WHERE title LIKE 'Karaoke%' 
        AND original_artist IS NOT NULL 
        AND song_title IS NOT NULL
    """
    
    cursor.execute(query)
    records = cursor.fetchall()
    
    logger.info(f"Found {len(records)} karaoke records to examine")
    
    # Statistics
    stats = {
        'examined': 0,
        'swapped': 0,
        'skipped_high_confidence_mb': 0,
        'skipped_already_correct': 0,
        'errors': 0
    }
    
    # Process each record
    for video_id, title, artist, song, mb_id, mb_conf in records:
        stats['examined'] += 1
        
        try:
            # Skip records with very high confidence MusicBrainz data
            # as they might be correct already
            if mb_conf and mb_conf > 0.85:
                stats['skipped_high_confidence_mb'] += 1
                logger.debug(f"Skipping high-confidence MB record: {title}")
                continue
            
            # For karaoke titles, the pattern should be:
            # "Karaoke [Song Title] - [Artist Name]"
            # But currently we have:
            # Artist = [Song Title], Song = [Artist Name] (SWAPPED)
            
            # Check if this looks like it needs swapping
            # Heuristics:
            # 1. Artist field is longer than song field (unusual)
            # 2. Song field looks like an artist name (short, proper noun)
            # 3. Artist field has song-like characteristics
            
            needs_swap = False
            reasons = []
            
            # Length heuristic: artist names are usually shorter than song titles
            if len(artist) > len(song) * 1.5:
                needs_swap = True
                reasons.append("artist_too_long")
            
            # Artist field contains song-like words
            song_words = ['remix', 'version', 'mix', 'edit', 'unplugged', 'live', 'acoustic']
            if any(word in artist.lower() for word in song_words):
                needs_swap = True
                reasons.append("artist_has_song_words")
            
            # Song field is very short (likely an artist name)
            if len(song.split()) <= 2 and len(song) < 20:
                needs_swap = True
                reasons.append("song_too_short")
            
            # Song field looks like a proper name (capitalized words)
            if all(word[0].isupper() for word in song.split() if word):
                needs_swap = True
                reasons.append("song_looks_like_name")
            
            # Skip if no strong indicators for swapping
            if not needs_swap:
                stats['skipped_already_correct'] += 1
                continue
            
            # Perform the swap
            logger.info(f"Swapping artist/title for: {title}")
            logger.info(f"  Before: Artist='{artist}', Song='{song}'")
            logger.info(f"  After:  Artist='{song}', Song='{artist}'")
            logger.info(f"  Reasons: {', '.join(reasons)}")
            
            # Update the database
            update_query = """
                UPDATE videos 
                SET original_artist = ?, song_title = ?
                WHERE video_id = ?
            """
            
            cursor.execute(update_query, (song, artist, video_id))
            stats['swapped'] += 1
            
        except Exception as e:
            logger.error(f"Error processing record {video_id}: {e}")
            stats['errors'] += 1
    
    # Commit changes
    conn.commit()
    conn.close()
    
    # Print summary
    logger.info("MIGRATION SUMMARY:")
    logger.info(f"Records examined: {stats['examined']}")
    logger.info(f"Records swapped: {stats['swapped']}")
    logger.info(f"Skipped (high MB confidence): {stats['skipped_high_confidence_mb']}")
    logger.info(f"Skipped (already correct): {stats['skipped_already_correct']}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info(f"Backup created: {backup_file}")
    
    return stats

def verify_fixes():
    """Verify that the fixes worked correctly."""
    
    conn = sqlite3.connect('karaoke_videos.db')
    cursor = conn.cursor()
    
    # Sample some fixed records
    cursor.execute("""
        SELECT title, original_artist, song_title, musicbrainz_confidence
        FROM videos 
        WHERE title LIKE 'Karaoke%' 
        AND original_artist IS NOT NULL 
        LIMIT 10
    """)
    
    results = cursor.fetchall()
    
    logger.info("VERIFICATION - Sample records after fix:")
    for title, artist, song, mb_conf in results:
        logger.info(f"  '{title}' -> Artist: '{artist}', Song: '{song}' (MB: {mb_conf})")
    
    conn.close()

if __name__ == "__main__":
    logger.info("Starting database artist/title swap fix...")
    
    # Run the fix
    stats = fix_swapped_karaoke_records()
    
    # Verify the results
    verify_fixes()
    
    logger.info("Migration completed!")