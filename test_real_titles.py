#!/usr/bin/env python3
"""Test with real titles from database to see where truncation happens."""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from collector.config import CollectorConfig
from collector.processor import VideoProcessor

def test_real_extraction():
    """Test extraction with real titles from the database."""
    config = CollectorConfig()
    processor = VideoProcessor(config)
    
    # Real titles from the database that show the issue
    test_titles = [
        "Karaoke - What Child Is This - Christmas Traditional",
        "Karaoke - Since I Fell For You - Charlie Rich", 
        "Karaoke - Amigos Para Siempre(Friends For Life) - Sarah Brightman & Jose Carreras"
    ]
    
    for title in test_titles:
        print(f"Testing: '{title}'")
        print("=" * 80)
        
        # Call the extraction method directly
        result = processor._extract_artist_song_info(title, "", "")
        
        print("Extraction result:")
        for key, value in result.items():
            print(f"  {key}: '{value}' (length: {len(str(value)) if value else 'None'})")
        
        # Let's trace through the pattern matching manually
        print("\nManual pattern testing:")
        
        # Get the actual patterns from the processor
        karaoke_patterns = [
            # From processor.py - the problematic pattern
            (
                r"^([^-–—]+?)\s*[-–—]\s*([^(\[]+)(?:\s*[\(\[][^)\]]*[Kk]araoke[^)\]]*[\)\]])?",
                2,  # artist_group  
                1,  # title_group
                0.7,
            )
        ]
        
        clean_title = processor._clean_title_for_parsing(title)
        print(f"Clean title: '{clean_title}'")
        
        for pattern, artist_group, title_group, confidence in karaoke_patterns:
            import re
            match = re.search(pattern, clean_title, re.IGNORECASE | re.UNICODE)
            if match:
                print(f"Pattern matched!")
                print(f"  Groups: {match.groups()}")
                print(f"  artist_group={artist_group}, title_group={title_group}")
                
                if artist_group and artist_group <= len(match.groups()):
                    raw_artist = match.group(artist_group)
                    cleaned_artist = processor._clean_extracted_text(raw_artist)
                    valid_artist = processor._is_valid_artist_name(cleaned_artist)
                    print(f"  Artist: raw='{raw_artist}' -> cleaned='{cleaned_artist}' -> valid={valid_artist}")
                
                if title_group and title_group <= len(match.groups()):
                    raw_title = match.group(title_group)
                    cleaned_title = processor._clean_extracted_text(raw_title) 
                    valid_title = processor._is_valid_song_title(cleaned_title)
                    print(f"  Title: raw='{raw_title}' -> cleaned='{cleaned_title}' -> valid={valid_title}")
                    
                    # Check if _is_valid_song_title is rejecting it
                    if not valid_title:
                        print(f"  *** TITLE REJECTED by validation! ***")
                        print(f"      Reason: ")
                        if not cleaned_title or len(cleaned_title.strip()) < 2:
                            print(f"        Too short (len={len(cleaned_title.strip())})")
                        title_lower = cleaned_title.lower().strip()
                        invalid_terms = {"karaoke", "instrumental", "backing track", "minus one", "mr", "inst"}
                        if title_lower in invalid_terms:
                            print(f"        Contains invalid term: '{title_lower}'")
                        if len(cleaned_title.strip()) > 200:
                            print(f"        Too long (len={len(cleaned_title.strip())})")
                break
        
        print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    test_real_extraction()