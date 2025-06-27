#!/usr/bin/env python3
"""Debug script to test artist/song extraction."""

import re
import sys
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from collector.config import CollectorConfig
from collector.processor import VideoProcessor


def debug_extraction():
    """Debug the artist/song extraction issue."""
    config = CollectorConfig()
    processor = VideoProcessor(config)

    test_title = "Artist Name - Song Title"
    print(f"Testing title: '{test_title}'")
    print("=" * 50)

    # Call the extraction method
    result = processor._extract_artist_song_info(test_title, "", "")

    print("Extraction result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    print()

    # Let's also test the individual patterns manually
    print("Testing individual patterns:")

    # Get the patterns from the processor (this is a copy of the relevant patterns)
    karaoke_patterns = [
        # Pattern: Title - Artist (with various separators and optional karaoke indicator)
        (
            r"^([^-–—]+?)\s*[-–—]\s*([^(\[]+?)(?:\s*[\(\[][^)\]]*[Kk]araoke[^)\]]*[\)\]])?",
            2,  # artist_group
            1,  # title_group
            0.7,
        ),
        # Pattern: Artist - Title (Karaoke Version)
        (r"^([^-]+?)\s*-\s*([^(]+?)\s*\([^)]*[Kk]araoke[^)]*\)", 1, 2, 0.85),
    ]

    clean_title = processor._clean_title_for_parsing(test_title)
    print(f"Clean title: '{clean_title}'")
    print()

    for i, (pattern, artist_group, title_group, confidence) in enumerate(karaoke_patterns):
        print(f"Pattern {i+1}: {pattern}")
        print(f"  Artist group: {artist_group}, Title group: {title_group}")

        match = re.search(pattern, clean_title, re.IGNORECASE | re.UNICODE)
        if match:
            print(f"  MATCH! Groups: {match.groups()}")
            if artist_group and artist_group <= len(match.groups()):
                raw_artist = match.group(artist_group)
                cleaned_artist = processor._clean_extracted_text(raw_artist)
                valid_artist = processor._is_valid_artist_name(cleaned_artist)
                print(f"  Raw artist (group {artist_group}): '{raw_artist}'")
                print(f"  Cleaned artist: '{cleaned_artist}'")
                print(f"  Valid artist: {valid_artist}")

            if title_group and title_group <= len(match.groups()):
                raw_title = match.group(title_group)
                cleaned_title = processor._clean_extracted_text(raw_title)
                valid_title = processor._is_valid_song_title(cleaned_title)
                print(f"  Raw title (group {title_group}): '{raw_title}'")
                print(f"  Cleaned title: '{cleaned_title}'")
                print(f"  Valid title: {valid_title}")
            break
        else:
            print("  No match")
        print()


if __name__ == "__main__":
    debug_extraction()
