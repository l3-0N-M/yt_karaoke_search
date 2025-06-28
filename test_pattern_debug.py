#!/usr/bin/env python3
"""Debug the pattern issue to understand what's really happening."""

import re

# The pattern from advanced_parser.py
pattern = r"^[Kk]araoke\s*[-–—]\s*([^-]+?)\s*[-–—]\s*(.+)$"
artist_group = 2
title_group = 1

# Test titles from the database
test_titles = [
    "Karaoke Gib mir sonne - Rosenstolz",
    "Karaoke Alles Rot - Silly",
    "Karaoke Von allein - Culcha Candela",
]

print("Testing the karaoke pattern:")
print(f"Pattern: {pattern}")
print(f"Artist group: {artist_group}, Title group: {title_group}")
print("=" * 80)

for title in test_titles:
    print(f"\nTitle: '{title}'")
    match = re.search(pattern, title, re.IGNORECASE | re.UNICODE)
    
    if match:
        print(f"MATCH! Groups: {match.groups()}")
        print(f"Group 1: '{match.group(1)}'")
        print(f"Group 2: '{match.group(2)}'")
        
        # Using the pattern's group assignments
        artist = match.group(artist_group) if artist_group <= len(match.groups()) else None
        song_title = match.group(title_group) if title_group <= len(match.groups()) else None
        
        print("Pattern extracts as:")
        print(f"  original_artist: '{artist}'")
        print(f"  song_title: '{song_title}'")
        print()
        print("Expected (correct) assignment:")
        print(f"  original_artist: '{match.group(2)}' (Group 2)")
        print(f"  song_title: '{match.group(1)}' (Group 1)")
    else:
        print("No match")