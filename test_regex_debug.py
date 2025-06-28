#!/usr/bin/env python3
"""Debug the regex pattern to understand why it's not matching."""

import re

# Test titles from the database
test_titles = [
    "Karaoke Gib mir sonne - Rosenstolz",
    "Karaoke Alles Rot - Silly", 
    "Karaoke Von allein - Culcha Candela",
]

# Different pattern variations to test
patterns = [
    # Original pattern
    (r"^[Kk]araoke\s*[-–—]\s*([^-]+?)\s*[-–—]\s*(.+)$", "Original"),
    
    # Simpler version
    (r"^[Kk]araoke\s+(.+?)\s*-\s*(.+)$", "Simple dash"),
    
    # More flexible
    (r"^[Kk]araoke\s+(.+?)\s*[-–—]\s*(.+)$", "Flexible dash"),
    
    # With word boundaries
    (r"^[Kk]araoke\s+(.+?)\s+[-–—]\s+(.+)$", "Word boundaries"),
    
    # Check what character is actually there
    (r"^[Kk]araoke\s+(.+)$", "Just capture everything after Karaoke"),
]

for title in test_titles:
    print(f"\nTitle: '{title}'")
    print(f"Characters: {[ord(c) for c in title]}")
    
    for pattern, name in patterns:
        match = re.search(pattern, title, re.IGNORECASE | re.UNICODE)
        if match:
            print(f"  ✓ {name}: {match.groups()}")
        else:
            print(f"  ✗ {name}: No match")
    
    # Check what separator is actually used
    dash_match = re.search(r"Karaoke\s+(.+?)\s+([-–—])\s+(.+)", title)
    if dash_match:
        print(f"  Separator found: '{dash_match.group(2)}' (ord: {ord(dash_match.group(2))})")
    else:
        print("  No separator pattern found")